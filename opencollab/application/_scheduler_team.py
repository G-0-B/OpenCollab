"""Scheduler team views, topology checks, prebuilt roster, diffs, and events."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, NamedTuple

from opencollab.application._scheduler_constants import (
    SPAWN_REFUSAL_TASK_CHARS,
    WORKTREE_DIFF_KEEP_CHARS,
    WORKTREE_DIFF_MAX_CHARS,
)
from opencollab.application.async_timeout import await_owned_operation
from opencollab.application.ports import (
    CapturedChangesDiffPort,
    DiffCapablePort,
    EnvironmentPort,
    RecoverableChangesPort,
)
from opencollab.application.scheduler_types import TeamPrebuiltError
from opencollab.domain.events import SchedulerEvent
from opencollab.domain.identity import role_collision_key
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase

logger = logging.getLogger(__name__)

_DIFF_HEADER = "diff --git "
_BODY_MARKERS = ("@@", "GIT binary patch", "Binary files ")

class _CompletedCoderChanges(NamedTuple):
    parent_turn: float | None
    step_count: int
    patch: str

def _shell_state(agent: Any, env: Any) -> str:
    """Classify effective shell access for the run record.

    Return ``absent`` when the role lacks ``bash``, ``sandbox_required`` when
    its required OS sandbox is unavailable, and ``available`` otherwise.
    """
    bash = next(
        (
            tool
            for tool in getattr(agent, "tools", ())
            if getattr(tool, "name", None) == "bash"
        ),
        None,
    )
    if bash is None:
        return "absent"
    if not getattr(bash, "require_process_isolation", False):
        return "available"
    if getattr(env, "process_isolated", False):
        return "available"
    return "sandbox_required"

def _git_unquote(token: str) -> str:
    """Undo the C-style quoting git applies to paths with unusual bytes.

    ``"b/\\344\\270\\255.txt"`` is how git writes a UTF-8 path; the escapes are
    per *byte*, so the round trip decodes the escapes to latin-1 code points,
    packs those back into bytes, and decodes the bytes as UTF-8. Anything that
    does not survive the trip is returned untouched — a mangled path is a
    weaker record than a quoted one, but neither may raise.
    """
    if len(token) < 2 or not token.startswith('"') or not token.endswith('"'):
        return token
    try:
        raw = token[1:-1].encode("utf-8").decode("unicode_escape").encode("latin-1")
        return raw.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return token

def _side_path(token: str, prefix: str) -> str | None:
    """One side of a file header (``--- a/x`` / ``+++ b/x``), or None for /dev/null.

    Git appends a tab after the path when the path contains a space, so the
    tab is stripped before anything else looks at the token.
    """
    token = _git_unquote(token.split("\t", 1)[0])
    if token == "/dev/null":
        return None
    return token[len(prefix):] if token.startswith(prefix) else token

def _header_line_paths(line: str) -> tuple[str | None, str | None]:
    """The (old, new) paths off a ``diff --git a/X b/Y`` line.

    Only needed where git omits the ``---``/``+++`` pair — binary payloads and
    mode-only changes. Unquoted paths containing " b/" are genuinely ambiguous
    on this line; the same-path split is tried first because that is the shape
    of every non-rename entry, and a rename carries explicit ``rename
    from``/``rename to`` lines anyway.
    """
    rest = line[len(_DIFF_HEADER):]
    if rest.startswith('"'):
        closing = rest.find('" "')
        if closing == -1:
            return None, None
        return (
            _side_path(rest[: closing + 1], "a/"),
            _side_path(rest[closing + 2:], "b/"),
        )
    if not rest.startswith("a/"):
        return None, None
    body = rest[2:]
    half = (len(body) - 3) // 2
    if half > 0 and body[half: half + 3] == " b/" and body[:half] == body[half + 3:]:
        return body[:half], body[:half]
    marker = body.find(" b/")
    if marker == -1:
        return None, None
    return body[:marker], body[marker + 3:]

def _classify_block(block: list[str]) -> list[tuple[str, str]]:
    """Turn one file's diff header into ``(path, op)`` entries."""
    old_path = new_path = rename_from = rename_to = None
    added = deleted = False
    for line in block[1:]:
        if line.startswith(_BODY_MARKERS):
            break
        if line.startswith("new file mode"):
            added = True
        elif line.startswith("deleted file mode"):
            deleted = True
        elif line.startswith("rename from "):
            rename_from = _git_unquote(line[len("rename from "):])
        elif line.startswith("rename to "):
            rename_to = _git_unquote(line[len("rename to "):])
        elif line.startswith("--- "):
            old_path = _side_path(line[4:], "a/")
        elif line.startswith("+++ "):
            new_path = _side_path(line[4:], "b/")
    if rename_from and rename_to:
        # A rename is reported as its two endpoints so ``op`` stays inside the
        # added/modified/deleted vocabulary the record documents.
        return [(rename_from, "deleted"), (rename_to, "added")]
    fallback_old, fallback_new = _header_line_paths(block[0])
    if deleted:
        path = old_path or fallback_old
        return [(path, "deleted")] if path else []
    path = new_path or fallback_new or old_path or fallback_old
    if not path:
        return []
    return [(path, "added" if added else "modified")]

def _parse_worktree_diff(diff: str) -> list[tuple[str, str]]:
    """Every file an *untruncated* worktree diff touches, in diff order.

    Splitting on ``diff --git `` at column zero is safe inside hunks: a body
    line is always prefixed by ``+``, ``-``, or a space.
    """
    blocks: list[list[str]] = []
    for line in diff.split("\n"):
        if line.startswith(_DIFF_HEADER):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    return [entry for block in blocks for entry in _classify_block(block)]

class SchedulerTeamMixin:
    @property
    def turns_serialized(self) -> bool:
        """Whether teammate turns must wait for the current turn to end."""
        return self._serialize_turns

    @property
    def lead_session(self) -> Any:
        """Agent 0's session (the interactive entry)."""
        return self._lead_session

    def agent_environment(self, aid: int) -> EnvironmentPort | None:
        """Return a registered agent's usable execution environment."""
        if self._shutting_down or isinstance(aid, bool) or not isinstance(aid, int):
            return None
        environment = getattr(self._sessions.get(aid), "env", None)
        if environment is None or getattr(environment, "revoked", False):
            return None
        return environment

    def _role_of(self, aid: int) -> str:
        scb = self.table.get(aid)
        return scb.agent.name if scb is not None else "?"

    def _topology_forbids(self, src_role: str, dst_role: str) -> bool:
        """Whether the configured topology forbids ``src_role`` from reaching ``dst_role``.

        Accept role names because spawn targets have no aid yet.
        """
        return self._topology is not None and not self._topology.allows(src_role, dst_role)

    def _check_topology(self, src_aid: int, dst_role: str, *, verb: str) -> None:
        """Raise ``PermissionError`` if the topology forbids src → dst_role."""
        src_role = self._role_of(src_aid)
        if self._topology_forbids(src_role, dst_role):
            raise PermissionError(f"Role '{src_role}' is not permitted to {verb} '{dst_role}' under the team topology.")

    def team_snapshot(self) -> list[dict[str, Any]]:
        """Read-only roster of every tracked (live) agent, ordered by aid."""
        snapshot: list[dict[str, Any]] = []
        for aid in sorted(self.table.entries):
            scb = self.table.entries[aid]
            task = self._tasks.get(aid)
            snapshot.append(
                {
                    "aid": aid,
                    "role": scb.agent.name,
                    "parent_aid": scb.parent_aid,
                    "phase": scb.state.phase.value,
                    "busy": task is not None and not task.done(),
                }
            )
        return snapshot

    def team_status_rows(self) -> list[dict[str, Any]]:
        """Add turn-lock wait state to the live roster shown by team_status."""
        return [
            {**entry, "turn_queued": entry["aid"] in self._turn_waiters}
            for entry in self.team_snapshot()
        ]

    def team_roster(self) -> list[dict[str, Any]]:
        """Full configured team for the prompt toolbar: every live agent plus
        each configured role that has no live agent yet (``aid=None``, phase
        ``"available"``). Unlike ``team_snapshot`` (live agents only, used to
        message teammates by aid), this surfaces the team the user defined in
        the team config before anything has spawned.
        """
        live = self.team_snapshot()
        live_roles = {entry["role"] for entry in live}
        available = [
            {
                "aid": None,
                "role": role,
                "parent_aid": None,
                "phase": "available",
                "busy": False,
            }
            for role in self._roles
            if role not in live_roles
        ]
        return live + available

    async def ensure_team_prebuilt(self) -> tuple[int, ...]:
        """Seat declared teammates before the first turn and return their aids.

        With ``prebuild_team`` disabled, return ``()``. Otherwise leave agent 0 in
        place and build the remaining roles in declaration order through the normal
        worktree and session factories. Tasks arrive later through teammate messages.

        A construction failure rolls back the partial team and raises before model
        execution. Seating reserves no shared tokens. Each seat's later calls are
        bounded by its per-agent cap.
        """
        if not self._prebuild_team or self._team_prebuilt:
            return ()
        if self._shutting_down:
            raise RuntimeError("Cannot prebuild the team: scheduler is shutting down.")
        if self._prebuild_lock is None:
            self._prebuild_lock = asyncio.Lock()
        async with self._prebuild_lock:
            if self._shutting_down:
                raise RuntimeError("Cannot prebuild the team: scheduler is shutting down.")
            if self._team_prebuilt:
                return ()
            lead = self.table.get(0)
            if lead is None:
                raise RuntimeError(
                    "Cannot prebuild the team: agent 0 does not exist yet. "
                    "Call create_init_process() first."
                )
            entry_key = role_collision_key(lead.agent.name)
            built: list[tuple[int, Any]] = []
            try:
                for role in self._roles:
                    if role_collision_key(role) == entry_key:
                        continue
                    built.append(await self._prebuild_peer(role))
            except BaseException:
                for aid, env in reversed(built):
                    await self._rollback_failed_spawn(aid, env)
                raise
            self._team_prebuilt = True
            self._trace_assigned_topology()
            self._write_manifest()
            return tuple(aid for aid, _ in built)

    async def _prebuild_peer(self, role: str) -> tuple[int, Any]:
        """Create one declared teammate and return ``(aid, env)`` for rollback.

        Use ``parent_aid=0`` so hooks report ``SubagentStop`` rather than root
        ``Stop``. Leave the peer out of ``_spawn_origin`` so completion cannot
        join a waiting parent. It receives tasks through teammate messages.
        """
        aid = self.table.allocate_aid()
        env: Any | None = None
        try:
            budget = self._reserve_child_budget(aid, role)
            if budget <= 0:
                raise RuntimeError(
                    f"Cannot prebuild role '{role}': a team token budget of "
                    f"{self._max_budget_tokens} leaves nothing for it."
                )
            env = await self._worktree_pool.acquire(role)
            if self._shutting_down:
                raise RuntimeError("Cannot prebuild the team: scheduler is shutting down.")
            session = self._session_factory.build_spawn_session(
                role=role,
                env=env,
                budget=budget,
                aid=aid,
                scheduler=self,
                task=None,
                context="",
            )
            session.agent.name = role
            self.table.add(
                SessionControlBlock(
                    aid=aid,
                    parent_aid=0,
                    agent=session.agent,
                    state=session.state,
                )
            )
            self._sessions[aid] = session
            await self.emit_scheduler_event(
                self._events.agent_spawned(aid, 0, role, "")
            )
            if self._shutting_down:
                raise RuntimeError("Cannot prebuild the team: scheduler is shutting down.")
        except BaseException:
            await self._rollback_failed_spawn(aid, env)
            raise
        return aid, env

    def _assigned_topology_nodes(self) -> list[dict[str, Any]]:
        """One record per seated agent: who it is and what it may do."""
        entry_aid = 0
        nodes: list[dict[str, Any]] = []
        for aid in sorted(self.table.entries):
            scb = self.table.entries[aid]
            session = self._sessions.get(aid)
            agent = getattr(session, "agent", None) or scb.agent
            tools = sorted(
                str(name)
                for tool in getattr(agent, "tools", ())
                if (name := getattr(tool, "name", None)) is not None
            )
            env = getattr(session, "env", None)
            nodes.append(
                {
                    "aid": aid,
                    "role": scb.agent.name,
                    "entry": aid == entry_aid,
                    "model": getattr(agent, "model", None),
                    "tools": tools,
                    # "confirm" means an external permission policy is wired for risky actions.
                    "permission_mode": (
                        "confirm"
                        if getattr(session, "permission_policy", None) is not None
                        else "auto"
                    ),
                    "workspace": getattr(env, "workspace", None),
                    # Shared-directory runs report False even when every agent has an environment.
                    "workspace_isolated": isinstance(env, DiffCapablePort),
                    # Record actual shell capability per node so comparisons can be checked from the run.
                    "shell": _shell_state(agent, env),
                    "token_allowance": self._agent_cap(aid, scb.agent.name),
                }
            )
        return nodes

    def _assigned_topology_edges(self) -> list[dict[str, str]]:
        """Every edge the team config declares, as ``(from_role, to_role)`` pairs.

        Read from the same ``Topology`` object ``_topology_forbids`` consults, so
        the recorded edge set is the one actually enforced. Roles declared with
        no outgoing edge contribute no pair — they appear in the node record and
        in ``roles`` instead.
        """
        topology = self._topology
        if topology is None:
            return []
        return sorted(
            (
                {"from_role": source, "to_role": destination}
                for source, destinations in topology.edges.items()
                for destination in destinations
            ),
            key=lambda edge: (edge["from_role"], edge["to_role"]),
        )

    def _trace_assigned_topology(self) -> None:
        """Record the prebuilt roster and declared communication edges once.

        ``assigned.topology_nodes`` captures tools, permissions, workspace, shell
        access and ``turns_serialized``. ``assigned.topology_edges`` retains the
        configured edges and ``allow_all``, which distinguishes an open topology
        from an empty closed graph.

        Recording is disabled without prebuilding, and trace failures leave the run
        unaffected.
        """
        tracer = self._tracer
        if tracer is None:
            return
        lead = self.table.get(0)
        topology = self._topology
        try:
            tracer.log_step(
                step_type="assigned.topology_nodes",
                payload={
                    "entry_role": lead.agent.name if lead is not None else None,
                    "declared_roles": list(self._roles),
                    # Record serialization as part of the run configuration.
                    "turns_serialized": bool(self._serialize_turns),
                    "budget_source": "team_file" if self._role_budgets else "shared_rule",
                    "nodes": self._assigned_topology_nodes(),
                },
            )
            tracer.log_step(
                step_type="assigned.topology_edges",
                payload={
                    "allow_all": bool(topology is not None and topology.allow_all),
                    "declared_roles": list(self._roles),
                    "edges": self._assigned_topology_edges(),
                },
            )
        except Exception as exc:
            logger.error("assigned topology trace failed: %s", exc)

    def _live_roster(self) -> list[dict[str, Any]]:
        return [
            {"aid": aid, "role": self.table.entries[aid].agent.name}
            for aid in sorted(self.table.entries)
        ]

    def _refuse_spawn_when_prebuilt(
        self,
        parent_aid: int,
        role: str,
        task: str,
        context: str,
    ) -> None:
        """Record and raise a roster-bearing spawn refusal for prebuilt teams.

        ``spawn_agent`` stays available so the trace captures attempted delegation.
        With ``prebuild_team=False``, return immediately.
        """
        if not self._prebuild_team:
            return
        roster = self._live_roster()
        declared = {role_collision_key(name) for name in self._roles}
        requested_key = role_collision_key(role)
        seated = next(
            (
                entry
                for entry in roster
                if role_collision_key(entry["role"]) == requested_key
            ),
            None,
        )
        self._trace_spawn_refused(
            parent_aid=parent_aid,
            role=role,
            task=task,
            context=context,
            roster=roster,
            declared=requested_key in declared,
        )
        listing = ", ".join(f"{entry['role']} (aid {entry['aid']})" for entry in roster)
        head = (
            f"Not spawned: '{role}' is already on this team as aid "
            f"{seated['aid']}."
            if seated is not None
            else f"Not spawned: '{role}' is not a role on this team."
        )
        raise TeamPrebuiltError(
            f"{head} This team's roster is fixed before the run starts and no "
            f"agent can be added to it. Your teammates are already running: "
            f"{listing}. Send one of them this work with send_message instead — "
            f"nothing about your request failed, only the way you asked for it.",
            roster=tuple((entry["aid"], entry["role"]) for entry in roster),
        )

    def _trace_spawn_refused(
        self,
        *,
        parent_aid: int,
        role: str,
        task: str,
        context: str,
        roster: list[dict[str, Any]],
        declared: bool,
    ) -> None:
        """Record a refused spawn with role and topology attribution.

        ``requested_role_declared`` identifies requests outside the declared roster.
        ``topology_allowed`` distinguishes fixed-roster refusal from a forbidden
        edge. Capped task text retains its full length for counting.

        Trace failures leave the refusal unchanged.
        """
        tracer = self._tracer
        if tracer is None:
            return
        try:
            tracer.log_step(
                step_type="spawn_refused",
                payload={
                    "reason": "team_prebuilt",
                    "requester_aid": parent_aid,
                    "requester_role": self._role_of(parent_aid),
                    "requested_role": role,
                    "requested_role_declared": declared,
                    "topology_allowed": not self._topology_forbids(
                        self._role_of(parent_aid), role
                    ),
                    "declared_roles": list(self._roles),
                    "live_roster": roster,
                    "task": str(task)[:SPAWN_REFUSAL_TASK_CHARS],
                    "task_chars": len(task or ""),
                    "context_chars": len(context or ""),
                },
            )
        except Exception as exc:
            logger.error("spawn_refused trace failed for aid %s: %s", parent_aid, exc)

    def agent_step_count(self, aid: int) -> int:
        """Return the cumulative step count for one registered agent."""
        session = self._sessions.get(aid)
        if session is None or self.table.get(aid) is None:
            raise ValueError(f"No agent with aid {aid}.")
        return int(getattr(session, "step_count", session.state.step_count))

    async def _append_worktree_diff(
        self, env: EnvironmentPort, result: str, *, aid: int | None = None, role: str = "",
        partial: bool = False,
    ) -> str:
        """Capture worktree changes once for the result and its trace."""
        diff = await env.get_diff() if isinstance(env, DiffCapablePort) else ""
        scb = self.table.get(aid) if aid is not None else None
        if (
            not partial and scb is not None and scb.parent_aid == 0
            and role_collision_key(scb.agent.name) == "coder"
            and scb.state.phase is SessionPhase.DONE
            and self._can_adopt_agent(0)
        ):
            parent_env = getattr(self._sessions.get(0), "env", None)
            shared = (
                parent_env is not None
                and getattr(parent_env, "workspace", None) == getattr(env, "workspace", None)
            )
            self._completed_coder_diffs[aid] = _CompletedCoderChanges(
                self._turn_started_at.get(0), scb.state.step_count, "" if shared else diff,
            )
            result += f"\n\n[Completed coder aid={aid}; adopt_agent can apply these changes]"
        if aid is not None:
            await self._trace_worktree_changes(aid, role, env, diff=diff)
        if not diff:
            return result
        recovery = ""
        if len(diff) > WORKTREE_DIFF_MAX_CHARS:
            if partial and isinstance(env, RecoverableChangesPort):
                recovery = f"\n\n[Full partial changes retained at {env.retain_changes()}]"
            diff = (
                diff[:WORKTREE_DIFF_KEEP_CHARS]
                + f"\n\n... [{len(diff) - WORKTREE_DIFF_MAX_CHARS} chars truncated] ...\n\n"
                + diff[-WORKTREE_DIFF_KEEP_CHARS:]
            )
        return result + f"\n\n[Changes made in worktree]\n```diff\n{diff}\n```" + recovery

    def _can_adopt_agent(self, aid: int) -> bool:
        scb = self.table.get(aid)
        return scb is not None and any(
            getattr(tool, "name", None) == "adopt_agent" for tool in getattr(scb.agent, "tools", ())
        )

    async def adopt_agent(self, from_aid: int, aid: int) -> str:
        """Adopt only this entry turn's completed direct coding child."""
        if self._shutting_down or from_aid != 0 or not self._can_adopt_agent(from_aid):
            raise PermissionError("adopt_agent requires the entry agent's declared capability")
        if isinstance(aid, bool) or not isinstance(aid, int):
            raise ValueError("adopt_agent requires an integer coder aid")
        async with self._adoption_lock:
            child = self.table.get(aid)
            captured = self._completed_coder_diffs.get(aid)
            if (
                child is None or child.parent_aid != from_aid
                or role_collision_key(child.agent.name) != "coder"
                or child.state.phase is not SessionPhase.DONE
                or captured is None or captured.parent_turn != self._turn_started_at.get(from_aid)
            ):
                raise ValueError("adopt_agent requires a successfully completed direct coder in this turn")
            adopted = self._adopted_coders.get(aid)
            if adopted == captured:
                await self._release_adopted_retention(aid, captured.patch)
                return f"Coder aid={aid} was already adopted."

            async def apply_completed_changes() -> None:
                patch = captured.patch
                try:
                    if adopted is not None:
                        if adopted.patch == captured.patch:
                            patch = ""
                        else:
                            env = getattr(self._sessions.get(aid), "env", None)
                            if not isinstance(env, CapturedChangesDiffPort):
                                raise RuntimeError(
                                    "coder environment cannot calculate an increment between completed changes"
                                )
                            patch = await env.diff_captured(adopted.patch, captured.patch)
                    if patch.strip():
                        await self._worktree_pool.adopt(patch)
                except Exception as failure:
                    location = await self._preserve_adoption_candidate(aid, patch)
                    raise RuntimeError(f"{failure}; full candidate retained at {location}") from failure
                self._adopted_coders[aid] = captured
                await self._release_adopted_retention(aid, captured.patch)

            await await_owned_operation(apply_completed_changes(), propagate_cancellation=True)
            child_env = getattr(self._sessions.get(aid), "env", None)
            parent_env = getattr(self._sessions.get(from_aid), "env", None)
            if (
                child_env is not None
                and getattr(child_env, "workspace", None) == getattr(parent_env, "workspace", None)
            ):
                return f"Coder aid={aid} changes are already present in the shared delivery workspace."
            return f"Adopted coder aid={aid}. Spawn a fresh tester to verify the delivery workspace."

    async def _preserve_adoption_candidate(self, aid: int, patch: str) -> str:
        if self._candidate_diff_writer is not None:
            try:
                return await self._candidate_diff_writer(aid, self._role_of(aid), patch)
            except Exception:
                logger.warning("candidate patch persistence failed for aid %s", aid, exc_info=True)
        env = getattr(self._sessions.get(aid), "env", None)
        if isinstance(env, RecoverableChangesPort):
            return env.retain_changes()
        raise RuntimeError("completed candidate recovery environment is unavailable")

    async def _release_adopted_retention(self, aid: int, patch: str) -> None:
        env = getattr(self._sessions.get(aid), "env", None)
        if (
            isinstance(env, RecoverableChangesPort) and isinstance(env, DiffCapablePort)
            and env.recovery_location is not None
        ):
            current = await env.get_diff()
            if current != patch:
                env.retain_changes()

    async def _trace_worktree_changes(
        self, aid: int, role: str, env: EnvironmentPort, *, diff: str,
    ) -> None:
        """Record per-file evidence from an agent's untruncated worktree diff.

        ``content_sha`` hashes resulting file content and is ``None`` for deletions
        or unreadable files. The row also records the whole diff's hash, length and
        whether the parent's prose copy was truncated. An unchanged tree still emits
        an empty file list.

        ``diff_base`` moves from the creation base to an adopted commit after a
        handoff. ``head_commit`` and ``commits`` identify the agent's own commits,
        allowing ``tester.diff_base in coder.commits`` to link adoption. The capped
        list has an uncapped ``commit_count``, or ``None`` when it cannot be read.

        Individual reads and the complete record are guarded so tracing failures
        leave execution unaffected. Required diff delivery is handled separately by
        ``_append_worktree_diff``.
        """
        tracer = self._tracer
        if tracer is None or not isinstance(env, DiffCapablePort):
            return
        try:
            files: list[dict[str, Any]] = []
            path_prefix = getattr(env, "diff_path_prefix", "")
            for path, op in _parse_worktree_diff(diff):
                entry: dict[str, Any] = {"path": path, "op": op, "content_sha": None}
                if op != "deleted":
                    try:
                        read_path = path
                        if path_prefix:
                            prefix = path_prefix + "/"
                            if not path.startswith(prefix):
                                raise PermissionError("diff file is outside the execution workspace")
                            read_path = path[len(prefix):]
                        content = await env.read_file(read_path)
                        entry["content_sha"] = hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest()
                    except Exception as exc:
                        entry["sha_error"] = type(exc).__name__
                files.append(entry)
            tracer.log_step(
                step_type="worktree_changes",
                payload={
                    "aid": aid,
                    "role": role,
                    # A handoff changes the diff base to the adopted commit rather than the creation base.
                    "diff_base": getattr(env, "diff_base", None),
                    # Record resulting commits so later teammate checkouts can be joined to this handoff.
                    "head_commit": getattr(env, "head_commit", None),
                    "commits": list(getattr(env, "own_commits", ()) or ()),
                    "commit_count": getattr(env, "own_commit_count", None),
                    "files": files,
                    "diff_sha": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
                    "diff_chars": len(diff),
                    "truncated_in_result": len(diff) > WORKTREE_DIFF_MAX_CHARS,
                },
            )
        except Exception as exc:
            logger.error("worktree_changes trace failed for aid %s: %s", aid, exc)

    async def emit_scheduler_event(self, event: SchedulerEvent) -> None:
        """Emit a scheduler event through the configured sink."""
        await self._event_sink.emit(event)

    async def _safe_emit_scheduler_event(self, event: SchedulerEvent) -> None:
        """Emit an observational lifecycle event without changing scheduler state."""
        try:
            await self.emit_scheduler_event(event)
        except Exception as exc:
            logger.error("scheduler event %s failed: %s", event.type, exc)
