"""Compose CLI/SDK team dependencies and initialize agent 0."""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from opencollab.adapters.env import Environment, LocalEnvironment
from opencollab.adapters.hooks import ShellHookRunner
from opencollab.adapters.safe_files import write_regular_bytes_atomic
from opencollab.adapters.storage import SessionStore
from opencollab.adapters.working_tree import EnvWorkingTreeProbe
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.application.event_bus import EventBus
from opencollab.application.hooks import HookEventSubscriber
from opencollab.application.scheduler import LaunchSpec, Scheduler
from opencollab.bootstrap.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING,
    DEFAULT_TOP_P,
    resolve_thinking_params,
)
from opencollab.bootstrap.container import RuntimeContext, build_workspace_safety_policy
from opencollab.bootstrap.context_builder import SpawnConfig
from opencollab.bootstrap.session_factory import (
    SESSION_MAX_STEPS,
    DefaultSessionFactory,
    agent_save_path,
    make_run_dir,
)
from opencollab.bootstrap.team_config import (
    TeamConfig,
    load_team_config,
    resolve_team_file,
)
from opencollab.domain.agent import DEFAULT_MAX_TOKENS_PER_STEP

# Prebuilt peers need this tool to use their declared communication edges.
_MESSAGING_TOOL_NAME = "message_agent"


def _reject_unwalkable_edges(team: TeamConfig) -> None:
    """Reject prebuilt topology edges whose source lacks ``message_agent``.

    Prebuilt teams refuse spawning, so these edges cannot be used. Fail before
    creating agent 0; otherwise a wiring error could look like a model declining
    to collaborate. Dynamic teams may legitimately use spawn/join without messaging.
    """
    offenders: list[tuple[str, list[str], list[str]]] = []
    for source in sorted(team.topology.edges):
        destinations = sorted(team.topology.edges[source])
        role = team.roles.get(source)
        if not destinations or role is None:
            continue
        if _MESSAGING_TOOL_NAME in role.tools:
            continue
        offenders.append((source, destinations, sorted(role.tools)))
    if not offenders:
        return
    detail = "\n".join(
        f"  - role '{source}' may address {', '.join(destinations)} "
        f"but its tools are [{', '.join(tools)}]\n"
        f"    unwalkable edges: "
        + ", ".join(f"{source} -> {destination}" for destination in destinations)
        for source, destinations, tools in offenders
    )
    named = ", ".join(f"'{source}'" for source, _, _ in offenders)
    raise ValueError(
        "prebuild_team: this team declares edges that no agent can walk.\n"
        f"{detail}\n"
        f"A prebuilt team refuses spawn_agent, so '{_MESSAGING_TOOL_NAME}' is "
        "the only channel between seated agents. Add "
        f"'{_MESSAGING_TOOL_NAME}' to the `tools:` list of {named}, or remove "
        "those roles' entries from `topology:`."
    )



def build_scheduler(
    ctx: RuntimeContext,
    *,
    use_worktrees: bool,
    interactive: bool,
    session_file: str | None = None,
    auto_save: bool = True,
    enable_hooks: bool = True,
    team_config_path: str | os.PathLike[str] | None = None,
    resolved_team_config: TeamConfig | None = None,
    save_dir: str | os.PathLike[str] | None = None,
    prebuild_team: bool = False,
    allow_unisolated_shell: bool | None = None,
    allow_unisolated_child_shell: bool = False,
    max_steps: int | None = SESSION_MAX_STEPS,
    serialize_turns: bool = False,
    environment: Environment | None = None,
    record_delivery_tree: bool = False,
    run_id: str | None = None,
) -> Scheduler:
    """Compose a team and initialize agent 0 with the selected configuration.

    ``session_file`` resumes entry history. ``auto_save`` writes per-agent JSON
    and a team manifest under ``save_dir`` or a timestamped workspace session
    folder. ``enable_hooks`` permits configured lifecycle shell commands.

    ``prebuild_team`` seats the complete roster before execution and refuses
    later spawning. ``serialize_turns`` prevents concurrent turns without changing
    topology permissions or requiring agents to communicate. ``max_steps`` bounds
    each seat. A supplied ``environment`` hosts agent 0 and provides the source
    for isolated teammates; otherwise the source is the host workspace.

    ``interactive`` controls access to a human. ``allow_unisolated_shell`` controls
    host commands for entry/prebuilt seats; None inherits ``interactive``.
    ``allow_unisolated_child_shell`` separately authorizes dynamically spawned
    children and defaults to False. A worktree is not a process sandbox;
    command confirmation still uses the permission policy.

    ``record_delivery_tree`` observes agent 0's delivered workspace at seat
    boundaries. It is off by default to avoid otherwise unnecessary diff probes.
    """
    if session_file is not None and not SessionStore().has_snapshot(session_file):
        raise ValueError(f"session file does not exist: {session_file}")

    cfg = ctx.config
    team_cfg = (
        resolved_team_config
        if resolved_team_config is not None
        else load_team_config(ctx.workspace, path=team_config_path)
    )
    if prebuild_team:
        _reject_unwalkable_edges(team_cfg)
    event_bus = EventBus(ctx.event_sink)

    # Per-run folder: every agent's transcript plus a team.json manifest land
    # here. Known before the factory so spawned children inherit the same dir.
    run_dir: str | None = None
    if auto_save:
        run_dir = (
            os.path.abspath(os.fspath(save_dir))
            if save_dir is not None
            else make_run_dir(ctx.workspace)
        )
    lead_save_path = agent_save_path(run_dir, 0, team_cfg.entry) if run_dir else None
    # Reuse the manifest identity for every native member record, including
    # trace-off runs whose existing identity is their reserved folder name.
    manifest_run_id = (
        run_id
        if run_id is not None
        else ctx.tracer.run_id
        if run_dir is not None and ctx.tracer is not None
        else os.path.basename(run_dir)
        if run_dir is not None
        else None
    )

    session_factory = DefaultSessionFactory(
        SpawnConfig(
            model=cfg["model"],
            provider=cfg["provider"],
            wire_protocol=cfg.get("wire_protocol", "chat_completions"),
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            llm_timeout=cfg.get("llm_timeout", 600.0),
            temperature=cfg.get("temperature", DEFAULT_TEMPERATURE),
            top_p=cfg.get("top_p", DEFAULT_TOP_P),
            max_output_tokens=cfg.get(
                "max_output_tokens", DEFAULT_MAX_TOKENS_PER_STEP
            ),
            context_window=cfg.get("context_window"),
            thinking=cfg.get("thinking", DEFAULT_THINKING),
            thinking_params=resolve_thinking_params(cfg.get("thinking_params")),
            thinking_params_explicit=cfg.get("thinking_params_explicit", cfg.get("thinking_params") is not None),
            reasoning_effort=cfg.get("reasoning_effort"),
            llm_max_retries=cfg.get("llm_max_retries", 3),
            llm_connect_timeout=cfg.get("llm_connect_timeout", 30.0),
            llm_first_event_timeout=cfg.get("llm_first_event_timeout", 180.0),
            llm_stream_idle_timeout=cfg.get("llm_stream_idle_timeout", 180.0),
            llm_stream_chat=bool(cfg.get("llm_stream_chat", False)),
            provider_error_time_budget=cfg.get("provider_error_time_budget", 0.0),
            tracer=ctx.tracer,
            event_bus=event_bus,
            permission_policy=ctx.permission_policy,
            ask_policy=ctx.ask_policy,
            safety_policy_factory=build_workspace_safety_policy,
        ),
        team_cfg=team_cfg,
        lead_workspace=ctx.workspace,
        lead_environment=environment,
        interactive=interactive,
        save_dir=run_dir,
        # A prebuilt roster's teammates are declared nodes seated before the
        # first model call, so the factory gives them agent 0's shell instead of
        # the hardened default it gives a child a model spawned mid-run.
        prebuilt_roster=prebuild_team,
        allow_unisolated_shell=allow_unisolated_shell,
        allow_unisolated_child_shell=allow_unisolated_child_shell,
        max_steps=max_steps,
        run_id=manifest_run_id,
    )
    worktree_pool = WorktreePool(
        ctx.workspace,
        use_worktrees=use_worktrees,
        base_environment=environment,
        owned_save_dir=run_dir,
        enable_adoption=any("adopt_agent" in role.tools for role in team_cfg.roles.values()),
        owned_output_paths=getattr(ctx.tracer, "owned_paths", ()),
    )

    # The graded tree is where agent 0 works: the environment when the run was
    # given one (a container the repository lives in), the host workspace
    # otherwise. Same choice ``_workflow_runtime_session`` makes for the probe
    # it hands a workflow, so both regimes measure the same directory.
    delivery_tree_probe = (
        EnvWorkingTreeProbe(
            environment if environment is not None else LocalEnvironment(ctx.workspace),
            from_initial_head=True,
        )
        if record_delivery_tree
        else None
    )

    recovery_dir = run_dir

    async def write_candidate_diff(aid: int, role: str, patch: str) -> str:
        nonlocal recovery_dir
        if recovery_dir is None:
            recovery_dir = await asyncio.to_thread(tempfile.mkdtemp, prefix="opencollab-team-recovery-")
        path = str(Path(agent_save_path(recovery_dir, aid, role)).with_suffix(".patch"))
        await asyncio.to_thread(write_regular_bytes_atomic, path, patch.encode("utf-8"))
        return path

    scheduler = Scheduler(
        session_factory=session_factory,
        worktree_pool=worktree_pool,
        event_sink=event_bus,
        tracer=ctx.tracer,
        # Declared per-role allowances are independent, so the team total is
        # their sum; otherwise one shared pool of the configured budget.
        max_budget_tokens=(
            sum(team_cfg.role_budgets.values()) if team_cfg.role_budgets else cfg["budget"]
        ),
        permission_policy=ctx.permission_policy,
        topology=team_cfg.topology,
        roles=tuple(team_cfg.roles),
        role_budgets=team_cfg.role_budgets,
        entry_role=team_cfg.entry,
        prebuild_team=prebuild_team,
        serialize_turns=serialize_turns,
        delivery_tree_probe=delivery_tree_probe,
        candidate_diff_writer=write_candidate_diff,
    )

    # Attach hooks after the scheduler exists so the runner can hold its handle
    # (the coordination-ready seam for a future ``agent`` executor). Subscribing
    # appends to the same bus, leaving the TUI sink at target[0] untouched.
    if enable_hooks and team_cfg.hooks:
        runner = ShellHookRunner(
            team_cfg.hooks,
            scheduler=scheduler,
            workspace=ctx.workspace,
        )
        scheduler.register_lifecycle_resource(runner, description="hook processes")
        event_bus.subscribe(HookEventSubscriber(runner))

    # Persist a team.json manifest from the live roster on every roster change.
    # Wired before create_init_process so agent 0 is captured on registration.
    if run_dir is not None:
        store = SessionStore()
        manifest_path = os.path.join(run_dir, "team.json")
        team_file = (
            Path(os.path.abspath(os.fspath(team_config_path)))
            if team_config_path is not None
            else resolve_team_file(ctx.workspace)
        )
        started_at = datetime.now(timezone.utc).isoformat()

        def _manifest_payload() -> dict:
            return {
                "run_id": manifest_run_id,
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "team_file": str(team_file) if team_file else None,
                "agents": scheduler.team_snapshot(),
            }

        def _write_manifest() -> None:
            store.save_manifest(manifest_path, _manifest_payload())

        def _prepare_manifest() -> Callable[[], None]:
            payload = _manifest_payload()
            return lambda: store.save_manifest(manifest_path, payload)

        scheduler.set_manifest_writer(
            _write_manifest,
            prepare_fn=_prepare_manifest,
        )

    scheduler.create_init_process(
        LaunchSpec(session_file=session_file, auto_save_path=lead_save_path)
    )
    return scheduler


__all__ = ["build_scheduler"]
