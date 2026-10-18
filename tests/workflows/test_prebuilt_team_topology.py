"""Prebuilt teams record assigned roles and edges before model calls and refuse later spawn."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from opencollab.application.scheduler_types import TeamPrebuiltError
from opencollab.application.tool_execution import ToolRuntime
from opencollab.bootstrap import build_runtime_context, build_scheduler
from opencollab.bootstrap.context_builder import SpawnConfig
from opencollab.bootstrap.session_factory import SESSION_MAX_STEPS, DefaultSessionFactory
from opencollab.bootstrap.team_config import (
    CODER_TOOL_NAMES,
    TESTER_TOOL_NAMES,
    default_team_config,
    load_team_config,
)
from opencollab.domain.scheduler import PER_AGENT_BUDGET_SHARE, per_agent_cap
from tests.support.paths import REPO_ROOT
from tests.support.prebuilt_team_test_support import (
    CONFIG,
    _roles,
    _scheduler,
)
from tests.support.scheduler_awaiting_test_support import ScriptedFactory, ScriptedSession, terminal

HANDOFF_TEAM_FILE = (
    REPO_ROOT / "configs" / "team.handoff.experiment.yaml"
)

SCHEDULER_STANDIN = object()

def _spawn_config(ctx) -> SpawnConfig:
    return SpawnConfig(
        model=CONFIG["model"],
        provider=CONFIG["provider"],
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"],
        llm_timeout=600.0,
        tracer=None,
        event_bus=EventBus(ctx.event_sink),
        permission_policy=None,
    )

def _payloads(path: str, step_type: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [record["payload"] for record in records if record["type"] == step_type]

def _git(repo, *args) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

def _repo(path):
    """A committed git repository — the precondition for real worktrees."""
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "OpenCollab Tests")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "base")
    return path

async def test_with_the_switch_off_nothing_is_seated_and_nothing_is_recorded(tmp_path):
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=False)
    try:
        assert await scheduler.ensure_team_prebuilt() == ()
        assert _roles(scheduler) == {0: "analyst"}
        assert sorted(scheduler._sessions) == [0]
        assert [entry["phase"] for entry in scheduler.team_roster()] == [
            "idle",
            "available",
            "available",
        ]
        tracer.flush()
    finally:
        tracer.close()
        await scheduler.cleanup()

    assert _payloads(tracer.path, "assigned.topology_nodes") == []
    assert _payloads(tracer.path, "assigned.topology_edges") == []

async def test_with_the_switch_off_spawn_still_creates_an_agent():
    lead = ScriptedSession("analyst", [terminal("done")])
    child = ScriptedSession("coder", [terminal("built")])
    scheduler = Scheduler(
        session_factory=ScriptedFactory([child], {}),
        worktree_pool=WorktreePool(".", use_worktrees=False),
        event_sink=EventBus(_noop_sink),
        roles=("analyst", "coder"),
    )
    lead.scheduler = scheduler
    scheduler.register_lead(lead)

    assert await scheduler.spawn(0, "coder", "implement it") == 1
    assert _roles(scheduler) == {0: "analyst", 1: "coder"}
    await scheduler.cleanup()

async def _noop_sink(event) -> None:
    return None

async def test_every_declared_role_is_seated_before_the_first_model_call(tmp_path):
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        assert sorted(scheduler._sessions) == [0], "agent 0 alone until prebuild runs"

        assert await scheduler.ensure_team_prebuilt() == (1, 2)

        assert _roles(scheduler) == {0: "analyst", 1: "coder", 2: "tester"}
        assert sorted(scheduler._sessions) == [0, 1, 2]
        assert [scb.state.phase.value for scb in scheduler.table.entries.values()] == [
            "idle",
            "idle",
            "idle",
        ]
        assert [session.step_count for session in scheduler._sessions.values()] == [0, 0, 0]
        assert [session.used_tokens for session in scheduler._sessions.values()] == [0, 0, 0]
        assert [entry["phase"] for entry in scheduler.team_roster()] == ["idle"] * 3
    finally:
        tracer.close()
        await scheduler.cleanup()

async def test_a_prebuilt_peer_is_a_child_of_the_init_process_with_no_join_route(tmp_path):
    """Peers use parent_aid=0 without a spawn join route.

    The hook bridge reserves parent_aid=None for the root team lifecycle.
    """
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        await scheduler.ensure_team_prebuilt()
        parents = {aid: scb.parent_aid for aid, scb in scheduler.table.entries.items()}
        assert parents == {0: None, 1: 0, 2: 0}
        assert scheduler._spawn_origin == {}
        assert scheduler._inflight == {}
    finally:
        tracer.close()
        await scheduler.cleanup()

async def test_each_prebuilt_teammate_gets_its_own_worktree(tmp_path):
    workspace = _repo(tmp_path / "repo")
    scheduler, tracer = _scheduler(
        tmp_path, prebuild_team=True, use_worktrees=True, workspace=workspace
    )
    try:
        await scheduler.ensure_team_prebuilt()
        spaces = {aid: scheduler._sessions[aid].env.workspace for aid in sorted(scheduler._sessions)}
        assert spaces[0] == str(workspace.resolve())
        assert spaces[1] != spaces[0] and spaces[2] != spaces[0]
        assert spaces[1] != spaces[2]
        for aid in (1, 2):
            assert (Path(spaces[aid]) / "tracked.txt").read_text(encoding="utf-8") == "base\n"
    finally:
        tracer.close()
        await scheduler.cleanup()

async def test_prebuilding_twice_seats_nobody_twice(tmp_path):
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        assert await scheduler.ensure_team_prebuilt() == (1, 2)
        assert await scheduler.ensure_team_prebuilt() == ()
        assert sorted(scheduler._sessions) == [0, 1, 2]
    finally:
        tracer.close()
        await scheduler.cleanup()

async def test_a_five_role_team_seats_and_every_seat_gets_the_same_cap(tmp_path):
    """Seating reserves no tokens, so roster size is independent of the shared pool's spend."""
    team = tmp_path / "team.yaml"
    names = ("lead", "a", "b", "c", "d")
    team.write_text(
        "entry: lead\n"
        "roles:\n"
        + "".join(
            f"  {name}:\n    prompt: work as the {name}\n    tools: [file_read]\n"
            for name in names
        ),
        encoding="utf-8",
    )
    scheduler, tracer = _scheduler(
        tmp_path, prebuild_team=True, team_config_path=str(team)
    )
    try:
        assert await scheduler.ensure_team_prebuilt() == (1, 2, 3, 4)
        assert sorted(scheduler._sessions) == [0, 1, 2, 3, 4]
        assert list(_roles(scheduler).values()) == list(names)

        total = scheduler._max_budget_tokens
        cap = per_agent_cap(total, len(names))
        assert cap == int(total * PER_AGENT_BUDGET_SHARE / len(names))
        assert scheduler._per_agent_cap() == cap
        assert [
            scheduler._sessions[aid].max_budget_tokens for aid in sorted(scheduler._sessions)
        ] == [cap] * len(names)
        assert scheduler.allocated_tokens == 0
    finally:
        tracer.close()
        await scheduler.cleanup()

def _spawn_tool(scheduler):
    tool = next(t for t in scheduler.lead_session.agent.tools if t.name == "spawn_agent")
    runtime = ToolRuntime(
        environment=scheduler.lead_session.env,
        safety_policy=None,
        permission_policy=None,
        aid=0,
        tool_call_id="call_1",
    )
    return tool, runtime

async def test_the_spawn_tool_survives_but_refuses_and_names_the_live_roster(tmp_path):
    """Keeping spawn_agent exposes attempted roster expansion as a recorded refusal."""
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        await scheduler.ensure_team_prebuilt()
        assert "spawn_agent" in {t.name for t in scheduler.lead_session.agent.tools}

        tool, runtime = _spawn_tool(scheduler)
        answer = await tool.execute_with_runtime(
            {"role": "reviewer", "task": "check the patch", "context": "ctx"}, runtime
        )

        assert isinstance(answer, str)
        assert answer.startswith("Not spawned: 'reviewer' is not a role on this team.")
        assert "analyst (aid 0), coder (aid 1), tester (aid 2)" in answer
        assert "send_message" in answer
        assert sorted(scheduler._sessions) == [0, 1, 2]
        assert sorted(scheduler.table.entries) == [0, 1, 2]
    finally:
        tracer.close()
        await scheduler.cleanup()

async def test_a_refused_spawn_records_who_wanted_which_role_and_whether_it_existed(tmp_path):
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        await scheduler.ensure_team_prebuilt()
        tool, runtime = _spawn_tool(scheduler)
        await tool.execute_with_runtime(
            {"role": "reviewer", "task": "check the patch", "context": "abc"}, runtime
        )
        await tool.execute_with_runtime({"role": "coder", "task": "implement it"}, runtime)
        tracer.flush()
    finally:
        tracer.close()
        await scheduler.cleanup()

    roster = [
        {"aid": 0, "role": "analyst"},
        {"aid": 1, "role": "coder"},
        {"aid": 2, "role": "tester"},
    ]
    assert _payloads(tracer.path, "spawn_refused") == [
        {
            "reason": "team_prebuilt",
            "requester_aid": 0,
            "requester_role": "analyst",
            "requested_role": "reviewer",
            "requested_role_declared": False,
            "topology_allowed": False,
            "declared_roles": ["analyst", "coder", "tester"],
            "live_roster": roster,
            "task": "check the patch",
            "task_chars": 15,
            "context_chars": 3,
        },
        {
            "reason": "team_prebuilt",
            "requester_aid": 0,
            "requester_role": "analyst",
            "requested_role": "coder",
            "requested_role_declared": True,
            "topology_allowed": True,
            "declared_roles": ["analyst", "coder", "tester"],
            "live_roster": roster,
            "task": "implement it",
            "task_chars": 12,
            "context_chars": 0,
        },
    ]

async def test_a_long_delegation_is_still_measurable_as_long(tmp_path):
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        await scheduler.ensure_team_prebuilt()
        with pytest.raises(TeamPrebuiltError):
            await scheduler.spawn(0, "reviewer", "x" * 5000)
        tracer.flush()
    finally:
        tracer.close()
        await scheduler.cleanup()

    payload = _payloads(tracer.path, "spawn_refused")[0]
    assert payload["task"] == "x" * 1000
    assert payload["task_chars"] == 5000

async def test_the_recorded_edges_are_the_team_config_edges(tmp_path):
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        await scheduler.ensure_team_prebuilt()
        tracer.flush()
    finally:
        tracer.close()
        await scheduler.cleanup()

    declared = default_team_config().topology
    expected = sorted(
        ({"from_role": source, "to_role": destination}
         for source, destinations in declared.edges.items()
         for destination in destinations),
        key=lambda edge: (edge["from_role"], edge["to_role"]),
    )
    assert _payloads(tracer.path, "assigned.topology_edges") == [
        {
            "allow_all": False,
            "declared_roles": ["analyst", "coder", "tester"],
            "edges": expected,
        }
    ]
    assert expected == [
        {"from_role": "analyst", "to_role": "coder"},
        {"from_role": "analyst", "to_role": "tester"},
    ]

async def test_the_recorded_nodes_are_the_agents_that_were_actually_seated(tmp_path):
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=True)
    try:
        await scheduler.ensure_team_prebuilt()
        workspaces = {
            aid: scheduler._sessions[aid].env.workspace for aid in sorted(scheduler._sessions)
        }
        tracer.flush()
    finally:
        tracer.close()
        await scheduler.cleanup()

    (payload,) = _payloads(tracer.path, "assigned.topology_nodes")
    per_agent_cap = scheduler._per_agent_cap()
    assert payload["budget_source"] == "shared_rule"
    assert payload["entry_role"] == "analyst"
    assert payload["declared_roles"] == ["analyst", "coder", "tester"]
    assert payload["nodes"] == [
        {
            "aid": 0,
            "role": "analyst",
            "entry": True,
            "model": CONFIG["model"],
            # _prebuildable_default adds message_agent so the Analyst can use its declared edges.
            "tools": [
                "adopt_agent",
                "file_read",
                "grep",
                "message_agent",
                "spawn_agent",
                "use_skill",
            ],
            "permission_mode": "auto",
            "workspace": workspaces[0],
            "workspace_isolated": False,
            "shell": "absent",
            "token_allowance": per_agent_cap,
        },
        {
            "aid": 1,
            "role": "coder",
            "entry": False,
            "model": CONFIG["model"],
            "tools": sorted(CODER_TOOL_NAMES),
            "permission_mode": "auto",
            "workspace": workspaces[1],
            "workspace_isolated": False,
            # Noninteractive mode applies the same sandboxed shell policy to every seat.
            "shell": "sandbox_required",
            "token_allowance": per_agent_cap,
        },
        {
            "aid": 2,
            "role": "tester",
            "entry": False,
            "model": CONFIG["model"],
            "tools": sorted(TESTER_TOOL_NAMES),
            "permission_mode": "auto",
            "workspace": workspaces[2],
            "workspace_isolated": False,
            "shell": "sandbox_required",
            "token_allowance": per_agent_cap,
        },
    ]

def _bash(session):
    for tool in session.agent.tools:
        if tool.name == "bash":
            return tool
    raise AssertionError(f"{session.agent.name} was built without bash")

async def test_a_seated_peer_gets_the_entry_agent_s_shell(tmp_path):
    """Prebuilt peers lack ask_user but still need the entry agent's shell capability."""
    team = load_team_config(path=str(HANDOFF_TEAM_FILE))
    scheduler, tracer = _scheduler(
        tmp_path,
        prebuild_team=True,
        use_worktrees=True,
        workspace=_repo(tmp_path / "repo"),
        interactive=True,
        resolved_team_config=team,
    )
    try:
        await scheduler.ensure_team_prebuilt()
        seats = {
            session.agent.name: _bash(session).require_process_isolation
            for aid, session in sorted(scheduler._sessions.items())
            if any(tool.name == "bash" for tool in session.agent.tools)
        }
        entry = scheduler._sessions[0]
    finally:
        tracer.close()
        await scheduler.cleanup()

    # The roster declares agent 0 as entry because unattended teams omit ask_user.
    assert entry.agent.name == team.entry == "analyst"
    assert seats == {"analyst": False, "coder": False, "tester": False}

async def test_a_child_the_model_spawns_is_not_a_seat_and_keeps_the_hard_default(
    tmp_path,
):
    """Model-spawned children retain sandbox requirements independently of declared seats."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = build_runtime_context(str(workspace), dict(CONFIG), trace=False)
    factory = DefaultSessionFactory(
        _spawn_config(ctx),
        team_cfg=load_team_config(path=str(HANDOFF_TEAM_FILE)),
        lead_workspace=str(workspace),
        interactive=True,
    )
    env = LocalEnvironment(str(workspace))
    ad_hoc = factory.build_spawn_session(
        role="coder", env=env, budget=10_000, aid=1, scheduler=SCHEDULER_STANDIN
    )
    assert _bash(ad_hoc).require_process_isolation is True

    seated = DefaultSessionFactory(
        _spawn_config(ctx),
        team_cfg=load_team_config(path=str(HANDOFF_TEAM_FILE)),
        lead_workspace=str(workspace),
        interactive=True,
        prebuilt_roster=True,
    ).build_spawn_session(
        role="coder", env=env, budget=10_000, aid=1, scheduler=SCHEDULER_STANDIN
    )
    assert _bash(seated).require_process_isolation is False

async def test_no_seat_gets_a_shell_the_entry_agent_would_not_have(tmp_path):
    """Entry and peer shells share the same restriction when interactive approval is unavailable."""
    scheduler, tracer = _scheduler(
        tmp_path,
        prebuild_team=True,
        use_worktrees=True,
        workspace=_repo(tmp_path / "repo"),
        interactive=False,
        resolved_team_config=load_team_config(path=str(HANDOFF_TEAM_FILE)),
    )
    try:
        await scheduler.ensure_team_prebuilt()
        seats = {
            session.agent.name: _bash(session).require_process_isolation
            for session in scheduler._sessions.values()
            if any(tool.name == "bash" for tool in session.agent.tools)
        }
    finally:
        tracer.close()
        await scheduler.cleanup()

    assert seats == {"analyst": True, "coder": True, "tester": True}

def _mute_team(tmp_path, *, messaging: bool) -> Path:
    """Create a Coder return edge with optional message_agent capability."""
    coder_tools = "[file_read, message_agent]" if messaging else "[file_read]"
    team = tmp_path / f"team-{'talking' if messaging else 'mute'}.yaml"
    team.write_text(
        "entry: analyst\n"
        "roles:\n"
        "  analyst:\n"
        "    prompt: plan the work\n"
        "    tools: [file_read, message_agent]\n"
        "  coder:\n"
        "    prompt: do the work\n"
        f"    tools: {coder_tools}\n"
        "topology:\n"
        "  analyst: [coder]\n"
        "  coder: [analyst]\n",
        encoding="utf-8",
    )
    return team

async def test_a_prebuilt_role_with_an_edge_and_no_way_to_walk_it_is_refused(tmp_path):
    """A prebuilt seat needs message_agent to traverse any outgoing topology edge."""
    with pytest.raises(ValueError) as raised:
        _scheduler(
            tmp_path,
            prebuild_team=True,
            team_config_path=str(_mute_team(tmp_path, messaging=False)),
        )
    message = str(raised.value)
    assert "prebuild_team: this team declares edges that no agent can walk." in message
    assert "role 'coder' may address analyst" in message
    assert "unwalkable edges: coder -> analyst" in message
    assert "Add 'message_agent' to the `tools:` list of 'coder'" in message
    assert "role 'analyst'" not in message

async def test_the_same_team_starts_once_the_missing_tool_is_given_back(tmp_path):
    scheduler, tracer = _scheduler(
        tmp_path,
        prebuild_team=True,
        team_config_path=str(_mute_team(tmp_path, messaging=True)),
    )
    try:
        assert await scheduler.ensure_team_prebuilt() == (1,)
        assert _roles(scheduler) == {0: "analyst", 1: "coder"}
    finally:
        tracer.close()
        await scheduler.cleanup()

async def test_the_refusal_lands_before_agent_0_exists(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = build_runtime_context(str(workspace), dict(CONFIG), trace=False)
    with pytest.raises(ValueError):
        build_scheduler(
            ctx,
            use_worktrees=False,
            interactive=False,
            auto_save=True,
            prebuild_team=True,
            team_config_path=str(_mute_team(tmp_path, messaging=False)),
        )
    assert not (workspace / ".opencollab").exists()

async def test_the_shipped_team_still_starts_with_the_switch_off(tmp_path):
    """Dynamic rosters use spawn_agent and its join path instead of message_agent."""
    scheduler, tracer = _scheduler(tmp_path, prebuild_team=False)
    try:
        shipped = default_team_config()
        assert "message_agent" not in shipped.roles["analyst"].tools
        assert sorted(shipped.topology.edges["analyst"]) == ["coder", "tester"]
        assert _roles(scheduler) == {0: "analyst"}
        assert await scheduler.spawn(0, "coder", "implement it") == 1
    finally:
        tracer.close()
        await scheduler.cleanup()

async def test_every_seat_gets_the_same_step_ceiling(tmp_path):
    """Equal token budgets can permit different step counts because histories differ in size."""
    scheduler, tracer = _scheduler(
        tmp_path,
        prebuild_team=True,
        team_config_path=str(HANDOFF_TEAM_FILE),
        max_steps=777,
    )
    try:
        await scheduler.ensure_team_prebuilt()
        ceilings = {
            scb.agent.name: scheduler._sessions[aid].max_steps
            for aid, scb in sorted(scheduler.table.entries.items())
        }
    finally:
        tracer.close()
        await scheduler.cleanup()

    assert ceilings == {"analyst": 777, "coder": 777, "tester": 777}

async def test_the_default_ceiling_is_one_number_not_a_privilege(tmp_path):
    scheduler, tracer = _scheduler(
        tmp_path,
        prebuild_team=True,
        team_config_path=str(HANDOFF_TEAM_FILE),
    )
    try:
        await scheduler.ensure_team_prebuilt()
        ceilings = {
            scheduler._sessions[aid].max_steps
            for aid in scheduler.table.entries
        }
    finally:
        tracer.close()
        await scheduler.cleanup()

    assert ceilings == {SESSION_MAX_STEPS}

async def test_a_run_records_whether_its_turns_were_serialized(tmp_path):
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    serialized, serialized_tracer = _scheduler(
        on_dir, prebuild_team=True, serialize_turns=True
    )
    try:
        await serialized.ensure_team_prebuilt()
        serialized_tracer.flush()
    finally:
        serialized_tracer.close()
        await serialized.cleanup()

    concurrent, concurrent_tracer = _scheduler(off_dir, prebuild_team=True)
    try:
        await concurrent.ensure_team_prebuilt()
        concurrent_tracer.flush()
    finally:
        concurrent_tracer.close()
        await concurrent.cleanup()

    (on,) = _payloads(serialized_tracer.path, "assigned.topology_nodes")
    (off,) = _payloads(concurrent_tracer.path, "assigned.topology_nodes")
    assert on["turns_serialized"] is True
    assert off["turns_serialized"] is False

async def test_serializing_turns_does_not_narrow_who_may_address_whom(tmp_path):
    scheduler, tracer = _scheduler(
        tmp_path, prebuild_team=True, serialize_turns=True
    )
    try:
        await scheduler.ensure_team_prebuilt()
        tracer.flush()
    finally:
        tracer.close()
        await scheduler.cleanup()

    (payload,) = _payloads(tracer.path, "assigned.topology_edges")
    assert payload["edges"] == [
        {"from_role": "analyst", "to_role": "coder"},
        {"from_role": "analyst", "to_role": "tester"},
    ]
    assert payload["allow_all"] is False
