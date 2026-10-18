"""Declared rosters share a token pool with equal per-agent ceilings.

Idle seats reserve no tokens, and cap refusals remain distinct from pool exhaustion.
"""

from __future__ import annotations

import asyncio
from typing import Any

from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from opencollab.application.steering import build_steering_block
from opencollab.domain.scheduler import (
    PER_AGENT_BUDGET_SHARE,
    dynamic_roster_share,
    per_agent_cap,
)
from opencollab.domain.session import SessionPhase, SessionState

TOTAL = 600_000
ROLES = ("lead", "coder", "tester")
CAP = per_agent_cap(TOTAL, len(ROLES))

def run(coro):
    return asyncio.run(coro)

class FakeSession:
    """The little a scheduler needs of a session to do budget arithmetic."""

    def __init__(self, role: str, budget: int):
        self.agent = type("_Agent", (), {"name": role})()
        self.state = SessionState(messages=[])
        self.max_budget_tokens = budget
        self.used_tokens = 0

    async def add_user_message(self, content: str) -> None:
        self.state.append_message({"role": "user", "content": content})

    async def run_loop(self) -> str:
        self.state.set_phase(SessionPhase.DONE)
        return "done"

class RecordingFactory:
    """Records the budget every session is built with."""

    def __init__(self) -> None:
        self.lead_budget: int | None = None
        self.spawn_budgets: list[int] = []

    def create_lead_session(self, *, scheduler, launch, budget, aid=0):
        self.lead_budget = budget
        return FakeSession("lead", budget)

    def build_spawn_session(self, *, role, env, budget, aid=-1, **kwargs):
        self.spawn_budgets.append(budget)
        session = FakeSession(role, budget)
        session.state.aid = aid
        return session

class _NoWorktrees:
    async def acquire(self, role):
        return None

    async def release(self):
        return None

class RecordingTracer:
    def __init__(self) -> None:
        self.steps: list[tuple[str, dict[str, Any]]] = []

    def log_step(self, *, step_type: str, payload: dict[str, Any]) -> None:
        self.steps.append((step_type, payload))

    def payloads(self, step_type: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.steps if name == step_type]

def _scheduler(*, prebuild_team: bool, tracer: RecordingTracer | None = None):
    factory = RecordingFactory()
    scheduler = Scheduler(
        session_factory=factory,
        worktree_pool=_NoWorktrees(),
        event_sink=EventBus(None),
        tracer=tracer,
        max_budget_tokens=TOTAL,
        roles=ROLES,
        prebuild_team=prebuild_team,
    )
    lead = factory.create_lead_session(
        scheduler=scheduler, launch=None, budget=scheduler._entry_start_budget()
    )
    scheduler.register_lead(lead)
    return scheduler, factory

def _spend(scheduler: Scheduler, aid: int, tokens: int) -> None:
    scheduler.table.get(aid).state.used_tokens = tokens

def test_the_cap_is_the_share_constant_times_an_equal_split():
    scheduler, _ = _scheduler(prebuild_team=True)
    assert scheduler._declared_team_size() == 3
    assert CAP == int(TOTAL * PER_AGENT_BUDGET_SHARE / 3) == 200_000
    assert scheduler._per_agent_cap() == CAP
    # An equal split gives each seat a solo-sized cap.
    assert CAP == TOTAL // len(ROLES)
    assert CAP * len(ROLES) == TOTAL

def test_every_seat_gets_the_same_cap_agent_zero_included():
    """Agent 0's allowance is the rule's, not a privilege of being the root."""

    async def scenario():
        scheduler, factory = _scheduler(prebuild_team=True)
        assert await scheduler.ensure_team_prebuilt() == (1, 2)
        caps = [scheduler._sessions[aid].max_budget_tokens for aid in (0, 1, 2)]
        assert caps == [CAP, CAP, CAP]
        assert factory.spawn_budgets == [CAP, CAP]

    run(scenario())

def test_agent_zero_no_longer_leases_the_whole_pool_for_a_turn():
    """Dynamic rosters lease agent zero all unreserved tokens because their size is unknown."""

    async def scenario():
        declared, _ = _scheduler(prebuild_team=True)
        await declared.ensure_team_prebuilt()
        assert declared._entry_agent_takes_the_pool(0) is False
        assert declared._reserve_turn_lease(0) == CAP < TOTAL

        dynamic, _ = _scheduler(prebuild_team=False)
        assert dynamic._entry_agent_takes_the_pool(0) is True
        assert dynamic._reserve_turn_lease(0) == TOTAL

    run(scenario())

def test_agent_zero_is_told_the_budget_it_can_actually_spend():
    scheduler, factory = _scheduler(prebuild_team=True)
    assert factory.lead_budget == CAP == scheduler._per_agent_cap()

    block, _override, _level = build_steering_block(
        used_tokens=0,
        max_budget_tokens=scheduler._sessions[0].max_budget_tokens,
        step_count=0,
        max_steps=50,
        reads=0,
        has_write=False,
        has_structured_output=False,
        structured_override=None,
    )
    assert block["content"] == "[Budget: ~200k/200k tokens left, ~50 steps left.]"

def test_an_idle_teammate_freezes_nothing_for_the_agent_doing_the_work():
    """At a share of one, equal ceilings still leave idle seats with zero committed spend."""

    async def scenario():
        scheduler, _ = _scheduler(prebuild_team=True)
        await scheduler.ensure_team_prebuilt()

        equal_split = TOTAL // len(ROLES)
        assert CAP == equal_split
        half_a_share = equal_split // 2
        _spend(scheduler, 0, half_a_share)
        assert scheduler.allocated_tokens == half_a_share

        grant = scheduler._reserve_turn_lease(0)
        assert grant == CAP - half_a_share == 100_000
        assert scheduler._sessions[0].max_budget_tokens == CAP
        assert scheduler.allocated_tokens == half_a_share

    run(scenario())

def test_the_cap_still_bounds_the_agent_that_is_working():
    async def scenario():
        scheduler, _ = _scheduler(prebuild_team=True)
        await scheduler.ensure_team_prebuilt()
        _spend(scheduler, 0, CAP)
        assert scheduler._reserve_turn_lease(0) == 0
        assert scheduler.used_tokens == CAP < TOTAL

    run(scenario())

def test_a_turn_the_cap_refused_is_recorded_with_the_pool_beside_it():
    async def scenario():
        tracer = RecordingTracer()
        scheduler, _ = _scheduler(prebuild_team=True, tracer=tracer)
        await scheduler.ensure_team_prebuilt()
        _spend(scheduler, 0, CAP)
        assert scheduler._reserve_turn_lease(0) == 0

        assert tracer.payloads("agent_cap_reached") == [
            {
                "agent_id": 0,
                "role": "lead",
                "step_count": 0,
                "requested": TOTAL - CAP,
                "per_agent_cap": CAP,
                "used": CAP,
                "remaining": 0,
                "pool_remaining": TOTAL - CAP,
                "spent": CAP,
                "total": TOTAL,
                "team_size": 3,
                "share": PER_AGENT_BUDGET_SHARE,
                "would_exceed_by": TOTAL - CAP,
            }
        ]

    run(scenario())

def test_an_empty_pool_is_not_recorded_as_a_cap_refusal():
    """At a share of one, in-flight overspend can exhaust the pool below another seat's cap."""

    async def scenario():
        tracer = RecordingTracer()
        scheduler, _ = _scheduler(prebuild_team=True, tracer=tracer)
        await scheduler.ensure_team_prebuilt()
        overspent = TOTAL // 2
        assert overspent > CAP
        _spend(scheduler, 0, overspent)
        _spend(scheduler, 1, overspent)
        assert scheduler.used_tokens == TOTAL
        assert scheduler.table.get(2).state.used_tokens == 0
        assert scheduler._reserve_turn_lease(2) == 0
        assert tracer.payloads("agent_cap_reached") == []

    run(scenario())

def test_a_failing_tracer_never_overturns_the_allocation():
    class Broken(RecordingTracer):
        def log_step(self, *, step_type, payload):
            raise RuntimeError("tracer down")

    async def scenario():
        scheduler, _ = _scheduler(prebuild_team=True, tracer=Broken())
        await scheduler.ensure_team_prebuilt()
        _spend(scheduler, 0, CAP)
        assert scheduler._reserve_turn_lease(0) == 0
        assert scheduler._sessions[0].max_budget_tokens == CAP

    run(scenario())

def test_without_a_declared_roster_the_reservation_rule_still_applies():
    scheduler, factory = _scheduler(prebuild_team=False)
    assert scheduler._per_agent_cap() is None
    assert factory.lead_budget == TOTAL
    assert scheduler._turn_lease[0] == dynamic_roster_share(TOTAL) == 150_000
    assert scheduler.allocated_tokens == 150_000
    assert scheduler._entry_agent_takes_the_pool(0) is True
