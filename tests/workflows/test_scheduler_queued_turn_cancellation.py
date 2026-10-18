"""A waiting public turn acquires cleanup ownership only after its run lock."""

from __future__ import annotations

import asyncio
import copy

import pytest

from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from opencollab.bootstrap import build_session
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase
from tests.support.session_characterization_test_support import (
    FakeAgent,
    FakeLLMClient,
    fake_worktree_pool,
    llm_response,
)


class ControlledProvider:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.cancelled = False

    async def complete(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        if len(self.calls) == 1:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return llm_response(content=f"answer-{len(self.calls)}")

def build_addressed_scheduler(aid):
    provider = ControlledProvider()
    target = build_session(agent=FakeAgent(), llm=provider)
    scheduler = Scheduler(
        session_factory=object(), worktree_pool=fake_worktree_pool(), event_sink=EventBus(),
    )
    if aid == 0:
        scheduler.register_lead(target)
    else:
        lead = build_session(agent=FakeAgent(), llm=FakeLLMClient())
        scheduler.register_lead(lead)
        target.state.aid = aid
        target.scheduler = scheduler
        scheduler.table.add(SessionControlBlock(aid=aid, parent_aid=0, agent=target.agent, state=target.state))
        scheduler._sessions[aid] = target
        scheduler._reserve_child_budget(aid)
    return scheduler, target, provider

async def start_queued_turn(scheduler, aid, provider, first_event):
    def run(message, cancel_event=None):
        if aid == 0:
            return scheduler.run(message, cancel_event=cancel_event)
        return scheduler.run_turn(aid, message, cancel_event=cancel_event)

    first = asyncio.create_task(run("first request", first_event))
    await asyncio.wait_for(provider.started.wait(), 1)
    queued = asyncio.create_task(run("queued request", asyncio.Event()))

    async def wait_until_registered():
        while queued not in scheduler._active_run_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_registered(), 1)
    assert len(provider.calls) == 1
    return first, queued

@pytest.mark.parametrize("aid", [0, 1])
async def test_normal_queued_turn_runs_after_the_active_turn(aid):
    scheduler, target, provider = build_addressed_scheduler(aid)
    first, queued = await start_queued_turn(scheduler, aid, provider, asyncio.Event())
    try:
        provider.release.set()
        assert await asyncio.wait_for(first, 1) == "answer-1"
        assert await asyncio.wait_for(queued, 1) == "answer-2"
        assert len(provider.calls) == 2
        assert any(message.get("content", "").startswith("queued request") for message in target.messages)
        assert not scheduler._shutting_down
        assert scheduler._active_run_tasks == {}
    finally:
        provider.release.set()
        await scheduler.cleanup()
        await asyncio.gather(first, queued, return_exceptions=True)

@pytest.mark.parametrize("aid", [0, 1])
async def test_cancelling_a_queued_turn_preserves_the_active_turn_and_future_turns(aid):
    scheduler, target, provider = build_addressed_scheduler(aid)
    first_event = asyncio.Event()
    first, queued = await start_queued_turn(scheduler, aid, provider, first_event)
    try:
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(queued, 1)
        assert not provider.cancelled
        assert not first.done()
        assert not scheduler._shutting_down
        assert scheduler._turn_cancel_events[aid] is first_event
        assert scheduler._active_run_tasks == {first: aid}
        assert not any(message.get("content", "").startswith("queued request") for message in target.messages)
        provider.release.set()
        assert await asyncio.wait_for(first, 1) == "answer-1"
        assert await asyncio.wait_for(scheduler.run_turn(aid, "retry request"), 1) == "answer-2"
        assert len(provider.calls) == 2
        assert scheduler._active_run_tasks == {}
    finally:
        provider.release.set()
        await scheduler.cleanup()
        await asyncio.gather(first, queued, return_exceptions=True)

@pytest.mark.parametrize("aid", [0, 1])
@pytest.mark.parametrize("stop_mode", ["cancel_owner", "cleanup"])
async def test_execution_owner_cancellation_and_explicit_cleanup_still_stop_the_team(aid, stop_mode):
    scheduler, target, provider = build_addressed_scheduler(aid)
    first, queued = await start_queued_turn(scheduler, aid, provider, asyncio.Event())
    try:
        if stop_mode == "cancel_owner":
            first.cancel()
        else:
            await asyncio.wait_for(scheduler.cleanup(), 1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 1)
        with pytest.raises((asyncio.CancelledError, RuntimeError)) as stopped:
            await asyncio.wait_for(queued, 1)
        if isinstance(stopped.value, RuntimeError):
            assert "agent is still running" in str(stopped.value) or "scheduler is shutting down" in str(stopped.value)
        assert provider.cancelled
        assert scheduler._shutting_down
        assert target.phase is SessionPhase.STOPPED
        assert scheduler._tasks == {}
        assert scheduler._active_run_tasks == {}
        with pytest.raises(RuntimeError, match="scheduler is shutting down"):
            await scheduler.run_turn(aid, "retry request")
    finally:
        provider.release.set()
        await scheduler.cleanup()
        await asyncio.gather(first, queued, return_exceptions=True)
