"""Accepted messages wait for the prior provider cleanup to settle."""

from __future__ import annotations

import asyncio
import copy

import pytest

from opencollab.adapters.tools.base import Tool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase
from tests.support.session_characterization_test_support import fake_worktree_pool, llm_response


class _ChildProvider:
    def __init__(self, ending):
        self.ending = ending
        self.started = asyncio.Event()
        self.cancel_seen = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        if len(self.calls) > 1:
            return llm_response(content="follow-up processed")
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_seen.set()
            if self.ending == "cancelled":
                raise
            await self.release.wait()
            if self.ending == "error":
                raise RuntimeError("late provider failure")
            return llm_response(content="late abandoned answer", input_tokens=8, output_tokens=9)

class _Factory:
    def __init__(self, provider):
        self.provider = provider
        self.session = None

    def build_spawn_session(self, *, role, budget, aid, task, **_kwargs):
        self.session = build_session(
            agent=Agent(name=role, system_prompt="worker"),
            llm=self.provider,
            aid=aid,
            max_budget_tokens=budget,
            llm_timeout=0.01,
            seed_user_messages=[{"role": "user", "content": task}],
        )
        return self.session

class _StartAndMessage(Tool):
    name = "start_and_message"
    description = "Start worker work and queue a follow-up."

    def __init__(self, scheduler, provider, *, send_after_timeout=False):
        self.scheduler = scheduler
        self.provider = provider
        self.send_after_timeout = send_after_timeout
        self.ack = None

    async def execute_with_runtime(self, params, runtime):
        aid = await self.scheduler.spawn(0, "worker", "initial work")
        await self.provider.started.wait()
        if self.send_after_timeout:
            driver = self.scheduler._tasks[aid]
            await self.provider.cancel_seen.wait()
            await driver
        self.ack = await self.scheduler.send_message(0, aid, "follow-up", "Process this accepted follow-up")
        return self.ack

class _LeadProvider:
    def __init__(self):
        self.calls = 0
        self.requests = []

    async def complete(self, **kwargs):
        self.calls += 1
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        if self.calls == 1:
            return llm_response(tool_calls=[{
                "id": "start", "type": "function",
                "function": {"name": "start_and_message", "arguments": "{}"},
            }], finish_reason="tool_calls")
        return llm_response(content="lead done")

@pytest.mark.parametrize("ending", ["success", "error", "cancelled"])
@pytest.mark.parametrize("send_after_timeout", [False, True])
async def test_accepted_followup_runs_after_timeout_cleanup(ending, send_after_timeout):
    provider = _ChildProvider(ending)
    factory = _Factory(provider)
    scheduler = Scheduler(
        session_factory=factory, worktree_pool=fake_worktree_pool(),
        event_sink=EventBus(), max_budget_tokens=400_000,
    )
    starter = _StartAndMessage(scheduler, provider, send_after_timeout=send_after_timeout)
    lead_provider = _LeadProvider()
    scheduler.register_lead(build_session(
        agent=Agent(name="lead", system_prompt="lead", tools=[starter]), llm=lead_provider,
    ))
    active = asyncio.create_task(scheduler.run("Start the work"))
    try:
        await asyncio.wait_for(provider.cancel_seen.wait(), 2)

        async def wait_for_terminal_driver_and_message():
            while scheduler._tasks.get(1) is not None or starter.ack is None:
                await asyncio.sleep(0)

        if ending != "cancelled":
            await asyncio.wait_for(wait_for_terminal_driver_and_message(), 2)
            assert not active.done()
            assert starter.ack == "Message queued to aid 1."
            assert len(provider.calls) == 1
            assert len(scheduler._message_inbox[1]) == 1
            assert factory.session.phase is SessionPhase.ERROR
            assert factory.session.runner.pending_cleanup_tasks
        provider.release.set()
        assert await asyncio.wait_for(active, 2) == "lead done"
        assert len(provider.calls) == 2
        assert "Process this accepted follow-up" in str(provider.calls[1])
        assert "late abandoned answer" not in str(factory.session.messages)
        assert not scheduler._message_inbox[1]
        assert factory.session.state.pending_user_messages == []
        assert factory.session.phase is SessionPhase.DONE
        assert factory.session.runner.late_provider_usage == ((17,) if ending == "success" else ())
        assert factory.session.used_tokens == (19 if ending == "success" else 2)
        assert scheduler._pending_provider_budget_leases == {}
        assert not any("will not answer your last message" in str(request) for request in lead_provider.requests)
    finally:
        provider.release.set()
        await scheduler.cleanup()
        await asyncio.gather(active, return_exceptions=True)

async def test_cancelling_public_turn_preserves_the_provider_cleanup_owner():
    provider = _ChildProvider("success")
    factory = _Factory(provider)
    scheduler = Scheduler(
        session_factory=factory, worktree_pool=fake_worktree_pool(), event_sink=EventBus(),
    )
    starter = _StartAndMessage(scheduler, provider)
    scheduler.register_lead(build_session(
        agent=Agent(name="lead", system_prompt="lead", tools=[starter]), llm=_LeadProvider(),
    ))
    active = asyncio.create_task(scheduler.run("Start the work"))
    try:
        await asyncio.wait_for(provider.cancel_seen.wait(), 2)

        async def wait_for_drivers():
            while scheduler._tasks:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_drivers(), 2)
        owners = factory.session.runner.pending_cleanup_tasks
        assert owners
        active.cancel()
        await asyncio.sleep(0)
        provider.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, 2)
        assert all(owner.done() and not owner.cancelled() for owner in owners)
        assert factory.session.runner.late_provider_usage == (17,)
        assert len(provider.calls) == 1
        assert factory.session.state.pending_user_messages
        assert scheduler._shutting_down
    finally:
        provider.release.set()
        await scheduler.cleanup()
        await asyncio.gather(active, return_exceptions=True)
