"""Workflow capacity follows providers and their late completion callbacks."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from opencollab.application.autosave import AutoSaveSubscriber
from opencollab.application.session_run import ENFORCEMENT_ON
from opencollab.application.workflow import WorkflowContext
from opencollab.bootstrap import build_session
from tests.support.session_characterization_test_support import FakeAgent, FakeTool, llm_response, tool_call
from tests.support.workflow_context_test_support import FakeFactory, FakeSession


class LateProvider:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancel_seen = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.active = 0
        self.peak = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        first = self.calls == 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if first:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancel_seen.set()
                    await self.release.wait()
            return llm_response(content="answer", input_tokens=40, output_tokens=40)
        finally:
            self.active -= 1

class RealSessionFactory:
    def __init__(self, provider, timeout):
        self.provider = provider
        self.timeout = timeout
        self.sessions = []

    def build_workflow_session(self, **kwargs):
        session = build_session(
            agent=FakeAgent(tools=kwargs.get("tools")),
            llm=self.provider,
            llm_timeout=self.timeout,
            max_budget_tokens=kwargs["budget"],
        )
        self.sessions.append(session)
        return session

@pytest.mark.asyncio
async def test_pending_autosave_allows_synthesis_and_keeps_final_cleanup_owned():
    class Provider:
        calls = 0

        async def complete(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return llm_response(tool_calls=[tool_call(name="read_file", arguments="{}")])
            if self.calls == 2:
                raise RuntimeError("provider failed after gathering evidence")
            report = {"summary": "Recovered evidence", "findings": [], "insufficient_evidence": True}
            return llm_response(tool_calls=[
                tool_call(name="submit_findings", arguments=json.dumps(report)),
            ])

    release = threading.Event()

    class SavingFactory(RealSessionFactory):
        def build_workflow_session(self, **kwargs):
            session = super().build_workflow_session(**kwargs)
            if len(self.sessions) == 1:
                session.event_bus.subscribe(AutoSaveSubscriber(lambda: release.wait(5)))
            return session

    provider = Provider()
    factory = SavingFactory(provider, 1.0)
    ctx = WorkflowContext(factory, max_concurrency=1, budget_total=200000)
    try:
        result = await asyncio.wait_for(ctx.agent(
            "inspect", tools=[FakeTool(name="read_file", result="observed evidence")],
            budget=100000, enforcement_strength=ENFORCEMENT_ON,
        ), 2.0)
        assert "Recovered evidence" in result
        assert provider.calls == 3
        assert len(factory.sessions) == 2
        assert not factory.sessions[0].pending_execution_tasks
        assert factory.sessions[0].event_bus.pending_tasks
        assert len(ctx.budget._leases) == 1
        assert ctx._semaphore._value == 0
    finally:
        release.set()
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 2.0)
    assert ctx.budget._leases == []
    assert ctx._semaphore._value == 1

@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["provider_timeout", "workflow_timeout", "caller_cancel"])
@pytest.mark.parametrize("composition", ["direct", "nested_collection", "permit_reentry"])
async def test_provider_cleanup_keeps_lease_and_capacity(interruption, composition):
    provider = LateProvider()
    factory = RealSessionFactory(provider, 0.05 if interruption == "provider_timeout" else 10.0)
    ctx = WorkflowContext(factory, max_concurrency=1, task_concurrency=1, budget_total=200000)

    async def first_call():
        return await ctx.agent(
            "first", budget=100000,
            timeout=0.05 if interruption == "workflow_timeout" else None,
        )

    async def run_call(operation):
        if composition == "direct":
            return await operation()
        if composition == "permit_reentry":
            return await ctx._run_with_concurrency_permit(
                lambda: ctx._run_with_concurrency_permit(operation)
            )
        return (await ctx.parallel([lambda: ctx.parallel([operation])]))[0][0]

    first = asyncio.create_task(run_call(first_call))
    second = None
    try:
        await asyncio.wait_for(provider.started.wait(), 1.0)
        if interruption == "caller_cancel":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 1.0)
        else:
            assert await asyncio.wait_for(first, 1.0) is None
        await asyncio.wait_for(provider.cancel_seen.wait(), 1.0)
        for _ in range(20):
            await asyncio.sleep(0)
        assert provider.active == 1
        assert factory.sessions[0].pending_cleanup_tasks
        assert ctx.tokens_remaining() == 100000
        assert len(ctx.budget._leases) == 1

        second = asyncio.create_task(run_call(lambda: ctx.agent("second", budget=100000)))
        for _ in range(20):
            await asyncio.sleep(0)
        assert provider.calls == 1
        assert not second.done()

        provider.release.set()
        assert await asyncio.wait_for(second, 1.0) == "answer"
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)
        assert provider.peak == 1
        assert ctx.tokens_spent() == 160
        assert ctx.tokens_remaining() == 199840
        assert ctx.budget._leases == []
        assert ctx._semaphore._value == 1
    finally:
        provider.release.set()
        if not first.done():
            first.cancel()
        if second is not None and not second.done():
            second.cancel()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)

@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_agent_reuses_one_permit_only_after_prior_provider_stops(nested):
    provider = LateProvider()
    factory = RealSessionFactory(provider, 0.05)
    ctx = WorkflowContext(factory, max_concurrency=1, budget_total=200000)
    first_finished = asyncio.Event()

    async def operation():
        assert await ctx.agent("first", budget=100000) is None
        first_finished.set()
        if nested:
            return await ctx._run_with_concurrency_permit(lambda: ctx.agent("second", budget=100000))
        return await ctx.agent("second", budget=100000)

    call = asyncio.create_task(ctx._run_with_concurrency_permit(operation))
    try:
        await asyncio.wait_for(first_finished.wait(), 1.0)
        for _ in range(20):
            await asyncio.sleep(0)
        assert provider.calls == 1
        assert not call.done()
        assert len(ctx.budget._leases) == 1
        provider.release.set()
        assert await asyncio.wait_for(call, 1.0) == "answer"
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)
        assert provider.peak == 1
        assert ctx._semaphore._value == 1
    finally:
        provider.release.set()
        await asyncio.gather(call, return_exceptions=True)
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)

@pytest.mark.asyncio
async def test_cancelled_permit_reentry_keeps_prior_cleanup_owned():
    provider = LateProvider()
    factory = RealSessionFactory(provider, 0.05)
    ctx = WorkflowContext(factory, max_concurrency=1, budget_total=200000)
    first_finished = asyncio.Event()

    async def operation():
        assert await ctx.agent("first", budget=100000) is None
        first_finished.set()
        return await ctx.agent("second", budget=100000)

    call = asyncio.create_task(ctx._run_with_concurrency_permit(operation))
    try:
        await asyncio.wait_for(first_finished.wait(), 1.0)
        for _ in range(20):
            await asyncio.sleep(0)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        for _ in range(20):
            await asyncio.sleep(0)
        assert provider.calls == 1
        assert provider.active == 1
        assert len(ctx.budget._leases) == 1
        assert ctx._semaphore._value == 0
    finally:
        provider.release.set()
        await asyncio.gather(call, return_exceptions=True)
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)
    assert ctx.budget._leases == []
    assert ctx._semaphore._value == 1

@pytest.mark.asyncio
async def test_inner_timeout_keeps_dead_scout_synthesizer_from_overlapping_provider():
    class EvidenceProvider(LateProvider):
        async def complete(self, messages, **kwargs):
            factory.sessions[0].state.turn.scout_ledger.append(
                {"tool": "read_file", "summary": "observed evidence"}
            )
            return await super().complete(messages, **kwargs)

    provider = EvidenceProvider()
    factory = RealSessionFactory(provider, 0.05)
    ctx = WorkflowContext(factory, max_concurrency=1, budget_total=200000)
    try:
        await asyncio.wait_for(
            ctx.agent("scout", budget=100000, enforcement_strength=ENFORCEMENT_ON),
            1.0,
        )
        await asyncio.wait_for(provider.cancel_seen.wait(), 1.0)
        assert provider.calls == 1
        assert len(factory.sessions) == 1
        assert provider.active == 1
        assert len(ctx.budget._leases) == 1
    finally:
        provider.release.set()
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)

@pytest.mark.asyncio
async def test_completion_callback_cleanup_keeps_the_same_lease_and_slot():
    first_release = asyncio.Event()
    checkpoint_release = asyncio.Event()
    checkpoint_started = asyncio.Event()

    class CallbackSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.tasks = set()

        @property
        def pending_cleanup_tasks(self):
            return tuple(self.tasks)

        async def run_loop(self, cancel_event=None):
            task = asyncio.create_task(first_release.wait())
            self.tasks.add(task)

            async def checkpoint():
                checkpoint_started.set()
                await checkpoint_release.wait()

            def completed(finished):
                self.tasks.discard(finished)
                followup = asyncio.create_task(checkpoint())
                self.tasks.add(followup)
                followup.add_done_callback(self.tasks.discard)

            task.add_done_callback(completed)
            return "first"

    factory = FakeFactory([CallbackSession(), FakeSession(reply="second")])
    ctx = WorkflowContext(factory, max_concurrency=1, budget_total=200000)
    second = None
    try:
        assert await ctx.agent("first", budget=100000) == "first"
        first_release.set()
        await asyncio.wait_for(checkpoint_started.wait(), 1.0)
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(ctx.budget._leases) == 1
        assert ctx.tokens_remaining() == 100000
        second = asyncio.create_task(ctx.agent("second", budget=100000))
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(factory.builds) == 1
        checkpoint_release.set()
        assert await asyncio.wait_for(second, 1.0) == "second"
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)
        assert ctx.budget._leases == []
        assert ctx._semaphore._value == 1
    finally:
        first_release.set()
        checkpoint_release.set()
        if second is not None:
            await asyncio.gather(second, return_exceptions=True)
        await asyncio.wait_for(ctx.wait_for_pending_cleanup(), 1.0)
