"""Parent tail handoffs retain a producer through driver replacement."""

from __future__ import annotations

import asyncio

import pytest

from opencollab.application.scheduler import SchedulerStalledError
from opencollab.domain.pending import PendingRow, PendingRowError, RowKind, RowStatus
from opencollab.domain.session import SessionPhase
from tests.support.scheduler_awaiting_test_support import (
    ScriptedSession,
    build_scheduler,
    event_types,
    resume_done,
)


def _tail_scheduler():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def suspended_tail(session):
        session.state.pending_events.add(PendingRow(
            tool_call_id="child", kind=RowKind.CHILD_AGENT, order=0, ref=1,
        ))
        session.state.set_phase(SessionPhase.AWAITING_EVENTS)
        entered.set()
        await release.wait()
        return ""

    lead = ScriptedSession("lead", [suspended_tail, resume_done(lambda rows: rows[0])])
    scheduler, events = build_scheduler(lead, [])
    return scheduler, lead, events, entered, release

@pytest.mark.parametrize("recover_route", [False, True])
async def test_parent_replacement_is_owned_before_the_existing_waiter_runs(recover_route):
    scheduler, lead, events, entered, release = _tail_scheduler()
    original = scheduler._start_agent_task(0, lead)
    await entered.wait()
    waiter_entered = asyncio.Event()

    async def wait_for_both_drivers():
        waiter_entered.set()
        # The public run loop already waits on the old driver when the child
        # completes during its terminal checkpoint. Keep that callback order.
        await scheduler._wait_for_scheduler_progress(0)
        await scheduler._wait_for_scheduler_progress(0)

    waiter = asyncio.create_task(wait_for_both_drivers())
    await waiter_entered.wait()
    if recover_route:
        await scheduler._recover_delivery_route(9, 0, PendingRowError("missing child route"))
        await scheduler._recover_delivery_route(9, 0, PendingRowError("duplicate child completion"))
    else:
        await scheduler._wake(0, "child", "child output", RowStatus.DONE)
        with pytest.raises(PendingRowError):
            await scheduler._wake(0, "child", "duplicate output", RowStatus.DONE)

    release.set()
    try:
        await asyncio.wait_for(waiter, timeout=1)
        await scheduler.wait_until_terminal(0)
        assert lead.state.phase is SessionPhase.DONE
        assert lead.state.pending_events.is_empty()
        assert len(event_types(events, "agent_resumed")) == 1
        result = scheduler.table.get(0).result
        if recover_route:
            assert "child completion routing failed" in result
        else:
            assert result == "child output"
    finally:
        await asyncio.gather(original, return_exceptions=True)
        await scheduler.cleanup()
    assert scheduler._tasks == {}

@pytest.mark.parametrize("cleanup", [False, True])
async def test_cancelled_parent_tail_keeps_its_owner_and_cannot_resume(cleanup):
    scheduler, lead, events, entered, release = _tail_scheduler()
    original = scheduler._start_agent_task(0, lead)
    await entered.wait()
    await scheduler._wake(0, "child", "child output", RowStatus.DONE)
    assert scheduler._tasks[0] is original

    if cleanup:
        await scheduler.cleanup()
        assert original.cancelled()
    else:
        await scheduler._settle_cancelled_suspended_turn(0)
        release.set()
        await original
        await scheduler.cleanup()

    assert lead.state.phase is SessionPhase.STOPPED
    assert len(event_types(events, "agent_resumed")) == 0
    assert scheduler._tasks == {}

async def test_cleanup_owns_resumed_event_delivery():
    scheduler, lead, _events, entered, release = _tail_scheduler()
    event_entered = asyncio.Event()
    event_cancelled = asyncio.Event()
    original_emit = scheduler.emit_scheduler_event

    async def blocked_resume_event(event):
        if event.type == "agent_resumed":
            event_entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                event_cancelled.set()
        await original_emit(event)

    scheduler.emit_scheduler_event = blocked_resume_event
    original = scheduler._start_agent_task(0, lead)
    await entered.wait()
    await scheduler._wake(0, "child", "child output", RowStatus.DONE)
    release.set()
    await asyncio.wait_for(event_entered.wait(), timeout=1)
    await scheduler.cleanup()
    assert event_cancelled.is_set()
    assert scheduler._tasks == {}
    assert lead.state.phase is SessionPhase.STOPPED
    await original

async def test_completed_child_row_without_a_resume_owner_still_stalls():
    scheduler, lead, _events, _entered, _release = _tail_scheduler()
    lead.state.set_phase(SessionPhase.AWAITING_EVENTS)
    lead.state.pending_events.add(PendingRow(
        tool_call_id="child", kind=RowKind.CHILD_AGENT, order=0, ref=1,
    ))
    lead.state.pending_events.fill("child", result="child output", status=RowStatus.DONE)
    with pytest.raises(SchedulerStalledError, match="no active producer"):
        await scheduler.wait_until_terminal(0)
    await scheduler.cleanup()
