"""A finished sender reopens on reply, and queued messages prevent team quiescence."""

from __future__ import annotations

import pytest

from opencollab.adapters.tools.message import _DELIVERY_NOTE
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase
from tests.support.scheduler_awaiting_test_support import (
    ScriptedSession,
    build_scheduler,
    run,
    terminal,
)


def _seat_coder(scheduler, coder: ScriptedSession, aid: int = 1) -> None:
    coder.state.aid = aid
    coder.scheduler = scheduler
    scheduler.table.add(
        SessionControlBlock(aid=aid, parent_aid=0, agent=coder.agent, state=coder.state)
    )
    scheduler._sessions[aid] = coder

def _hand_off_then_finish(step_result: str):
    async def step(sess: ScriptedSession) -> str:
        await sess.scheduler.send_message(0, 1, "implement", "please write the fix")
        sess.state.set_phase(SessionPhase.DONE)
        sess.state.append_message({"role": "assistant", "content": step_result})
        return step_result

    return step

def _reply_then_finish(sha: str):
    async def step(sess: ScriptedSession) -> str:
        await sess.scheduler.send_message(1, 0, "done", f"committed {sha}")
        sess.state.set_phase(SessionPhase.DONE)
        sess.state.append_message({"role": "assistant", "content": "sent the sha"})
        return "sent the sha"

    return step

def _land_what_arrived():
    async def step(sess: ScriptedSession) -> str:
        assert sess.added, "resumed without a teammate message"
        assert "6b5c529" in sess.added[-1]
        sess.state.set_phase(SessionPhase.DONE)
        answer = "landed 6b5c529"
        sess.state.append_message({"role": "assistant", "content": answer})
        return answer

    return step

def test_a_finished_sender_is_reopened_by_its_teammate_s_reply():
    lead = ScriptedSession("analyst", [_hand_off_then_finish("handed off"), _land_what_arrived()])
    coder = ScriptedSession("coder", [_reply_then_finish("6b5c529")])
    scheduler, _ = build_scheduler(lead, [])
    _seat_coder(scheduler, coder)

    answer = run(scheduler.run("fix the bug"))

    # The reply must reopen the lead after its initial handoff-only completion.
    assert answer == "landed 6b5c529"
    assert lead._steps == []

def test_the_team_turn_does_not_end_while_a_message_is_undelivered():
    lead = ScriptedSession("analyst", [terminal("done on my own")])
    coder = ScriptedSession("coder", [])
    scheduler, _ = build_scheduler(lead, [])
    _seat_coder(scheduler, coder)
    lead.state.set_phase(SessionPhase.DONE)
    coder.state.set_phase(SessionPhase.DONE)

    assert scheduler._quiescent()

    scheduler._message_inbox.setdefault(1, []).append(object())

    assert not scheduler._quiescent()

@pytest.mark.parametrize("promise", ["reopens your turn", "finishing is how"])
def test_the_delivery_note_says_finishing_is_a_way_to_wait(promise: str):
    # The tool description must prevent budget-consuming polling.
    assert promise in _DELIVERY_NOTE
