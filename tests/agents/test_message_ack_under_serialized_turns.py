"""A serialized sender must end its turn before a messaged teammate can run."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from opencollab.adapters.tools.message import _DELIVERY_NOTE, MessageAgentTool
from tests.support.scheduler_awaiting_test_support import ScriptedSession, build_scheduler

ACK = "Message queued for aid 1."

class _Team:
    def __init__(self, *, serialized: bool | None):
        if serialized is not None:
            self.turns_serialized = serialized

    def team_snapshot(self):
        return [{"aid": 0, "role": "adopter"}, {"aid": 1, "role": "coder_a"}]

    async def send_message(self, from_aid, to_aid, summary, content):
        return ACK

def _send(team) -> str:
    tool = MessageAgentTool(team)
    params = {"to_role": "coder_a", "summary": "fix", "content": "please fix it"}
    return asyncio.run(tool.execute_with_runtime(params, runtime=SimpleNamespace(aid=0)))

def test_a_serialized_sender_is_told_to_submit_now():
    out = _send(_Team(serialized=True))

    assert out.startswith(ACK)
    assert "cannot start until you end your turn" in out
    assert "call submit now" in out.lower()
    assert "reopens your turn" in out
    assert "meanwhile" not in out

def test_a_concurrent_sender_keeps_the_general_note():
    assert _send(_Team(serialized=False)) == f"{ACK} {_DELIVERY_NOTE}"

def test_a_scheduler_without_the_flag_keeps_the_general_note():
    assert _send(_Team(serialized=None)) == f"{ACK} {_DELIVERY_NOTE}"

def test_the_scheduler_reports_whether_turns_are_serialized():
    lead = ScriptedSession("lead", [])
    assert build_scheduler(lead, [], serialize_turns=True)[0].turns_serialized is True
    lead = ScriptedSession("lead", [])
    assert build_scheduler(lead, [], serialize_turns=False)[0].turns_serialized is False
