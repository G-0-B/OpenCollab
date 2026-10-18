"""Capture executed tool evidence and synthesize dead-scout findings from the transcript.

Synthesis uses a bounded submit-only call without new exploration.
"""

from __future__ import annotations

import asyncio
import copy

from opencollab.application.submit_findings import (
    SUBMIT_TOOL_NAME,
    build_dead_scout_synthesis_prompt,
    harvest_findings,
)
from opencollab.domain.session import SessionState
from tests.support.tool_execution_test_support import build_sensor_use_case as _use_case
from tests.tools.scout_test_support import _FakeState, _ReadStub


def run(coro):
    return asyncio.run(coro)

class ScriptedTool:
    def __init__(self, name, outputs):
        self.name = name
        self._outputs = list(outputs)
        self.calls = []

    async def execute_with_runtime(self, args, runtime):
        self.calls.append(args)
        return self._outputs.pop(0) if self._outputs else ""

def _call(name, args, cid="c1"):
    return {"id": cid, "function": {"name": name, "arguments": args}}

def _run_one(state, tool, name, args):
    result = run(_use_case(state, tool).process([_call(name, args)]))
    result.apply_to(state)
    return result

def test_t1_ledger_accumulates_correct_cards_across_tool_calls():
    state = SessionState(messages=[])

    _run_one(state, ScriptedTool("grep", ["fs.py:42: end = start + n"]), "grep", '{"pattern":"end"}')
    _run_one(state, ScriptedTool("grep", ["fs.py:42: end = start + n"]), "grep", '{"pattern":"start"}')
    _run_one(state, ScriptedTool("file_read", [""]), "file_read", '{"path":"empty.py"}')
    _run_one(state, ScriptedTool("grep", ["No matches found for pattern: zzz"]), "grep", '{"pattern":"zzz"}')
    _run_one(
        state,
        ScriptedTool("file_read", ["File: b.py (2 lines total)\n1\tdef f(): pass"]),
        "file_read",
        '{"path":"b.py"}',
    )

    ledger = state.turn.scout_ledger
    assert len(ledger) == 5
    assert [c["outcome"] for c in ledger] == ["hit", "duplicate", "NO-MATCH", "NO-MATCH", "hit"]
    assert [c["tool"] for c in ledger] == ["grep", "grep", "file_read", "grep", "file_read"]
    assert ledger[0]["target"] == "end"
    assert ledger[2]["target"] == "empty.py"
    assert ledger[4]["target"] == "b.py"
    assert "end = start + n" in ledger[0]["snippet"]

def test_t1_ledger_resets_on_a_fresh_user_turn():
    state = SessionState(messages=[])
    _run_one(state, ScriptedTool("grep", ["a hit"]), "grep", '{"pattern":"a"}')
    assert len(state.turn.scout_ledger) == 1
    state.reset_for_user_turn()
    assert state.turn.scout_ledger == []

def test_t1_harvest_backstop_reads_the_ledger_for_a_partial_salvage():
    # Without captured output or prose, harvest prefers the classified ledger.
    ledger = [
        {"tool": "grep", "target": "end", "outcome": "hit", "snippet": "fs.py:42: end = start + n"},
        {"tool": "file_read", "target": "fs.py", "outcome": "duplicate", "snippet": "def slice(...)"},
    ]
    report = harvest_findings(None, fallback_text="", messages=[], ledger=ledger)
    assert "partial" in report
    assert "fs.py:42" in report
    assert "[hit]" in report
    assert "scout died" not in report

def test_dead_scout_prompt_has_total_bound_and_omission_counts():
    ledger = [
        {
            "tool": "grep",
            "target": f"target-{index}-" + "t" * 500,
            "outcome": "hit",
            "snippet": f"file.py:{index} " + "s" * 500,
        }
        for index in range(256)
    ]
    messages = [
        {"role": "tool", "tool_call_id": f"call-{index}", "content": "r" * 2_000}
        for index in range(100)
    ]

    prompt = build_dead_scout_synthesis_prompt(
        ledger,
        messages,
        max_chars=4_000,
    )

    assert len(prompt) <= 4_000
    assert "omitted" in prompt

from opencollab.application.session_run import ENFORCEMENT_OFF, ENFORCEMENT_ON  # noqa: E402


def _cited_payload():
    return {
        "findings": [
            {
                "aspect": "bug origin",
                "claim": "off-by-one in slice bound",
                "evidence_anchor": "fs.py:42",
                "verified": True,
                "confidence": "high",
            }
        ],
        "summary": "root cause located from the gathered evidence",
        "insufficient_evidence": False,
    }

def _insufficient_payload():
    return {
        "findings": [],
        "summary": "The available tool evidence did not establish this dimension.",
        "insufficient_evidence": True,
    }

class _FakeTracer:
    def __init__(self):
        self.steps = []

    def log_step(self, step_type, payload, tokens=0, latency=0.0):
        self.steps.append({"step_type": step_type, "payload": copy.deepcopy(payload)})

class _FakeRunner:
    def __init__(self):
        self.configured = None

    def configure_enforcement(self, *, enforcement_strength, commit_reserve):
        self.configured = (enforcement_strength, commit_reserve)

class _FakeSession:
    """Script a submit capture from the injected tool during run_loop."""

    def __init__(self, *, capture=None, reply="", used_tokens=30_000,
                 max_budget_tokens=100_000, scout_ledger=None, messages=None):
        self.runner = _FakeRunner()
        self.state = _FakeState(used_tokens, scout_ledger=scout_ledger, messages=messages)
        self.max_budget_tokens = max_budget_tokens
        self._capture = capture
        self.reply = reply
        self.submit_tool = None

    async def add_user_message(self, content):
        self.state.messages.append({"role": "user", "content": content})

    async def run_loop(self, cancel_event=None):
        if self._capture is not None and self.submit_tool is not None:
            self.submit_tool.captured = self._capture
        return self.reply

class _ScriptedFactory:
    """Build sessions in script order and bind their injected submit tools."""

    def __init__(self, sessions):
        self.sessions = list(sessions)
        self.builds = []

    def build_workflow_session(self, *, prompt, budget, tools=None, isolation=False,
                               label=None, tool_choice=None, thinking=None, env=None):
        self.builds.append(
            {"tools": list(tools or []), "label": label, "tool_choice": tool_choice, "prompt": prompt}
        )
        session = self.sessions.pop(0)
        for tool in tools or []:
            if getattr(tool, "name", None) == SUBMIT_TOOL_NAME:
                session.submit_tool = tool
        return session

def _ctx_with(factory, tracer):
    from opencollab.application.workflow import WorkflowContext

    return WorkflowContext(factory, tracer=tracer, budget_total=None)

_DEAD_LEDGER = [
    {"tool": "grep", "target": "slice", "outcome": "hit", "snippet": "fs.py:42: end = start + n"},
    {"tool": "file_read", "target": "fs.py", "outcome": "duplicate", "snippet": "def slice(...)"},
]
_DEAD_MESSAGES = [
    {"role": "user", "content": "investigate"},
    {"role": "tool", "tool_call_id": "c1", "content": "fs.py:42: end = start + n"},
]

def test_t2_dead_scout_triggers_synthesizer_yielding_cited_findings():
    tracer = _FakeTracer()
    dead = _FakeSession(capture=None, reply="", scout_ledger=list(_DEAD_LEDGER),
                        messages=list(_DEAD_MESSAGES))
    synth = _FakeSession(capture=_cited_payload(), reply="")
    factory = _ScriptedFactory([dead, synth])
    ctx = _ctx_with(factory, tracer)

    result = run(
        ctx.agent(
            "scout the bug",
            tools=[_ReadStub()],
            label="scout:0:bug-origin",
            enforcement_strength=ENFORCEMENT_ON,
            commit_reserve=25_000,
        )
    )

    assert "root cause located" in result
    assert "fs.py:42" in result
    assert "scout died" not in result
    assert len(factory.builds) == 2
    synth_build = factory.builds[1]
    assert [getattr(t, "name", None) for t in synth_build["tools"]] == [SUBMIT_TOOL_NAME]
    assert synth_build["tool_choice"] is not None
    assert "fs.py:42" in synth_build["prompt"]
    assert any(s["step_type"] == "dead_scout_synthesis" for s in tracer.steps)

def test_t2_dead_scout_synthesizer_accepts_valid_insufficient_evidence():
    tracer = _FakeTracer()
    dead = _FakeSession(capture=None, reply="", scout_ledger=list(_DEAD_LEDGER),
                        messages=list(_DEAD_MESSAGES))
    synth = _FakeSession(capture=_insufficient_payload(), reply="")
    factory = _ScriptedFactory([dead, synth])
    ctx = _ctx_with(factory, tracer)

    result = run(
        ctx.agent(
            "scout the bug",
            tools=[_ReadStub()],
            label="scout:0",
            enforcement_strength=ENFORCEMENT_ON,
        )
    )

    assert "insufficient_evidence" in result
    assert len(factory.builds) == 2

def test_t2_committed_scout_does_not_trigger_synthesizer():
    tracer = _FakeTracer()
    committed = _FakeSession(capture=_cited_payload(), reply="", scout_ledger=list(_DEAD_LEDGER))
    factory = _ScriptedFactory([committed])
    ctx = _ctx_with(factory, tracer)

    result = run(
        ctx.agent("scout", tools=[_ReadStub()], label="scout:0", enforcement_strength=ENFORCEMENT_ON)
    )

    assert "root cause located" in result
    assert len(factory.builds) == 1
    assert not any(s["step_type"] == "dead_scout_synthesis" for s in tracer.steps)

def test_t2_dead_scout_with_no_ledger_does_not_trigger_synthesizer():
    tracer = _FakeTracer()
    dead = _FakeSession(capture=None, reply="", scout_ledger=[], messages=[])
    factory = _ScriptedFactory([dead])
    ctx = _ctx_with(factory, tracer)

    run(ctx.agent("scout", tools=[_ReadStub()], label="scout:0", enforcement_strength=ENFORCEMENT_ON))

    assert len(factory.builds) == 1

def test_t2_off_path_never_synthesizes():
    tracer = _FakeTracer()
    session = _FakeSession(capture=None, reply="plain report", scout_ledger=list(_DEAD_LEDGER))
    factory = _ScriptedFactory([session])
    ctx = _ctx_with(factory, tracer)

    result = run(ctx.agent("scout", tools=[_ReadStub()], label="scout:0", enforcement_strength=ENFORCEMENT_OFF))

    assert result == "plain report"
    assert len(factory.builds) == 1
    assert not any(s["step_type"] == "dead_scout_synthesis" for s in tracer.steps)
