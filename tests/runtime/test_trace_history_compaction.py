"""Persist model-derived or fallback compaction thresholds for each session."""

from __future__ import annotations

import json

from opencollab.adapters.trace import Tracer
from opencollab.bootstrap.container import build_session_runtime
from opencollab.domain.agent import Agent
from tests.support.session_run_loop_test_support import FakeTracer

RECORD_TYPE = "session.history_compaction"

class _Llm:
    """Just the one thing the shaper wiring asks a resolved LLM for."""

    def __init__(self, window):
        self._window = window

    def context_window(self):
        return self._window

    async def complete(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("no model call is made while building a session")

def _agent() -> Agent:
    return Agent(name="coder", system_prompt="sys", model="fake-model")

def _records(tracer: FakeTracer) -> list[dict]:
    return [step["payload"] for step in tracer.steps if step["step_type"] == RECORD_TYPE]

def test_history_compaction_record_names_the_thresholds_in_force():
    tracer = FakeTracer()

    build_session_runtime(agent=_agent(), tracer=tracer, llm=_Llm(1_015_576), aid=3)

    records = _records(tracer)
    assert len(records) == 1
    assert records[0] == {
        "aid": 3,
        "model": "fake-model",
        "context_policy": "default",
        "context_window_tokens": 1_015_576,
        "history_trigger_tokens": 982_576,
        "history_target_tokens": 736_932,
        "history_thresholds_from": "context_window",
    }

def test_history_compaction_record_says_when_the_fixed_defaults_were_used():
    tracer = FakeTracer()

    build_session_runtime(agent=_agent(), tracer=tracer, llm=_Llm(None), aid=0)

    records = _records(tracer)
    assert len(records) == 1
    assert records[0]["context_window_tokens"] is None
    assert records[0]["history_thresholds_from"] == "fixed_default"
    assert records[0]["history_trigger_tokens"] == 120_000
    assert records[0]["history_target_tokens"] == 90_000

def test_history_compaction_record_declares_an_injected_shaper():
    """An injected shaper bypasses threshold derivation."""
    tracer = FakeTracer()

    class _Shaper:
        def shape(self, messages):  # pragma: no cover - never called here
            return messages

    build_session_runtime(
        agent=_agent(), tracer=tracer, llm=_Llm(200_000), aid=1, shaper=_Shaper()
    )

    records = _records(tracer)
    assert len(records) == 1
    assert records[0]["history_thresholds_from"] == "injected_shaper"
    assert records[0]["history_trigger_tokens"] is None
    assert records[0]["history_target_tokens"] is None

def test_history_compaction_record_reaches_the_trajectory_file(tmp_path):
    """Drive the real tracer: a field that never reaches disk fails here."""
    tracer = Tracer(run_id="history-compaction", output_dir=str(tmp_path))
    path = tracer.path
    try:
        build_session_runtime(agent=_agent(), tracer=tracer, llm=_Llm(200_000), aid=2)
        tracer.flush()
    finally:
        tracer.close()

    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    payloads = [r["payload"] for r in records if r["type"] == RECORD_TYPE]
    assert len(payloads) == 1
    assert payloads[0]["history_trigger_tokens"] == 167_000
    assert payloads[0]["history_target_tokens"] == 125_250

def test_session_builds_without_a_tracer():
    """Observation only: no tracer, no record, no failure."""
    runtime = build_session_runtime(agent=_agent(), tracer=None, llm=_Llm(200_000))
    assert runtime is not None
