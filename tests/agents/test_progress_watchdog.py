"""Progress watchdog and low-yield triggers share the wind-down actuator.

Pending tool results defer the brake until the session ingests them.
"""

from __future__ import annotations

import asyncio
import json

from opencollab.application.event_bus import EventBus
from opencollab.application.session_run import (
    DEFAULT_LOW_YIELD_M,
    DEFAULT_WATCHDOG_K,
    ENFORCEMENT_OFF,
    ENFORCEMENT_ON,
)
from opencollab.application.submit_findings import (
    SUBMIT_TOOL_NAME,
    SubmitFindingsTool,
    commitment_terminus_payload,
)
from opencollab.application.tool_execution import ToolExecutionUseCase
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase, SessionState
from tests.support.session_run_test_support import (
    CapturingToolExecution,
    FakeLLM,
    build_runner,
    llm_response,
    run,
    tool_call,
)
from tests.support.session_run_test_support import (
    ReadStub as _ReadStub,
)
from tests.support.session_run_test_support import (
    agent_with_submit as _agent_with_submit,
)
from tests.support.tool_execution_test_support import build_sensor_use_case as _use_case


def _captured():
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
        "summary": "root cause located",
        "insufficient_evidence": False,
    }

_FORCED_SUBMIT_CHOICE = {"type": "function", "function": {"name": SUBMIT_TOOL_NAME}}

def test_default_knobs():
    assert DEFAULT_WATCHDOG_K == 4
    assert DEFAULT_LOW_YIELD_M == 3

def test_t1_watchdog_trips_with_budget_remaining_and_forces_commit():
    # Budget stays below the wind-down threshold so only the watchdog can trip.
    capture_done = asyncio.Event()
    submit = SubmitFindingsTool(on_capture=capture_done.set)
    agent = Agent(name="scout", system_prompt="s", tools=[_ReadStub(), submit])
    state = SessionState(messages=[{"role": "user", "content": "investigate"}], used_tokens=1_000)
    state.turn.steps_since_progress = DEFAULT_WATCHDOG_K
    llm = FakeLLM(
        [llm_response(tool_calls=[tool_call(arguments=json.dumps(_captured()))], finish_reason="tool_calls")]
    )
    runner = build_runner(
        state=state,
        agent=agent,
        llm=llm,
        tool_execution=CapturingToolExecution(agent),
        max_budget_tokens=100_000,
        commit_reserve=25_000,
        enforcement_strength=ENFORCEMENT_ON,
    )

    run(runner.run_loop(capture_done))

    assert state.wind_down_done is True
    assert state.used_tokens < 75_000
    assert [t.name for t in agent.tools] == [SUBMIT_TOOL_NAME]
    assert agent.tool_choice == _FORCED_SUBMIT_CHOICE
    assert llm.calls[0]["tool_choice"] == _FORCED_SUBMIT_CHOICE
    assert len(llm.calls) == 1
    assert submit.captured is not None
    assert state.phase.is_terminal()
    payload = commitment_terminus_payload(
        role="scout:0", captured=submit.captured, wind_down_done=state.wind_down_done,
        used_tokens=state.used_tokens, max_budget_tokens=100_000,
        wind_down_token_mark=state.wind_down_token_mark, artifact="x",
    )
    assert payload["terminus"] == "forced"

def test_watchdog_trips_after_three_malformed_tool_steps():
    capture_done = asyncio.Event()
    submit = SubmitFindingsTool(on_capture=capture_done.set)
    agent = Agent(name="scout", system_prompt="s", tools=[_ReadStub(), submit])
    state = SessionState(
        messages=[{"role": "user", "content": "investigate"}],
        used_tokens=1_000,
    )
    malformed = [
        llm_response(
            tool_calls=[
                tool_call(
                    call_id=f"broken-{index}",
                    name="file_read",
                    arguments='{"path":',
                )
            ],
            finish_reason="tool_calls",
        )
        for index in range(DEFAULT_WATCHDOG_K)
    ]
    final = llm_response(
        tool_calls=[
            tool_call(
                call_id="submit",
                arguments=json.dumps(_captured()),
            )
        ],
        finish_reason="tool_calls",
    )
    llm = FakeLLM([*malformed, final])
    tool_execution = ToolExecutionUseCase(
        agent=agent,
        environment=None,
        state=state,
        event_publisher=EventBus(None),
    )
    runner = build_runner(
        state=state,
        agent=agent,
        llm=llm,
        tool_execution=tool_execution,
        max_budget_tokens=100_000,
        commit_reserve=25_000,
        enforcement_strength=ENFORCEMENT_ON,
    )

    run(runner.run_loop(capture_done))

    assert len(llm.calls) == DEFAULT_WATCHDOG_K + 1
    assert llm.calls[-1]["tool_choice"] == _FORCED_SUBMIT_CHOICE
    assert state.wind_down_done is True
    assert submit.captured is not None

def test_t1_watchdog_just_below_k_does_not_trip():
    # K-1 no-progress steps must NOT brake (off-by-one guard).
    agent = _agent_with_submit()
    state = SessionState(messages=[{"role": "user", "content": "x"}], used_tokens=1_000)
    state.turn.steps_since_progress = DEFAULT_WATCHDOG_K - 1
    llm = FakeLLM([llm_response(content="still exploring")])
    runner = build_runner(
        state=state, agent=agent, llm=llm,
        max_budget_tokens=100_000, commit_reserve=25_000, enforcement_strength=ENFORCEMENT_ON,
    )
    run(runner.run_loop())
    assert state.wind_down_done is False
    assert [t.name for t in agent.tools] == ["file_read", SUBMIT_TOOL_NAME]

def test_t2_low_yield_brake_trips_at_m_and_forces_commit():
    capture_done = asyncio.Event()
    submit = SubmitFindingsTool(on_capture=capture_done.set)
    agent = Agent(name="scout", system_prompt="s", tools=[_ReadStub(), submit])
    state = SessionState(messages=[{"role": "user", "content": "investigate"}], used_tokens=2_000)
    state.turn.low_yield_since_progress = DEFAULT_LOW_YIELD_M
    llm = FakeLLM(
        [llm_response(tool_calls=[tool_call(arguments=json.dumps(_captured()))], finish_reason="tool_calls")]
    )
    runner = build_runner(
        state=state,
        agent=agent,
        llm=llm,
        tool_execution=CapturingToolExecution(agent),
        max_budget_tokens=100_000,
        commit_reserve=25_000,
        enforcement_strength=ENFORCEMENT_ON,
    )

    run(runner.run_loop(capture_done))

    assert state.wind_down_done is True
    assert state.used_tokens < 75_000
    assert [t.name for t in agent.tools] == [SUBMIT_TOOL_NAME]
    assert agent.tool_choice == _FORCED_SUBMIT_CHOICE
    assert submit.captured is not None
    assert state.phase.is_terminal()

def test_t2_off_pinned_counters_never_brake_reference_behavior():
    # Disabled enforcement must ignore both progress counters.
    agent = _agent_with_submit()
    state = SessionState(messages=[{"role": "user", "content": "investigate"}], used_tokens=2_000)
    state.turn.steps_since_progress = 99
    state.turn.low_yield_since_progress = 99
    llm = FakeLLM([llm_response(content="here is my answer")])
    runner = build_runner(
        state=state,
        agent=agent,
        llm=llm,
        max_budget_tokens=100_000,
        commit_reserve=25_000,
        enforcement_strength=ENFORCEMENT_OFF,
    )

    result = run(runner.run_loop())

    assert result == "here is my answer"
    assert state.phase is SessionPhase.DONE
    assert state.wind_down_done is False
    assert [t.name for t in agent.tools] == ["file_read", SUBMIT_TOOL_NAME]
    assert agent.tool_choice is None
    assert llm.calls[0]["tool_choice"] is None
    assert len(llm.calls) == 1

def test_watchdog_not_entered_while_a_tool_result_is_pending():
    from opencollab.domain.pending import PendingRow, RowKind, RowStatus

    state = SessionState(messages=[{"role": "user", "content": "x"}], used_tokens=1_000)
    state.turn.steps_since_progress = 99
    state.turn.low_yield_since_progress = 99
    state.pending_events.add(
        PendingRow(tool_call_id="t1", kind=RowKind.CHILD_AGENT, order=0, status=RowStatus.PENDING)
    )
    agent = _agent_with_submit()
    llm = FakeLLM([llm_response(content="done")])
    runner = build_runner(
        state=state, agent=agent, llm=llm,
        max_budget_tokens=100_000, commit_reserve=25_000, enforcement_strength=ENFORCEMENT_ON,
    )
    run(runner.run_loop())
    assert state.wind_down_done is False
    assert [t.name for t in agent.tools] == ["file_read", SUBMIT_TOOL_NAME]

def test_watchdog_forced_turn_ignored_escalates_once_then_commits():
    # The scout ignores the forced tool choice once, then commits on the sole retry.
    capture_done = asyncio.Event()
    submit = SubmitFindingsTool(on_capture=capture_done.set)
    agent = Agent(name="scout", system_prompt="s", tools=[_ReadStub(), submit])
    state = SessionState(messages=[{"role": "user", "content": "x"}], used_tokens=1_000)
    state.turn.steps_since_progress = DEFAULT_WATCHDOG_K
    llm = FakeLLM(
        [
            llm_response(tool_calls=[tool_call(name="grep", arguments="{}")], finish_reason="tool_calls"),
            llm_response(
                tool_calls=[tool_call(name="submit_findings", arguments=json.dumps(_captured()))],
                finish_reason="tool_calls",
            ),
        ]
    )
    runner = build_runner(
        state=state,
        agent=agent,
        llm=llm,
        tool_execution=CapturingToolExecution(agent),
        max_budget_tokens=100_000,
        commit_reserve=25_000,
        enforcement_strength=ENFORCEMENT_ON,
    )

    run(runner.run_loop(capture_done))

    assert len(llm.calls) == 2
    assert [t.name for t in agent.tools] == [SUBMIT_TOOL_NAME]
    assert llm.calls[0]["tool_choice"] == _FORCED_SUBMIT_CHOICE
    assert llm.calls[1]["tool_choice"] == _FORCED_SUBMIT_CHOICE
    assert submit.captured is not None

class ScriptedTool:
    def __init__(self, name, outputs):
        self.name = name
        self._outputs = list(outputs)

    async def execute_with_runtime(self, args, runtime):
        return self._outputs.pop(0) if self._outputs else ""

def _run_one(state, tool, name, args):
    call = {"id": "c1", "function": {"name": name, "arguments": args}}
    run(_use_case(state, tool).process([call])).apply_to(state)

def test_steps_since_progress_resets_on_user_turn():
    state = SessionState(messages=[])
    state.turn.steps_since_progress = 5
    state.reset_for_user_turn()
    assert state.turn.steps_since_progress == 0
