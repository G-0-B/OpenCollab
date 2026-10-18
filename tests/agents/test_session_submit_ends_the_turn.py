"""Explicit submit ends a turn and preserves the model-provided summary."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.submit import MAX_SUMMARY_CHARS, SubmitTool
from opencollab.adapters.tui import TUI
from opencollab.application.event_bus import EventBus
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase, SessionState
from opencollab.domain.tools import ToolProcessingResult
from tests.support.session_characterization_test_support import FakeLLMClient
from tests.support.session_run_loop_test_support import (
    FakeLLM,
    FakeToolExecution,
    build_runner,
    llm_response,
    run,
    tool_call,
)

SUMMARY = "Fixed the off-by-one in _rows() and ran the suite."

def _runner(*, submitted: bool, responses):
    state = SessionState(messages=[{"role": "system", "content": "sys"}])
    result = ToolProcessingResult(
        messages_to_append=[{"role": "tool", "tool_call_id": "t1", "content": "ok"}],
        turn_submitted=submitted,
        submitted_summary=SUMMARY if submitted else None,
    )
    llm = FakeLLM(responses=responses)
    return build_runner(state=state, llm=llm, tool_execution=FakeToolExecution(result)), llm

def test_a_submitted_step_ends_the_turn_at_done_with_its_summary():
    # Two responses are scripted; only the first should ever be consumed.
    runner, llm = _runner(
        submitted=True,
        responses=[
            llm_response(tool_calls=[tool_call("t1", "submit", '{"summary": "s"}')]),
            llm_response(content="this second turn must never run"),
        ],
    )

    answer = run(runner.run_loop())

    assert runner.state.phase is SessionPhase.DONE
    assert runner.state.terminal_reason == "submitted"
    assert answer == SUMMARY
    assert len(llm.calls) == 1

def test_a_step_that_did_not_submit_still_loops_back_to_another_call():
    runner, llm = _runner(
        submitted=False,
        responses=[
            llm_response(tool_calls=[tool_call("t1", "bash", '{"command": "ls"}')]),
            llm_response(content="done exploring"),
        ],
    )

    answer = run(runner.run_loop())

    assert runner.state.phase is SessionPhase.DONE
    assert answer == "done exploring"
    assert len(llm.calls) == 2

def test_submit_is_a_success_terminal_not_a_stop():
    runner, _ = _runner(
        submitted=True,
        responses=[llm_response(tool_calls=[tool_call("t1", "submit", '{"summary": "s"}')])],
    )

    run(runner.run_loop())

    # STOPPED is the controlled-halt terminal and the scheduler turns it into
    # "Error: agent stopped"; an agent that said it was finished must not be
    # reported as one that was cut off.
    assert runner.state.phase is not SessionPhase.STOPPED
    assert runner.state.phase is SessionPhase.DONE

@pytest.mark.parametrize("content", [None, SUMMARY, SUMMARY + "\n", "I have checked the requested change."])
@pytest.mark.parametrize("aid", [0, 7])
async def test_public_session_delivers_submitted_text_once_to_its_agent(tmp_path, content, aid):
    from tests.support.session_characterization_test_support import llm_response as response

    model = FakeLLMClient([response(
        content=content, tool_calls=[tool_call("t1", "submit", '{"summary": "%s"}' % SUMMARY)],
    )])
    events = []
    console = Console(file=StringIO(), width=120, color_system=None)
    tui = TUI(console)
    tui._live_paused = True
    event_bus = EventBus(events.append)
    event_bus.subscribe(tui.event_handler)
    session = build_session(
        agent=Agent(name="lead" if aid == 0 else "worker", system_prompt="Finish.", tools=[SubmitTool()]),
        llm=model, env=LocalEnvironment(str(tmp_path)), aid=aid,
        event_sink=event_bus,
    )
    try:
        await session.add_user_message("Return the verified result.")
        assert await session.run_loop() == SUMMARY
        assert session.phase is SessionPhase.DONE
        assert session.state.terminal_reason == "submitted"
        assert len(model.calls) == 1
        assert session.messages[-1]["content"] == SUMMARY
        expected_text = [content] if content else []
        if content is None or content.strip() != SUMMARY:
            expected_text.append(SUMMARY)
        text_events = [event for event in events if event.type == "text_delta"]
        assert [event.data["content"] for event in text_events] == expected_text
        assert all(event.data["aid"] == aid for event in text_events)
        tui.settle_turn()
        if aid:
            assert SUMMARY not in console.file.getvalue()
            tui.select_agent(aid)
        assert console.file.getvalue().count(SUMMARY) == 1
    finally:
        await session.aclose()

def _call(tool, params):
    return run(tool.execute_with_runtime(params, None))

def test_the_tool_refuses_an_empty_or_oversized_summary():
    tool = SubmitTool()

    assert _call(tool, {}).startswith("Error")
    assert tool.turn_submitted is False
    assert _call(tool, {"summary": "   "}).startswith("Error")
    assert tool.turn_submitted is False
    assert _call(tool, {"summary": "x" * (MAX_SUMMARY_CHARS + 1)}).startswith("Error")
    assert tool.turn_submitted is False

def test_a_bad_call_after_a_good_one_does_not_leave_the_turn_submitted():
    tool = SubmitTool()

    _call(tool, {"summary": SUMMARY})
    assert tool.turn_submitted is True

    # One instance serves every step of a session, so the flag has to be reset
    # on entry rather than only set on success.
    _call(tool, {"summary": ""})
    assert tool.turn_submitted is False
    assert tool.submitted_summary is None

def test_the_real_executor_carries_the_flag_out_of_the_batch():
    """Exercise the real dispatcher so a fake pre-built result cannot hide a lost flag."""
    from opencollab.application.tool_execution import ToolExecutionUseCase
    from tests.support.tool_execution_test_support import (
        AlwaysAllowPermissionPolicy,
        FakeAgent,
        NullEventPublisher,
    )

    use_case = ToolExecutionUseCase(
        agent=FakeAgent([SubmitTool()]),
        environment=object(),
        state=SessionState(messages=[]),
        event_publisher=NullEventPublisher(),
        safety_policy=object(),
        permission_policy=AlwaysAllowPermissionPolicy(),
    )

    result = run(
        use_case.process(
            [
                {
                    "id": "c1",
                    "function": {
                        "name": "submit",
                        "arguments": '{"summary": "%s"}' % SUMMARY,
                    },
                },
                {
                    "id": "c2",
                    "function": {"name": "submit", "arguments": '{"summary": "again"}'},
                },
            ]
        )
    )

    assert result.turn_submitted is True
    assert result.submitted_summary == SUMMARY
    assert len(result.messages_to_append) == 2
    assert "Submitted" in result.messages_to_append[0]["content"]
    assert result.messages_to_append[1]["tool_call_id"] == "c2"
    assert "Submitted" not in result.messages_to_append[1]["content"]
