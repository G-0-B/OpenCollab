"""Session wind-down preserves a reusable Agent configuration template."""

from __future__ import annotations

import asyncio

import pytest

from opencollab.adapters.tools.submit import SubmitTool
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from tests.support.session_run_test_support import FakeLLM, agent_with_submit, llm_response, tool_call


@pytest.mark.parametrize("create_second_before_first_turn", [False, True])
def test_wind_down_preserves_tools_for_a_session_reusing_the_template(
    create_second_before_first_turn,
):
    template = agent_with_submit()
    first_model = FakeLLM([llm_response(content="first protected answer")])
    second_model = FakeLLM([llm_response(content="second answer")])
    first = build_session(agent=template, llm=first_model, max_budget_tokens=50_000)

    async def scenario():
        second = None
        if create_second_before_first_turn:
            second = build_session(agent=template, llm=second_model, max_budget_tokens=50_000)
            await second.add_user_message("Read for an independent turn.")
        await first.add_user_message("Inspect and submit.")
        first.runner.configure_enforcement(enforcement_strength="needs-enforcement")
        first.used_tokens = 25_000
        await first.run_loop()
        if second is None:
            second = build_session(agent=template, llm=second_model, max_budget_tokens=50_000)
            await second.add_user_message("Read for a later turn.")
        await second.run_loop()
        return second

    second = asyncio.run(scenario())

    assert first.state.wind_down_done is True
    assert [tool.name for tool in first.agent.tools] == ["submit_findings"]
    assert [spec["function"]["name"] for spec in second_model.calls[0]["tools"]] == [
        "file_read", "submit_findings"
    ]
    assert second_model.calls[0]["tool_choice"] is None
    assert [tool.name for tool in template.tools] == ["file_read", "submit_findings"]
    assert template.tool_choice is None
    assert first.agent is first.runner.agent is first.tool_execution.agent
    assert second.agent is second.runner.agent is second.tool_execution.agent
    assert first.agent is not second.agent
    assert first.agent.tools[0] is template.tools[1]

def test_separate_templates_keep_their_tool_configuration():
    first_template = agent_with_submit()
    second_template = agent_with_submit()
    first = build_session(agent=first_template, llm=FakeLLM([llm_response(content="first")]))
    second = build_session(agent=second_template, llm=FakeLLM([llm_response(content="second")]))

    async def scenario():
        await first.add_user_message("first")
        await first.run_loop()
        await second.add_user_message("second")
        await second.run_loop()

    asyncio.run(scenario())
    assert [tool.name for tool in first.agent.tools] == ["file_read", "submit_findings"]
    assert [tool.name for tool in second.agent.tools] == ["file_read", "submit_findings"]

@pytest.mark.asyncio
@pytest.mark.parametrize("shared, concurrent", [(True, True), (False, True), (True, False)])
async def test_reused_submit_tool_keeps_each_sessions_accepted_answer(shared, concurrent):
    template = Agent(name="lead", system_prompt="Finish the task.", tools=[SubmitTool()])
    second_template = template if shared else Agent(
        name="lead", system_prompt="Finish the task.", tools=[SubmitTool()],
    )
    sessions = [
        build_session(
            agent=agent,
            llm=FakeLLM([llm_response(tool_calls=[tool_call(
                name="submit", arguments='{"summary": "%s"}' % summary,
            )])]),
        )
        for agent, summary in zip([template, second_template], ["first", "second"])
    ]
    for session, task in zip(sessions, ["first task", "second task"]):
        await session.add_user_message(task)

    if concurrent:
        answers = await asyncio.gather(*(session.run_loop() for session in sessions))
    else:
        answers = [await session.run_loop() for session in sessions]

    assert answers == ["first", "second"]
    assert [session.messages[-1]["content"] for session in sessions] == answers
    assert all(session.state.terminal_reason == "submitted" for session in sessions)
    assert sessions[0].agent.tools[0] is template.tools[0]
    assert sessions[1].agent.tools[0] is second_template.tools[0]

@pytest.mark.asyncio
async def test_invalid_concurrent_submit_keeps_the_other_sessions_accepted_answer():
    template = Agent(name="lead", system_prompt="Finish the task.", tools=[SubmitTool()])
    first = build_session(agent=template, llm=FakeLLM([llm_response(tool_calls=[
        tool_call(name="submit", arguments='{"summary": "accepted"}'),
    ])]))
    second = build_session(agent=template, llm=FakeLLM([
        llm_response(tool_calls=[tool_call(name="submit", arguments='{"summary": ""}')]),
        llm_response(content="corrected answer"),
    ]))
    await first.add_user_message("first task")
    await second.add_user_message("second task")

    assert await asyncio.gather(first.run_loop(), second.run_loop()) == ["accepted", "corrected answer"]
    assert first.state.terminal_reason == "submitted"
    assert second.state.terminal_reason == "completed"
