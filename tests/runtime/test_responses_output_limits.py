"""Provider output limits preserve terminal status and reported consumption."""

import pytest

from opencollab.adapters.llm.responses_provider import _consume_stream, _parse_stream
from tests.support.responses_provider_test_support import FakeStream, completed_response, function_item, ns


@pytest.mark.parametrize("streamed, emit_done", [(False, False), (True, False), (True, True)])
@pytest.mark.asyncio
async def test_output_limit_keeps_usage_without_executing_partial_tool(streamed, emit_done):
    from opencollab.adapters.llm.responses_provider import parse_responses_response
    from opencollab.bootstrap import build_session
    from opencollab.domain.session import SessionPhase
    from tests.support.session_characterization_test_support import FakeAgent, FakeLLMClient

    item = {**function_item("partial", "file_write", '{"path":'), "status": "incomplete"}
    response = completed_response(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[item],
    )
    messages = [{"role": "user", "content": "write the file"}]
    if streamed:
        events = [ns(type="response.function_call_arguments.delta", output_index=0, delta=item["arguments"])]
        if emit_done:
            events.append(ns(type="response.output_item.done", output_index=0, item=item))
        events.append(ns(type="response.incomplete", response=response))
        state = await _consume_stream(FakeStream(events), 1, 1)
        parsed = _parse_stream(state, messages, "gpt-fake")
    else:
        parsed = parse_responses_response(response, messages, expected_model="gpt-fake")
    assert parsed.finish_reason == "max_tokens"
    assert parsed.tool_calls == []
    assert parsed.provider_items == []
    assert parsed.usage.total_tokens == 150
    assert parsed.usage.estimated is False
    session = build_session(agent=FakeAgent(), llm=FakeLLMClient([parsed]))
    await session.run_loop()
    assert session.phase is SessionPhase.STOPPED
    assert session.used_tokens == 150
    assert not any(message.get("tool_calls") for message in session.messages)

@pytest.mark.parametrize("encrypted", [None, "opaque encrypted reasoning"])
@pytest.mark.asyncio
async def test_hidden_reasoning_output_limit_returns_once_with_reported_usage(encrypted):
    from opencollab.adapters.llm.responses_provider import complete_responses
    from opencollab.bootstrap import build_session
    from opencollab.domain.session import SessionPhase
    from tests.support.session_characterization_test_support import FakeAgent, FakeLLMClient

    item = {"type": "reasoning", "id": "rs_hidden", "summary": []}
    if encrypted is not None:
        item["encrypted_content"] = encrypted
    response = completed_response(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[item],
    )

    class Responses:
        calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            return response

    wire = Responses()
    parsed = await complete_responses(
        ns(responses=wire), "gpt-fake", [{"role": "user", "content": "think"}],
        None, 1.0, 1, stream=False,
    )
    assert wire.calls == 1
    assert parsed.finish_reason == "max_tokens"
    assert parsed.content is None
    assert parsed.provider_items == [item]
    assert parsed.usage.total_tokens == 150
    assert parsed.usage.reasoning_tokens == 9
    session = build_session(agent=FakeAgent(), llm=FakeLLMClient([parsed]))
    await session.run_loop()
    assert session.phase is SessionPhase.STOPPED
    assert session.used_tokens == 150

def test_completed_partial_tool_remains_a_protocol_error():
    from opencollab.adapters.llm.responses_provider import ResponsesProtocolError, parse_responses_response

    item = {**function_item("partial", "file_write", '{"path":'), "status": "incomplete"}
    with pytest.raises(ResponsesProtocolError):
        parse_responses_response(
            completed_response(output=[item]), [{"role": "user", "content": "write"}], expected_model="gpt-fake",
        )

def test_completed_hidden_reasoning_remains_empty_output():
    from opencollab.adapters.llm.responses_provider import ResponsesEmptyOutputError, parse_responses_response

    with pytest.raises(ResponsesEmptyOutputError):
        parse_responses_response(
            completed_response(output=[{"type": "reasoning", "id": "rs_1", "summary": []}]),
            [{"role": "user", "content": "think"}], expected_model="gpt-fake",
        )
