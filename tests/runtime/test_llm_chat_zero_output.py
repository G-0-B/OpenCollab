"""Explicit zero output usage remains usable for a completed empty Chat turn."""

from __future__ import annotations

import pytest

from opencollab.adapters.llm.errors import StreamedUsageUnavailableError
from opencollab.adapters.tools.fs import FileReadTool, FileWriteTool
from opencollab.domain.session import SessionPhase
from tests.runtime.test_llm_chat_streaming import (
    FakeClient,
    chunk,
    run_stream,
    tool_delta,
    usage_chunk,
)
from tests.runtime.test_llm_chat_streaming_session import USAGE, completion, session_with_wire

ZERO_OUTPUT = {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}

async def test_complete_empty_stream_preserves_reported_zero_output():
    response, client = await run_stream([
        chunk(delta={"role": "assistant"}), chunk(finish_reason="stop"),
        usage_chunk(ZERO_OUTPUT),
    ])
    assert not response.content and not response.tool_calls
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 0
    assert response.usage.estimated is False
    assert all(stream.closed for stream in client.streams)

@pytest.mark.parametrize("delta", [
    {"content": "Visible answer"},
    {"content": " "},
    {"reasoning_content": "Think before answering."},
    {"refusal": "Unable to answer."},
    tool_delta(0, call_id="call-1", name="read", arguments="{}"),
])
async def test_nonempty_stream_still_rejects_zero_output_usage(delta):
    with pytest.raises(StreamedUsageUnavailableError):
        await run_stream([chunk(delta=delta), chunk(finish_reason="stop"), usage_chunk(ZERO_OUTPUT)])

@pytest.mark.parametrize("usage", [
    None,
    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    {**ZERO_OUTPUT, "total_tokens": 101},
    {**ZERO_OUTPUT, "completion_tokens_details": {"reasoning_tokens": 1}},
])
async def test_empty_stream_still_requires_consistent_reported_usage(usage):
    script = [chunk(delta={"role": "assistant"}), chunk(finish_reason="stop")]
    if usage is not None:
        script.append(usage_chunk(usage))
    with pytest.raises(StreamedUsageUnavailableError):
        await run_stream(script)

@pytest.mark.parametrize("stream", [False, True])
async def test_empty_zero_output_turn_reaches_session_correction_and_tools(tmp_path, stream):
    (tmp_path / "input.txt").write_text("source content\n", encoding="utf-8")
    calls = [
        {"id": "read", "type": "function", "function": {
            "name": "file_read", "arguments": '{"path":"input.txt"}',
        }},
        {"id": "write", "type": "function", "function": {
            "name": "file_write", "arguments": '{"path":"output.txt","mode":"create","content":"delivered"}',
        }},
    ]
    if stream:
        scripts = [
            [chunk(delta={"role": "assistant"}), chunk(finish_reason="stop"), usage_chunk(ZERO_OUTPUT)],
            [chunk(delta={"tool_calls": [{"index": index, **call} for index, call in enumerate(calls)]}),
             chunk(finish_reason="tool_calls"), usage_chunk(USAGE)],
            [chunk(delta={"content": "done"}), chunk(finish_reason="stop"),
             usage_chunk(USAGE)],
        ]
    else:
        first = completion(None, [], "stop")
        first.usage.completion_tokens = 0
        first.usage.total_tokens = 100
        scripts = [first, completion(None, calls, "tool_calls"), completion("done", [], "stop")]
    wire = FakeClient(scripts)
    session = await session_with_wire(tmp_path, wire, stream=stream, tools=[FileReadTool(), FileWriteTool()])

    assert await session.run_loop() == "done"
    assert session.phase is SessionPhase.DONE
    assert session.used_tokens == 500
    assert len(wire.calls) == 3
    assert len(wire.calls[1]["messages"]) > len(wire.calls[0]["messages"])
    receipts = {row["tool_call_id"]: row["content"] for row in wire.calls[2]["messages"] if row["role"] == "tool"}
    assert "source content" in receipts["read"]
    assert "Created/wrote" in receipts["write"]
    assert (tmp_path / "output.txt").read_text() == "delivered"
    assert all(stream.closed for stream in wire.streams)

@pytest.mark.parametrize("stream", [False, True])
async def test_zero_output_provider_limit_still_stops_without_executing_tools(tmp_path, stream):
    if stream:
        script = [chunk(delta={"role": "assistant"}), chunk(finish_reason="length"), usage_chunk(ZERO_OUTPUT)]
    else:
        script = completion(None, [], "length")
        script.usage.completion_tokens = 0
        script.usage.total_tokens = 100
    wire = FakeClient([script])
    session = await session_with_wire(tmp_path, wire, stream=stream, tools=[FileWriteTool()])

    await session.run_loop()

    assert session.phase is SessionPhase.STOPPED
    assert session.state.terminal_reason == "output truncated: provider reached its generation limit"
    assert session.used_tokens == 100
    assert len(wire.calls) == 1
    assert list(tmp_path.iterdir()) == []
    assert all(stream.closed for stream in wire.streams)
