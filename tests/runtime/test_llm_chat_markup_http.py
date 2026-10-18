"""Quoted tool markers and valid tool markup through the real Chat SDK."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import openai
import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient
from opencollab.tools import builtin_tools

_BEGIN = "<|tool_calls_section_begin|>"
_END = "<|tool_calls_section_end|>"
_REVERSE = f"The end boundary is `{_END}`. The start boundary is `{_BEGIN}`."
_FORWARD = f"The start boundary is `{_BEGIN}`. The end boundary is `{_END}`."
_CALL = {"id": "read-1", "type": "function", "function": {
    "name": "file_read", "arguments": '{"path":"facts.txt"}',
}}
_MARKUP = (
    _BEGIN + "<|tool_call_begin|>functions.file_read:read-1"
    '<|tool_call_argument_begin|>{"path":"facts.txt"}<|tool_call_end|>' + _END
)
_USAGE = {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}

def _reply(stream, content, structured):
    common = {"id": "chatcmpl-markers", "created": 1, "model": "gpt-4o"}
    finish = "tool_calls" if structured else "stop"
    if not stream:
        message = {"role": "assistant", "content": content}
        if structured:
            message["tool_calls"] = [_CALL]
        return httpx.Response(200, json={
            **common, "object": "chat.completion", "usage": _USAGE,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        })
    common["object"] = "chat.completion.chunk"
    deltas = [{"content": content[offset:offset + 7]} for offset in range(0, len(content), 7)]
    if structured:
        deltas.append({"tool_calls": [{"index": 0, **_CALL}]})
    frames = [
        {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        for delta in deltas
    ]
    frames.extend([
        {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        {**common, "choices": [], "usage": _USAGE},
    ])
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body.encode())

@asynccontextmanager
async def _client(stream, content, structured=False):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return _reply(stream, content, structured)

    client = LLMClient(model="gpt-4o", api_key="unused", max_retries=0, stream_chat=stream)  # pragma: allowlist secret
    await client._openai.close()
    client._openai = openai.AsyncOpenAI(
        api_key="unused", base_url="https://markers.invalid/v1", max_retries=0,  # pragma: allowlist secret
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        yield client, requests
    finally:
        await client.close()

@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "content", [_REVERSE, _FORWARD, _REVERSE + "\n" + _REVERSE, f"The start boundary is `{_BEGIN}`."],
    ids=["reversed", "forward", "repeated", "missing-end"],
)
async def test_quoted_marker_explanation_remains_plain_text(tmp_path, stream, content):
    async with _client(stream, content) as (client, requests):
        result = await OpenCollab(
            tmp_path, model="gpt-4o", provider="openai", api_key="unused",  # pragma: allowlist secret
            config={"thinking": False, "top_p": None},
        ).agent(
            "Explain the tool protocol boundaries.", llm=client, tools=(),
            max_steps=3, budget=10_000, timeout=10, cleanup_timeout=1,
            trace=False, artifacts=tmp_path / "artifacts",
        )

    assert result.status == "completed"
    assert result.output == content
    assert result.tokens == 20
    assert len(requests) == 1
    history = json.loads((tmp_path / "artifacts" / "agent.json").read_text())["messages"]
    assert history[-1]["content"] == content

@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("structured", [False, True], ids=["markup", "structured"])
async def test_valid_markup_and_structured_calls_keep_their_tool_calls(stream, structured):
    content = _REVERSE if structured else _MARKUP
    async with _client(stream, content, structured) as (client, requests):
        response = await client.complete([{"role": "user", "content": "Read the facts."}])

    assert response.tool_calls == [_CALL]
    assert response.content == (_REVERSE if structured else None)
    assert response.finish_reason == ("tool_calls" if structured else "stop")
    assert response.usage.total_tokens == 20
    assert response.usage.markup_recovered == (0 if structured else 1)
    assert len(requests) == 1

@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("wrapper", [
    "```text\n{}\n```",
    "~~~text\n{}\n~~~",
    "````text\n```\n{}\n```\n````",
    "   ```text\n{}\n   ```",
    "```text\n{}",
    "> {}",
    "Use `{}`.",
    "Use ``{}``.",
    "    {}",
    "\t{}",
], ids=["backticks", "tildes", "longer-fence", "indented-fence", "unclosed-fence", "blockquote",
        "inline-code", "long-inline-code", "indented-code", "tab-code"])
async def test_complete_protocol_example_remains_text_with_tools_available(tmp_path, stream, wrapper):
    content = "The tool protocol example follows.\n" + wrapper.format(_MARKUP)
    (tmp_path / "facts.txt").write_text("facts remain available", encoding="utf-8")
    async with _client(stream, content) as (client, requests):
        result = await OpenCollab(tmp_path, model="gpt-4o").agent(
            "Explain the tool protocol with an example.", llm=client,
            tools=builtin_tools("file_read", headless=False),
            max_steps=3, budget=10_000, timeout=10, trace=False, artifacts=tmp_path / "artifacts",
        )
    assert result.status == "completed"
    assert result.output == content
    assert result.tokens == 20
    assert result.metrics.get("markup_recovered", 0) == 0
    assert len(requests) == 1
    history = json.loads((tmp_path / "artifacts" / "agent.json").read_text())["messages"]
    assert all(message["role"] != "tool" for message in history)

@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_bare_protocol_after_fenced_example_is_recovered(stream):
    example = "Protocol example.\n```text\n" + _MARKUP + "\n```\n"
    async with _client(stream, example + _MARKUP) as (client, requests):
        response = await client.complete([{"role": "user", "content": "Read the facts."}])
    assert response.content == example.strip()
    assert response.tool_calls == [_CALL]
    assert response.usage.markup_recovered == 1
    assert len(requests) == 1

@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_structured_call_with_fenced_example_keeps_native_call(stream):
    content = "```text\n" + _MARKUP + "\n```"
    async with _client(stream, content, structured=True) as (client, requests):
        response = await client.complete([{"role": "user", "content": "Read the facts."}])
    assert response.content == content
    assert response.tool_calls == [_CALL]
    assert response.usage.markup_recovered == 0
    assert len(requests) == 1

@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_bare_markup_keeps_indented_json_and_backticks_inside_arguments(stream):
    arguments = json.dumps({"path": "facts`edition`.txt"}, indent=4)
    markup = (
        _BEGIN + "\n<|tool_call_begin|>functions.file_read:read-1"
        "<|tool_call_argument_begin|>" + arguments + "<|tool_call_end|>\n" + _END
    )
    async with _client(stream, markup) as (client, requests):
        response = await client.complete([{"role": "user", "content": "Read the facts."}])
    assert response.tool_calls == [{
        **_CALL, "function": {"name": "file_read", "arguments": arguments},
    }]
    assert response.usage.markup_recovered == 1
    assert len(requests) == 1
