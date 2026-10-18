"""Declared DeepSeek thinking modes reach native SDKs over local HTTP."""

from __future__ import annotations

import copy
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import httpx
import openai
import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.anthropic_provider import _build_request_kwargs as _build_anthropic_kwargs
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.openai_provider import _build_request_kwargs
from opencollab.adapters.llm.responses_provider import _build_request_kwargs as _build_responses_kwargs
from opencollab.adapters.tools.fs import FileReadTool
from opencollab.bootstrap.container import build_session
from opencollab.domain.agent import Agent
from tests.runtime.test_llm_chat_reasoning_replay_http import _completion_response
from tests.runtime.test_responses_http_contract import _completed_event, _item_event
from tests.support.provider_sdk_http import completion_http_response

_MODELS = ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"]
_PROTOCOLS = ["chat", "responses", "anthropic"]
_REASONING = "Inspect f.py before answering."
_READ_CALL = {"id": "read-1", "type": "function", "function": {
    "name": "file_read", "arguments": '{"path":"f.py"}',
}}

@contextmanager
def _deepseek_server(*, tool_round=False, keep_prior_reasoning=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path not in {"/v1/chat/completions", "/v1/responses", "/v1/messages"}:
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            calls = [_READ_CALL] if tool_round and len(requests) == 1 else None
            prior = next((message for message in body.get("messages", []) if message.get("tool_calls")), None)
            invalid_replay = prior is not None and keep_prior_reasoning is not None and (
                prior.get("reasoning_content") != _REASONING if keep_prior_reasoning
                else "reasoning_content" in prior
            )
            if invalid_replay:
                response = httpx.Response(400, json={"error": {
                    "type": "invalid_request_error", "message": "Unexpected prior reasoning_content",
                }})
            elif self.path == "/v1/chat/completions":
                response = _completion_response(
                    body["model"], body.get("stream", False), calls=calls,
                    reasoning=_REASONING if calls else None, content=None if calls else "done",
                )
            elif calls and self.path == "/v1/responses":
                item = {"id": "fc_read", "type": "function_call", "status": "completed",
                        "call_id": _READ_CALL["id"], **_READ_CALL["function"]}
                events = [_item_event(item, 0), _completed_event("resp_read", 2, [item])]
                content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
                response = httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=content)
            elif calls:
                response = httpx.Response(200, json={
                    "id": "msg_read", "type": "message", "role": "assistant", "model": body["model"],
                    "content": [{"type": "tool_use", "id": _READ_CALL["id"], "name": "file_read",
                                 "input": {"path": "f.py"}}], "stop_reason": "tool_use", "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 100},
                })
            else:
                request = httpx.Request("POST", "http://127.0.0.1" + self.path, json=body)
                response = completion_http_response(request, text="done")
            self.send_response(response.status_code)
            self.send_header("Content-Type", response.headers["Content-Type"])
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("thinking", ["default", False, True])
@pytest.mark.parametrize("entry", ["workflow", "team", "agent", "session", "client"])
async def test_deepseek_default_mode_reaches_all_public_entries(
    monkeypatch, tmp_path, model, protocol, thinking, entry,
):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.delenv("OPENCOLLAB_THINKING", raising=False)
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    mode = {} if thinking == "default" else {"thinking": thinking}
    if protocol == "anthropic" and thinking is True:
        mode["thinking_params"] = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    route = {"provider": "anthropic" if protocol == "anthropic" else "openai",
             "wire_protocol": "responses" if protocol == "responses" else "chat_completions"}
    with _deepseek_server() as (base_url, requests):
        if protocol == "anthropic":
            base_url = base_url.removesuffix("/v1")
        app = OpenCollab(
            tmp_path, model=model, api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url, config={"llm_max_retries": 0, **route, **mode},
        )
        if entry == "workflow":
            async def flow(ctx, _inputs):
                return await ctx.agent("answer", tools=())

            result = await app.workflow(flow, trace=False, max_steps=1)
            assert result.ok and result.output == "done"
        elif entry == "team":
            team = tmp_path / "team.yaml"
            team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n")
            result = await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)
            assert result.ok and result.output == "done"
        elif entry == "agent":
            result = await app.agent("answer", tools=(), trace=False, max_steps=1)
            assert result.ok and result.output == "done"
        elif entry == "session":
            session = build_session(agent=Agent(
                name="reader", system_prompt="Reader", model=model,
                api_key="fake-key-for-test",  # pragma: allowlist secret
                base_url=base_url, llm_max_retries=0, **route, **mode,
            ))
            try:
                await session.add_user_message("answer")
                assert await session.run_loop() == "done"
            finally:
                await session.aclose()
        else:
            async with LLMClient(
                model=model, api_key="fake-key-for-test",  # pragma: allowlist secret
                base_url=base_url, max_retries=0, **route,
            ) as client:
                response = await client.complete([{"role": "user", "content": "answer"}], **mode)
                assert response.content == "done"
    assert len(requests) == 1
    assert requests[0]["model"] == model
    if thinking is True:
        if protocol == "anthropic":
            assert requests[0]["thinking"] == mode["thinking_params"]["thinking"]
        else:
            assert "thinking" not in requests[0]
        assert "reasoning" not in requests[0]
        assert requests[0].get("enable_thinking") is not False
    else:
        if protocol == "responses":
            assert requests[0].get("reasoning") == {"effort": "none"}, requests[0]
            assert "thinking" not in requests[0]
        else:
            assert requests[0].get("thinking") == {"type": "disabled"}, requests[0]
        assert "enable_thinking" not in requests[0]

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("prefix", ["", "deepseek/", "gateway/"])
@pytest.mark.parametrize("suffix", ["", "-2026", "-2026-09", "-2026-09-21"])
def test_deepseek_native_disable_accepts_existing_model_aliases(model, protocol, prefix, suffix):
    builder = {"chat": _build_request_kwargs, "responses": _build_responses_kwargs,
               "anthropic": _build_anthropic_kwargs}[protocol]
    kwargs = builder(prefix + model + suffix, [{"role": "user", "content": "answer"}], None, 0.2)
    if protocol == "chat":
        assert kwargs.get("extra_body") == {"thinking": {"type": "disabled"}}
    elif protocol == "responses":
        assert kwargs.get("reasoning") == {"effort": "none"}
        assert "extra_body" not in kwargs
    else:
        assert kwargs.get("thinking") == {"type": "disabled"}

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("params", [
    {"enable_thinking": True},
    {"thinking": {"type": "enabled", "keep": "all"}},
    {"thinking": {"type": "disabled"}},
])
def test_deepseek_explicit_protocol_keeps_priority(model, thinking, params):
    original = copy.deepcopy(params)
    kwargs = _build_request_kwargs(
        model, [{"role": "user", "content": "answer"}], None, 0.2,
        thinking=thinking, thinking_params=params, reasoning_effort="max",
    )
    expected = params if thinking else (
        {"enable_thinking": False} if "enable_thinking" in params else {"thinking": {"type": "disabled"}}
    )
    assert kwargs["extra_body"] == expected
    assert kwargs["reasoning_effort"] == "max"
    assert params == original

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("effort", [None, "high", "max"])
@pytest.mark.parametrize("params", [None, {"enable_thinking": True}, {"thinking": {"type": "enabled"}}])
def test_deepseek_responses_keeps_explicit_effort_and_extensions(model, thinking, effort, params):
    kwargs = _build_responses_kwargs(
        model, [{"role": "user", "content": "answer"}], None, 0.2,
        thinking=thinking, reasoning_effort=effort, thinking_params=params, native_openai=False,
    )
    expected = effort if effort is not None else (None if thinking else "none")
    assert kwargs.get("reasoning") == (None if expected is None else {"effort": expected})
    if params is None:
        assert "extra_body" not in kwargs
    else:
        assert kwargs["extra_body"] == (params if thinking else (
            {"enable_thinking": False} if "enable_thinking" in params else {"thinking": {"type": "disabled"}}
        ))

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_deepseek_native_sdk_disable_control(model, protocol):
    with _deepseek_server() as (base_url, requests):
        if protocol == "anthropic":
            async with anthropic.AsyncAnthropic(
                api_key="fake-key-for-test", max_retries=0,  # pragma: allowlist secret
                base_url=base_url.removesuffix("/v1"),
            ) as client:
                response = await client.messages.create(
                    model=model, messages=[{"role": "user", "content": "answer"}],
                    max_tokens=8192, thinking={"type": "disabled"},
                )
                assert response.content[0].text == "done"
        else:
            async with openai.AsyncOpenAI(
                api_key="fake-key-for-test", base_url=base_url, max_retries=0,  # pragma: allowlist secret
            ) as client:
                if protocol == "chat":
                    response = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": "answer"}],
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    assert response.choices[0].message.content == "done"
                else:
                    stream = await client.responses.create(
                        model=model, input="answer", reasoning={"effort": "none"}, stream=True,
                    )
                    async with stream:
                        events = [event async for event in stream]
                    assert events[-1].type == "response.completed"
    assert len(requests) == 1
    if protocol == "responses":
        assert requests[0]["reasoning"] == {"effort": "none"}
        assert "thinking" not in requests[0]
    else:
        assert requests[0]["thinking"] == {"type": "disabled"}

@pytest.mark.parametrize("model", [
    "deepseek-pro", "deepseek-reasoner", "deepseek-v4.1-flash", "deepseek-flash-preview",
    "deepseek-v4-pro-custom", "deepseek-v4-flash-vision-custom",
    "deepseek-v4-flash-vision-exp-preview", "gateway/unknown", "gpt-4o",
])
@pytest.mark.parametrize("prefix", ["", "gateway/"])
@pytest.mark.parametrize("protocol", _PROTOCOLS)
def test_unverified_models_keep_existing_default_fields(model, prefix, protocol):
    builder = {"chat": _build_request_kwargs, "responses": _build_responses_kwargs,
               "anthropic": _build_anthropic_kwargs}[protocol]
    kwargs = builder(prefix + model, [{"role": "user", "content": "answer"}], None, 0.2)
    assert "extra_body" not in kwargs
    assert "thinking" not in kwargs
    assert "reasoning" not in kwargs

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_deepseek_workflow_override_selects_disabled_mode(monkeypatch, tmp_path, model, protocol):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    route = {"provider": "anthropic" if protocol == "anthropic" else "openai",
             "wire_protocol": "responses" if protocol == "responses" else "chat_completions"}
    config = {"thinking": True, "thinking_params": {"thinking": {"type": "enabled", "budget_tokens": 1024}}}
    if protocol == "responses":
        config.update(thinking_params={}, reasoning_effort="high")
    with _deepseek_server() as (base_url, requests):
        if protocol == "anthropic":
            base_url = base_url.removesuffix("/v1")
        app = OpenCollab(
            tmp_path, model=f"gateway/{model}-2026-09-21",
            api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url, config={"llm_max_retries": 0, **route, **config},
        )

        async def flow(ctx, _inputs):
            output = await ctx.agent("answer", tools=(), thinking=False)
            assert ctx.sessions[0].agent.thinking is False
            return output

        result = await app.workflow(flow, trace=False, max_steps=1)
        assert result.ok and result.output == "done"
    assert len(requests) == 1
    if protocol == "responses":
        assert requests[0]["reasoning"] == {"effort": "none"}
        assert "thinking" not in requests[0]
    else:
        assert requests[0]["thinking"] == {"type": "disabled"}

@pytest.mark.parametrize("model", ["gpt-5.6-luna", "k3", "kimi-for-coding"])
async def test_fixed_thinking_models_keep_workflow_protection(monkeypatch, tmp_path, model):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    with _deepseek_server() as (base_url, requests):
        app = OpenCollab(
            tmp_path, model=model, api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url, config={"thinking": True, "thinking_params": {"thinking": {"type": "enabled"}},
                                       "llm_max_retries": 0},
        )

        async def flow(ctx, _inputs):
            output = await ctx.agent("answer", tools=(), thinking=False)
            assert ctx.sessions[0].agent.thinking is True
            return output

        result = await app.workflow(flow, trace=False, max_steps=1)
        assert result.ok and result.output == "done"
    assert requests[0]["thinking"] == {"type": "enabled"}

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("protocol,stream", [
    ("chat", False), ("chat", True), ("responses", False), ("anthropic", False),
], ids=["chat-json", "chat-sse", "responses", "anthropic"])
async def test_deepseek_tool_continuation_keeps_selected_mode(
    monkeypatch, tmp_path, model, thinking, protocol, stream,
):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    (tmp_path / "f.py").write_text("x = 1\n")
    params = {"thinking_params": {"thinking": {"type": "enabled", "budget_tokens": 1024}}} if thinking else {}
    route = {"provider": "anthropic" if protocol == "anthropic" else "openai",
             "wire_protocol": "responses" if protocol == "responses" else "chat_completions"}
    if protocol == "responses":
        params = {"reasoning_effort": "high"} if thinking else {}
    with _deepseek_server(tool_round=True, keep_prior_reasoning=protocol == "chat") as (base_url, requests):
        if protocol == "anthropic":
            base_url = base_url.removesuffix("/v1")
        app = OpenCollab(
            tmp_path, model=model, api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url, config={"thinking": thinking, "llm_stream_chat": stream,
                                       "llm_max_retries": 0, **route, **params},
        )
        result = await app.agent("inspect the file", tools=(FileReadTool(),), trace=False, max_steps=3)
        assert result.ok and result.output == "done"
    assert len(requests) == 2
    for request in requests:
        assert request["model"] == model
        if protocol == "responses":
            assert request["reasoning"] == {"effort": "high" if thinking else "none"}
            assert "thinking" not in request
        else:
            assert request.get("stream", False) is stream
            expected = params["thinking_params"]["thinking"] if thinking else {"type": "disabled"}
            assert request["thinking"] == expected
        assert "enable_thinking" not in request
    if protocol == "responses":
        tool = next(item for item in requests[1]["input"] if item.get("type") == "function_call_output")
        assert tool["call_id"] == _READ_CALL["id"]
        assert "x = 1" in tool["output"]
    elif protocol == "anthropic":
        tool = next(block for message in requests[1]["messages"] for block in message["content"]
                    if isinstance(block, dict) and block["type"] == "tool_result")
        assert tool["tool_use_id"] == _READ_CALL["id"]
        assert "x = 1" in tool["content"]
    else:
        assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
        assert assistant["reasoning_content"] == _REASONING
        tool = next(message for message in requests[1]["messages"] if message["role"] == "tool")
        assert tool["tool_call_id"] == _READ_CALL["id"]
        assert "x = 1" in tool["content"]

@pytest.mark.parametrize("model", _MODELS + [f"gateway/{model}-2026-09-21" for model in _MODELS])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("params", [None, {"thinking": {"type": "disabled"}}])
async def test_thinking_change_preserves_required_tool_reasoning(monkeypatch, tmp_path, model, stream, params):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    tools = [{"type": "function", "function": {"name": "file_read", "parameters": {"type": "object"}}}]
    with _deepseek_server(tool_round=True, keep_prior_reasoning=True) as (base_url, requests):
        app = OpenCollab(
            tmp_path, model=model, api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url, config={"llm_stream_chat": stream, "llm_max_retries": 0},
        )
        async with app.create_model_client() as transport:
            messages = [{"role": "user", "content": "inspect the file"}]
            first = await transport.complete(
                messages, tools=tools, thinking=True, thinking_params={"thinking": {"type": "enabled"}},
            )
            assert first.reasoning == _REASONING and first.tool_calls == [_READ_CALL]
            messages.extend([
                {"role": "assistant", "content": first.content, "reasoning_content": first.reasoning,
                 "tool_calls": first.tool_calls},
                {"role": "tool", "tool_call_id": _READ_CALL["id"], "content": "x = 1"},
            ])
            original = copy.deepcopy(messages)
            error = None
            try:
                followup = await transport.complete(messages, tools=tools, thinking=False, thinking_params=params)
            except openai.BadRequestError as exc:
                error = exc
        async with openai.AsyncOpenAI(
            api_key="fake-key-for-test", base_url=base_url, max_retries=0,  # pragma: allowlist secret
        ) as sdk:
            reference = await sdk.chat.completions.create(
                model=model, messages=messages, tools=tools, extra_body={"thinking": {"type": "disabled"}},
            )
        assert reference.choices[0].message.content == "done"
        assert error is None, str(error)
        assert followup.content == "done"
        assert messages == original
    assert len(requests) == 3
    for request in requests[1:]:
        assert request["model"] == model
        assert request["thinking"] == {"type": "disabled"}
        prior = next(message for message in request["messages"] if message.get("tool_calls"))
        assert prior["reasoning_content"] == _REASONING

@pytest.mark.parametrize("model", ["gpt-4o", "gateway/unknown"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_ordinary_followup_cleans_unsupported_reasoning(monkeypatch, tmp_path, model, stream):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    tools = [{"type": "function", "function": {"name": "file_read", "parameters": {"type": "object"}}}]
    with _deepseek_server(tool_round=True, keep_prior_reasoning=False) as (base_url, requests):
        app = OpenCollab(
            tmp_path, model=model, api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url, config={"llm_stream_chat": stream, "llm_max_retries": 0},
        )
        async with app.create_model_client() as transport:
            messages = [{"role": "user", "content": "inspect the file"}]
            params = {"thinking": {"type": "enabled"}}
            first = await transport.complete(messages, tools=tools, thinking=True, thinking_params=params)
            assert first.reasoning == _REASONING
            messages.extend([
                {"role": "assistant", "content": first.content, "reasoning_content": first.reasoning,
                 "tool_calls": first.tool_calls},
                {"role": "tool", "tool_call_id": _READ_CALL["id"], "content": "x = 1"},
            ])
            followup = await transport.complete(messages, tools=tools, thinking=False, thinking_params=params)
            assert followup.content == "done"
    assert len(requests) == 2
    assert requests[1]["thinking"] == {"type": "disabled"}
    prior = next(message for message in requests[1]["messages"] if message.get("tool_calls"))
    assert "reasoning_content" not in prior
    assert messages[1]["reasoning_content"] == _REASONING
