"""Native DeepSeek thinking preserves forced tools over real SDK HTTP."""

from __future__ import annotations

import copy
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.anthropic_provider import _build_request_kwargs
from opencollab.adapters.llm.types import model_capabilities
from opencollab.adapters.tools.fs import FileReadTool
from opencollab.bootstrap import _workflow_runtime_session as wiring
from opencollab.domain.session import SessionPhase

_MODELS = ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"]
_MESSAGES = [{"role": "user", "content": "Read f.py"}]
_TOOLS = [{"type": "function", "function": {
    "name": "file_read", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}}]
_NAMED = {"type": "function", "function": {"name": "file_read"}}
_THINKING = {"thinking": {"type": "enabled"}}

@contextmanager
def _choice_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            choice = body.get("tool_choice", {})
            forced = choice.get("type") in {"any", "tool"}
            call_id = f"call_read_{len(requests)}"
            name = choice.get("name") or body.get("tools", [{}])[0].get("name")
            arguments = {"x": 7} if name == "structured_output" else {"path": "f.py"}
            content = (
                [{"type": "tool_use", "id": call_id, "name": name, "input": arguments}]
                if forced else [{"type": "text", "text": "done"}]
            )
            payload = json.dumps({
                "id": "msg_choice", "type": "message", "role": "assistant", "model": body["model"],
                "content": content, "stop_reason": "tool_use" if forced else "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 3},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_UNBOUNDED_LIMITS", "false")
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")

def _app(tmp_path, base_url, model="deepseek-flash"):
    return OpenCollab(
        tmp_path, provider="anthropic", model=model,
        api_key="fake-key-for-test", base_url=base_url,  # pragma: allowlist secret
        config={"llm_max_retries": 0, "thinking": True, "thinking_params": _THINKING},
    )

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("choice,native", [
    ("required", {"type": "any"}), (_NAMED, {"type": "tool", "name": "file_read"}),
])
async def test_public_model_client_matches_native_sdk_forced_tool_use(tmp_path, model, choice, native):
    original = copy.deepcopy(choice)
    with _choice_server() as (base_url, requests):
        async with anthropic.AsyncAnthropic(
            api_key="fake-key-for-test", base_url=base_url, max_retries=0,  # pragma: allowlist secret
        ) as sdk:
            control = await sdk.messages.create(
                model=model, messages=_MESSAGES, max_tokens=8192, thinking={"type": "enabled"},
                tools=[{"name": "file_read", "input_schema": _TOOLS[0]["function"]["parameters"]}],
                tool_choice=native,
            )
        async with _app(tmp_path, base_url, model).create_model_client() as client:
            response = await client.complete(
                _MESSAGES, tools=_TOOLS, thinking=True, thinking_params=_THINKING, tool_choice=choice,
            )
    assert control.content[0].type == "tool_use"
    assert control.content[0].id == "call_read_1"
    assert response.tool_calls == [{"id": "call_read_2", "type": "function", "function": {
        "name": "file_read", "arguments": '{"path": "f.py"}',
    }}]
    assert response.usage.total_tokens == 5
    assert len(requests) == 2
    assert all(body["tool_choice"] == native for body in requests)
    assert choice == original
    rows = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "success"
    assert rows[0]["usage"]["total_tokens"] == 5

@pytest.mark.parametrize("entry", ["named", "structured"])
async def test_public_workflow_executes_forced_tool_and_counts_usage(tmp_path, monkeypatch, entry):
    (tmp_path / "f.py").write_text("observed file content\n")
    sessions = []
    build = wiring.build_session

    def capture(**kwargs):
        session = build(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(wiring, "build_session", capture)

    async def flow(ctx, _inputs):
        if entry == "structured":
            return await ctx.agent("Return x", tools=(), schema={
                "type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"],
            })
        return await ctx.agent("Read f.py", tools=[FileReadTool()], tool_choice=_NAMED)

    with _choice_server() as (base_url, requests):
        result = await _app(tmp_path, base_url).workflow(flow, budget=50_000, max_steps=3, trace=False)
    assert result.status == "completed", result.reason
    if entry == "structured":
        assert result.output == {"x": 7}
        assert len(requests) == 2 and result.tokens == 10
        assert sessions[0].phase is SessionPhase.DONE
        assert sessions[-1].phase is SessionPhase.STOPPED
        assert sessions[-1].state.terminal_reason == "interrupted by user"
        assert requests[0].get("tool_choice", {"type": "auto"}) == {"type": "auto"}
        assert requests[1]["tool_choice"] == {"type": "tool", "name": "structured_output"}
    else:
        assert len(requests) == 3 and result.tokens == 15
        assert result.output == ""
        assert len(sessions) == 1
        session = sessions[0]
        assert session.phase is SessionPhase.STOPPED
        assert session.state.terminal_reason == "step limit reached: 3 steps"
        assert session.state.step_count == 3 and session.used_tokens == 15
        assert all(body["tool_choice"] == {"type": "tool", "name": "file_read"} for body in requests)
        calls = [call for message in session.messages for call in message.get("tool_calls", [])]
        assert len(calls) == 3 and len({call["id"] for call in calls}) == 3
        assert all(call["function"]["name"] == "file_read" for call in calls)
        results = [message for message in session.messages if message["role"] == "tool"]
        assert [message["tool_call_id"] for message in results] == [call["id"] for call in calls]
        assert all("observed file content" in message["content"] for message in results)
        for index, request in enumerate(requests[1:], start=1):
            blocks = [
                block for message in request["messages"]
                for block in message["content"] if isinstance(block, dict)
            ]
            uses = [block for block in blocks if block.get("type") == "tool_use"]
            replies = [block for block in blocks if block.get("type") == "tool_result"]
            assert [block["id"] for block in uses] == [call["id"] for call in calls[:index]]
            assert [block["tool_use_id"] for block in replies] == [call["id"] for call in calls[:index]]
            assert all("observed file content" in str(block["content"]) for block in replies)
    rows = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(rows) == len(requests)
    assert all(row["status"] == "success" and row["usage"]["total_tokens"] == 5 for row in rows)
    assert sum(row["usage"]["total_tokens"] for row in rows) == result.tokens

@pytest.mark.parametrize("choice", ["auto", "none"])
async def test_normal_choices_reach_http_without_forcing_tools(tmp_path, choice):
    with _choice_server() as (base_url, requests):
        async with _app(tmp_path, base_url).create_model_client() as client:
            response = await client.complete(
                _MESSAGES, tools=_TOOLS, thinking=True, thinking_params=_THINKING, tool_choice=choice,
            )
    assert response.content == "done" and response.tool_calls == []
    assert len(requests) == 1 and requests[0]["tool_choice"] == {"type": choice}

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("prefix,suffix", [("", ""), ("gateway/", "-2026-09-21")])
def test_exact_model_aliases_keep_protocol_capabilities_independent(model, prefix, suffix):
    capabilities = model_capabilities(prefix + model + suffix)
    assert capabilities.supports_anthropic_thinking_forced_tool_choice is True
    request = _build_request_kwargs(
        prefix + model + suffix, _MESSAGES, _TOOLS, 0.2,
        thinking=True, thinking_params=_THINKING, tool_choice="required",
    )
    assert request["tool_choice"] == {"type": "any"}
    if model == "deepseek-v4-flash":
        assert capabilities.supports_forced_tool_choice is False

@pytest.mark.parametrize("model", [
    "claude-sonnet-4-5", "gateway/unknown", "deepseek-pro", "deepseek-v4.1-flash",
    "deepseek-flash-preview", "deepseek-v4-flash-vision-exp-preview",
])
@pytest.mark.parametrize("choice", ["required", _NAMED])
def test_manual_thinking_keeps_existing_policy_for_unverified_models(model, choice):
    request = _build_request_kwargs(
        model, _MESSAGES, _TOOLS, 0.2, thinking=True,
        thinking_params={"thinking": {"type": "enabled", "budget_tokens": 1024}}, tool_choice=choice,
    )
    assert request["tool_choice"] == {"type": "auto"}

@pytest.mark.parametrize("tools,choice", [
    (None, "required"), (_TOOLS, {"type": "function", "function": {"name": "missing"}}),
])
async def test_invalid_target_fails_before_http(tmp_path, tools, choice):
    with _choice_server() as (base_url, requests):
        async with _app(tmp_path, base_url).create_model_client() as client:
            with pytest.raises(ValueError):
                await client.complete(
                    _MESSAGES, tools=tools, thinking=True, thinking_params=_THINKING, tool_choice=choice,
                )
    assert requests == []
