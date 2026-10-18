"""Cooperative cancellation settles generation before accepting new tools."""

from __future__ import annotations

import asyncio
import json

import pytest

import opencollab
from opencollab.adapters.tools.human import AskUserTool
from opencollab.application.event_bus import EventBus
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase
from opencollab.environments import local_environment
from opencollab.tools import builtin_tools
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport
from tests.support.responses_provider_test_support import function_item, message_item
from tests.support.session_runtime_test_support import FakeTracer, tool_call


def _tool_response(request, mode, calls, text):
    body = json.loads(request.content)
    if mode == "anthropic":
        return http_response(request, 200, json={
            "id": "msg_late", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": text}, *[
                {"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                 "input": json.loads(call["function"]["arguments"])} for call in calls
            ]],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 2},
        })
    if mode == "responses":
        items = [message_item(text), *[
            function_item(call["id"], call["function"]["name"], call["function"]["arguments"])
            for call in calls
        ]]
        response = {
            "id": "resp_late", "object": "response", "status": "completed", "model": body["model"],
            "output": items, "error": None, "incomplete_details": None,
            "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        }
        events = [
            *[{"type": "response.output_item.done", "output_index": index, "item": item}
              for index, item in enumerate(items)],
            {"type": "response.completed", "response": response},
        ]
        stream = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return http_response(request, 200, headers={"content-type": "text/event-stream"}, text=stream)
    return http_response(request, 200, json={
        "id": "chat_late", "object": "chat.completion", "created": 1, "model": body["model"],
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": text, "tool_calls": calls,
        }}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    })

@pytest.mark.parametrize("mode", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("kind", ["ask", "write", "text"])
@pytest.mark.parametrize("interrupt_boundary", ["none", "generation", "accepted"])
@pytest.mark.parametrize("restored", [False, True])
async def test_cancelled_generation_preserves_text_usage_and_reusable_native_history(
    tmp_path, monkeypatch, mode, kind, interrupt_boundary, restored,
):
    entered, release, cancel = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests, questions, events = [], [], []
    tracer = FakeTracer()
    text = "I identified the parser file."
    name = "ask_user" if kind == "ask" else "file_write"
    arguments = {"question": "Which parser?"} if kind == "ask" else {
        "mode": "create", "path": "config.py", "content": "FEATURE_ON = True\n",
    }
    calls = [tool_call("late_call", name, json.dumps(arguments))]
    target = tmp_path / "config.py"
    target.write_text("FEATURE_ON = False\n", encoding="utf-8")
    path = tmp_path / "session.json"

    class AskPolicy:
        async def ask(self, question):
            questions.append(question)
            return "parser.py"

    async def sink(event):
        events.append(event)
        if interrupt_boundary == "accepted" and event.type == "text_delta":
            cancel.set()

    async def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) > 1:
            return completion_http_response(request, text="Completed.")
        entered.set()
        await release.wait()
        if kind == "text":
            return completion_http_response(request, text=text)
        return _tool_response(request, mode, calls, text)

    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    install_sdk_transport(monkeypatch, "anthropic" if mode == "anthropic" else "openai", handler)
    environment = local_environment(tmp_path)

    def agent():
        return Agent(
            name="writer", system_prompt="Complete the task.",
            model="claude-sonnet-4-6" if mode == "anthropic" else "gpt-4o",
            provider="anthropic" if mode == "anthropic" else "openai",
            api_key="controlled-test", base_url="https://controlled.invalid/v1",  # pragma: allowlist secret
            wire_protocol="responses" if mode == "responses" else "chat_completions",
            tools=[AskUserTool(), *builtin_tools("file_write")],
        )

    session = build_session(
        agent=agent(), env=environment, ask_policy=AskPolicy(), event_sink=EventBus(sink),
        auto_save_path=str(path), tracer=tracer, max_budget_tokens=30_000,
    )
    next_session = session
    await session.add_user_message("Start the task.")
    task = asyncio.create_task(session.run_loop(cancel))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if interrupt_boundary == "generation":
            cancel.set()
        release.set()
        answer = await asyncio.wait_for(task, 5)
        interrupted = interrupt_boundary != "none"
        skipped = interrupt_boundary == "generation" and kind != "text"
        stopped = interrupted and kind != "text"
        assert session.phase is (SessionPhase.STOPPED if stopped else SessionPhase.DONE)
        assert session.state.terminal_reason == ("interrupted by user" if stopped else "completed")
        assert answer == (text if stopped or kind == "text" else "Completed.")
        assert questions == (["Which parser?"] if kind == "ask" and not skipped else [])
        assert target.read_text(encoding="utf-8") == (
            "FEATURE_ON = True\n" if kind == "write" and not skipped else "FEATURE_ON = False\n"
        )
        expected_steps = 1 if interrupted or kind == "text" else 2
        assert len(requests) == expected_steps
        assert session.used_tokens == 4 * expected_steps
        assert session.step_count == expected_steps
        assert session.state.context_tokens == 2
        assert session._open_tool_call_ids() == []
        results = [message for message in session.messages if message["role"] == "tool"]
        assert [message["tool_call_id"] for message in results] == (
            ["late_call"] if kind != "text" and not skipped else []
        )
        if skipped:
            assert {"role": "assistant", "content": text} in session.messages
            assert all(not message.get("tool_calls") for message in session.messages)
            assert all(not message.get("response_items") for message in session.messages)
            assert all(not message.get("provider_state") for message in session.messages)
            assert [event.type for event in events].count("step_end") == 1
            assert [event.data["reason"] for event in events if event.type == "error"] == ["interrupted by user"]
        assert text in [event.data["content"] for event in events if event.type == "text_delta"]
        traces = [step["payload"] for step in tracer.steps if step["step_type"] == "llm_call"]
        assert traces[0]["content"] == text
        assert traces[0]["usage"]["total_tokens"] == 4
        assert traces[0]["tool_calls"] == (None if kind == "text" else [{
            "id": "late_call", "name": name, "arguments": json.dumps(arguments),
        }])
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        assert snapshot["session_state"]["phase"] == session.phase.value
        assert snapshot["session_state"]["used_tokens"] == session.used_tokens
        assert snapshot["session_state"]["step_count"] == expected_steps
        if restored:
            next_session = load_session(
                str(path), agent=agent(), env=environment, ask_policy=AskPolicy(),
                auto_save_path=str(tmp_path / "restored.json"), max_budget_tokens=30_000,
            )
            assert next_session.messages == session.messages
            assert next_session.used_tokens == session.used_tokens
        await next_session.add_user_message("Continue the conversation.")
        assert await next_session.run_loop() == "Completed."
        assert len(requests) == expected_steps + 1
        assert next_session.used_tokens == 4 * (expected_steps + 1)
        assert next_session._open_tool_call_ids() == []
        if skipped:
            assert "late_call" not in json.dumps(requests[-1])
            assert text in json.dumps(requests[-1])
        receipt = {
            "module": opencollab.__file__,
            "mode": mode, "kind": kind, "interrupt_boundary": interrupt_boundary, "restored": restored,
            "questions": questions, "target": target.read_text(encoding="utf-8"),
            "requests": requests, "stopped_snapshot": snapshot, "traces": tracer.steps,
            "followup_phase": next_session.phase.value, "followup_tokens": next_session.used_tokens,
        }
        (tmp_path / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=True, indent=2), encoding="utf-8")
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if next_session is not session:
            await next_session.aclose()
        await session.aclose()
        await environment.cleanup()
