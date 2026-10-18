"""Parallel Responses items keep their output order through the public workflow."""

from __future__ import annotations

import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.tools.fs import FileWriteTool
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport


def native_events(reverse):
    calls = [
        {
            "id": f"fc_{index}",
            "type": "function_call",
            "status": "completed",
            "call_id": f"call_{index}",
            "name": "file_write",
            "arguments": json.dumps({"path": f"{index}.txt", "mode": "create", "content": f"{index}\n"}),
        }
        for index in range(2)
    ]
    terminal = {
        "id": "resp_parallel",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o",
        "output": calls,
        "error": None,
        "incomplete_details": None,
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "temperature": 0,
        "top_p": 1,
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    events = [
        {"type": "response.created", "response": {**terminal, "status": "in_progress", "output": [], "usage": None}}
    ]
    for index, item in enumerate(calls):
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "status": "in_progress", "arguments": ""},
            }
        )
    for index in range(1, -1, -1) if reverse else range(2):
        item = calls[index]
        events.extend(
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": item["arguments"],
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "arguments": item["arguments"],
                },
                {"type": "response.output_item.done", "output_index": index, "item": item},
            ]
        )
    events.append({"type": "response.completed", "response": terminal})
    for sequence, event in enumerate(events):
        event["sequence_number"] = sequence
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)

@pytest.mark.parametrize("reverse", [False, True], ids=["in-order", "reverse-order"])
async def test_public_workflow_executes_parallel_calls_in_output_order(tmp_path, monkeypatch, reverse):
    monkeypatch.delenv("OPENCOLLAB_API_USAGE_LOG", raising=False)
    monkeypatch.delenv("OPENCOLLAB_TRUST_STREAMED_OUTPUT_ON_TERMINAL_MISMATCH", raising=False)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return http_response(
                request, 200, headers={"content-type": "text/event-stream"}, text=native_events(reverse)
            )
        return completion_http_response(request, text="both files written")

    install_sdk_transport(monkeypatch, "openai", handler)

    async def flow(ctx, inputs):
        return await ctx.agent("Write 0.txt and 1.txt, then report.", tools=[FileWriteTool()])

    result = await OpenCollab(
        tmp_path,
        model="gpt-4o",
        api_key="fixture-local-key",  # pragma: allowlist secret
        base_url="https://fixture.invalid/v1",
        config={"wire_protocol": "responses", "llm_max_retries": 0},
    ).workflow(flow, budget=50_000, max_steps=3, timeout=15, trace=False)

    assert result.ok and result.output == "both files written"
    assert [(tmp_path / f"{index}.txt").read_text() for index in range(2)] == ["0\n", "1\n"]
    assert len(requests) == 2
    assert [item["call_id"] for item in requests[1]["input"] if item.get("type") == "function_call_output"] == [
        "call_0",
        "call_1",
    ]
