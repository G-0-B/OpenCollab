"""The Responses client preserves developer instructions as native messages."""

import json

import httpx
import openai
import pytest

from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.responses_errors import ResponsesProtocolError
from opencollab.adapters.llm.responses_messages import messages_to_input


@pytest.mark.parametrize("content", ["Use a concise answer.", [{"type": "text", "text": "Use a concise answer."}]])
async def test_public_responses_client_preserves_developer_role(monkeypatch, content):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0, "status": "completed",
            "error": None, "incomplete_details": None, "model": "o1-pro",
            "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "answer", "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        })

    original = openai.AsyncOpenAI

    def sdk(**kwargs):
        return original(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai, "AsyncOpenAI", sdk)
    async with LLMClient(
        model="o1-pro", wire_protocol="responses", max_retries=0,
        api_key="test-placeholder",  # pragma: allowlist secret
    ) as client:
        response = await client.complete([
            {"role": "system", "content": "System instructions."},
            {"role": "developer", "content": content},
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "first reply"},
            {"role": "developer", "content": "Use the latest policy."},
            {"role": "user", "content": "second request"},
        ])
    assert response.content == "answer"
    assert len(requests) == 1
    request = requests[0]
    assert request["instructions"] == "System instructions."
    assert [item["role"] for item in request["input"]] == ["developer", "user", "assistant", "developer", "user"]
    expected = content if isinstance(content, str) else [{"type": "input_text", "text": "Use a concise answer."}]
    assert request["input"][0]["content"] == expected
    assert request["input"][3]["content"] == "Use the latest policy."

def test_developer_history_keeps_tool_call_pairing():
    instructions, items = messages_to_input([
        {"role": "developer", "content": "Use the lookup result."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_test", "function": {"name": "lookup", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_test", "content": "42"},
    ])
    assert instructions is None
    assert items[0] == {"role": "developer", "content": "Use the lookup result."}
    assert items[1]["call_id"] == items[2]["call_id"] == "call_test"

def test_unknown_roles_remain_rejected():
    with pytest.raises(ResponsesProtocolError, match="unsupported message role"):
        messages_to_input([{"role": "unknown", "content": "invalid"}])
