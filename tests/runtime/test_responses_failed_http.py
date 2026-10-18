"""HTTP 200 failed Responses retain provider error and retry semantics."""

import json

import httpx
import openai
import pytest

from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.responses_errors import ResponsesProtocolError
from opencollab.adapters.llm.responses_provider import parse_responses_response
from opencollab.adapters.llm.retry import is_retryable_error
from tests.support.responses_provider_test_support import completed_response, message_item


def _response(body, error=None):
    return {
        "id": "resp_failed_test", "object": "response", "created_at": 1,
        "status": "failed" if error else "completed", "model": body["model"],
        "instructions": body.get("instructions"), "incomplete_details": None,
        "error": error, "output": [] if error else [message_item("connected")],
        "usage": None,
    }

def _wire_response(body, error=None):
    response = _response(body, error)
    if not body["stream"]:
        return httpx.Response(200, json=response)
    events = (
        [{"type": "response.failed", "sequence_number": 0, "response": response}]
        if error else [
            {"type": "response.output_item.done", "sequence_number": 0,
             "output_index": 0, "item": response["output"][0]},
            {"type": "response.completed", "sequence_number": 1, "response": response},
        ]
    )
    payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)

def _patch_transport(monkeypatch, handler):
    sdk_type = openai.AsyncOpenAI

    def create_sdk(**kwargs):
        return sdk_type(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai, "AsyncOpenAI", create_sdk)

@pytest.mark.parametrize("model,stream", [("o1-pro", False), ("gpt-4o", True)])
async def test_failed_response_server_error_retries_and_recovers(monkeypatch, model, stream):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        error = {"code": "server_error", "message": "An internal server error occurred."}
        return _wire_response(body, error if len(requests) == 1 else None)

    async def skip_delay(_seconds):
        pass

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr("opencollab.adapters.llm.retry.asyncio.sleep", skip_delay)
    async with LLMClient(
        model=model, wire_protocol="responses", api_key="test-placeholder",  # pragma: allowlist secret
        base_url="https://provider.invalid/v1", max_retries=1,
    ) as client:
        result = await client.complete([{"role": "user", "content": "connect"}])

    assert result.content == "connected"
    assert len(requests) == 2
    assert all(body["stream"] is stream for body in requests)

@pytest.mark.parametrize(
    "code,message,param,retryable",
    [
        ("server_error", "An internal server error occurred.", None, True),
        ("invalid_prompt", "Invalid prompt supplied.", None, False),
        ("context_length_exceeded", "Request exceeds context window.", "input", False),
    ],
)
async def test_failed_non_stream_response_preserves_error_identity(monkeypatch, code, message, param, retryable):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return _wire_response(body, {"code": code, "message": message, "param": param})

    _patch_transport(monkeypatch, handler)
    async with LLMClient(
        model="o1-pro", wire_protocol="responses", api_key="test-placeholder",  # pragma: allowlist secret
        base_url="https://provider.invalid/v1", max_retries=0 if retryable else 1,
    ) as client:
        with pytest.raises(ResponsesProtocolError) as captured:
            await client.complete([{"role": "user", "content": "connect"}])

    error = captured.value
    assert error.code == code
    assert str(error) == message
    assert error.param == param
    assert error.body["error"]["message"] == message
    assert is_retryable_error(error) is retryable
    assert len(requests) == 1

def test_failed_non_stream_response_requires_model_identity():
    response = completed_response(
        status="failed", model="", error={"code": "server_error", "message": "upstream failure"},
    )
    with pytest.raises(ResponsesProtocolError, match="missing model identity"):
        parse_responses_response(response, [{"role": "user", "content": "connect"}], expected_model="o1-pro")
