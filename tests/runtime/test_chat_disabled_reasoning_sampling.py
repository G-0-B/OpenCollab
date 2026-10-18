"""Documented non-reasoning GPT-5 requests retain caller sampling controls."""

import json

import httpx
import openai
import pytest

from opencollab.adapters.llm.openai_provider import _build_request_kwargs, complete_openai


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4-2026-03-05", "openai/gpt-5.4", "gpt-5.1", "gpt-5.2"])
@pytest.mark.parametrize("effort", [None, "none", "low", "high"])
def test_sampling_respects_effective_reasoning_mode(model, effort):
    request = _build_request_kwargs(
        model, [{"role": "user", "content": "answer"}], None, 0.2,
        top_p=0.7, reasoning_effort=effort, max_output_tokens=64,
    )
    if effort in (None, "none"):
        assert request["temperature"] == 0.2
        assert request["top_p"] == 0.7
    else:
        assert "temperature" not in request
        assert "top_p" not in request
    assert request["max_completion_tokens"] == 64
    assert "max_tokens" not in request

@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-codex", "o3-mini"])
def test_other_reasoning_models_keep_existing_sampling_rules(model):
    request = _build_request_kwargs(
        model, [{"role": "user", "content": "answer"}], None, 0.2,
        top_p=0.7, reasoning_effort="none",
    )
    assert "temperature" not in request
    assert "top_p" not in request

async def test_sdk_receives_sampling_controls_with_reasoning_disabled():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "gpt-5.4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    async with openai.AsyncOpenAI(
        api_key="test-placeholder",  # pragma: allowlist secret
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as sdk:
        response = await complete_openai(
            sdk, "gpt-5.4", [{"role": "user", "content": "answer"}], None, 0.2, 0,
            top_p=0.7, reasoning_effort="none", max_output_tokens=64,
        )
    assert response.content == "answer"
    assert requests[0]["temperature"] == 0.2
    assert requests[0]["top_p"] == 0.7
    assert requests[0]["reasoning_effort"] == "none"
    assert requests[0]["max_completion_tokens"] == 64
