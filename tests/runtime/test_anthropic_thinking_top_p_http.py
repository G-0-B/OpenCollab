"""Native Claude sampling retains supported ranges and default settings."""

from __future__ import annotations

import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.anthropic_provider import _build_request_kwargs
from opencollab.adapters.llm.client import LLMClient
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport

_MANUAL_THINKING = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
_ADAPTIVE_THINKING = {"thinking": {"type": "adaptive"}}

@pytest.fixture(autouse=True)
def isolate_usage_log(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))

@pytest.mark.parametrize("model", [
    "claude-sonnet-4", "claude-opus-4",
    "claude-sonnet-4-20250514", "claude-opus-4-20250514",
    "gateway/claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929",
    "gateway/claude-sonnet-4-5-20250929",
])
@pytest.mark.parametrize("thinking", [False, True])
async def test_older_claude_snapshots_send_sampling_and_manual_thinking(monkeypatch, model, thinking):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "anthropic", handler)
    top_p = 0.95 if thinking else 0.8
    async with LLMClient(
        provider="anthropic", model=model,
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0,
    ) as client:
        result = await client.complete(
            [{"role": "user", "content": "Return ok"}], temperature=0.7,
            thinking=thinking, thinking_params=_MANUAL_THINKING if thinking else None,
            max_output_tokens=2048, top_p=top_p,
        )

    assert result.content == "ok"
    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert requests[0]["max_tokens"] == 2048
    assert requests[0]["top_p"] == top_p
    assert "temperature" not in requests[0]
    if thinking:
        assert requests[0]["thinking"] == _MANUAL_THINKING["thinking"]
    else:
        assert "thinking" not in requests[0]

@pytest.mark.parametrize("top_p", [None, 0.95, 0.975, 1.0])
@pytest.mark.parametrize("public_agent", [False, True])
@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-4-6"])
@pytest.mark.parametrize("thinking_params", [_MANUAL_THINKING, _ADAPTIVE_THINKING])
async def test_claude_4_6_thinking_sends_supported_top_p(
    monkeypatch, tmp_path, top_p, public_agent, model, thinking_params,
):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "anthropic", handler)
    app = OpenCollab(
        tmp_path, provider="anthropic", model=model,
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"llm_max_retries": 0, "thinking": True, "thinking_params": thinking_params,
                "top_p": top_p, "max_output_tokens": 2048},
    )
    if public_agent:
        result = await app.agent("Return ok", tools=(), trace=False, budget=10000, max_steps=1)
        assert result.status == "completed"
        assert result.output == "ok"
    else:
        async with app.create_model_client() as client:
            result = await client.complete(
                [{"role": "user", "content": "Return ok"}], thinking=True,
                thinking_params=thinking_params, max_output_tokens=2048, top_p=top_p,
            )
            assert result.content == "ok"

    assert len(requests) == 1
    assert requests[0]["thinking"] == thinking_params["thinking"]
    assert requests[0]["max_tokens"] == 2048
    assert "temperature" not in requests[0]
    assert ("top_p" in requests[0]) is (top_p is not None)
    if top_p is not None:
        assert requests[0]["top_p"] == top_p

@pytest.mark.parametrize("model", [
    "claude-sonnet-4-6", "gateway/claude-sonnet-4-6-20260217", "claude-opus-4-6",
    "claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5",
])
@pytest.mark.parametrize("top_p", [0.95, 1.0])
def test_known_older_claude_manual_thinking_keeps_sampling(model, top_p):
    request = _build_request_kwargs(
        model, [{"role": "user", "content": "Return ok"}], None, 0.2,
        thinking=True, thinking_params=_MANUAL_THINKING, max_output_tokens=2048, top_p=top_p,
    )

    assert request["top_p"] == top_p
    assert "temperature" not in request

@pytest.mark.parametrize("top_p", [0.94, 0.0, 1.01, float("nan"), float("inf"), True])
@pytest.mark.parametrize("thinking_params", [_MANUAL_THINKING, _ADAPTIVE_THINKING])
async def test_thinking_rejects_out_of_range_sampling_before_http(monkeypatch, top_p, thinking_params):
    requests = []

    async def handler(request):
        requests.append(request)
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "anthropic", handler)
    async with LLMClient(
        provider="anthropic", model="claude-sonnet-4-6",
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0,
    ) as client:
        with pytest.raises(ValueError, match="top_p"):
            await client.complete(
                [{"role": "user", "content": "Return ok"}], thinking=True,
                thinking_params=thinking_params, max_output_tokens=2048, top_p=top_p,
            )

    assert requests == []

@pytest.mark.parametrize("model,params", [
    ("claude-opus-4-7", {"thinking": {"type": "adaptive"}}),
    ("claude-opus-4-8", {"thinking": {"type": "adaptive"}}),
    ("claude-sonnet-5", {"thinking": {"type": "adaptive"}}),
    ("claude-opus-5-5", {"thinking": {"type": "adaptive"}}),
    ("claude-mythos-preview", _MANUAL_THINKING),
    ("claude-haiku-4-6", {"thinking": {"type": "adaptive"}}),
    ("compatible-unknown-model", _MANUAL_THINKING),
    ("compatible-unknown-model", {"thinking": {"type": "adaptive"}}),
])
@pytest.mark.parametrize("top_p", [0.95, 1.0])
def test_other_thinking_models_and_modes_keep_existing_sampling_rules(model, params, top_p):
    with pytest.raises(ValueError, match="provider-default top_p"):
        _build_request_kwargs(
            model, [{"role": "user", "content": "Return ok"}], None, 0.2,
            thinking=True, thinking_params=params, max_output_tokens=2048, top_p=top_p,
        )

def test_snapshot_adaptive_thinking_keeps_supported_sampling():
    request = _build_request_kwargs(
        "gateway/claude-sonnet-4-6-20260217", [{"role": "user", "content": "Return ok"}],
        None, 0.2, thinking=True, thinking_params=_ADAPTIVE_THINKING,
        max_output_tokens=2048, top_p=0.95,
    )

    assert request["top_p"] == 0.95
    assert request["thinking"] == _ADAPTIVE_THINKING["thinking"]
    assert "temperature" not in request

def test_thinking_disabled_sends_only_explicit_top_p():
    request = _build_request_kwargs(
        "claude-sonnet-4-6", [{"role": "user", "content": "Return ok"}], None, 0.2,
        top_p=0.8, max_output_tokens=2048,
    )

    assert request["top_p"] == 0.8
    assert "temperature" not in request

@pytest.mark.parametrize("model,thinking", [
    ("claude-opus-4-7", False), ("claude-opus-4-7", True),
    ("claude-opus-4-8", False), ("claude-opus-4-8", True),
    ("claude-opus-5", False), ("claude-opus-5", True),
    ("claude-sonnet-5", False), ("claude-sonnet-5", True),
    ("claude-opus-5-5", True), ("claude-sonnet-5-5", True),
    ("claude-haiku-5-5", False), ("claude-haiku-5-5", True),
    ("claude-fable-5-1", True),
    ("claude-sonnet-4-6", False), ("claude-sonnet-4-6", True),
])
async def test_public_agent_accepts_documented_default_top_p(monkeypatch, tmp_path, model, thinking):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "anthropic", handler)
    result = await OpenCollab(
        tmp_path, provider="anthropic", model=model,
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={
            "llm_max_retries": 0, "thinking": thinking,
            "thinking_params": _ADAPTIVE_THINKING if thinking else None,
            "top_p": 0.99, "max_output_tokens": 2048,
        },
    ).agent("Return ok", tools=(), trace=False, budget=10000, max_steps=1)

    assert result.ok, result.error
    assert result.output == "ok" and len(requests) == 1
    if model == "claude-sonnet-4-6":
        assert requests[0]["top_p"] == 0.99
    else:
        assert "top_p" not in requests[0]
    if thinking:
        assert requests[0]["thinking"] == _ADAPTIVE_THINKING["thinking"]
