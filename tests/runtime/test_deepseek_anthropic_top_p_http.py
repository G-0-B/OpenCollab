"""Declared DeepSeek thinking sampling reaches the native SDK over HTTP."""

from __future__ import annotations

from dataclasses import replace

import anthropic
import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm import types
from opencollab.adapters.llm.anthropic_provider import _build_request_kwargs
from tests.runtime.test_deepseek_thinking_off_http import _deepseek_server

_MODELS = ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"]
_MESSAGES = [{"role": "user", "content": "Return done"}]

@pytest.fixture(autouse=True)
def isolate_usage_log(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("top_p", [None, 0.95, 0.98, 1.0])
async def test_public_agent_matches_native_sdk_thinking_sampling(tmp_path, model, top_p):
    thinking = {"type": "enabled"}
    with _deepseek_server() as (base_url, requests):
        app = OpenCollab(
            tmp_path, provider="anthropic", model=model,
            api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url.removesuffix("/v1"),
            config={"thinking": True, "thinking_params": {"thinking": thinking},
                    "top_p": top_p, "max_output_tokens": 2048, "llm_max_retries": 0},
        )
        result = await app.agent("Return done", tools=(), trace=False, max_steps=1)
        assert result.ok and result.output == "done", result.reason
        assert len(requests) == 1
        async with anthropic.AsyncAnthropic(
            api_key="fake-key-for-test", max_retries=0,  # pragma: allowlist secret
            base_url=base_url.removesuffix("/v1"),
        ) as sdk:
            response = await sdk.messages.create(
                model=model, messages=_MESSAGES, max_tokens=2048, thinking=thinking,
                extra_body={} if top_p is None else {"top_p": top_p},
            )
        assert response.content[-1].text == "done"

    assert len(requests) == 2
    for request in requests:
        assert request["thinking"] == thinking
        assert ("top_p" in request) is (top_p is not None)
        if top_p is not None:
            assert request["top_p"] == top_p
    assert "temperature" not in requests[0]

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("prefix", ["", "deepseek/", "gateway/"])
@pytest.mark.parametrize("suffix", ["", "-2026", "-2026-09", "-2026-09-21"])
def test_existing_exact_routes_accept_thinking_sampling(model, prefix, suffix):
    request = _build_request_kwargs(
        prefix + model + suffix, _MESSAGES, None, 0.2, thinking=True,
        thinking_params={"thinking": {"type": "enabled"}}, max_output_tokens=2048, top_p=0.98,
    )
    assert request["top_p"] == 0.98

@pytest.mark.parametrize("model", [
    "deepseek-pro", "deepseek-reasoner", "deepseek-v4.1-flash", "deepseek-flash-preview",
    "deepseek-v4-pro-custom", "deepseek-v4-flash-vision-exp-preview", "gateway/unknown",
])
def test_unverified_models_keep_default_thinking_sampling(model):
    with pytest.raises(ValueError, match="provider-default top_p"):
        _build_request_kwargs(
            model, _MESSAGES, None, 0.2, thinking=True,
            thinking_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            max_output_tokens=2048, top_p=0.98,
        )

@pytest.mark.parametrize("top_p", [0.94, 0.0, 1.01, float("nan"), float("inf"), True])
def test_declared_thinking_sampling_validates_effective_range(top_p):
    with pytest.raises(ValueError, match="between 0.95 and 1.0"):
        _build_request_kwargs(
            "deepseek-flash", _MESSAGES, None, 0.2, thinking=True,
            thinking_params={"thinking": {"type": "enabled"}}, top_p=top_p,
        )

@pytest.mark.parametrize("requires_budget,supports_sampling", [(False, False), (True, True)])
def test_thinking_sampling_capability_is_independent_of_budget(monkeypatch, requires_budget, supports_sampling):
    capability = replace(
        types.model_capabilities("deepseek-flash"),
        requires_anthropic_thinking_budget=requires_budget,
        supports_anthropic_thinking_top_p=supports_sampling,
    )
    monkeypatch.setitem(types._EXACT_MODEL_CAPABILITIES, "deepseek-flash", capability)
    params = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    if supports_sampling:
        request = _build_request_kwargs(
            "deepseek-flash", _MESSAGES, None, 0.2, thinking=True,
            thinking_params=params, max_output_tokens=2048, top_p=0.98,
        )
        assert request["top_p"] == 0.98
    else:
        with pytest.raises(ValueError, match="provider-default top_p"):
            _build_request_kwargs(
                "deepseek-flash", _MESSAGES, None, 0.2, thinking=True,
                thinking_params=params, max_output_tokens=2048, top_p=0.98,
            )
