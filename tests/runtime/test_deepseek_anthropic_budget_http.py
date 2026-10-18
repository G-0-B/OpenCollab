"""DeepSeek native thinking uses the real output and session token limits."""

from __future__ import annotations

import copy

import anthropic
import pytest

from opencollab import OpenCollab
from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.anthropic_provider import _build_request_kwargs
from opencollab.adapters.llm.client import LLMClient
from opencollab.bootstrap.session_factory import build_session
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase
from tests.runtime.test_deepseek_thinking_off_http import _deepseek_server

_MODELS = ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"]
_MESSAGES = [{"role": "user", "content": "Return done"}]

@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_UNBOUNDED_LIMITS", "false")
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("limit,mode", [
    (None, {"type": "enabled"}),
    (1000, {"type": "enabled", "budget_tokens": 1024}),
    (128, {"type": "enabled", "budget_tokens": 1}),
])
async def test_public_agent_sends_native_thinking_without_claude_budget_requirement(
    tmp_path, model, limit, mode,
):
    config = {"thinking": True, "thinking_params": {"thinking": mode}, "llm_max_retries": 0}
    if limit is not None:
        config["max_output_tokens"] = limit
    original = copy.deepcopy(config)
    with _deepseek_server() as (base_url, requests):
        app = OpenCollab(
            tmp_path, provider="anthropic", model=model,
            api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url=base_url.removesuffix("/v1"), config=config,
        )
        result = await app.agent("Return done", tools=(), system_prompt="system", trace=False, max_steps=1)
    assert result.ok and result.output == "done", result.reason
    assert result.tokens == 4
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == (8192 if limit is None else limit)
    assert requests[0]["thinking"] == mode
    assert config == original
    assert app.configuration["thinking_params"] == original["thinking_params"]

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("limit,mode", [
    (8192, {"type": "enabled"}),
    (1000, {"type": "enabled", "budget_tokens": 1024}),
])
async def test_native_sdk_accepts_the_same_deepseek_thinking_request(model, limit, mode):
    with _deepseek_server() as (base_url, requests):
        async with anthropic.AsyncAnthropic(
            api_key="fake-key-for-test", max_retries=0,  # pragma: allowlist secret
            base_url=base_url.removesuffix("/v1"),
        ) as client:
            response = await client.messages.create(
                model=model, max_tokens=limit, thinking=mode, messages=_MESSAGES,
            )
    assert response.content[-1].text == "done"
    assert requests[0]["thinking"] == mode
    assert requests[0]["max_tokens"] == limit

@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("prefix", ["", "deepseek/", "gateway/"])
@pytest.mark.parametrize("suffix", ["", "-2026", "-2026-09", "-2026-09-21"])
def test_existing_exact_aliases_share_the_native_budget_capability(model, prefix, suffix):
    request = _build_request_kwargs(
        prefix + model + suffix, _MESSAGES, None, 0.2,
        thinking=True, thinking_params={"thinking": {"type": "enabled"}}, max_output_tokens=128,
    )
    assert request["thinking"] == {"type": "enabled"}
    assert request["max_tokens"] == 128

@pytest.mark.parametrize("model", [
    "claude-sonnet-4-6", "gateway/unknown", "deepseek-pro", "deepseek-reasoner",
    "deepseek-v4.1-flash", "deepseek-flash-preview", "deepseek-v4-pro-custom",
    "deepseek-v4-flash-vision-exp-preview",
])
@pytest.mark.parametrize("mode,limit,error", [
    ({"type": "enabled"}, 8192, "integer budget_tokens"),
    ({"type": "enabled", "budget_tokens": 1000}, 2000, "at least 1024"),
    ({"type": "enabled", "budget_tokens": 1024}, 1000, "less than max_output_tokens"),
])
def test_claude_and_unverified_models_keep_strict_budget_validation(model, mode, limit, error):
    with pytest.raises(ValueError, match=error):
        _build_request_kwargs(
            model, _MESSAGES, None, 0.2, thinking=True,
            thinking_params={"thinking": mode}, max_output_tokens=limit,
        )

@pytest.mark.parametrize("params,error", [
    ({"thinking": {"type": "enabled", "budget_tokens": True}}, "integer budget_tokens"),
    ({"thinking": {"type": "enabled", "budget_tokens": "1024"}}, "integer budget_tokens"),
    ({"thinking": {"type": "enabled", "unexpected": True}}, "unsupported Anthropic thinking field"),
    ({"thinking": {"type": "enabled", "display": "invalid"}}, "thinking display"),
    ({"thinking": {"type": "adaptive", "budget_tokens": 1024}}, "unsupported Anthropic thinking field"),
    ({"thinking": {"type": "enabled"}, "extra": True}, "unsupported Anthropic thinking parameter"),
    ({"thinking": {"type": "enabled"}, "output_config": {"format": {}}}, "format requires"),
])
def test_deepseek_keeps_other_native_schema_validation(params, error):
    with pytest.raises(ValueError, match=error):
        _build_request_kwargs("deepseek-flash", _MESSAGES, None, 0.2, thinking=True, thinking_params=params)

@pytest.mark.parametrize("mode", [{"type": "enabled"}, {"type": "enabled", "budget_tokens": 2048}])
@pytest.mark.parametrize("output_budget", [0, 128])
async def test_actual_session_caps_output_with_ignored_thinking_budget(tmp_path, mode, output_budget):
    params = {"thinking": mode}
    with _deepseek_server() as (base_url, requests):
        async with LLMClient(
            provider="anthropic", model="deepseek-flash",
            api_key="fake-key-for-test", max_retries=0,  # pragma: allowlist secret
            base_url=base_url.removesuffix("/v1"),
        ) as client:
            input_budget = client.estimate_request_tokens(_MESSAGES, thinking=True, thinking_params=params)
            budget = input_budget + output_budget
            session = build_session(
                agent=Agent(name="budget", system_prompt="system", provider="anthropic", model="deepseek-flash",
                            thinking=True, thinking_params=params, max_tokens_per_step=1000),
                llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=budget,
            )
            session.messages = copy.deepcopy(_MESSAGES)
            try:
                answer = await session.run_loop()
                if output_budget:
                    assert answer == "done"
                    assert len(requests) == 1
                    assert requests[0]["max_tokens"] == output_budget
                    assert requests[0]["thinking"] == mode
                    assert session.used_tokens == 4
                    assert session.used_tokens <= budget
                else:
                    assert session.phase is SessionPhase.STOPPED
                    assert session.used_tokens == 0
                    assert requests == []
            finally:
                await session.aclose()
