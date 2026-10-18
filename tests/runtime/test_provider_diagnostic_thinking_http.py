"""Connectivity probes retain the caller's configured reasoning mode."""

from __future__ import annotations

import json
import os

import pytest

from scripts.check_dashscope import request_completion
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport


@pytest.fixture
def diagnostic_config(monkeypatch, tmp_path):
    for name in tuple(os.environ):
        if name.startswith(("OPENCOLLAB_", "OPENAI_", "ANTHROPIC_", "DASHSCOPE_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "selected.env"
    monkeypatch.setenv("OPENCOLLAB_CONFIG_FILE", str(path))

    def configure(provider, model, *, thinking=False, params=None, effort=None, maximum=4096):
        values = {
            "OPENCOLLAB_PROVIDER": provider,
            "OPENCOLLAB_MODEL": model,
            "OPENCOLLAB_API_KEY": "diagnostic-fixture-key",  # pragma: allowlist secret
            "OPENCOLLAB_BASE_URL": "https://provider.invalid/v1",
            "OPENCOLLAB_LLM_MAX_RETRIES": "0",
            "OPENCOLLAB_MAX_OUTPUT_TOKENS": str(maximum),
            "OPENCOLLAB_THINKING": str(thinking).lower(),
        }
        if params is not None:
            values["OPENCOLLAB_THINKING_PARAMS"] = json.dumps(params)
        if effort is not None:
            values["OPENCOLLAB_REASONING_EFFORT"] = effort
        if provider == "responses":
            values["OPENCOLLAB_PROVIDER"] = "openai"
            values["OPENCOLLAB_WIRE_PROTOCOL"] = "responses"
        path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")

    return configure

@pytest.mark.parametrize("mode", [
    "plain", "always_on", "adaptive", "manual", "chat", "responses",
    "chat_default", "responses_default", "responses_implicit",
])
async def test_diagnostic_native_request_keeps_reasoning(monkeypatch, diagnostic_config, tmp_path, mode):
    provider = "responses" if mode.startswith("responses") else "openai" if mode.startswith("chat") else "anthropic"
    model = (
        "gpt-4o" if mode.endswith("_default")
        else "gpt-5.4" if provider in {"openai", "responses"}
        else "claude-sonnet-4-6"
    )
    params = None
    effort = "high" if mode in {"adaptive", "chat", "responses", "responses_implicit"} else None
    thinking = mode in {"always_on", "adaptive", "manual", "responses_implicit"}
    if mode == "always_on":
        model = "claude-fable-5-1"
    if thinking and mode != "responses_implicit":
        params = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if mode == "manual" else {
            "thinking": {"type": "adaptive"},
        }
    diagnostic_config(provider, model, thinking=thinking, params=params, effort=effort)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert "enable_thinking" not in requests[-1]
        return completion_http_response(request, text="connected")

    install_sdk_transport(monkeypatch, "anthropic" if provider == "anthropic" else "openai", handler)
    assert await request_completion("connect once", workspace=tmp_path) == "connected"
    assert len(requests) == 1
    body = requests[0]
    if provider == "anthropic" and thinking:
        assert body["thinking"] == params["thinking"]
    elif provider == "anthropic":
        assert "thinking" not in body
    if effort:
        key = "output_config" if provider == "anthropic" else "reasoning" if provider == "responses" else None
        assert (body[key]["effort"] if key else body["reasoning_effort"]) == effort
    output_limit = body.get("max_tokens", body.get("max_output_tokens", body.get("max_completion_tokens")))
    assert output_limit == (1025 if mode == "manual" else 256)

@pytest.mark.parametrize("provider", ["openai", "responses"])
async def test_explicit_compatible_protocol_still_turns_thinking_off(
    monkeypatch, diagnostic_config, tmp_path, provider,
):
    diagnostic_config(
        provider, "compatible-custom", thinking=False, params={"enable_thinking": True},
    )
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="connected")

    install_sdk_transport(monkeypatch, "openai", handler)
    assert await request_completion("connect once", workspace=tmp_path) == "connected"
    assert len(requests) == 1 and requests[0]["enable_thinking"] is False

@pytest.mark.parametrize("maximum", [512, 1024])
async def test_invalid_manual_budget_fails_without_http(monkeypatch, diagnostic_config, tmp_path, maximum):
    diagnostic_config(
        "anthropic", "claude-sonnet-4-6", thinking=True,
        params={"thinking": {"type": "enabled", "budget_tokens": 1024}}, maximum=maximum,
    )
    requests = []

    def handler(request):
        requests.append(request)
        return completion_http_response(request, text="connected")

    install_sdk_transport(monkeypatch, "anthropic", handler)
    with pytest.raises(ValueError, match="budget_tokens must be less than max_output_tokens"):
        await request_completion("connect once", workspace=tmp_path)
    assert not requests
