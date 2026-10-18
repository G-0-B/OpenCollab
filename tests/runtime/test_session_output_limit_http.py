"""Session output limits reach real SDK requests, including the default value."""

from __future__ import annotations

import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.bootstrap import container
from opencollab.bootstrap.session_factory import build_session
from opencollab.domain.agent import Agent
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport


@pytest.fixture(autouse=True)
def isolated_request_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")

@pytest.fixture(params=[
    ("openai", "chat_completions", "gpt-4o", "max_tokens"),
    ("openai", "chat_completions", "gpt-5", "max_completion_tokens"),
    ("openai", "responses", "gpt-4o", "max_output_tokens"),
    ("anthropic", "chat_completions", "claude-sonnet-4-6", "max_tokens"),
])
def sdk_configuration(request):
    provider, protocol, model, field = request.param
    return {"provider": provider, "wire_protocol": protocol, "model": model}, field

@pytest.mark.parametrize("configured_limit", [None, 8191, 8192, 8193])
async def test_public_agent_sends_default_and_explicit_output_limits(
    monkeypatch, tmp_path, sdk_configuration, configured_limit,
):
    configuration, field = sdk_configuration
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        output_tokens = min(9000, body[field]) if field in body else 9000
        return completion_http_response(request, output_tokens=output_tokens)

    install_sdk_transport(monkeypatch, configuration["provider"], handler)
    app = OpenCollab(
        tmp_path, model=configuration["model"], provider=configuration["provider"],
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"llm_max_retries": 0, "wire_protocol": configuration["wire_protocol"],
                **({"max_output_tokens": configured_limit} if configured_limit else {})},
    )
    result = await app.agent("Return ok", tools=(), budget=1_000_000, max_steps=2, trace=False)

    assert result.status == "completed"
    assert len(requests) == 1
    assert requests[0][field] == (8192 if configured_limit is None else configured_limit)
    assert result.tokens == 2 + requests[0][field]

async def test_finite_budget_reduces_the_default_limit_after_reserving_input(
    monkeypatch, tmp_path, sdk_configuration,
):
    configuration, field = sdk_configuration
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, configuration["provider"], handler)
    async with LLMClient(
        **configuration, api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0,
    ) as client:
        messages = [{"role": "user", "content": "Return ok"}]
        budget = client.estimate_request_tokens(messages) + 3000
        session = build_session(
            agent=Agent(name="budgeted", system_prompt="system", **configuration),
            llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=budget,
        )
        session.messages = messages
        assert await session.run_loop() == "ok"

    assert len(requests) == 1
    assert requests[0][field] == 3000

@pytest.mark.parametrize("injected", [False, True])
async def test_summary_sends_the_default_limit_through_the_provider_sdk(
    monkeypatch, tmp_path, sdk_configuration, injected,
):
    configuration, field = sdk_configuration
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="<summary>Preserved context.</summary>")

    install_sdk_transport(monkeypatch, configuration["provider"], handler)
    async with LLMClient(
        **configuration, api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0,
    ) as client:
        agent = Agent(
            name="summary", system_prompt="system", **configuration,
            api_key="controlled-test",  # pragma: allowlist secret
            base_url="https://controlled.invalid/v1",
        )
        session = build_session(agent=agent, llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=None)
        summarizer = container._build_summarizer(
            agent, client if injected else None, client, 600.0, None,
            completion_handler=session.runner._invoke_summary,
        )
        assert await summarizer.asummarize([{"role": "user", "content": "Keep this context"}]) == (
            "Summary:\nPreserved context."
        )
        assert session.used_tokens == 4

    assert len(requests) == 1
    assert requests[0][field] == 8192

@pytest.mark.parametrize("configured_limit", [8192, 7000])
@pytest.mark.parametrize("client_signature", ["legacy", "explicit", "kwargs"])
async def test_injected_client_keeps_legacy_defaults_and_receives_supported_limits(
    tmp_path, configured_limit, client_signature,
):
    calls = []

    class LegacyClient:
        async def complete(self, messages, tools=None, temperature=0.0):
            calls.append({})
            return LLMResponse(content="ok", finish_reason="stop", usage=Usage())

    class ExplicitClient:
        async def complete(self, messages, tools=None, temperature=0.0, max_output_tokens=None):
            calls.append({"max_output_tokens": max_output_tokens})
            return LLMResponse(content="ok", finish_reason="stop", usage=Usage())

    class KeywordClient:
        async def complete(self, messages, tools=None, **kwargs):
            calls.append(kwargs)
            return LLMResponse(content="ok", finish_reason="stop", usage=Usage())

    client = {"legacy": LegacyClient, "explicit": ExplicitClient, "kwargs": KeywordClient}[client_signature]()
    session = build_session(
        agent=Agent(name="injected", system_prompt="system", max_tokens_per_step=configured_limit),
        llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=None,
    )
    session.messages = [{"role": "user", "content": "Return ok"}]
    if client_signature == "legacy" and configured_limit != 8192:
        with pytest.raises(TypeError, match="max_output_tokens"):
            await session.run_loop()
        assert calls == []
    else:
        assert await session.run_loop() == "ok"
        assert len(calls) == 1
        assert calls[0].get("max_output_tokens") == (None if client_signature == "legacy" else configured_limit)

async def test_legacy_injected_client_summary_retains_the_original_signature(tmp_path):
    class LegacyClient:
        async def complete(self, messages, tools=None, temperature=0.0):
            return LLMResponse(content="<summary>Preserved context.</summary>", finish_reason="stop", usage=Usage())

    client = LegacyClient()
    agent = Agent(name="legacy", system_prompt="system")
    session = build_session(agent=agent, llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=None)
    summarizer = container._build_summarizer(
        agent, client, client, 600.0, None, completion_handler=session.runner._invoke_summary,
    )
    assert await summarizer.asummarize([{"role": "user", "content": "Keep this context"}]) == (
        "Summary:\nPreserved context."
    )
