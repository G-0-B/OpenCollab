"""Small context windows retain short turns and their available token budget."""

from __future__ import annotations

import json

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.client import LLMClient
from opencollab.application.run_control import RunControl
from opencollab.bootstrap import build_session
from opencollab.bootstrap.agent_profiles import resolve_agent_profile
from opencollab.bootstrap.container import _history_compaction_settings
from opencollab.domain.agent import Agent
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport
from tests.support.session_run_loop_test_support import FakeTracer


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))

@pytest.mark.parametrize("configuration", [
    {"provider": "openai", "wire_protocol": "chat_completions", "model": "gpt-4o"},
    {"provider": "openai", "wire_protocol": "responses", "model": "gpt-4o"},
    {"provider": "anthropic", "wire_protocol": "chat_completions", "model": "claude-sonnet-4-6"},
])
@pytest.mark.parametrize("window", [8_192, 32_768, 33_001, 33_002, 40_000, 128_000])
@pytest.mark.parametrize("budget", [None, 1_024])
@pytest.mark.parametrize("profile", [None, "single2"])
async def test_short_public_turns_complete_without_unneeded_summary_requests(
    monkeypatch, tmp_path, configuration, window, budget, profile,
):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        is_summary = "create a detailed summary of the conversation" in json.dumps(body)
        return completion_http_response(request, text=(
            "<summary>Use src and run pytest.</summary>" if is_summary else "Noted."
        ))

    install_sdk_transport(monkeypatch, configuration["provider"], handler)
    async with LLMClient(
        **configuration, api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0, context_window=window,
    ) as client:
        session = build_session(
            agent=Agent(
                name="notes", system_prompt="Keep the user's task notes and answer the current request.",
                **configuration, max_tokens_per_step=1_024,
            ),
            llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=budget,
            agent_profile=None if profile is None else resolve_agent_profile(profile),
        )
        try:
            for text in (
                "The task directory is src.", "The test command is pytest.", "Continue with these notes.",
            ):
                await session.add_user_message(text)
                assert await session.run_loop() == "Noted."
                assert session.phase.value == "done"
            assert len(requests) == 3
            assert not any("create a detailed summary of the conversation" in json.dumps(body) for body in requests)
            assert session.used_tokens == 12
        finally:
            await session.aclose()

@pytest.mark.parametrize("window", [None, 40_000, 64_000, 128_000, 1_048_576])
def test_unknown_and_larger_windows_keep_existing_reservations(window):
    class Model:
        def context_window(self):
            return window

    settings = _history_compaction_settings(Model(), max_output_tokens=1_024)
    expected = 120_000 if window is None else window - 33_000
    assert settings["history_trigger_tokens"] == expected
    assert settings["history_target_tokens"] == (90_000 if window is None else int(expected * 0.75))

@pytest.mark.parametrize("output_limit", [1_024, 2_048])
@pytest.mark.parametrize("cap", [None, 2_000, 7_000])
def test_small_window_uses_configured_output_limit_and_records_effective_cap(tmp_path, output_limit, cap):
    class Model:
        def context_window(self):
            return 8_192

    tracer = FakeTracer()
    session = build_session(
        agent=Agent(name="notes", system_prompt="Keep task notes.", max_tokens_per_step=output_limit),
        llm=Model(), env=LocalEnvironment(str(tmp_path)), tracer=tracer,
        run_control=RunControl(history_trigger_tokens=cap),
    )
    record = next(step["payload"] for step in tracer.steps if step["step_type"] == "session.history_compaction")
    trigger = 8_192 - output_limit - 2_730
    if cap is not None:
        trigger = min(trigger, cap)
    assert record["history_trigger_tokens"] == trigger
    assert record["history_target_tokens"] == int(trigger * 0.75)
    for rung in session.runner.shaper._shapers[1:]:
        assert rung.trigger_tokens == record["history_trigger_tokens"]
        assert rung.target_tokens == record["history_target_tokens"]
