"""Owned main and auxiliary transports share one session retry allowance."""

from __future__ import annotations

import json
import re

import openai
import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm import retry
from opencollab.adapters.llm.retry import RetryTimeBudget
from opencollab.bootstrap import build_session, container
from opencollab.domain.agent import Agent
from opencollab.domain.token_estimation import estimate_request_tokens
from opencollab.environments import local_environment
from opencollab.tools import builtin_tools
from tests.support.provider_sdk_http import (
    completion_http_response,
    http_response,
    install_sdk_transport,
)


async def install(monkeypatch, handler):
    monkeypatch.setattr(retry.random, "uniform", lambda *_args: 0.0)
    for name in ("OPENCOLLAB_UNBOUNDED_LIMITS", "OPENCOLLAB_API_USAGE_LOG",
                 "OPENCOLLAB_EXTERNAL_PROVIDER_ISOLATION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    def prepared_handler(request):
        if "scope-sdk-warmup" in str(json.loads(request.content)):
            return completion_http_response(request, text="done")
        return handler(request)

    install_sdk_transport(monkeypatch, "openai", prepared_handler)
    models = []
    native_client = container.LLMClient
    warmup = native_client(
        model="gpt-4o", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
    )
    try:
        response = await warmup.complete([
            {"role": "system", "content": "Warm the controlled SDK."},
            {"role": "user", "content": "scope-sdk-warmup"},
        ], max_output_tokens=128)
        assert response.content == "done"
    finally:
        await warmup.close()

    class RecordedClient(native_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            models.append(self)

    monkeypatch.setattr(container, "LLMClient", RecordedClient)
    return models

def limited(request):
    return http_response(request, 429, headers={"retry-after": "1"}, json={"error": {
        "message": "Controlled rate limit.", "type": "rate_limit_error",
    }})

@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("allowance", [1.5, 0])
async def test_agent_tool_history_triggers_summary_under_the_same_retry_allowance(
    tmp_path, monkeypatch, borrowed, allowance,
):
    for index in range(3):
        (tmp_path / f"note-{index}.txt").write_text(f"Note {index}.\n")
    counts = {"main": 0, "summary": 0}
    successes = 0

    def handler(request):
        nonlocal successes
        body = json.loads(request.content)
        summary = "create a detailed summary of the conversation" in json.dumps(body)
        purpose = "summary" if summary else "main"
        counts[purpose] += 1
        if counts[purpose] == 1:
            return limited(request)
        if summary:
            return completion_http_response(request, text="<summary>Inspect the local notes.</summary>")
        successes += 1
        if successes > 2:
            return completion_http_response(request, text="Finished.")
        prompt_tokens = estimate_request_tokens(body["messages"], body.get("tools"))
        return http_response(request, 200, json={
            "id": f"chatcmpl-{successes}", "object": "chat.completion", "created": 1,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Inspect the local project notes before answering. " * 300,
                "tool_calls": [{"id": f"read-{successes}", "type": "function", "function": {
                    "name": "file_read", "arguments": json.dumps({"path": f"note-{successes}.txt"}),
                }}],
            }}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 3000,
                           "total_tokens": prompt_tokens + 3000},
        })

    models = await install(monkeypatch, handler)
    oc = OpenCollab(tmp_path, model="gpt-4o", api_key="controlled-test",  # pragma: allowlist secret
                    base_url="https://controlled.invalid/v1", config={
                        "provider_error_time_budget": allowance, "llm_max_retries": 2,
                        "max_output_tokens": 8192, "context_window": 40_000, "budget": 100_000,
                    })
    model = oc.create_model_client() if borrowed else None
    try:
        result = await oc.agent("Inspect the local notes.", llm=model, tools=builtin_tools("file_read"),
                                trace=False, max_steps=6)
        assert result.ok
        assert counts == {"main": 4, "summary": 1 if allowance else 2}
        if borrowed:
            assert models == []
            assert not model._openai.is_closed()
        else:
            assert len(models) == 2
            assert models[0].provider_retry_budget is models[1].provider_retry_budget
            assert all(owned._openai.is_closed() for owned in models)
    finally:
        if model is not None:
            await model.close()

@pytest.mark.parametrize("budget_kind", ["implicit", "explicit", "zero"])
async def test_owned_session_turns_and_repeated_summaries_keep_the_original_allowance(
    tmp_path, monkeypatch, budget_kind,
):
    counts = {}

    def handler(request):
        body = json.loads(request.content)
        key = re.findall(r"scope-task-(?:turn|summary)-\d+", str(body))[-1]
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1:
            return limited(request)
        summary = "create a detailed summary of the conversation" in str(body)
        return completion_http_response(request, text="<summary>Retained context.</summary>" if summary else "done")

    models = await install(monkeypatch, handler)
    allowance = 0 if budget_kind == "zero" else 2.5
    supplied = RetryTimeBudget(allowance) if budget_kind == "explicit" else None
    environment = local_environment(tmp_path)
    session = build_session(
        agent=Agent(name="owned-session", system_prompt="Finish briefly.", model="gpt-4o", provider="openai",
                    api_key="controlled-test",  # pragma: allowlist secret
                    base_url="https://controlled.invalid/v1",
                    context_window=40_000, provider_error_time_budget=allowance, llm_max_retries=2),
        env=environment, provider_retry_budget=supplied, max_budget_tokens=100_000,
    )
    try:
        await session.add_user_message("scope-task-turn-0")
        assert await session.run_loop() == "done"
        cacheable = []
        for index in range(3):
            await session._context_summarizer.asummarize([{"role": "user", "content": f"scope-task-summary-{index}"}])
            cacheable.append(session._context_summarizer.last_call_cacheable)
            assert not models[0]._openai.is_closed()
            assert all(model._openai.is_closed() for model in models[1:])
        await session.add_user_message("scope-task-turn-1")
        if allowance:
            with pytest.raises(openai.RateLimitError):
                await session.run_loop()
        else:
            assert await session.run_loop() == "done"
        assert cacheable == ([True, False, False] if allowance else [True, True, True])
        assert counts == {"scope-task-turn-0": 2, "scope-task-summary-0": 2,
                          "scope-task-summary-1": 1 if allowance else 2,
                          "scope-task-summary-2": 1 if allowance else 2,
                          "scope-task-turn-1": 1 if allowance else 2}
        assert len(models) == 4
        assert all(model.provider_retry_budget is models[0].provider_retry_budget for model in models)
        if supplied is not None:
            assert models[0].provider_retry_budget is supplied
        elif not allowance:
            assert models[0].provider_retry_budget is None
        await environment.write_file("caller-owned.txt", "usable")
    finally:
        await session.aclose()
        await environment.cleanup()
    assert all(model._openai.is_closed() for model in models)
