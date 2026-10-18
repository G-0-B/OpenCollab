"""Summary thinking matches provider requirements and session budget estimates."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.client import LLMClient
from opencollab.application._session_run_shared import _TokenBudgetStop
from opencollab.application.compaction_prompt import build_summary_request
from opencollab.application.session_run import GenerationTimeoutError
from opencollab.bootstrap import build_session, container
from opencollab.domain.agent import Agent
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport
from tests.support.session_run_loop_test_support import FakeTracer


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")

def destination_history():
    return [
        {"role": "system", "content": "Follow the latest destination selected by the user."},
        {"role": "user", "content": "DEST=BLUE\n" + "Earlier project specifications. " * 600},
        {"role": "assistant", "content": "Acknowledged the original destination."},
        {"role": "user", "content": (
            "Change the destination to DEST=GREEN.\n" + "Updated project specifications. " * 600
        )},
        {"role": "assistant", "content": "Acknowledged the revised destination."},
        {"role": "user", "content": "Continue checking the specifications."},
        {"role": "assistant", "content": "The specification checks are complete."},
        {"role": "user", "content": "Report the chosen destination from our earlier conversation."},
    ]

def native_client(model):
    return LLMClient(
        model=model, provider="anthropic", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0, context_window=40_000,
    )

@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("model,effort", [
    ("claude-fable-5-1", None), ("claude-fable-5-1", "low"),
    ("claude-mythos-5-1", None), ("claude-sonnet-4-6", None),
])
async def test_restored_session_summary_preserves_latest_revision(monkeypatch, tmp_path, borrowed, model, effort):
    requests = []
    params = {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "low"}}

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        summary = "Your task is to create a detailed summary" in json.dumps(body["messages"])
        text = "<summary>The user revised the destination to DEST=GREEN.</summary>" if summary else (
            "GREEN" if "DEST=GREEN" in json.dumps(body["messages"]) else "BLUE"
        )
        return completion_http_response(request, text=text)

    install_sdk_transport(monkeypatch, "anthropic", handler)
    client = native_client(model) if borrowed else None
    tracer = FakeTracer()
    session = build_session(
        agent=Agent(
            name="destination-reader", system_prompt="Follow the latest destination.",
            provider="anthropic", model=model, api_key="controlled-test",  # pragma: allowlist secret
            base_url="https://controlled.invalid/v1", context_window=40_000,
            thinking=True, thinking_params=params, reasoning_effort=effort, llm_max_retries=0,
        ),
        llm=client, env=LocalEnvironment(str(tmp_path)), auto_save_path=str(tmp_path / "session.json"),
        max_budget_tokens=40_000, tracer=tracer,
    )
    original = destination_history()
    session.messages = copy.deepcopy(original)
    try:
        session.save(tmp_path / "session.json")
        session.restore(tmp_path / "session.json")
        assert await session.run_loop() == "GREEN"
        assert session.messages[:len(original)] == original
        assert session._context_summarizer.last_call_cacheable is True
        assert session.used_tokens == 8
        assert len(requests) == 2
        assert "DEST=GREEN" in json.dumps(requests[1]["messages"])
        if model != "claude-sonnet-4-6":
            assert requests[0]["thinking"] == params["thinking"]
            assert requests[0]["output_config"] == params["output_config"]
        else:
            assert "thinking" not in requests[0]
            assert "output_config" not in requests[0]
        assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
        traces = [step["payload"] for step in tracer.steps if step["step_type"] == "llm_call"]
        assert len(traces) == 2
        assert traces[0]["purpose"] == "summary"
        assert traces[0]["thinking"] is ("thinking" in requests[0])
        assert traces[0]["usage"]["total_tokens"] == 4
        assert traces[1]["thinking"] is True
        assert traces[1]["usage"]["total_tokens"] == 4
    finally:
        await session.aclose()
        if client is not None:
            assert client._closed is False
            await client.close()

@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("model,params,output_budget,minimum", [
    ("claude-fable-5-1", {"thinking": {"type": "adaptive", "display": "omitted"}}, 0, 1),
    ("claude-fable-5-1", {"thinking": {"type": "adaptive", "display": "omitted"}}, 128, 1),
    ("claude-mythos-preview", {"thinking": {"type": "enabled", "budget_tokens": 1024}}, 1024, 1025),
    ("claude-mythos-preview", {"thinking": {"type": "enabled", "budget_tokens": 1024}}, 1025, 1025),
])
async def test_always_on_summary_respects_finite_budget(monkeypatch, tmp_path, borrowed, model, params,
                                                      output_budget, minimum):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="<summary>Preserved context.</summary>")

    install_sdk_transport(monkeypatch, "anthropic", handler)
    client = native_client(model) if borrowed else None
    session = build_session(
        agent=Agent(
            name="bounded-summary", system_prompt="system", provider="anthropic", model=model,
            api_key="controlled-test", base_url="https://controlled.invalid/v1",  # pragma: allowlist secret
            thinking=True, thinking_params=params, max_tokens_per_step=2048, llm_max_retries=0,
        ),
        llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=None,
    )
    segment = [{"role": "user", "content": "Keep this context"}]
    prepared = build_summary_request(segment)
    session.runner.max_budget_tokens = session.runner.llm.estimate_request_tokens(
        prepared, thinking=True, thinking_params=params,
    ) + output_budget
    try:
        if output_budget < minimum:
            with pytest.raises(_TokenBudgetStop) as error:
                await session._context_summarizer.asummarize(segment)
            assert error.value.minimum_output_tokens == minimum
            assert requests == []
            assert session.used_tokens == 0
        else:
            assert await session._context_summarizer.asummarize(segment) == "Summary:\nPreserved context."
            assert requests[0]["max_tokens"] == output_budget
            assert requests[0]["thinking"] == params["thinking"]
            assert session.used_tokens == 4
    finally:
        await session.aclose()
        if client is not None:
            await client.close()

async def test_standalone_always_on_summary_defaults_to_adaptive(monkeypatch):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="<summary>Preserved context.</summary>")

    install_sdk_transport(monkeypatch, "anthropic", handler)
    async with native_client("claude-fable-5-1") as client:
        agent = Agent(name="standalone-summary", system_prompt="system", model=client.model, provider="anthropic")
        summarizer = container._build_summarizer(agent, client, client, 600.0, None)
        assert await summarizer.asummarize([{"role": "user", "content": "Keep this context"}]) == (
            "Summary:\nPreserved context."
        )
        assert requests[0]["thinking"] == {"type": "adaptive"}
        assert agent.thinking_params == {}
        assert client._closed is False

async def test_summary_budget_uses_explicit_native_thinking_params(monkeypatch, tmp_path):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="<summary>Preserved context.</summary>")

    install_sdk_transport(monkeypatch, "anthropic", handler)
    async with native_client("claude-sonnet-4-6") as client:
        session = build_session(
            agent=Agent(name="explicit-summary", system_prompt="system", max_tokens_per_step=2048),
            llm=client, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=None,
        )
        params = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
        prepared = build_summary_request([{"role": "user", "content": "Keep this context"}])
        reserved = client.estimate_request_tokens(prepared, thinking=True, thinking_params=params)

        async def complete(messages, *, on_response, **kwargs):
            response = await client.complete(messages, **kwargs)
            on_response(response)
            return response

        try:
            session.runner.max_budget_tokens = reserved + 1024
            with pytest.raises(_TokenBudgetStop) as error:
                await session.runner._invoke_summary(complete, prepared, thinking=True, thinking_params=params)
            assert error.value.minimum_output_tokens == 1025
            assert requests == []
            session.runner.max_budget_tokens = reserved + 1025
            await session.runner._invoke_summary(complete, prepared, thinking=True, thinking_params=params)
            assert requests[0]["max_tokens"] == 1025
            assert requests[0]["thinking"] == params["thinking"]
            assert session.used_tokens == 4
        finally:
            await session.aclose()

@pytest.mark.parametrize("thinking", [False, True])
async def test_ordinary_completion_trace_preserves_configured_thinking(monkeypatch, tmp_path, thinking):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="ok")

    install_sdk_transport(monkeypatch, "anthropic", handler)
    tracer = FakeTracer()
    session = build_session(
        agent=Agent(
            name="ordinary-completion", system_prompt="Return ok.", provider="anthropic",
            model="claude-sonnet-4-6", api_key="controlled-test",  # pragma: allowlist secret
            base_url="https://controlled.invalid/v1", thinking=thinking,
            thinking_params={"thinking": {"type": "adaptive"}}, llm_max_retries=0,
        ), env=LocalEnvironment(str(tmp_path)), tracer=tracer,
    )
    try:
        await session.add_user_message("Return ok.")
        assert await session.run_loop() == "ok"
        calls = [step["payload"] for step in tracer.steps if step["step_type"] == "llm_call"]
        assert len(requests) == len(calls) == 1
        assert ("thinking" in requests[0]) is thinking
        assert calls[0]["thinking"] is thinking
        assert "purpose" not in calls[0]
        assert session.used_tokens == calls[0]["usage"]["total_tokens"] == 4
    finally:
        await session.aclose()

@pytest.mark.parametrize("model,thinking", [("claude-sonnet-4-6", False), ("claude-fable-5-1", True)])
async def test_late_summary_trace_keeps_resolved_thinking(monkeypatch, tmp_path, model, thinking):
    requests = []
    release = asyncio.Event()

    async def handler(request):
        requests.append(json.loads(request.content))
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return completion_http_response(request, text="<summary>Preserved context.</summary>")

    install_sdk_transport(monkeypatch, "anthropic", handler)
    tracer = FakeTracer()
    session = build_session(
        agent=Agent(
            name="late-summary", system_prompt="Keep the context.", provider="anthropic", model=model,
            api_key="controlled-test", base_url="https://controlled.invalid/v1",  # pragma: allowlist secret
            thinking=True, thinking_params={"thinking": {"type": "adaptive"}}, llm_max_retries=0,
        ), env=LocalEnvironment(str(tmp_path)), tracer=tracer, llm_timeout=0.1,
    )
    try:
        with pytest.raises(GenerationTimeoutError):
            await session._context_summarizer.asummarize([{"role": "user", "content": "Keep this context."}])
        assert len(requests) == 1
        assert ("thinking" in requests[0]) is thinking
        assert session.used_tokens == 0
        release.set()
        await asyncio.wait_for(asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True), 1)
        calls = [step["payload"] for step in tracer.steps if step["step_type"] == "llm_call"]
        assert len(calls) == 1
        assert calls[0]["purpose"] == "summary"
        assert calls[0]["thinking"] is thinking
        assert session.used_tokens == calls[0]["usage"]["total_tokens"] == 4
        assert session.runner.late_provider_usage == (4,)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True), 1)
        await session.aclose()
