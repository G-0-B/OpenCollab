"""Default enforcement reserve follows the session's approved token budget."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from opencollab import OpenCollab
from opencollab.application.session_run import DEFAULT_COMMIT_RESERVE, ENFORCEMENT_ON
from opencollab.bootstrap._workflow_runtime_session import WorkflowSessionFactory


@pytest.fixture
def recorded_provider(monkeypatch):
    monkeypatch.delenv("OPENCOLLAB_UNBOUNDED_LIMITS", raising=False)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    requests = []
    sessions = []
    original_client = openai.AsyncOpenAI
    original_build = WorkflowSessionFactory.build_workflow_session

    async def respond(request):
        requests.append(json.loads(request.content))
        arguments = json.dumps({
            "summary": "Scout findings", "findings": [], "insufficient_evidence": True,
        })
        return httpx.Response(200, json={
            "id": "test-response", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "submit", "type": "function", "function": {
                    "name": "submit_findings", "arguments": arguments,
                }}],
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def build_client(**kwargs):
        return original_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    def build_session(factory, **kwargs):
        session = original_build(factory, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(openai, "AsyncOpenAI", build_client)
    monkeypatch.setattr(WorkflowSessionFactory, "build_workflow_session", build_session)
    return requests, sessions

def _client(tmp_path):
    return OpenCollab(
        tmp_path, model="test-model", provider="openai",
        api_key="test-key",  # pragma: allowlist secret -- in-memory transport
        base_url="https://provider.example.invalid/v1",
    )

@pytest.mark.parametrize("unbounded", [False, True])
@pytest.mark.parametrize("count", [1, 3])
async def test_default_scouts_execute_after_internal_budget_split(
    tmp_path, monkeypatch, recorded_provider, unbounded, count,
):
    requests, sessions = recorded_provider
    if unbounded:
        monkeypatch.setenv("OPENCOLLAB_UNBOUNDED_LIMITS", "true")

    async def workflow(ctx, _args):
        return await ctx.parallel([
            lambda: ctx.agent("Scout", enforcement_strength=ENFORCEMENT_ON) for _ in range(count)
        ])

    result = await _client(tmp_path).workflow(
        workflow, budget=60_000, concurrency=count, system_prompt="Scout test", timeout=None, trace=False,
    )

    approved = None if unbounded else 60_000 // count
    reserve = 5_000 if approved == 20_000 else DEFAULT_COMMIT_RESERVE
    assert result.ok
    assert all("Scout findings" in value for value in result.output)
    assert len(requests) == len(sessions) == count
    assert [session.max_budget_tokens for session in sessions] == [approved] * count
    assert [session.runner._commit_reserve for session in sessions] == [reserve] * count
    assert result.tokens == 15 * count

@pytest.mark.parametrize("cap,reserve", [(20_000, 5_000), (25_000, 25_000), (30_000, 25_000)])
async def test_default_reserve_preserves_valid_existing_values(tmp_path, recorded_provider, cap, reserve):
    requests, sessions = recorded_provider

    async def workflow(ctx, _args):
        return await ctx.agent("Scout", budget=cap, enforcement_strength=ENFORCEMENT_ON)

    result = await _client(tmp_path).workflow(
        workflow, budget=60_000, system_prompt="Scout test", timeout=None, trace=False,
    )

    assert result.ok and "Scout findings" in result.output
    assert len(requests) == 1
    assert sessions[0].max_budget_tokens == cap
    assert sessions[0].runner._commit_reserve == reserve
    expected_choice = {"type": "function", "function": {"name": "submit_findings"}} if cap == 25_000 else "auto"
    assert requests[0].get("tool_choice") == expected_choice

@pytest.mark.parametrize("reserve", [1, 5_000, 20_000])
async def test_explicit_valid_reserve_is_used_unchanged(tmp_path, recorded_provider, reserve):
    requests, sessions = recorded_provider

    async def workflow(ctx, _args):
        return await ctx.agent("Scout", budget=20_000, enforcement_strength=ENFORCEMENT_ON, commit_reserve=reserve)

    result = await _client(tmp_path).workflow(
        workflow, budget=60_000, system_prompt="Scout test", timeout=None, trace=False,
    )

    assert result.ok and "Scout findings" in result.output
    assert len(requests) == 1
    assert sessions[0].runner._commit_reserve == reserve

@pytest.mark.parametrize("unbounded", [False, True])
@pytest.mark.parametrize("reserve", [0, -1, True, False, 1.5, "5000", 25_000])
async def test_explicit_reserve_retains_parameter_validation(
    tmp_path, monkeypatch, recorded_provider, unbounded, reserve,
):
    requests, sessions = recorded_provider
    if unbounded:
        monkeypatch.setenv("OPENCOLLAB_UNBOUNDED_LIMITS", "true")
    errors = []

    async def workflow(ctx, _args):
        try:
            return await ctx.agent("Scout", budget=20_000, enforcement_strength=ENFORCEMENT_ON, commit_reserve=reserve)
        except ValueError as exc:
            errors.append(str(exc))
            return "invalid reserve"

    result = await _client(tmp_path).workflow(
        workflow, budget=60_000, system_prompt="Scout test", timeout=None, trace=False,
    )

    assert len(sessions) == 1
    if unbounded and reserve == 25_000:
        assert result.ok and "Scout findings" in result.output
        assert len(requests) == 1
        assert sessions[0].runner._commit_reserve == reserve
        assert errors == []
    else:
        assert result.output == "invalid reserve"
        assert requests == []
        assert errors == [
            "commit_reserve cannot exceed max_budget_tokens" if reserve == 25_000
            else "commit_reserve must be a positive integer"
        ]
