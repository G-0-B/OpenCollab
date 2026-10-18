"""Workflow step limits reach real sessions through the public SDK."""

from __future__ import annotations

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.types import LLMResponse, Usage


class ReplyLLM:
    async def complete(self, messages, **kwargs):
        return LLMResponse(content="finished", usage=Usage(4, 2), finish_reason="stop")

@pytest.mark.parametrize("global_unbounded", [False, True], ids=["finite-env", "unbounded-env"])
@pytest.mark.parametrize("limit_mode", ["environment", "explicit"])
@pytest.mark.parametrize(
    ("step_options", "expected_steps"),
    [({}, 100), ({"max_steps": None}, None), ({"max_steps": 7}, 7)],
    ids=["default", "none", "integer"],
)
async def test_workflow_step_limits_reach_sessions(
    tmp_path, monkeypatch, global_unbounded, limit_mode, step_options, expected_steps,
):
    monkeypatch.setenv("OPENCOLLAB_UNBOUNDED_LIMITS", "true" if global_unbounded else "false")

    async def flow(ctx, _inputs):
        answer = await ctx.agent("Finish once.", tools=[])
        return {"answer": answer, "step_limits": [s.max_steps for s in ctx.sessions],
                "budget": ctx.budget.total}

    result = await OpenCollab(tmp_path, model="fixture-model").workflow(
        flow, budget=1_000, llm=ReplyLLM(), trace=False, limit_mode=limit_mode, **step_options,
    )

    unbounded = global_unbounded and limit_mode == "environment"
    assert result.status == "completed"
    assert result.output == {
        "answer": "finished",
        "step_limits": [None if unbounded else expected_steps],
        "budget": None if unbounded else 1_000,
    }
    assert result.tokens == 6

@pytest.mark.parametrize("max_steps", [True, False, 0, -1, 1.5, "1", []])
@pytest.mark.parametrize("limit_mode", ["environment", "explicit"])
async def test_workflow_rejects_invalid_step_limits_before_callback(
    tmp_path, monkeypatch, max_steps, limit_mode,
):
    monkeypatch.setenv("OPENCOLLAB_UNBOUNDED_LIMITS", "true" if limit_mode == "explicit" else "false")
    called = False

    async def flow(_ctx, _inputs):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        await OpenCollab(tmp_path, model="fixture-model").workflow(
            flow, max_steps=max_steps, llm=ReplyLLM(), trace=False, limit_mode=limit_mode,
        )

    assert called is False
