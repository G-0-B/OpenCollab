"""Composed public workflow collections share budgets while producing real files."""

from __future__ import annotations

import asyncio
import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.application.workflow import WorkflowBudgetExceeded
from opencollab.tools import builtin_tools


class MarkerLLM:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak = 0
        self.observe = lambda: None

    async def complete(self, messages, tools=None, **kwargs):
        name = next(m["content"].splitlines()[0] for m in messages if m["role"] == "user")
        tool_reply = any(m["role"] == "tool" for m in messages)
        self.calls.append({"name": name, "tool_reply": tool_reply})
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.observe()
        try:
            await asyncio.sleep(0)
            if tool_reply:
                return LLMResponse(content=f"saved {name}", usage=Usage(300, 40), finish_reason="stop")
            return LLMResponse(tool_calls=[{
                "id": "save", "type": "function", "function": {
                    "name": "file_write", "arguments": json.dumps({
                        "path": f"{name}.txt", "mode": "create", "content": f"landed {name}",
                    }),
                },
            }], usage=Usage(300, 40), finish_reason="tool_calls")
        finally:
            self.active -= 1

async def _run_collection_workflow(
    workspace, composition, *, call_budget="omitted", workflow_budget=100_000,
    task_concurrency=3, limit_mode="explicit", artifacts=None,
):
    model = MarkerLLM()
    rows, contexts, observations = [], [], []

    async def flow(ctx, _inputs):
        contexts.append(ctx)

        def observe():
            observations.append({
                "spent": ctx.budget.spent(), "remaining": ctx.budget.remaining(),
                "leases": [{"total": lease.total, "remaining": lease.remaining(),
                            "session_ids": [session.state.aid for session in lease.sessions]}
                           for lease in ctx.budget._leases],
            })

        model.observe = observe

        async def unit(name):
            kwargs = {} if call_budget == "omitted" else {"budget": call_budget}
            row = await ctx.agent_run(
                name, tools=builtin_tools("file_write", headless=False),
                system_prompt="Write the requested marker file.", **kwargs,
            )
            rows.append(row)
            observe()
            return row

        async def collection(kind, items, operation):
            if kind == "parallel":
                return await ctx.parallel([lambda item=item: operation(item) for item in items])

            async def stage(_previous, item, _index):
                return await operation(item)

            return await ctx.pipeline(items, stage)

        async def nested(level, name):
            if level == len(composition):
                return await unit(name)
            count = 3 if level == 0 else 2
            return await collection(
                composition[level], [f"{name}_{index}" for index in range(count)],
                lambda child: nested(level + 1, child),
            )

        if composition == "serial":
            return [await unit(f"serial_{index}") for index in range(6)]
        if composition == "flat":
            return await collection("parallel", [f"flat_{index}" for index in range(6)], unit)
        if composition == "sequential":
            first = await collection("parallel", [f"parallel_{index}" for index in range(6)], unit)
            second = await collection("pipeline", [f"pipeline_{index}" for index in range(6)], unit)
            return first, second
        return await nested(0, "marker")

    kwargs = {} if workflow_budget == "omitted" else {"budget": workflow_budget}
    result = await OpenCollab(
        workspace, model="test-model", config={"budget": 100_000, "max_output_tokens": 128},
    ).workflow(
        flow, llm=model, concurrency=3, task_concurrency=task_concurrency,
        limit_mode=limit_mode, agent_profile="single2", timeout=30,
        trace=artifacts is not None, artifacts=artifacts, **kwargs,
    )
    return result, rows, model, contexts[0], observations

def _assert_delivery(workspace, execution, expected_count, task_concurrency):
    result, rows, model, ctx, observations = execution
    assert result.status == "completed", result.reason
    assert len(rows) == expected_count
    assert all(row.status == "completed" for row in rows)
    assert all(row.cleanup_complete and row.workspace_ready for row in rows)
    assert len({row.session_id for row in rows}) == expected_count
    assert result.tokens == sum(row.tokens for row in rows) == expected_count * 680
    assert len(model.calls) == expected_count * 2
    assert sum(call["tool_reply"] for call in model.calls) == expected_count
    assert 1 <= model.peak <= min(3, task_concurrency)
    names = {call["name"] for call in model.calls}
    assert {path.name: path.read_text() for path in workspace.glob("*.txt")} == {
        f"{name}.txt": f"landed {name}" for name in names
    }
    assert ctx.budget._leases == []
    assert not ctx.pending_cleanup_tasks
    assert ctx._task_semaphore._value == task_concurrency
    assert ctx._active_collection_budget.get() is None
    if ctx.budget.total is None:
        assert ctx.budget.remaining() == float("inf")
    else:
        assert ctx.budget.remaining() == ctx.budget.total - result.tokens
        assert all(observation["remaining"] >= 0 for observation in observations)

@pytest.mark.asyncio
@pytest.mark.parametrize("call_budget", ["omitted", None])
@pytest.mark.parametrize("composition", [
    ("parallel", "parallel"), ("parallel", "pipeline"),
    ("pipeline", "parallel"), ("pipeline", "pipeline"),
    ("parallel", "pipeline", "parallel"),
])
async def test_nested_collections_deliver_all_files_with_uncapped_calls(tmp_path, composition, call_budget):
    execution = await _run_collection_workflow(tmp_path, composition, call_budget=call_budget)
    expected_count = 3 * 2 ** (len(composition) - 1)
    _assert_delivery(tmp_path, execution, expected_count, 3)

@pytest.mark.asyncio
@pytest.mark.parametrize("composition,workflow_budget,call_budget,task_concurrency,unbounded,limit_mode", [
    (("parallel", "parallel"), "omitted", "omitted", 3, False, "explicit"),
    (("parallel", "pipeline"), None, None, 3, False, "explicit"),
    (("pipeline", "parallel"), 1_000_000, "omitted", 3, False, "explicit"),
    (("parallel", "parallel"), 100_000, 10_000, 3, False, "explicit"),
    (("parallel", "parallel"), 10_000, "omitted", 3, False, "explicit"),
    (("parallel", "parallel"), 100_000, "omitted", 1, False, "explicit"),
    ("flat", 100_000, "omitted", 3, False, "explicit"),
    ("serial", 100_000, "omitted", 3, False, "explicit"),
    ("sequential", 100_000, "omitted", 3, False, "explicit"),
    (("parallel", "parallel"), None, None, 3, True, "environment"),
    (("pipeline", "parallel"), 100_000, 10_000, 3, True, "environment"),
    (("parallel", "pipeline"), 100_000, "omitted", 3, True, "explicit"),
])
async def test_collection_budget_controls(
    tmp_path, monkeypatch, composition, workflow_budget, call_budget, task_concurrency, unbounded, limit_mode,
):
    monkeypatch.setenv("OPENCOLLAB_UNBOUNDED_LIMITS", "true" if unbounded else "false")
    execution = await _run_collection_workflow(
        tmp_path, composition, workflow_budget=workflow_budget, call_budget=call_budget,
        task_concurrency=task_concurrency, limit_mode=limit_mode,
    )
    _assert_delivery(tmp_path, execution, 12 if composition == "sequential" else 6, task_concurrency)
    _, rows, _, ctx, _ = execution
    if unbounded and limit_mode == "environment":
        assert ctx.budget.total is None
        assert all(row.hard_budget_tokens is None for row in rows)
    else:
        assert ctx.budget.total == (100_000 if workflow_budget in ("omitted", None) else workflow_budget)
        if isinstance(call_budget, int):
            assert all(row.hard_budget_tokens == call_budget for row in rows)

@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [2_000, 6_000])
async def test_small_collection_shares_retain_session_budget_stops(tmp_path, budget):
    result, rows, model, ctx, _ = await _run_collection_workflow(
        tmp_path, ("parallel", "parallel"), workflow_budget=budget,
    )
    assert result.status == "completed", result.reason
    assert len(rows) == 6
    assert all(row.status == "stopped" and "budget exhausted before model call" in row.reason for row in rows)
    assert all(row.cleanup_complete and row.workspace_ready for row in rows)
    assert result.tokens == sum(row.tokens for row in rows)
    assert all(row.tokens <= row.hard_budget_tokens for row in rows)
    assert len(model.calls) <= 6
    assert ctx.budget.remaining() == budget - result.tokens
    assert ctx.budget._leases == []
    assert not ctx.pending_cleanup_tasks

@pytest.mark.asyncio
async def test_exhausted_workflow_rejects_nested_work_and_retains_one_escape(tmp_path):
    calls, contexts = [], []

    class SpendingLLM:
        async def complete(self, messages, tools=None, **kwargs):
            calls.append(messages)
            return LLMResponse(
                content="finished", usage=Usage(100_000 if len(calls) == 1 else 340, 0), finish_reason="stop",
            )

    async def flow(ctx, _inputs):
        contexts.append(ctx)
        first = await ctx.agent_run("spend the budget", tools=[], system_prompt="Finish.")
        assert ctx.budget.remaining() == 0

        async def group():
            return await ctx.parallel([lambda: ctx.agent_run("blocked", tools=[])])

        with pytest.raises(WorkflowBudgetExceeded):
            await ctx.parallel([group])
        escaped = await ctx.agent("forced finish", tools=[], over_budget_ok=True)
        with pytest.raises(WorkflowBudgetExceeded):
            await ctx.agent("second escape", tools=[], over_budget_ok=True)
        return first, escaped

    result = await OpenCollab(tmp_path, model="test-model", config={"max_output_tokens": 128}).workflow(
        flow, budget=100_000, llm=SpendingLLM(), limit_mode="explicit", agent_profile="single2",
    )
    assert result.status == "completed", result.reason
    assert len(calls) == 2
    assert result.tokens == 100_340
    ctx = contexts[0]
    assert ctx.budget.spent() == result.tokens
    assert ctx.budget.remaining() == -340
    assert ctx.budget._leases == []
    assert not ctx.pending_cleanup_tasks
