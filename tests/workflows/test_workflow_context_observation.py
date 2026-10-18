"""Observation and event sink tests for WorkflowContext."""

from __future__ import annotations

from typing import Any

import pytest

from opencollab.application.workflow import (
    WorkflowContext,
)
from opencollab.application.workflow_candidates import _CandidateLeaseTreeProbe
from tests.support.workflow_context_test_support import (
    FakeFactory,
    FakeProbe,
    FakeSession,
    RecordingSink,
    RecordingTracer,
)


@pytest.mark.asyncio
async def test_phase_and_log_emit_via_sink():
    sink = RecordingSink()
    ctx = WorkflowContext(FakeFactory([]), event_sink=sink)

    await ctx.phase("planning")
    await ctx.log("hello world")

    assert len(sink.events) == 2

@pytest.mark.asyncio
async def test_phase_and_log_noop_without_sink():
    ctx = WorkflowContext(FakeFactory([]))
    await ctx.phase("planning")
    await ctx.log("hello world")

@pytest.mark.asyncio
async def test_phase_and_log_ignore_observer_failures():
    class FailingSink:
        async def emit(self, event: Any) -> None:
            raise RuntimeError("sink failed")

    class FailingTracer:
        def log_step(self, **kwargs: Any) -> None:
            raise RuntimeError("trace failed")

    ctx = WorkflowContext(
        FakeFactory([]), event_sink=FailingSink(), tracer=FailingTracer()
    )

    await ctx.phase("planning")
    await ctx.log("still running")

@pytest.mark.asyncio
async def test_tree_changed_is_none_without_probe():
    ctx = WorkflowContext(FakeFactory([]))
    assert await ctx.tree_changed() is None

@pytest.mark.asyncio
async def test_tree_changed_proxies_probe_result():
    ctx_yes = WorkflowContext(FakeFactory([]), tree_probe=FakeProbe(changed=True))
    ctx_no = WorkflowContext(FakeFactory([]), tree_probe=FakeProbe(changed=False))
    assert await ctx_yes.tree_changed() is True
    assert await ctx_no.tree_changed() is False

@pytest.mark.asyncio
async def test_tree_changed_swallows_probe_error_to_none():
    ctx = WorkflowContext(FakeFactory([]), tree_probe=FakeProbe(boom=True))
    assert await ctx.tree_changed() is None

@pytest.mark.asyncio
async def test_source_changed_excludes_injected_paths():
    # Exclude the injected test from source changes while retaining whole-tree dirtiness.
    probe = FakeProbe(changed=True, excluded_changed=False)
    ctx = WorkflowContext(FakeFactory([]), tree_probe=probe)

    assert await ctx.tree_changed() is True
    assert await ctx.source_changed(["t/test_x.py"]) is False
    assert probe.exclude_calls == [("t/test_x.py",)]

@pytest.mark.asyncio
async def test_source_changed_is_none_without_probe():
    ctx = WorkflowContext(FakeFactory([]))
    assert await ctx.source_changed(["t/test_x.py"]) is None

@pytest.mark.asyncio
async def test_source_changed_swallows_probe_error_to_none():
    ctx = WorkflowContext(FakeFactory([]), tree_probe=FakeProbe(boom=True))
    assert await ctx.source_changed(["t/test_x.py"]) is None

@pytest.mark.asyncio
async def test_diff_reports_probe_output_or_unknown() -> None:
    assert await WorkflowContext(FakeFactory([])).diff() is None
    assert (
        await WorkflowContext(
            FakeFactory([]),
            tree_probe=FakeProbe(),
        ).diff()
        == "diff"
    )
    assert (
        await WorkflowContext(
            FakeFactory([]),
            tree_probe=FakeProbe(boom=True),
        ).diff()
        is None
    )

@pytest.mark.asyncio
async def test_diff_exclusions_keep_legacy_probe_and_backend_behavior() -> None:
    probe = FakeProbe()
    context = WorkflowContext(FakeFactory([]), tree_probe=probe)
    assert await context.diff() == "diff"
    assert await context.diff(["test_probe.py"]) is None

    class Backend:
        def __init__(self):
            self.calls = []

        async def source_diff(self, paths):
            self.calls.append(tuple(paths))
            return "filtered backend diff"

    backend = Backend()
    context = WorkflowContext(FakeFactory([]), tree_probe=probe, candidate_workspace=backend)
    assert await context.diff() == "diff"
    assert await context.diff(["test_probe.py"]) == "filtered backend diff"
    assert backend.calls == [("test_probe.py",)]

@pytest.mark.asyncio
async def test_candidate_diff_exclusion_failure_remains_unknown() -> None:
    class FailingLease:
        async def diff(self, exclude_paths=()):
            raise OSError("candidate diff read failed")

    context = WorkflowContext(FakeFactory([]), tree_probe=_CandidateLeaseTreeProbe(FailingLease()))
    assert await context.diff() is None
    assert await context.diff(["test_probe.py"]) is None

@pytest.mark.asyncio
async def test_token_observation_reports_live_session_usage() -> None:
    ctx = WorkflowContext(
        FakeFactory([FakeSession(tokens=12)]),
        budget_total=100,
    )

    assert ctx.tokens_spent() == 0
    assert ctx.tokens_remaining() == 100
    await ctx.agent("measure usage")
    assert ctx.tokens_spent() == 12
    assert ctx.tokens_remaining() == 88

class _NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name

@pytest.mark.asyncio
async def test_starting_an_agent_records_which_agent_it_was() -> None:
    """A workflow says who ran, not just that something ran.

    The agent's own records — ``llm_call``, ``tool_exec``, ``session_terminal``
    — carry its ``aid`` and a role that is the same constant for every workflow
    agent. Without this record the script's own name for the agent survives only
    in a transcript filename, and only when the run was given a folder to save
    into.
    """
    tracer = RecordingTracer()
    ctx = WorkflowContext(
        FakeFactory([FakeSession()]),
        tracer=tracer,
        budget_total=100_000,
    )

    await ctx.phase("triage")
    await ctx.agent("read the failure", label="scout", tools=[_NamedTool("read")])

    assert tracer.payloads("workflow_agent_started") == [
        {
            "aid": 0,
            "label": "scout",
            "phase": "triage",
            "tools": ["read"],
            "budget": 100_000,
            "isolated": False,
        }
    ]

@pytest.mark.asyncio
async def test_each_agent_in_a_run_is_recorded_under_its_own_id() -> None:
    """Two agents, two ids — the join key that makes the run per-agent."""
    tracer = RecordingTracer()
    ctx = WorkflowContext(
        FakeFactory([FakeSession(), FakeSession()]),
        tracer=tracer,
        budget_total=100_000,
    )

    await ctx.phase("find")
    await ctx.agent("first", label="scout")
    await ctx.phase("fix")
    await ctx.agent("second", label="coder")

    recorded = tracer.payloads("workflow_agent_started")
    assert [(row["aid"], row["label"], row["phase"]) for row in recorded] == [
        (0, "scout", "find"),
        (1, "coder", "fix"),
    ]

@pytest.mark.asyncio
async def test_an_agent_that_never_built_is_not_recorded_as_started() -> None:
    """A build failure has no id to attribute anything to, so it claims none."""

    class ExplodingFactory(FakeFactory):
        def build_workflow_session(self, **kwargs):
            raise RuntimeError("factory down")

    tracer = RecordingTracer()
    ctx = WorkflowContext(ExplodingFactory([]), tracer=tracer, budget_total=100_000)

    assert await ctx.agent("never runs", label="scout") is None
    assert tracer.payloads("workflow_agent_started") == []
    assert [failure["label"] for failure in ctx.agent_failures] == ["scout"]

@pytest.mark.asyncio
async def test_an_agent_started_before_any_phase_records_no_phase() -> None:
    tracer = RecordingTracer()
    ctx = WorkflowContext(
        FakeFactory([FakeSession()]), tracer=tracer, budget_total=100_000
    )

    await ctx.agent("unstaged work", label="scout")

    (payload,) = tracer.payloads("workflow_agent_started")
    assert payload["phase"] is None
