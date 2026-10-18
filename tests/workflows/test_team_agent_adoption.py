"""Adoption uses successful child ownership and its complete captured result."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from opencollab.adapters.tools.adopt_agent import AdoptAgentTool
from opencollab.application._scheduler_constants import WORKTREE_DIFF_MAX_CHARS
from opencollab.application.scheduler import Scheduler
from opencollab.application.tool_execution import ToolRuntime
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase, SessionState


def _scheduler():
    applied = []

    async def adopt(patch):
        applied.append(patch)

    async def persist(_aid, _role, _patch):
        return "/saved/candidate.patch"

    scheduler = Scheduler(session_factory=object(), worktree_pool=SimpleNamespace(adopt=adopt),
                          event_sink=object(), candidate_diff_writer=persist)
    for aid, parent, role in ((0, None, "analyst"), (7, 0, "coder"), (8, 0, "tester"), (9, 7, "coder")):
        state = SessionState(messages=[])
        state.set_phase(SessionPhase.EXECUTING_TOOLS if aid == 0 else SessionPhase.DONE)
        tools = [SimpleNamespace(name="adopt_agent")] if aid == 0 else []
        scheduler.table.add(SessionControlBlock(aid, parent, SimpleNamespace(name=role, tools=tools), state))
    scheduler._turn_started_at[0] = 10.0
    return scheduler, applied

class _CapturedEnvironment:
    workspace = "/isolated-coder"

    def __init__(self, diff):
        self.diff = diff

    async def get_diff(self):
        return self.diff

async def test_full_captured_diff_is_adopted_once_after_text_is_bounded():
    scheduler, applied = _scheduler()
    diff = "diff --git a/file b/file\n" + "x" * (WORKTREE_DIFF_MAX_CHARS + 500)
    env = _CapturedEnvironment(diff)
    text = await scheduler._append_worktree_diff(env, "completed", aid=7, role="coder")
    assert "Completed coder aid=7" in text and "chars truncated" in text
    assert diff not in text
    env.diff = "later unreported edit"
    assert "Adopted coder aid=7" in await scheduler.adopt_agent(0, 7)
    assert applied == [diff]
    assert "already adopted" in await scheduler.adopt_agent(0, 7)
    assert applied == [diff]
    scheduler._turn_started_at[0] = 11.0
    with pytest.raises(ValueError, match="in this turn"):
        await scheduler.adopt_agent(0, 7)

@pytest.mark.parametrize("aid,phase,partial", [
    (7, SessionPhase.CALLING_LLM, False),
    (7, SessionPhase.ERROR, True),
    (7, SessionPhase.STOPPED, True),
    (8, SessionPhase.DONE, False),
    (9, SessionPhase.DONE, False),
])
async def test_only_a_completed_direct_coder_is_available(aid, phase, partial):
    scheduler, applied = _scheduler()
    scheduler.table.get(aid).state.set_phase(phase)
    await scheduler._append_worktree_diff(_CapturedEnvironment("change"), "result", aid=aid,
                                          role=scheduler.table.get(aid).agent.name, partial=partial)
    with pytest.raises(ValueError, match="successfully completed direct coder"):
        await scheduler.adopt_agent(0, aid)
    assert applied == []

async def test_runtime_identity_and_schema_reject_arbitrary_changes():
    scheduler, applied = _scheduler()
    await scheduler._append_worktree_diff(_CapturedEnvironment("captured"), "done", aid=7, role="coder")
    runtime = ToolRuntime(None, None, None, aid=0)
    tool = AdoptAgentTool(scheduler)
    for params in ({"aid": 7, "patch": "arbitrary"}, {"sha": "HEAD"}, {"aid": 7, "path": "answer.py"}):
        with pytest.raises(ValueError, match="only a completed coder aid"):
            await tool.execute_with_runtime(params, runtime)
    for aid in (True, "7", 7.0):
        with pytest.raises(ValueError, match="integer coder aid"):
            await scheduler.adopt_agent(0, aid)
    with pytest.raises(PermissionError, match="entry agent"):
        await scheduler.adopt_agent(7, 7)
    scheduler.table.get(0).agent.tools.clear()
    with pytest.raises(PermissionError, match="declared capability"):
        await scheduler.adopt_agent(0, 7)
    assert applied == []

async def test_cancellation_settles_apply_and_adopted_record_together():
    scheduler, applied = _scheduler()
    await scheduler._append_worktree_diff(_CapturedEnvironment("captured"), "done", aid=7, role="coder")
    written = asyncio.Event()
    release = asyncio.Event()

    async def interrupted_return(patch):
        applied.append(patch)
        written.set()
        await release.wait()

    scheduler._worktree_pool.adopt = interrupted_return
    operation = asyncio.create_task(scheduler.adopt_agent(0, 7))
    await written.wait()
    operation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert 7 in scheduler._adopted_coders and applied == ["captured"]
    assert "already adopted" in await scheduler.adopt_agent(0, 7)
    assert applied == ["captured"]

async def test_conflict_can_be_retried_without_marking_the_child_adopted():
    scheduler, applied = _scheduler()
    await scheduler._append_worktree_diff(_CapturedEnvironment("captured"), "done", aid=7, role="coder")
    apply = scheduler._worktree_pool.adopt

    async def conflict(_patch):
        raise RuntimeError("conflict")

    scheduler._worktree_pool.adopt = conflict
    with pytest.raises(RuntimeError, match="conflict"):
        await scheduler.adopt_agent(0, 7)
    assert 7 not in scheduler._adopted_coders
    scheduler._worktree_pool.adopt = apply
    await scheduler.adopt_agent(0, 7)
    assert applied == ["captured"]

async def test_custom_diff_environment_preserves_first_adoption_and_reports_missing_increment():
    scheduler, applied = _scheduler()
    env = _CapturedEnvironment("first")
    scheduler._sessions[7] = SimpleNamespace(env=env)
    await scheduler._append_worktree_diff(env, "done", aid=7, role="coder")
    await scheduler.adopt_agent(0, 7)
    scheduler.table.get(7).state.set_step_count(1)
    await scheduler._append_worktree_diff(env, "done again", aid=7, role="coder")
    assert "Adopted coder" in await scheduler.adopt_agent(0, 7)
    assert applied == ["first"]
    adopted = scheduler._adopted_coders[7]
    scheduler.table.get(7).state.set_step_count(2)
    env.diff = "second"
    await scheduler._append_worktree_diff(env, "revised", aid=7, role="coder")
    with pytest.raises(RuntimeError, match="cannot calculate an increment.*full candidate retained"):
        await scheduler.adopt_agent(0, 7)
    assert applied == ["first"]
    assert scheduler._adopted_coders[7] == adopted
    assert scheduler._completed_coder_diffs[7].patch == "second"

async def test_completion_during_adoption_keeps_the_actually_delivered_receipt():
    scheduler, applied = _scheduler()

    class IncrementEnvironment(_CapturedEnvironment):
        async def diff_captured(self, before_patch, after_patch):
            return f"{before_patch} -> {after_patch}"

    env = IncrementEnvironment("first")
    scheduler._sessions[7] = SimpleNamespace(env=env)
    await scheduler._append_worktree_diff(env, "done", aid=7, role="coder")
    await scheduler.adopt_agent(0, 7)
    started, release = asyncio.Event(), asyncio.Event()
    adopt = scheduler._worktree_pool.adopt

    async def delayed(patch):
        started.set()
        await release.wait()
        await adopt(patch)

    scheduler._worktree_pool.adopt = delayed
    env.diff = "second"
    scheduler.table.get(7).state.set_step_count(1)
    await scheduler._append_worktree_diff(env, "done", aid=7, role="coder")
    pending = asyncio.create_task(scheduler.adopt_agent(0, 7))
    await started.wait()
    env.diff = "third"
    scheduler.table.get(7).state.set_step_count(2)
    await scheduler._append_worktree_diff(env, "done", aid=7, role="coder")
    release.set()
    await pending
    assert scheduler._adopted_coders[7].patch == "second"
    assert scheduler._completed_coder_diffs[7].patch == "third"
    await scheduler.adopt_agent(0, 7)
    assert applied == ["first", "first -> second", "second -> third"]
