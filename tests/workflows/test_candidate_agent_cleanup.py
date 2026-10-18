"""A candidate waits for its own execution before capturing or releasing it."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path

import httpx
import openai
import pytest

from opencollab import OpenCollab
from opencollab.adapters import _env_local as local_module
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace, _CandidateLease
from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.application.tool_execution import ToolExecutionUseCase
from opencollab.application.workflow import WorkflowContext
from opencollab.bootstrap import _workflow_runtime_session as runtime
from opencollab.tools import builtin_tools, evidence_tools
from tests.support.workflow_context_test_support import FakeFactory, FakeSession


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCOLLAB_UNBOUNDED_LIMITS", raising=False)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    (tmp_path / "source.txt").write_text("initial\n")
    for args in (
        ("init", "-q"),
        ("add", "source.txt"),
        ("-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-qm", "\u521d\u59cb\u5316\u6d4b\u8bd5\u4ed3\u5e93"),
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path

@pytest.fixture
def provider_calls(monkeypatch):
    original_client = openai.AsyncOpenAI
    calls = []

    async def respond(request):
        calls.append(request)
        return httpx.Response(200, json={
            "id": "test-response", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def build_client(**kwargs):
        return original_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "AsyncOpenAI", build_client)
    return calls

def _client(repository):
    return OpenCollab(
        repository, model="test-model", provider="openai",
        api_key="test-key",  # pragma: allowlist secret -- in-memory transport
        base_url="https://provider.example.invalid/v1", config={"llm_max_retries": 0},
    )

@pytest.mark.parametrize("concurrency", [1, 2])
async def test_public_candidate_and_ordinary_agent_complete(repository, provider_calls, concurrency):
    async def workflow(ctx, _args):
        values = await ctx.parallel([
            lambda: ctx.candidate_agent("candidate", label="candidate"),
            lambda: ctx.agent("ordinary", label="ordinary"),
        ])
        return [values[0].output, values[1]]

    task = asyncio.create_task(_client(repository).workflow(
        workflow, budget=100_000, concurrency=concurrency, task_concurrency=2, timeout=None, trace=False,
    ))
    try:
        done, _ = await asyncio.wait({task}, timeout=3)
        assert done, "candidate completed its model call but blocked the queued ordinary agent"
        result = task.result()
        assert result.ok, result.reason
        assert result.output == ["answer", "answer"]
        assert len(provider_calls) == 2
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

async def test_public_candidate_waits_for_session_subscriber_write(repository, provider_calls, monkeypatch):
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    candidate_returned = asyncio.Event()
    context = None
    original_build = runtime.build_session

    class DelayedWriteSubscriber:
        def __init__(self, environment):
            self.environment = environment
            self.pending_tasks = []

        async def write(self):
            cleanup_started.set()
            await release_cleanup.wait()
            await self.environment.write_file("source.txt", "late candidate write\n")

        def emit(self, event):
            if event.type == "step_end" and not self.pending_tasks:
                self.pending_tasks.append(asyncio.create_task(self.write()))

    def build(**kwargs):
        session = original_build(**kwargs)
        session.event_bus.subscribe(DelayedWriteSubscriber(kwargs["env"]))
        return session

    monkeypatch.setattr(runtime, "build_session", build)

    async def workflow(ctx, _args):
        nonlocal context
        context = ctx
        candidate = await ctx.candidate_agent("candidate", label="candidate", budget=100_000)
        candidate_returned.set()
        return candidate.diff

    task = asyncio.create_task(_client(repository).workflow(
        workflow, budget=100_000, concurrency=1, timeout=None, trace=False,
    ))
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        await asyncio.sleep(0.02)
        assert not candidate_returned.is_set()
        assert context.tokens_remaining() == 0
        assert context._semaphore.locked()
        assert (repository / "source.txt").read_text() == "initial\n"
    finally:
        release_cleanup.set()
        result = await asyncio.wait_for(task, timeout=5)

    assert result.ok, result.reason
    assert "+late candidate write" in result.output
    assert len(provider_calls) == 1
    assert context.tokens_remaining() == 99_985
    assert (repository / "source.txt").read_text() == "initial\n"

@pytest.mark.parametrize("past_cleanup_deadline", [False, True])
async def test_public_candidate_timeout_captures_owned_atomic_write(repository, monkeypatch, past_cleanup_deadline):
    started, release, written = threading.Event(), threading.Event(), threading.Event()
    cancellation_started, capture_started = asyncio.Event(), asyncio.Event()
    sessions = []
    original_write, original_diff = local_module.write_regular_bytes_atomic, _CandidateLease.diff
    original_build = runtime.build_session
    original_cancel = ToolExecutionUseCase._cleanup_caller_cancelled_execution

    def controlled_write(*args, **kwargs):
        if args[2] == "source.txt":
            started.set()
            assert release.wait(5)
        result = original_write(*args, **kwargs)
        if args[2] == "source.txt":
            written.set()
        return result

    async def complete(_llm, messages, **kwargs):
        return LLMResponse(tool_calls=[{
            "id": "write", "type": "function", "function": {
                "name": "file_write", "arguments": json.dumps({
                    "path": "source.txt", "mode": "create", "overwrite": True,
                    "content": "late native candidate write\n",
                }),
            },
        }], finish_reason="tool_calls", usage=Usage(10, 5))

    def build(**kwargs):
        session = original_build(**kwargs)
        sessions.append(session)
        return session

    async def cancelled(executor, execution):
        cancellation_started.set()
        await original_cancel(executor, execution)

    async def diff(lease, *args, **kwargs):
        capture_started.set()
        assert written.is_set(), "candidate captured before its atomic write completed"
        return await original_diff(lease, *args, **kwargs)

    monkeypatch.setattr(local_module, "write_regular_bytes_atomic", controlled_write)
    monkeypatch.setattr(LLMClient, "complete", complete)
    monkeypatch.setattr(runtime, "build_session", build)
    monkeypatch.setattr(ToolExecutionUseCase, "_cleanup_caller_cancelled_execution", cancelled)
    monkeypatch.setattr(_CandidateLease, "diff", diff)

    async def workflow(ctx, _args):
        return await ctx.candidate_agent(
            "write source", label="candidate", tools=builtin_tools("file_write", headless=True), timeout=0.03,
        )

    task = asyncio.create_task(_client(repository).workflow(workflow, trace=False, timeout=None))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        await asyncio.wait_for(cancellation_started.wait(), timeout=2)
        if past_cleanup_deadline:
            async def wait_for_revocation():
                while not sessions[0].env.revoked:
                    await asyncio.sleep(0.005)
            await asyncio.wait_for(wait_for_revocation(), timeout=2)
        await asyncio.sleep(0.05)
        assert not capture_started.is_set()
        assert not task.done()
        assert Path(sessions[0].env.workspace, "source.txt").read_text() == "initial\n"
        release.set()
        result = await asyncio.wait_for(task, timeout=5)
        assert result.ok, result.reason
        candidate = result.output
        assert candidate.output is None
        assert "+late native candidate write" in candidate.diff
        assert candidate.lifecycle_errors == ()
        assert not Path(sessions[0].env.workspace).exists()
        assert (repository / "source.txt").read_text() == "initial\n"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

@pytest.mark.parametrize("kind", ["ordinary", "candidate_agent", "candidate_workflow"])
@pytest.mark.parametrize("background", [False, True])
async def test_public_workflow_waits_through_candidate_capture(repository, monkeypatch, kind, background):
    started, release_model = asyncio.Event(), asyncio.Event()
    capturing, release_capture = asyncio.Event(), asyncio.Event()
    holder, sessions = {}, []
    original_build, original_diff = runtime.build_session, _CandidateLease.diff

    async def complete(llm, messages, **kwargs):
        if not any(message.get("role") == "tool" for message in messages):
            return LLMResponse(tool_calls=[{
                "id": "write", "type": "function", "function": {
                    "name": "file_write", "arguments": json.dumps({
                        "path": "source.txt", "mode": "create", "content": "generated candidate work\n",
                    }),
                },
            }], finish_reason="tool_calls", usage=Usage(10, 5))
        started.set()
        await release_model.wait()
        return LLMResponse(content="done", usage=Usage(10, 5))

    def build(**kwargs):
        session = original_build(**kwargs)
        sessions.append(session)
        return session

    async def diff(lease, *args, **kwargs):
        capturing.set()
        await release_capture.wait()
        return await original_diff(lease, *args, **kwargs)

    monkeypatch.setattr(LLMClient, "complete", complete)
    monkeypatch.setattr(runtime, "build_session", build)
    monkeypatch.setattr(_CandidateLease, "diff", diff)

    async def child(ctx, _args):
        return await ctx.agent("write", tools=evidence_tools("file_write"))

    async def workflow(ctx, _args):
        holder["context"] = ctx
        if kind == "ordinary":
            operation = child(ctx, {})
        elif kind == "candidate_agent":
            operation = ctx.candidate_agent("write", label="candidate", tools=evidence_tools("file_write"))
        else:
            operation = ctx.candidate_workflow(child, {}, label="candidate")

        async def collect():
            holder["value"] = await operation

        if background:
            holder["background"] = asyncio.create_task(collect())
            await started.wait()
        else:
            await collect()
        return "workflow returned"

    task = asyncio.create_task(_client(repository).workflow(
        workflow, budget=1000, timeout=None, trace=False, cleanup_timeout=5,
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        await asyncio.sleep(0.02)
        assert Path(sessions[0].env.workspace, "source.txt").read_text() == "generated candidate work\n"
        assert not task.done(), "runtime finished while the model was still executing"
        release_model.set()
        if kind != "ordinary":
            await asyncio.wait_for(capturing.wait(), timeout=5)
            await asyncio.sleep(0.02)
            assert not task.done(), "runtime finished before the candidate diff was captured"
        release_capture.set()
        result = await asyncio.wait_for(task, timeout=5)
        if background:
            await holder["background"]
        assert result.ok, result.reason
        assert result.output == "workflow returned"
        assert result.tokens == 30
        assert len(holder["context"].sessions) == 1
        assert holder["context"].pending_cleanup_tasks == ()
        if kind == "ordinary":
            assert holder["value"] == "done"
        else:
            assert holder["value"].output == "done"
            assert "+generated candidate work" in holder["value"].diff
            assert holder["value"].lifecycle_errors == ()
            assert (repository / "source.txt").read_text() == "initial\n"
            assert not Path(sessions[0].env.workspace).exists()
    finally:
        release_model.set()
        release_capture.set()
        await asyncio.gather(task, *([holder["background"]] if "background" in holder else []), return_exceptions=True)
        for session in sessions:
            await session.aclose()
            if session.env.workspace != str(repository):
                await session.env.cleanup()
                if Path(session.env.workspace).exists():
                    subprocess.run(["git", "worktree", "remove", "--force", "--", session.env.workspace],
                                   cwd=repository, check=True, capture_output=True)

@pytest.mark.parametrize("kind", ["ordinary", "draft_findings", "candidate_agent", "candidate_workflow"])
async def test_nested_candidate_preserves_outer_call_ownership(repository, kind):
    parent = WorkflowContext(
        FakeFactory([FakeSession(reply="done")]), budget_total=100,
        candidate_workspace=EnvCandidateWorkspace(LocalEnvironment(str(repository))),
    )

    async def inner(child, _args):
        return await child.agent("inner")

    async def outer(_child, _args):
        if kind == "ordinary":
            output = await parent.agent("read source for candidate preparation", budget=10)
        elif kind == "draft_findings":
            assert await parent.draft_findings("read source for candidate preparation", budget=10) is None
            output = "done"
        elif kind == "candidate_agent":
            output = (await parent.candidate_agent("inner", label="inner", budget=10)).output
        else:
            output = (await parent.candidate_workflow(inner, {}, label="inner", budget=10)).output
        assert asyncio.current_task() in parent._active_call_tasks
        return output

    candidate = await parent.candidate_workflow(outer, {}, label="outer", budget=50)
    assert candidate.output == "done"
    assert parent.agent_failures == ()
    assert parent._active_call_tasks == set()

@pytest.mark.parametrize("ending", ["timeout", "cancel", "cancel-twice"])
async def test_candidate_keeps_lease_for_cleanup_spawned_by_cancelled_turn(repository, ending):
    started = asyncio.Event()
    cancel_seen = asyncio.Event()
    release_turn = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    later_entered = asyncio.Event()

    class LateCleanupSession(FakeSession):
        pending_cleanup_tasks = ()

        async def write(self):
            cleanup_started.set()
            await release_cleanup.wait()
            await self.environment.write_file("source.txt", "late candidate write\n")

        async def run_loop(self, cancel_event=None):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                await release_turn.wait()
                self.pending_cleanup_tasks = (asyncio.create_task(self.write()),)
                raise

    slow = LateCleanupSession(tokens=7)

    async def enter_later():
        later_entered.set()

    class Factory(FakeFactory):
        def build_workflow_session(self, **kwargs):
            session = super().build_workflow_session(**kwargs)
            session.environment = kwargs["env"]
            return session

    context = WorkflowContext(
        Factory([slow, FakeSession(reply="later", on_enter=enter_later)]),
        max_concurrency=1, budget_total=100,
        candidate_workspace=EnvCandidateWorkspace(LocalEnvironment(str(repository))),
    )
    task = asyncio.create_task(context.candidate_agent(
        "candidate", label="candidate", budget=90, timeout=0.1 if ending == "timeout" else None,
    ))
    later_task = None
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        if ending != "timeout":
            task.cancel()
        await asyncio.wait_for(cancel_seen.wait(), timeout=5)
        release_turn.set()
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        if ending == "cancel-twice":
            task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert context.tokens_remaining() == 10
        later_task = asyncio.create_task(context.agent("ordinary", budget=1))
        await asyncio.sleep(0.02)
        assert not later_entered.is_set()
        assert (repository / "source.txt").read_text() == "initial\n"
    finally:
        release_turn.set()
        release_cleanup.set()
        if ending != "timeout":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        else:
            candidate = await asyncio.wait_for(task, timeout=5)
            assert candidate.output is None
            assert "+late candidate write" in candidate.diff
        if later_task is not None:
            assert await asyncio.wait_for(later_task, timeout=5) == "later"
        await asyncio.gather(*slow.pending_cleanup_tasks, return_exceptions=True)

    assert context.tokens_remaining() == 93
    assert context.pending_cleanup_tasks == ()
    assert (repository / "source.txt").read_text() == "initial\n"
