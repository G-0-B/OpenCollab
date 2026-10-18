"""Minimal candidate leases retain real files when exporting evidence fails."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.application.workflow_candidates import CandidateCaptureError, CandidateWorkspaceTrackingError
from opencollab.tools import builtin_tools


class _MinimalLease:
    """Expose exactly the candidate lease port over a real local worktree."""

    def __init__(self, delegate, *, capture_error):
        self.environment = delegate.environment
        self._delegate = delegate
        self.capture_error = capture_error
        self.captured = []
        self.cleanups = 0

    async def diff(self, exclude_paths=()):
        if self.capture_error:
            raise OSError("temporary candidate export failure")
        patch = await self._delegate.diff(exclude_paths)
        self.captured.append(patch)
        return patch

    async def cleanup(self):
        self.cleanups += 1
        await self._delegate.cleanup()

class _MinimalCandidates(EnvCandidateWorkspace):
    def __init__(self, environment, failure):
        super().__init__(environment)
        self.failure = failure
        self.leases = []
        self.source_reads = 0

    async def acquire(self, label):
        lease = _MinimalLease(await super().acquire(label), capture_error=self.failure == "capture")
        self.leases.append(lease)
        return lease

    async def source_diff(self, exclude_paths=()):
        self.source_reads += 1
        if self.failure == "source_read" and self.source_reads == 2:
            raise OSError("temporary source export failure")
        return await super().source_diff(exclude_paths)

class _WritingModel:
    def __init__(self, content, *, mutate=None):
        self.content = content
        self.mutate = mutate
        self.calls = 0

    async def complete(self, messages, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(tool_calls=[{
                "id": "generate", "type": "function", "function": {
                    "name": "file_write", "arguments": json.dumps({
                        "path": "generated.txt", "mode": "create", "content": self.content,
                    }),
                },
            }], usage=Usage(4, 2), finish_reason="tool_calls")
        if self.calls == 2:
            assert any(message.get("tool_call_id") == "generate" for message in messages)
            if self.mutate is not None:
                self.mutate()
        return LLMResponse(content="done", usage=Usage(4, 2), finish_reason="stop")

@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "source.txt").write_text("original source\n")
    for arguments in (
        ("init", "-q"), ("add", "source.txt"),
        ("-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-qm",
         "test: \u521b\u5efa\u5019\u9009\u7559\u5b58\u7528\u4f8b"),
    ):
        subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)
    return root

@pytest.mark.parametrize("entry", ["agent", "workflow"])
@pytest.mark.parametrize("failure", ["capture", "source_read", "source_changed", "none"])
async def test_public_minimal_candidate_lease_retains_failed_exports(repository, entry, failure):
    environment = LocalEnvironment(str(repository))
    backend = _MinimalCandidates(environment, failure)
    content = f"unique generated content for {repository.parent.name}\n"
    mutate = (lambda: (repository / "source.txt").write_text("concurrent user edit\n")) if (
        failure == "source_changed"
    ) else None
    model = _WritingModel(content, mutate=mutate)

    async def edit(child, _inputs):
        return await child.agent("generate a file", tools=builtin_tools("file_write"))

    async def flow(ctx, inputs):
        if entry == "agent":
            return await ctx.candidate_agent("generate a file", label="candidate", tools=builtin_tools("file_write"))
        return await ctx.candidate_workflow(edit, inputs, label="candidate")

    try:
        result = await OpenCollab(repository, environment=environment, model="controlled").workflow(
            flow, llm=model, candidate_workspace=backend, trace=False,
            limit_mode="explicit", budget=10000, max_steps=4, timeout=20,
        )
        lease, = backend.leases
        candidate_path = Path(lease.environment.workspace)
        assert not hasattr(lease, "candidate_workspace")
        assert model.calls == 2
        assert not (repository / "generated.txt").exists()
        assert (repository / "source.txt").read_text() == (
            "concurrent user edit\n" if failure == "source_changed" else "original source\n"
        )
        if failure == "none":
            assert result.ok, result.reason
            assert result.output.output == "done"
            assert content.strip() in result.output.diff
            assert lease.cleanups == 1
            assert not candidate_path.exists()
        else:
            expected = CandidateCaptureError if failure == "capture" else CandidateWorkspaceTrackingError
            assert isinstance(result.error, expected), result.reason
            assert "worktree preserved" in result.reason
            assert str(candidate_path) in result.reason
            if failure in {"capture", "source_read"}:
                assert isinstance(result.error.__cause__, OSError)
                assert "export failure" in str(result.error.__cause__)
            assert lease.cleanups == 0
            assert (candidate_path / "generated.txt").read_text() == content
            if failure != "capture":
                assert lease.captured and content.strip() in lease.captured[0]
            lease.capture_error = False
            assert content.strip() in await lease.diff()
            await lease.cleanup()
            assert lease.cleanups == 1
            assert not candidate_path.exists()
    finally:
        for lease in backend.leases:
            await lease.cleanup()
        await environment.cleanup()

@pytest.mark.parametrize("capture_error", [False, True])
async def test_public_minimal_lease_survives_child_cleanup_failure(repository, monkeypatch, capture_error):
    environment = LocalEnvironment(str(repository))
    backend = _MinimalCandidates(environment, "capture" if capture_error else "none")
    model = _WritingModel("candidate survives child cleanup\n")
    contexts, isolated, retained_isolation = [], [], []

    async def unavailable():
        raise OSError("isolated environment cleanup unavailable")

    async def edit(child, _inputs):
        await child.agent("generate a file", tools=builtin_tools("file_write"))
        await child.agent("inspect in isolation", isolation=True)
        isolated_environment = child.sessions[-1].env
        isolated.append((isolated_environment, isolated_environment.cleanup))
        monkeypatch.setattr(isolated_environment, "cleanup", unavailable)
        return "done"

    async def flow(ctx, inputs):
        contexts.append(ctx)
        try:
            return await ctx.candidate_workflow(edit, inputs, label="candidate")
        finally:
            # Retry the backend cleanup before the public run finalizes its evidence.
            for isolated_environment, cleanup in isolated:
                retained_isolation.append(Path(isolated_environment.workspace).is_dir())
                monkeypatch.setattr(isolated_environment, "cleanup", cleanup)
            await ctx.release_isolated_workspaces()

    try:
        result = await OpenCollab(repository, environment=environment, model="controlled").workflow(
            flow, llm=model, candidate_workspace=backend, trace=False,
            limit_mode="explicit", budget=10000, max_steps=4, timeout=20,
        )
        lease, = backend.leases
        candidate_path = Path(lease.environment.workspace)
        assert len(isolated) == 1
        if capture_error:
            assert isinstance(result.error, CandidateCaptureError), result.reason
            details = getattr(result.error, "__notes__", ())
        else:
            assert result.output.output == "done", result.reason
            details = result.output.lifecycle_errors
        assert any("isolated environment cleanup unavailable" in detail for detail in details)
        assert any("candidate worktree retained" in detail for detail in details)
        assert any(str(candidate_path) in detail for detail in details)
        assert lease.cleanups == 0
        assert (candidate_path / "generated.txt").read_text() == model.content
        assert retained_isolation == [True]
        assert not Path(isolated[0][0].workspace).exists()
        await lease.cleanup()
        assert not candidate_path.exists()
    finally:
        for isolated_environment, cleanup in isolated:
            monkeypatch.setattr(isolated_environment, "cleanup", cleanup)
        for context in contexts:
            await context.release_isolated_workspaces()
        for lease in backend.leases:
            await lease.cleanup()
        await environment.cleanup()
