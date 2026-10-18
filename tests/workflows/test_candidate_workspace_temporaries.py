from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.application.workflow import WorkflowContext
from opencollab.bootstrap import _workflow_runtime_session as runtime
from tests.workflows.test_workflow_candidate_workspace import _EditingSession, _git, _repository


async def test_parallel_candidates_ignore_internal_capture_files(monkeypatch, tmp_path):
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    factory = runtime.WorkflowSessionFactory(
        model="offline", provider="openai", api_key=None, base_url=None, workspace=str(repo), env=base,
    )
    context = WorkflowContext(factory, budget_total=None, max_concurrency=2, candidate_workspace=workspace)
    both_ready = asyncio.Event()
    release_capture = asyncio.Event()
    arrived = 0
    writes = 0
    original_write = LocalEnvironment.write_temp_file

    class Model(_EditingSession):
        async def run_loop(self, cancel_event=None):
            nonlocal arrived
            result = await super().run_loop(cancel_event)
            arrived += 1
            if arrived == 2:
                both_ready.set()
            await both_ready.wait()
            return result

    async def scheduled_write(owner, content, *, prefix, suffix=".tmp"):
        nonlocal writes
        path = await original_write(owner, content, prefix=prefix, suffix=suffix)
        if prefix == ".candidate-capture-index-":
            writes += 1
            if writes == 2:
                await release_capture.wait()
        return path

    monkeypatch.setattr(runtime, "build_session", lambda **kwargs: Model(kwargs["env"]))
    monkeypatch.setattr(LocalEnvironment, "write_temp_file", scheduled_write)
    tasks = [asyncio.create_task(context.candidate_agent("edit", label=label)) for label in ("first", "second")]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=10)
        assert done, "a candidate must settle while the other capture is suspended"
    finally:
        # Let the surviving model settle if its peer failed before the barrier.
        both_ready.set()
        release_capture.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        for entry in _git(repo, "worktree", "list", "--porcelain").splitlines():
            if entry.startswith("worktree ") and Path(entry[9:]).resolve() != repo.resolve():
                _git(repo, "worktree", "remove", "--force", entry[9:])
    try:
        assert all(not isinstance(item, BaseException) for item in outcomes), outcomes
        assert all("+value = 2" in item.diff for item in outcomes)
        assert (repo / "source.py").read_text() == "value = 1\n"
        assert await workspace.source_diff() == ""
        (repo / ".candidate-capture-index-user.tmp").write_text("user note\n")
        assert ".candidate-capture-index-user.tmp" in await workspace.source_diff()
    finally:
        await base.cleanup()

@pytest.mark.parametrize("linked_source", [False, True])
@pytest.mark.parametrize("completion", ["success", "cancel", "write_error"])
@pytest.mark.parametrize("prefix", [
    ".candidate-source-", ".candidate-capture-index-", ".candidate-capture-paths-",
    ".candidate-adopt-", ".candidate-current-", ".candidate-target-",
])
async def test_internal_candidate_materials_do_not_enter_source_snapshot(
    monkeypatch, tmp_path, prefix, linked_source, completion,
):
    main = _repository(tmp_path)
    repo = main
    if linked_source:
        repo = tmp_path / "linked"
        _git(main, "worktree", "add", "--detach", str(repo))
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    (repo / "source.py").write_text("value = 9\n")
    (repo / "draft.txt").write_text("user draft\n")
    user_file = repo / f"{prefix}user.tmp"
    user_file.write_text("user note\n")
    lease = None
    created = asyncio.Event()
    resume = asyncio.Event()
    original_write = LocalEnvironment.write_temp_file
    temporary_paths = []

    async def suspended_write(owner, content, *, prefix: str, suffix=".tmp"):
        path = await original_write(owner, content, prefix=prefix, suffix=suffix)
        temporary_paths.append(Path(path))
        if prefix == selected_prefix:
            created.set()
            await resume.wait()
            if completion == "write_error":
                raise OSError("temporary write interrupted")
        return path

    selected_prefix = prefix
    if prefix != ".candidate-source-":
        lease = await workspace.acquire("prepared")
        await lease.environment.write_file("source.py", "value = 10\n")
        if prefix == ".candidate-capture-paths-":
            _git(Path(lease.candidate_workspace), "reset", "--mixed", "HEAD")
            await lease.environment.write_file(".gitignore", "*.txt\n")
        candidate_patch = await lease.diff()
    baseline = await workspace.source_diff()
    monkeypatch.setattr(LocalEnvironment, "write_temp_file", suspended_write)
    if prefix == ".candidate-source-":
        operation = asyncio.create_task(workspace.acquire("observed"))
    elif prefix in (".candidate-capture-index-", ".candidate-capture-paths-"):
        operation = asyncio.create_task(lease.diff())
    elif prefix == ".candidate-adopt-":
        operation = asyncio.create_task(workspace.adopt(candidate_patch))
    else:
        operation = asyncio.create_task(workspace.restore_source(""))
    try:
        await asyncio.wait_for(created.wait(), 10)
        observed = await workspace.source_diff()
        assert observed == baseline
        assert user_file.name in observed
        if completion == "cancel":
            operation.cancel()
    finally:
        resume.set()
        if completion == "success":
            result = await operation
            if prefix == ".candidate-source-":
                await result.cleanup()
        elif completion == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(OSError, match="temporary write interrupted"):
                await operation
        if completion != "success":
            assert await workspace.source_diff() == baseline
        if lease is not None:
            await lease.cleanup()
        await base.cleanup()
        if linked_source:
            _git(main, "worktree", "remove", "--force", str(repo))
    assert all(not path.exists() for path in temporary_paths)
    assert all(not path.parent.exists() for path in temporary_paths)

@pytest.mark.parametrize("operation", ["adopt", "restore"])
@pytest.mark.parametrize("invalid_patch", [False, True])
async def test_patch_cleanup_error_preserves_original_outcome(monkeypatch, tmp_path, operation, invalid_patch):
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    lease = await workspace.acquire("prepare-patch")
    await lease.environment.write_file("source.py", "value = 2\n")
    patch = await lease.diff()
    await lease.cleanup()
    if operation == "restore":
        (repo / "source.py").write_text("value = 3\n")
    original_cleanup = LocalEnvironment.cleanup
    cleanup_errors = []

    async def cleanup_then_fail(owner):
        await original_cleanup(owner)
        if owner is not base:
            cleanup_errors.append(owner.workspace)
            raise OSError("temporary cleanup failed")

    monkeypatch.setattr(LocalEnvironment, "cleanup", cleanup_then_fail)
    try:
        method = workspace.adopt if operation == "adopt" else workspace.restore_source
        if invalid_patch:
            with pytest.raises(RuntimeError, match="candidate (path inspection|source restoration) failed"):
                await method("invalid patch\n")
            expected = "value = 1\n" if operation == "adopt" else "value = 3\n"
        else:
            await method(patch)
            expected = "value = 2\n"
        assert (repo / "source.py").read_text() == expected
        assert cleanup_errors
        assert all(not Path(directory).exists() for directory in cleanup_errors)
    finally:
        await base.cleanup()
