"""Complete candidate increments reach fresh isolated teammates and source files."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencollab.adapters import candidate_workspace
from opencollab.adapters.env import DockerWorkspaceEnvironment
from opencollab.adapters.trace import Tracer
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.application._scheduler_constants import WORKTREE_DIFF_MAX_CHARS
from opencollab.application.scheduler import Scheduler
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase, SessionState
from tests.support.worktree_delivery_support import CONTAINER_ID
from tests.support.worktree_delivery_support import worktree_transport as worktree_transport


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout

def _project(path, git):
    path.mkdir()
    (path / "answer.py").write_text("def answer():\n    return 1\n")
    (path / "second.py").write_text("value = 0\n")
    (path / "notes.txt").write_text("original\n")
    if git:
        _git(path, "init", "-q")
        _git(path, "config", "user.name", "Tests")
        _git(path, "config", "user.email", "tests@example.invalid")
        _git(path, "config", "commit.gpgsign", "false")
        _git(path, "add", ".")
        _git(path, "commit", "-qm", "\u521d\u59cb\u5316")
    return path

def _adoption_scheduler(pool, coder):
    scheduler = Scheduler(session_factory=object(), worktree_pool=pool, event_sink=object())
    for aid, parent, role in ((0, None, "analyst"), (7, 0, "coder")):
        state = SessionState(messages=[])
        state.set_phase(SessionPhase.DONE)
        tools = [SimpleNamespace(name="adopt_agent")] if aid == 0 else []
        scheduler.table.add(SessionControlBlock(aid, parent, SimpleNamespace(name=role, tools=tools), state))
    scheduler._sessions[7] = SimpleNamespace(env=coder)
    return scheduler

@pytest.mark.parametrize("git", [False, True])
async def test_followup_captures_preserve_bytes_user_edits_and_empty_reverts(tmp_path, git):
    source = _project(tmp_path / "project", git)
    original = "def answer():\n    return 1\n" + "# spacing\n" * 25 + "# user marker\n"
    (source / "answer.py").write_text(original)
    (source / "binary.dat").write_bytes(b"\0before\xff")
    (source / "legacy.txt").write_bytes(b"\xc4\xe3\xba\xc3\n")
    (source / "run.sh").write_text("true\n")
    (source / "alias").symlink_to("answer.py")
    if git:
        _git(source, "add", ".")
        _git(source, "commit", "-qm", "\u51c6\u5907\u5b57\u8282\u9a8c\u8bc1")
        head = _git(source, "rev-parse", "HEAD")
        (source / "notes.txt").write_text("staged\n")
        _git(source, "add", "notes.txt")
        (source / "notes.txt").write_text("staged\nunstaged\n")
        index = (source / ".git/index").read_bytes()
    (source / "untracked.txt").write_text("user contents\n")
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    scheduler = _adoption_scheduler(pool, coder)

    async def complete():
        state = scheduler.table.get(7).state
        state.set_step_count(state.step_count + 1)
        await scheduler._append_worktree_diff(coder, "done", aid=7, role="coder")
        await scheduler.adopt_agent(0, 7)

    try:
        # An empty initial completion is still followed by real, adoptable work.
        await complete()
        await coder.write_file("answer.py", original.replace("return 1", "return 2"))
        await complete()
        assert "return 2" in (source / "answer.py").read_text()
        notes = (source / "notes.txt").read_text() + "later user edit\n"
        (source / "notes.txt").write_text(notes)
        (source / "answer.py").write_text((source / "answer.py").read_text().replace("user marker", "user changed"))
        await coder.write_file("answer.py", original.replace("return 1", "return 3"))
        Path(coder.workspace, "binary.dat").write_bytes(b"\0after\xff")
        Path(coder.workspace, "legacy.txt").write_bytes(b"\xc4\xe3\xba\xc3\xca\xc0\xbd\xe7\n")
        Path(coder.workspace, "run.sh").chmod(0o755)
        Path(coder.workspace, "alias").unlink()
        Path(coder.workspace, "alias").symlink_to("notes.txt")
        Path(coder.workspace, "second.py").rename(Path(coder.workspace, "renamed.py"))
        await coder.write_file("added.txt", "candidate addition\n")
        state = scheduler.table.get(7).state
        state.set_step_count(state.step_count + 1)
        await scheduler._append_worktree_diff(coder, "done", aid=7, role="coder")
        await coder.write_file("answer.py", original.replace("return 1", "return 4"))
        await scheduler.adopt_agent(0, 7)
        delivered = original.replace("return 1", "return 3").replace("user marker", "user changed")
        assert (source / "answer.py").read_text() == delivered
        assert "return 4" in await coder.read_file("answer.py")
        assert (source / "binary.dat").read_bytes() == b"\0after\xff"
        assert (source / "legacy.txt").read_bytes() == b"\xc4\xe3\xba\xc3\xca\xc0\xbd\xe7\n"
        assert (source / "run.sh").stat().st_mode & 0o111
        assert os.readlink(source / "alias") == "notes.txt"
        assert not (source / "second.py").exists() and (source / "renamed.py").read_text() == "value = 0\n"
        assert "already adopted" in await scheduler.adopt_agent(0, 7)
        assert (source / "notes.txt").read_text() == notes
        assert (source / "untracked.txt").read_text() == "user contents\n"

        # Returning the saved candidate to its original contents yields an
        # empty cumulative patch, but delivery must undo the previous adoption.
        await coder.write_file("answer.py", original)
        Path(coder.workspace, "binary.dat").write_bytes(b"\0before\xff")
        Path(coder.workspace, "legacy.txt").write_bytes(b"\xc4\xe3\xba\xc3\n")
        Path(coder.workspace, "run.sh").chmod(0o644)
        Path(coder.workspace, "alias").unlink()
        Path(coder.workspace, "alias").symlink_to("answer.py")
        Path(coder.workspace, "renamed.py").rename(Path(coder.workspace, "second.py"))
        Path(coder.workspace, "added.txt").unlink()
        await complete()
        assert scheduler._completed_coder_diffs[7].patch == ""
        assert (source / "answer.py").read_text() == original.replace("user marker", "user changed")
        assert (source / "binary.dat").read_bytes() == b"\0before\xff"
        assert (source / "legacy.txt").read_bytes() == b"\xc4\xe3\xba\xc3\n"
        assert not (source / "run.sh").stat().st_mode & 0o111
        assert os.readlink(source / "alias") == "answer.py"
        assert not (source / "renamed.py").exists() and not (source / "added.txt").exists()
        assert (source / "second.py").read_text() == "value = 0\n"
        assert (source / "notes.txt").read_text() == notes
        if git:
            assert (source / ".git/index").read_bytes() == index
            assert _git(source, "rev-parse", "HEAD") == head
    finally:
        await coder.get_diff()
        await pool.release()

@pytest.mark.parametrize("git", [False, True])
async def test_followup_conflict_retains_increment_and_retries_atomically(tmp_path, git):
    source = _project(tmp_path / "project", git)
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    scheduler = _adoption_scheduler(pool, coder)
    saved = []

    apply = pool.adopt

    async def record_increment(patch):
        saved.append(patch)
        await apply(patch)

    pool.adopt = record_increment
    try:
        await coder.write_file("answer.py", "def answer():\n    return 2\n")
        await scheduler._append_worktree_diff(coder, "done", aid=7, role="coder")
        await scheduler.adopt_agent(0, 7)
        saved.clear()
        first = scheduler._adopted_coders[7]
        await coder.write_file("answer.py", "def answer():\n    return 3\n")
        await coder.write_file("second.py", "value = 3\n")
        scheduler.table.get(7).state.set_step_count(1)
        await scheduler._append_worktree_diff(coder, "done", aid=7, role="coder")
        (source / "answer.py").write_text("def answer():\n    return 99\n")
        before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
        with pytest.raises(RuntimeError, match="full candidate retained at"):
            await scheduler.adopt_agent(0, 7)
        assert before == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
        assert scheduler._adopted_coders[7] == first
        assert "-    return 2" in saved[0] and "+    return 3" in saved[0]
        assert "-    return 1" not in saved[0]
        with pytest.raises(OSError, match="retained"):
            await pool.release()
        assert os.path.isdir(coder.workspace) and coder.recovery_location == coder.workspace
        assert "return 3" in await coder.read_file("answer.py")
        (source / "answer.py").write_text("def answer():\n    return 2\n")
        await scheduler.adopt_agent(0, 7)
        assert (source / "answer.py").read_text() == "def answer():\n    return 3\n"
        assert (source / "second.py").read_text() == "value = 3\n"
        assert coder.recovery_location is None
    finally:
        await coder.get_diff()
        await pool.release()

@pytest.mark.parametrize("git", [False, True])
async def test_parallel_increments_conflict_and_next_teammates(tmp_path, git):
    source = _project(tmp_path / "project", git)
    if git:
        (source / "notes.txt").write_text("staged\n")
        _git(source, "add", "notes.txt")
        (source / "notes.txt").write_text("staged\nunstaged\n")
        (source / "untracked.txt").write_text("user contents\n")
        index = (source / ".git/index").read_bytes()
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    a, b, conflict = await asyncio.gather(*(pool.acquire("coder") for _ in range(3)))
    try:
        assert len({a.workspace, b.workspace, conflict.workspace, str(source)}) == 4
        await a.write_file("answer.py", "def answer():\n    return 42\n")
        await b.write_file("second.py", "value = 2\n")
        await conflict.write_file("answer.py", "def answer():\n    return 99\n")
        await conflict.write_file("second.py", "value = 99\n")
        assert await b.read_file("answer.py") == "def answer():\n    return 1\n"
        if git:
            assert await a.read_file("notes.txt") == "staged\nunstaged\n"
            assert await a.read_file("untracked.txt") == "user contents\n"
            commit = shlex.join(("git", "commit", "-qm", "\u5b9e\u73b0"))
            committed = await a.exec_cmd(f"git add answer.py && {commit}")
            assert committed.returncode == 0, committed.stderr
        (source / "notes.txt").write_text("staged\nunstaged\nlater user edit\n")
        patch_a, patch_b = await a.get_diff(), await b.get_diff()
        assert "notes.txt" not in patch_a and "untracked.txt" not in patch_a
        await pool.adopt(patch_a)
        await pool.adopt(patch_b)
        before = {name: (source / name).read_bytes() for name in ("answer.py", "second.py", "notes.txt")}
        with pytest.raises(RuntimeError, match="candidate adoption failed"):
            await pool.adopt(await conflict.get_diff())
        assert before == {name: (source / name).read_bytes() for name in before}
        if git:
            assert (source / ".git/index").read_bytes() == index
        tester = await pool.acquire("tester")
        command = shlex.join((
            sys.executable, "-B", "-c", "from answer import answer; from second import value; "
            "assert (answer(), value) == (42, 2); print('DELIVERY_PASS')",
        ))
        result = await tester.exec_cmd(command)
        assert result.returncode == 0 and "DELIVERY_PASS" in result.stdout
        assert tester.workspace not in {a.workspace, b.workspace, conflict.workspace, str(source)}
        await tester.get_diff()
        next_coder = await pool.acquire("coder")
        assert await next_coder.read_file("answer.py") == "def answer():\n    return 42\n"
        await next_coder.write_file("answer.py", "def answer():\n    return 43\n")
        revision = await next_coder.get_diff()
        assert "-    return 42" in revision and "second.py" not in revision
        await pool.adopt(revision)
        result = subprocess.run(
            [sys.executable, "-B", "-c", "from answer import answer; assert answer() == 43"], cwd=source,
        )
        assert result.returncode == 0
        assert (source / "notes.txt").read_text().endswith("later user edit\n")
    finally:
        await pool.release()
    assert all(not os.path.exists(env.workspace) for env in (a, b, conflict, tester, next_coder))

async def test_git_subdirectory_and_local_port_keep_source_index(tmp_path):
    root = _project(tmp_path / "repository", True)
    subdir = root / "nested"
    subdir.mkdir()
    (subdir / "module.py").write_text("value = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "\u6dfb\u52a0\u5b50\u76ee\u5f55")

    class LocalPort:
        local_filesystem = True
        workspace = str(subdir)

    pool = WorktreePool(str(tmp_path), use_worktrees=True, base_environment=LocalPort(), enable_adoption=True)
    coder = await pool.acquire("coder")
    await coder.write_file("module.py", "value = 42\n")
    patch = await coder.get_diff()
    assert "nested/module.py" in patch
    await pool.adopt(patch)
    tester = await pool.acquire("tester")
    assert await tester.read_file("module.py") == "value = 42\n"
    await tester.get_diff()
    await pool.release()
    assert (subdir / "module.py").read_text() == "value = 42\n"
    assert (root / "answer.py").read_text() == "def answer():\n    return 1\n"

@pytest.mark.parametrize("key,value", [
    ("filter.probe.clean", "touch FILTER_EXECUTED; cat"),
    ("diff.probe.textconv", "touch DIFF_EXECUTED; cat"),
    ("core.fsmonitor", "touch MONITOR_EXECUTED"),
])
async def test_unsafe_source_config_is_rejected_before_capture(tmp_path, key, value):
    source = _project(tmp_path / "project", True)
    _git(source, "config", key, value)
    (source / ".gitattributes").write_text("* filter=probe diff=probe\n")
    before = _git(source, "worktree", "list", "--porcelain")
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    with pytest.raises(RuntimeError, match="unsafe repository Git configuration"):
        await pool.acquire("coder")
    assert _git(source, "worktree", "list", "--porcelain") == before
    assert not any(source.glob("*EXECUTED"))
    await pool.release()

async def test_ignored_candidate_is_retained_until_complete_export(tmp_path):
    source = _project(tmp_path / "project", True)
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    await coder.write_file(".gitignore", "hidden.py\n")
    await coder.write_file("hidden.py", "unique implementation\n")
    with pytest.raises(RuntimeError, match="ignored untracked files"):
        await coder.get_diff()
    with pytest.raises(OSError, match="retained"):
        await pool.release()
    assert os.path.exists(coder.workspace)
    await coder.write_file(".gitignore", "")
    assert "unique implementation" in await coder.get_diff()
    await pool.release()
    assert not os.path.exists(coder.workspace)

async def test_source_owned_outputs_exclude_only_untracked_files(tmp_path):
    source = _project(tmp_path / "project", True)
    outputs = source / "records"
    outputs.mkdir()
    (outputs / "tracked.txt").write_text("initial\n")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "\u6dfb\u52a0\u8ddf\u8e2a\u8bb0\u5f55")
    (outputs / "tracked.txt").write_text("user edit\n")
    (outputs / "run.json").write_text("run output\n")
    index = (source / ".git/index").read_bytes()
    pool = WorktreePool(str(source), use_worktrees=True, owned_save_dir=str(outputs), enable_adoption=True)
    coder = await pool.acquire("coder")
    assert await coder.read_file("records/tracked.txt") == "user edit\n"
    assert not os.path.exists(os.path.join(coder.workspace, "records/run.json"))
    assert await coder.get_diff() == ""
    assert (source / ".git/index").read_bytes() == index
    await pool.release()

async def test_custom_team_without_adoption_keeps_dirty_source_refusal(tmp_path):
    source = _project(tmp_path / "project", True)
    (source / "answer.py").write_text("user implementation\n")
    pool = WorktreePool(str(source), use_worktrees=True)
    with pytest.raises(RuntimeError, match="source workspace has uncommitted changes"):
        await pool.acquire("coder")
    await pool.release()

async def test_container_delivery_keeps_isolation_and_executes_fresh_tester(tmp_path, worktree_transport):
    source = _project(tmp_path / "project", True)
    attached = DockerWorkspaceEnvironment(container_id=CONTAINER_ID, repo_root=str(source))
    pool = WorktreePool(str(tmp_path), use_worktrees=True, base_environment=attached, enable_adoption=True)
    coder = await pool.acquire("coder")
    assert coder.process_isolated is True and coder.local_filesystem is False
    await coder.write_file("answer.py", "def answer():\n    return 42\n")
    await pool.adopt(await coder.get_diff())
    tester = await pool.acquire("tester")
    assert len({coder.workspace, tester.workspace, str(source)}) == 3
    assert tester.process_isolated is True
    result = await tester.exec_cmd(shlex.join((
        sys.executable, "-B", "-c",
        "from answer import answer; assert answer() == 42; print('CONTAINER_DELIVERY_PASS')",
    )))
    assert result.returncode == 0 and "CONTAINER_DELIVERY_PASS" in result.stdout
    await tester.get_diff()
    await pool.release()
    assert attached.revoked is False
    assert (source / "answer.py").read_text() == "def answer():\n    return 42\n"
    assert worktree_transport and all(command[0] == "docker" for command in worktree_transport)
    await attached.cleanup()

async def test_failed_setup_retains_cleanup_retry_ownership(tmp_path, monkeypatch):
    source = _project(tmp_path / "project", True)
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    original_environment = candidate_workspace.EnvCandidateWorkspace._candidate_environment
    original_cleanup = candidate_workspace._CandidateLease.cleanup

    async def setup_failure(self, _path):
        raise RuntimeError("setup failure")

    async def cleanup_failure(self):
        raise OSError("cleanup failure")

    monkeypatch.setattr(candidate_workspace.EnvCandidateWorkspace, "_candidate_environment", setup_failure)
    monkeypatch.setattr(candidate_workspace._CandidateLease, "cleanup", cleanup_failure)
    with pytest.raises(RuntimeError, match="setup failure"):
        await pool.acquire("coder")
    assert len(pool._envs) == 1
    retained = pool._envs[0].workspace
    assert os.path.isdir(retained)
    assert pool._inflight_acquires == 0
    monkeypatch.setattr(candidate_workspace.EnvCandidateWorkspace, "_candidate_environment", original_environment)
    monkeypatch.setattr(candidate_workspace._CandidateLease, "cleanup", original_cleanup)
    await pool.release()
    assert not os.path.exists(retained) and pool._envs == []

async def test_candidate_submodule_changes_remain_recoverable(tmp_path):
    source = _project(tmp_path / "project", True)
    dependency = _project(tmp_path / "dependency", True)
    _git(source, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(dependency), "vendor/module")
    _git(source, "commit", "-qam", "\u6dfb\u52a0\u5b50\u6a21\u5757")
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    assert await coder.read_file("vendor/module/answer.py") == "def answer():\n    return 1\n"
    await coder.write_file("vendor/module/answer.py", "def answer():\n    return 42\n")
    with pytest.raises(RuntimeError, match="submodule changes cannot be delivered"):
        await coder.get_diff()
    with pytest.raises(OSError, match="retained"):
        await pool.release()
    assert os.path.isdir(coder.workspace)
    await coder.write_file("vendor/module/answer.py", "def answer():\n    return 1\n")
    assert await coder.get_diff() == ""
    await pool.release()

@pytest.mark.parametrize("git", [False, True])
async def test_active_trace_outputs_are_omitted_without_hiding_tracked_edits(tmp_path, git):
    source = _project(tmp_path / "project", git)
    traces = source / "trajectories"
    traces.mkdir()
    if git:
        (source / ".gitignore").write_text("trajectories/\n")
        (traces / "tracked.jsonl").write_text("tracked before\n")
        _git(source, "add", ".gitignore")
        _git(source, "add", "-f", "trajectories/tracked.jsonl")
        _git(source, "commit", "-qm", "\u6dfb\u52a0\u8f68\u8ff9\u8bbe\u7f6e")
        (traces / "tracked.jsonl").write_text("tracked user change\n")
    tracer = Tracer("current-run", output_dir=str(traces))
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True, owned_output_paths=tracer.owned_paths)
    try:
        tracer.log_step(step_type="start", payload={})
        coder = await pool.acquire("coder")
        assert not os.path.exists(os.path.join(coder.workspace, "trajectories/current-run.jsonl"))
        if git:
            assert await coder.read_file("trajectories/tracked.jsonl") == "tracked user change\n"
        await coder.write_file("answer.py", "def answer():\n    return 42\n")
        patch = await coder.get_diff()
        assert "trajectories" not in patch
        await pool.adopt(patch)
        if git:
            (traces / "unrelated.jsonl").write_text("user output\n")
            tester = await pool.acquire("tester")
            assert not os.path.exists(os.path.join(tester.workspace, "trajectories/unrelated.jsonl"))
            assert (traces / "unrelated.jsonl").read_text() == "user output\n"
            assert await tester.get_diff() == ""
    finally:
        tracer.close()
        await pool.release()

async def test_large_conflict_without_saved_artifacts_retains_and_retries(tmp_path):
    source = _project(tmp_path / "project", True)
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    implementation = "def answer():\n    return 42\n# " + "full candidate " * 1500 + "\n"
    await coder.write_file("answer.py", implementation)
    scheduler = _adoption_scheduler(pool, coder)
    text = await scheduler._append_worktree_diff(coder, "done", aid=7, role="coder")
    assert "chars truncated" in text
    assert len(scheduler._completed_coder_diffs[7].patch) > WORKTREE_DIFF_MAX_CHARS
    (source / "answer.py").write_text("def answer():\n    return 99\n")
    with pytest.raises(RuntimeError, match="full candidate retained at"):
        await scheduler.adopt_agent(0, 7)
    with pytest.raises(OSError, match="retained"):
        await pool.release()
    assert os.path.isdir(coder.workspace)
    assert await coder.read_file("answer.py") == implementation
    (source / "answer.py").write_text("def answer():\n    return 1\n")
    await scheduler.adopt_agent(0, 7)
    assert (source / "answer.py").read_text() == implementation
    assert coder.recovery_location is None
    await pool.release()
    assert not os.path.exists(coder.workspace)

@pytest.mark.parametrize("dirty", [False, True])
async def test_strict_snapshot_neutralizes_external_attributes_through_checkout_and_apply(tmp_path, dirty):
    source = _project(tmp_path / "project", True)
    attributes = tmp_path / "external-attributes"
    attributes.write_text("*.txt text eol=crlf\n")
    _git(source, "config", "core.attributesFile", str(attributes))
    if dirty:
        (source / "notes.txt").write_bytes(b"user contents\n")
    contents = (source / "notes.txt").read_bytes()
    index = (source / ".git/index").read_bytes()
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    assert Path(coder.workspace, "notes.txt").read_bytes() == contents
    assert await coder.get_diff() == ""
    await coder.write_file("notes.txt", "coder contents\n")
    await pool.adopt(await coder.get_diff())
    assert (source / "notes.txt").read_bytes() == b"coder contents\n"
    assert (source / ".git/index").read_bytes() == index
    tester = await pool.acquire("tester")
    assert Path(tester.workspace, "notes.txt").read_bytes() == b"coder contents\n"
    assert await tester.get_diff() == ""
    await pool.release()
