"""Candidate delivery retains executable bits independently of Git filemode config."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from tests.workflows.test_workflow_candidate_workspace import _git, _repository


@pytest.mark.parametrize("filemode", [False, True])
@pytest.mark.parametrize("initial,target", [(0o644, 0o755), (0o755, 0o644)])
@pytest.mark.parametrize("edit_content", [False, True])
async def test_candidate_delivers_actual_script_mode(tmp_path, filemode, initial, target, edit_content):
    repo = _repository(tmp_path)
    script = repo / "message.sh"
    script.write_text("#!/bin/sh\necho original\n")
    script.chmod(initial)
    _git(repo, "add", "message.sh")
    _git(repo, "commit", "-m", "\u8bb0\u5f55\u811a\u672c\u6587\u4ef6")
    _git(repo, "config", "core.filemode", str(filemode).lower())
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("script-mode")
        candidate_script = Path(lease.candidate_workspace) / "message.sh"
        if edit_content:
            candidate_script.write_text("#!/bin/sh\necho edited\n")
        candidate_script.chmod(target)
        candidate_execution = await lease.environment.exec_cmd("./message.sh")
        assert (candidate_execution.returncode == 0) == (target == 0o755)
        patch = await lease.diff()
        assert f"old mode 100{initial:o}" in patch
        assert f"new mode 100{target:o}" in patch
        await workspace.adopt(patch)
        assert script.stat().st_mode & 0o777 == target
        assert script.read_text() == candidate_script.read_text()
        delivered_execution = await environment.exec_cmd("./message.sh")
        assert delivered_execution.returncode == candidate_execution.returncode
        assert delivered_execution.stdout == candidate_execution.stdout
        assert _git(repo, "config", "core.filemode").strip() == str(filemode).lower()
        assert _git(repo, "ls-files", "--stage", "message.sh").startswith(f"100{initial:o} ")
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()

@pytest.mark.parametrize("filemode", [False, True])
@pytest.mark.parametrize("mode", [0o644, 0o755])
@pytest.mark.parametrize("edit_content", [False, True])
async def test_public_candidate_rename_preserves_script_mode(tmp_path, filemode, mode, edit_content):
    repo = _repository(tmp_path)
    script = repo / "message.sh"
    contents = "#!/bin/sh\n" + "".join(f"# retained line {number}\n" for number in range(12)) + "echo original\n"
    script.write_text(contents)
    script.chmod(mode)
    _git(repo, "add", "message.sh")
    _git(repo, "commit", "-m", "\u8bb0\u5f55\u811a\u672c\u6587\u4ef6")
    _git(repo, "config", "core.filemode", str(filemode).lower())
    (repo / "source.py").write_text("value = 2\n")
    _git(repo, "add", "source.py")
    (repo / "source.py").write_text("value = 3\n")
    (repo / "draft.txt").write_text("user draft\n")
    index = (repo / ".git" / "index").read_bytes()
    config = (repo / ".git" / "config").read_bytes()
    candidate_paths = []
    expected = "edited\n" if edit_content else "original\n"

    async def rename(child, _inputs):
        candidate_repo = Path(child.workspace_root)
        candidate_paths.append(candidate_repo)
        _git(candidate_repo, "mv", "message.sh", "renamed.sh")
        renamed = candidate_repo / "renamed.sh"
        if edit_content:
            renamed.write_text(contents.replace("echo original", "echo edited"))
        assert renamed.stat().st_mode & 0o777 == mode
        if mode == 0o755:
            return subprocess.run([str(renamed)], check=True, capture_output=True, text=True).stdout
        with pytest.raises(PermissionError):
            subprocess.run([str(renamed)], check=True, capture_output=True)
        return "permission denied"

    async def flow(ctx, inputs):
        candidate = await ctx.candidate_workflow(rename, inputs, label="rename-script")
        assert candidate.output == (expected if mode == 0o755 else "permission denied")
        assert "rename from message.sh" in candidate.diff
        assert "rename to renamed.sh" in candidate.diff
        assert "old mode" not in candidate.diff
        assert "new mode" not in candidate.diff
        assert script.read_text() == contents
        assert script.stat().st_mode & 0o777 == mode
        await ctx.adopt_candidate(candidate)
        return candidate.diff

    result = await OpenCollab(repo).workflow(flow, trace=False)
    assert result.ok, result.reason
    renamed = repo / "renamed.sh"
    assert not script.exists()
    assert renamed.stat().st_mode & 0o777 == mode
    expected_contents = contents.replace("echo original", "echo edited") if edit_content else contents
    assert renamed.read_text() == expected_contents
    if mode == 0o755:
        execution = subprocess.run([str(renamed)], check=True, capture_output=True, text=True)
        assert execution.stdout == expected
    else:
        with pytest.raises(PermissionError):
            subprocess.run([str(renamed)], check=True, capture_output=True)
    assert (repo / "source.py").read_text() == "value = 3\n"
    assert (repo / "draft.txt").read_text() == "user draft\n"
    assert (repo / ".git" / "index").read_bytes() == index
    assert (repo / ".git" / "config").read_bytes() == config
    assert candidate_paths and all(not path.exists() for path in candidate_paths)
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1

@pytest.mark.parametrize("filemode", [False, True])
@pytest.mark.parametrize("invalid_patch", [False, True])
async def test_source_rename_restoration_and_rollback_preserve_executable(tmp_path, filemode, invalid_patch):
    repo = _repository(tmp_path)
    script = repo / "message.sh"
    contents = "#!/bin/sh\necho original\n"
    script.write_text(contents)
    script.chmod(0o755)
    _git(repo, "add", "message.sh")
    _git(repo, "commit", "-m", "\u8bb0\u5f55\u811a\u672c\u6587\u4ef6")
    _git(repo, "config", "core.filemode", str(filemode).lower())
    _git(repo, "mv", "message.sh", "renamed.sh")
    index = (repo / ".git" / "index").read_bytes()
    config = (repo / ".git" / "config").read_bytes()
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    try:
        assert "rename from message.sh" in await workspace.source_diff()
        if invalid_patch:
            with pytest.raises(RuntimeError, match="candidate source restoration failed"):
                await workspace.restore_source("invalid patch\n")
            restored = repo / "renamed.sh"
            assert not script.exists()
        else:
            await workspace.restore_source("")
            restored = script
            assert not (repo / "renamed.sh").exists()
        assert restored.read_text() == contents
        assert restored.stat().st_mode & 0o777 == 0o755
        assert subprocess.run([str(restored)], check=True, capture_output=True, text=True).stdout == "original\n"
        assert (repo / ".git" / "index").read_bytes() == index
        assert (repo / ".git" / "config").read_bytes() == config
    finally:
        await environment.cleanup()

@pytest.mark.parametrize("filemode", [False, True])
@pytest.mark.parametrize("initial,target", [(0o644, 0o755), (0o755, 0o644)])
@pytest.mark.parametrize("edit_source_content", [False, True])
async def test_candidate_inherits_source_script_mode_as_initial_contents(
    tmp_path, filemode, initial, target, edit_source_content,
):
    repo = _repository(tmp_path)
    script = repo / "message.sh"
    script.write_text("#!/bin/sh\necho original\n")
    script.chmod(initial)
    _git(repo, "add", "message.sh")
    _git(repo, "commit", "-m", "\u8bb0\u5f55\u811a\u672c\u6587\u4ef6")
    _git(repo, "config", "core.filemode", str(filemode).lower())
    script.chmod(target)
    if edit_source_content:
        script.write_text("#!/bin/sh\necho inherited\n")
        (repo / "source.py").write_text("value = 2\n")
        _git(repo, "add", "source.py")
    source_contents = script.read_text()
    source_status = _git(repo, "status", "--porcelain=v1")
    source_index = _git(repo, "ls-files", "--stage")
    source_config = (repo / ".git" / "config").read_bytes()
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        source_execution = await environment.exec_cmd("./message.sh")
        lease = await workspace.acquire("source-script-mode")
        candidate_script = Path(lease.candidate_workspace) / "message.sh"
        candidate_execution = await lease.environment.exec_cmd("./message.sh")
        assert candidate_execution.returncode == source_execution.returncode
        assert candidate_execution.stdout == source_execution.stdout
        assert candidate_script.stat().st_mode & 0o777 == target
        assert candidate_script.read_text() == source_contents
        assert await lease.diff() == ""
        source_patch = await workspace.source_diff()
        assert f"old mode 100{initial:o}" in source_patch
        assert f"new mode 100{target:o}" in source_patch
        assert _git(repo, "status", "--porcelain=v1") == source_status
        assert _git(repo, "ls-files", "--stage") == source_index
        assert (repo / ".git" / "config").read_bytes() == source_config

        candidate_script.write_text("#!/bin/sh\necho edited\n")
        patch = await lease.diff()
        assert f"-{source_contents.splitlines()[1]}" in patch
        assert "+echo edited" in patch
        assert "old mode" not in patch
        assert "new mode" not in patch
        assert "source.py" not in patch
        await workspace.adopt(patch)
        assert script.stat().st_mode & 0o777 == target
        assert script.read_text() == candidate_script.read_text()
        delivered_execution = await environment.exec_cmd("./message.sh")
        candidate_execution = await lease.environment.exec_cmd("./message.sh")
        assert delivered_execution.returncode == candidate_execution.returncode
        assert delivered_execution.stdout == candidate_execution.stdout
        assert _git(repo, "ls-files", "--stage") == source_index
        assert (repo / ".git" / "config").read_bytes() == source_config
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()
