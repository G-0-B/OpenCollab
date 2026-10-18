"""Subdirectory workspaces deliver the entire isolated repository patch."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from opencollab.bootstrap.programmatic import worktree_environment


def _git(repository: Path, *arguments: str, input: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        input=input, text=True, capture_output=True, check=True,
    )
    return result.stdout

def _commit(repository: Path) -> None:
    _git(repository, "add", "-A")
    _git(
        repository, "-c", "user.name=Kai", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm",
        "test: \u4fdd\u5b58\u6d4b\u8bd5\u5019\u9009",
    )

def _repository(tmp_path: Path, monkeypatch) -> Path:
    config = tmp_path / "gitconfig"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    repository = tmp_path / "source repository"
    repository.mkdir()
    (repository / "app").mkdir()
    (repository / "lib").mkdir()
    (repository / "app/main.py").write_text("initial = True\n")
    (repository / "lib/existing.txt").write_text("original library\n")
    (repository / "root.txt").write_text("original root\n")
    (repository / ".gitignore").write_text("root-cache.tmp\n")
    _git(repository, "init", "-q", "-b", "main")
    _commit(repository)
    return repository

@pytest.mark.parametrize(("subdirectory", "target", "committed"), [
    (False, "root new.txt", False),
    (True, "app/new.py", False),
    (True, "root new.txt", False),
    (True, "root new.txt", True),
    (True, "lib/new.conf", False),
    (True, "lib/new.conf", True),
    (True, "root.txt", False),
])
async def test_repository_patch_includes_paths_outside_exposed_subdirectory(
    tmp_path, monkeypatch, subdirectory, target, committed,
):
    source = _repository(tmp_path, monkeypatch)
    initial_head = _git(source, "rev-parse", "HEAD")
    env = worktree_environment(source / "app" if subdirectory else source)
    await env.setup()
    root_result = await env.exec_cmd("git rev-parse --show-toplevel")
    assert root_result.returncode == 0
    candidate_root = Path(root_result.stdout.strip())
    candidate = candidate_root / target
    candidate.write_text("candidate = True\n")
    if committed:
        _commit(candidate_root)
    original_index = _git(candidate_root, "ls-files", "--stage")
    try:
        patch = await env.get_diff()
        assert f"+++ b/{target}" in patch
        observed_cwd = await env.exec_cmd("pwd")
        assert Path(observed_cwd.stdout.strip()).resolve() == Path(env.workspace).resolve()
        assert _git(candidate_root, "ls-files", "--stage") == original_index
        assert _git(source, "rev-parse", "HEAD") == initial_head
        assert _git(source, "status", "--porcelain") == ""
        await env.cleanup()
        assert not candidate_root.exists()
        _git(source, "apply", "--check", "-", input=patch)
        _git(source, "apply", "-", input=patch)
        assert (source / target).read_text() == "candidate = True\n"
    finally:
        await env.cleanup()

@pytest.mark.parametrize("subdirectory", [False, True])
async def test_outside_ignored_file_preserves_existing_capture_refusal(
    tmp_path, monkeypatch, subdirectory,
):
    source = _repository(tmp_path, monkeypatch)
    env = worktree_environment(source / "app" if subdirectory else source)
    await env.setup()
    root = Path((await env.exec_cmd("git rev-parse --show-toplevel")).stdout.strip())
    ignored = root / "root-cache.tmp"
    ignored.write_text("generated cache\n")
    try:
        with pytest.raises(RuntimeError, match="ignored untracked files"):
            await env.get_diff()
        with pytest.raises(RuntimeError, match="undelivered Git worktree changes"):
            await env.cleanup()
        assert ignored.read_text() == "generated cache\n"
        assert env.recovery_location == env.workspace
    finally:
        ignored.unlink(missing_ok=True)
        await env.get_diff()
        await env.cleanup()
