"""Worktree delivery applies legacy CSV bytes before releasing the candidate."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from opencollab.adapters import worktree_pool
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.environments import attach_container
from tests.support.worktree_delivery_support import CONTAINER_ID, git
from tests.support.worktree_delivery_support import worktree_transport as worktree_transport

_FILENAME = "\u5ba2\u6237 \u8bb0\u5f55.csv"
_ORIGINAL = b"id,name\n" + b"1,caf\xe9 \x80\n" * 20
_UPDATED = _ORIGINAL + b"2,pi\xf1ata\n"

async def _candidate(tmp_path, monkeypatch, backend):
    source = tmp_path / "repo with spaces"
    source.mkdir()
    (source / _FILENAME).write_bytes(_ORIGINAL)
    (source / "source.py").write_text("value = 1\n", encoding="utf-8")
    if backend != "copy":
        git(source, "init", "-q", "-b", "main")
        git(source, "config", "user.name", "OpenCollab Tests")
        git(source, "config", "user.email", "tests@example.invalid")
        git(source, "add", ".")
        git(source, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    monkeypatch.setattr(worktree_pool, "CONTAINER_WORKTREE_ROOT", str(tmp_path / "worktrees"))
    base = attach_container(container_id=CONTAINER_ID, workspace=str(source)) if backend == "container" else None
    pool = WorktreePool(str(source), use_worktrees=True, base_environment=base)
    return source, pool, await pool.acquire("csv writer")

def _apply(source, patch):
    for arguments in (("apply", "--check", "-"), ("apply", "-")):
        subprocess.run(
            ["git", "-C", str(source), *arguments], input=patch.encode("utf-8"),
            check=True, capture_output=True,
        )

@pytest.mark.parametrize("backend", ["local", "container", "copy"])
@pytest.mark.parametrize("change", ["edit", "add", "delete", "rename", "executable"])
async def test_legacy_worktree_patch_applies_exact_bytes_and_releases(
    worktree_transport, tmp_path, monkeypatch, backend, change,
):
    source, pool, env = await _candidate(tmp_path, monkeypatch, backend)
    workspace = Path(env.workspace)
    target_name = _FILENAME
    try:
        target = workspace / _FILENAME
        if change == "delete":
            target.unlink()
        else:
            if change in {"add", "rename"}:
                target_name = "\u65b0 \u8bb0\u5f55.csv"
                target = workspace / target_name if change == "add" else target.rename(workspace / target_name)
            target.write_bytes(_UPDATED)
            if change == "executable":
                target.chmod(0o755)
        (workspace / "source.py").write_text("value = 2\n# caf\u00e9\n", encoding="utf-8")
        index = None if backend == "copy" else Path(git(workspace, "rev-parse", "--git-path", "index").strip())
        index_before = None if index is None else index.read_bytes()
        patch = await env.get_diff()
        assert "GIT binary patch" in patch
        assert "\ufffd" not in patch
        assert "+value = 2\n+# caf\u00e9\n" in patch
        assert str(workspace) not in patch
        assert env.recovery_location is None
        if index is not None:
            assert index.read_bytes() == index_before
        _apply(source, patch)
        if change in {"delete", "rename"}:
            assert not (source / _FILENAME).exists()
        if change != "delete":
            assert (source / target_name).read_bytes() == _UPDATED
        if change == "add":
            assert (source / _FILENAME).read_bytes() == _ORIGINAL
        if change == "executable":
            assert (source / target_name).stat().st_mode & 0o111 == 0o111
        assert (source / "source.py").read_text(encoding="utf-8") == "value = 2\n# caf\u00e9\n"
    finally:
        await pool.release()
    assert not workspace.exists()
    if backend != "copy":
        assert env._branch not in git(source, "branch", "--list")
    if backend == "container":
        assert any(call[:2] == ["docker", "exec"] for call in worktree_transport)

@pytest.mark.parametrize("backend", ["local", "container", "copy"])
async def test_failed_legacy_conversion_retains_candidate_until_complete_delivery(
    worktree_transport, tmp_path, monkeypatch, backend,
):
    source, pool, env = await _candidate(tmp_path, monkeypatch, backend)
    workspace = Path(env.workspace)
    (workspace / _FILENAME).write_bytes(_UPDATED)
    real_git = shutil.which("git")
    assert real_git is not None
    executable = tmp_path / "bin"
    executable.mkdir()
    (executable / "git").write_text(
        "#!/bin/sh\nfor arg; do\n"
        '  if [ "$arg" = --no-index ]; then echo "binary export unavailable" >&2; exit 86; fi\n'
        f"done\nexec {shlex.quote(real_git)} \"$@\"\n",
        encoding="utf-8",
    )
    (executable / "git").chmod(0o755)
    original_path = os.environ["PATH"]
    try:
        monkeypatch.setenv("PATH", f"{executable}{os.pathsep}{original_path}")
        with pytest.raises(RuntimeError, match="binary export unavailable"):
            await env.get_diff()
        with pytest.raises(OSError, match="retry state retained"):
            await pool.release()
        assert workspace.is_dir()
        assert (workspace / _FILENAME).read_bytes() == _UPDATED
        assert (source / _FILENAME).read_bytes() == _ORIGINAL
        monkeypatch.setenv("PATH", original_path)
        patch = await env.get_diff()
        _apply(source, patch)
        assert (source / _FILENAME).read_bytes() == _UPDATED
    finally:
        monkeypatch.setenv("PATH", original_path)
        await env.get_diff()
        await pool.release()
    assert not workspace.exists()

@pytest.mark.parametrize("backend", ["local", "container"])
async def test_binary_conversion_uses_the_captured_blobs_under_global_attributes(
    worktree_transport, tmp_path, monkeypatch, backend,
):
    source, pool, env = await _candidate(tmp_path, monkeypatch, backend)
    workspace = Path(env.workspace)
    updated = _UPDATED.replace(b"\n", b"\r\n")
    (workspace / _FILENAME).write_bytes(updated)
    attributes = tmp_path / "global-attributes"
    attributes.write_text("*.csv text\n", encoding="utf-8")
    config = tmp_path / "gitconfig"
    config.write_text(f"[core]\n\tattributesFile = {attributes}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    try:
        patch = await env.get_diff()
        # Applying outside the repository's index keeps the on-disk byte
        # comparison independent of the caller's checkout normalization.
        destination = tmp_path / "delivery"
        destination.mkdir()
        (destination / _FILENAME).write_bytes(_ORIGINAL)
        _apply(destination, patch)
        assert (destination / _FILENAME).read_bytes() == updated
        assert (workspace / _FILENAME).read_bytes() == updated
        assert (source / _FILENAME).read_bytes() == _ORIGINAL
        assert attributes.read_text(encoding="utf-8") == "*.csv text\n"
    finally:
        await pool.release()
    assert not workspace.exists()
