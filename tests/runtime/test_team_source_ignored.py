"""Source-owned ignored assets survive strict team snapshots and adoption."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from opencollab.adapters.worktree_pool import WorktreePool
from tests.runtime.test_team_candidate_delivery import _git, _project


def _native_test_command() -> list[str]:
    return [sys.executable, "-B", "-m", "pytest", "-q", "-o", "cache_dir=.pytest_cache", "test_native.py"]

def _source_with_generated_ignored(tmp_path: Path) -> Path:
    source = _project(tmp_path / "project", True)
    (source / ".gitignore").write_text(".venv/\n.pytest_cache/\n__pycache__/\nprivate-output/\n")
    (source / "test_native.py").write_text(
        "from answer import answer\ndef test_answer():\n    assert answer() > 0\n"
    )
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "\u6dfb\u52a0\u539f\u751f\u6d4b\u8bd5\u4e0e\u5ffd\u7565\u89c4\u5219")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(source / ".venv")], check=True)
    tested = subprocess.run(
        _native_test_command(), cwd=source,
        env={**os.environ, "PYTEST_ADDOPTS": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, check=True,
    )
    assert "1 passed" in tested.stdout
    assert (source / ".venv/pyvenv.cfg").is_file()
    assert (source / ".pytest_cache/CACHEDIR.TAG").is_file()
    (source / "private-output").mkdir()
    (source / "private-output/user.dat").write_bytes(b"user-owned ignored asset\x00\xff")
    return source

@pytest.mark.parametrize("dirty", [False, True])
async def test_source_venv_and_pytest_cache_stay_in_source_during_team_delivery(tmp_path, dirty):
    source = _source_with_generated_ignored(tmp_path)
    if dirty:
        (source / "notes.txt").write_text("staged\n")
        _git(source, "add", "notes.txt")
        (source / "notes.txt").write_text("staged\nunstaged\n")
        (source / "draft.txt").write_text("user draft\n")
    ignored_names = (
        ".venv/pyvenv.cfg", ".pytest_cache/CACHEDIR.TAG",
        ".pytest_cache/v/cache/nodeids", "private-output/user.dat",
    )
    ignored_bytes = {name: (source / name).read_bytes() for name in ignored_names}
    index = (source / ".git/index").read_bytes()
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    try:
        assert all(not (Path(coder.workspace) / name).exists() for name in ignored_names)
        if dirty:
            assert await coder.read_file("notes.txt") == "staged\nunstaged\n"
            assert await coder.read_file("draft.txt") == "user draft\n"
        assert await coder.get_diff() == ""
        await coder.write_file("answer.py", "def answer():\n    return 42\n")
        patch = await coder.get_diff()
        assert "+    return 42" in patch
        assert "notes.txt" not in patch and "draft.txt" not in patch
        assert all(name not in patch for name in ignored_names)
        await pool.adopt(patch)
        tester = await pool.acquire("tester")
        assert await tester.read_file("answer.py") == "def answer():\n    return 42\n"
        assert await tester.get_diff() == ""
        assert (source / ".git/index").read_bytes() == index
        assert {name: (source / name).read_bytes() for name in ignored_names} == ignored_bytes
        if dirty:
            assert (source / "notes.txt").read_text() == "staged\nunstaged\n"
            assert (source / "draft.txt").read_text() == "user draft\n"
    finally:
        await pool.release()

async def test_source_ignored_asset_blocks_same_path_candidate_adoption(tmp_path):
    source = _source_with_generated_ignored(tmp_path)
    protected = source / "private-output/user.dat"
    original = protected.read_bytes()
    ignore_rules = (source / ".gitignore").read_bytes()
    index = (source / ".git/index").read_bytes()
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    try:
        await coder.write_file(".gitignore", ".venv/\n.pytest_cache/\n__pycache__/\n")
        (Path(coder.workspace) / "private-output").mkdir()
        await coder.write_file("private-output/user.dat", "candidate replacement\n")
        await coder.write_file("answer.py", "def answer():\n    return 42\n")
        patch = await coder.get_diff()
        assert "+candidate replacement" in patch
        with pytest.raises(RuntimeError, match="already exists in working directory"):
            await pool.adopt(patch)
        assert protected.read_bytes() == original
        assert (source / ".gitignore").read_bytes() == ignore_rules
        assert (source / "answer.py").read_text() == "def answer():\n    return 1\n"
        assert (source / ".git/index").read_bytes() == index
    finally:
        await pool.release()

async def test_candidate_native_pytest_cache_still_blocks_export_and_retains_workspace(tmp_path):
    source = _source_with_generated_ignored(tmp_path)
    source_nodes = (source / ".pytest_cache/v/cache/nodeids").read_bytes()
    index = (source / ".git/index").read_bytes()
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    workspace = Path(coder.workspace)
    await coder.write_file("answer.py", "def answer():\n    return 42\n")
    result = await coder.exec_cmd("PYTEST_ADDOPTS= " + shlex.join(_native_test_command()))
    assert result.returncode == 0 and "1 passed" in result.stdout
    with pytest.raises(RuntimeError, match="ignored untracked files"):
        await coder.get_diff()
    with pytest.raises(OSError, match="retained"):
        await pool.release()
    try:
        assert workspace.is_dir()
        assert (workspace / ".pytest_cache/CACHEDIR.TAG").is_file()
        assert (workspace / "answer.py").read_text() == "def answer():\n    return 42\n"
        assert (source / ".pytest_cache/v/cache/nodeids").read_bytes() == source_nodes
        assert (source / ".git/index").read_bytes() == index
    finally:
        shutil.rmtree(workspace / ".pytest_cache")
        assert "+    return 42" in await coder.get_diff()
        await pool.release()
