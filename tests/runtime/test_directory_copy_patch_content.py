"""Directory-copy increments retain literal content and nested Git paths."""

from pathlib import Path

import pytest

from opencollab.adapters.worktree_pool import WorktreePool


@pytest.mark.parametrize("name", [
    "paths.txt",
    'nested/a/before/b/after/quoted " name\t.txt',
    "nested a/before/b/after/with b/after/file.txt",
])
@pytest.mark.parametrize("contents", [
    ("version 1\n", "paths a/before/input and b/after/output\n"),
    (
        "paths a/before/input and b/after/output\n-- a/before/old\nversion 1\n",
        "paths a/before/input and b/after/output\n++ b/after/new\nversion 2\n",
    ),
])
async def test_directory_copy_followup_preserves_paths_and_hunk_contents(tmp_path, name, contents):
    source = tmp_path / "plain"
    target = source / name
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    first_contents, second_contents = contents
    try:
        await coder.write_file(name, first_contents)
        first = await coder.get_diff()
        await pool.adopt(first)
        await coder.write_file(name, second_contents)
        second = await coder.get_diff()
        increment = await coder.diff_captured(first, second)
        await pool.adopt(increment)
        assert target.read_text(encoding="utf-8") == second_contents
        assert {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()} == {name}
    finally:
        await coder.get_diff()
        await pool.release()

async def test_directory_copy_followup_preserves_legacy_bytes_and_file_metadata(tmp_path):
    source = tmp_path / "plain"
    source.mkdir()
    (source / "removed.txt").write_bytes(b"remove me\n")
    name = "nested a/before/b/after/with b/after/legacy.csv"
    target = source / name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"name\ncaf\xe9\n")
    run_name = "nested a/before/b/after/with b/after/run.sh"
    (source / run_name).write_bytes(b"true\n")
    pool = WorktreePool(str(source), use_worktrees=True, enable_adoption=True)
    coder = await pool.acquire("coder")
    try:
        Path(coder.workspace, name).write_bytes(b"name\npi\xf1ata\n")
        first = await coder.get_diff()
        await pool.adopt(first)
        updated = b"name\npi\xf1ata\npaths a/before/input and b/after/output\n"
        Path(coder.workspace, name).write_bytes(updated)
        Path(coder.workspace, run_name).chmod(0o755)
        Path(coder.workspace, "removed.txt").unlink()
        await coder.write_file("added.txt", "paths a/before/added and b/after/added\n")
        second = await coder.get_diff()
        increment = await coder.diff_captured(first, second)
        assert "GIT binary patch" in increment
        await pool.adopt(increment)
        assert target.read_bytes() == updated
        assert (source / run_name).stat().st_mode & 0o111 == 0o111
        assert not (source / "removed.txt").exists()
        assert (source / "added.txt").read_text() == "paths a/before/added and b/after/added\n"
    finally:
        await coder.get_diff()
        await pool.release()
