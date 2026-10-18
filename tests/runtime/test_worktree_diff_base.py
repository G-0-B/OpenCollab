"""Worktree diffs attribute handoff commits to their original author.

After a checkout, the reflog supplies the commit HEAD moved onto as the new base.
Real linked worktrees exercise shared Git objects and checkout behavior.
"""

from __future__ import annotations

import subprocess

from opencollab.adapters.env import WorktreeEnvironment
from opencollab.application._scheduler_team import _parse_worktree_diff


def _git(cwd, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

def _repo(path, *, object_format: str = "sha1"):
    """A committed repository — the precondition for linked worktrees."""
    path.mkdir()
    _git(path, "init", "-q", "-b", "main", f"--object-format={object_format}")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "OpenCollab Tests")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "base")
    return path

def _changed_files(diff: str) -> list[str]:
    """What the scheduler's ``worktree_changes`` record would list."""
    return [path for path, _ in _parse_worktree_diff(diff)]

async def test_an_agents_own_commits_stay_in_its_own_diff(tmp_path) -> None:
    """Committing is how work is handed over, so it must not erase the record."""
    source = _repo(tmp_path / "repo")
    env = WorktreeEnvironment(str(source), branch_name="diff-base-coder")
    try:
        await env.setup()
        creation_base = _git(env.workspace, "rev-parse", "HEAD")
        await env.write_file("f1.txt", "coder work\n")
        assert (await env.exec_cmd("git add -A && git commit -qm coder")).returncode == 0

        diff = await env.get_diff()

        assert _changed_files(diff) == ["f1.txt"]
        # A descendant commit advances HEAD while retaining the original diff base.
        assert _git(env.workspace, "rev-parse", "HEAD") != creation_base
        assert env.diff_base == creation_base
    finally:
        await env.cleanup()

async def test_a_checked_out_teammate_commit_is_not_this_agents_work(tmp_path) -> None:
    """The bug this exists for: the tester's row must not claim the coder's file."""
    source = _repo(tmp_path / "repo")
    coder = WorktreeEnvironment(str(source), branch_name="handoff-coder")
    tester = WorktreeEnvironment(str(source), branch_name="handoff-tester")
    try:
        await coder.setup()
        await coder.write_file("f1.txt", "coder work\n")
        assert (await coder.exec_cmd("git add -A && git commit -qm coder")).returncode == 0
        handoff_sha = _git(coder.workspace, "rev-parse", "HEAD")
        assert _changed_files(await coder.get_diff()) == ["f1.txt"]

        await tester.setup()
        assert (await tester.exec_cmd(f"git checkout -q {handoff_sha}")).returncode == 0
        await tester.write_file("f2.txt", "tester work\n")

        diff = await tester.get_diff()

        assert _changed_files(diff) == ["f2.txt"]
        assert "f1.txt" not in diff
        assert tester.diff_base == handoff_sha
    finally:
        await tester.cleanup()
        await coder.cleanup()

async def test_work_committed_after_a_handoff_still_belongs_to_the_taker(tmp_path) -> None:
    """Adopting a base and then committing on it are different moves."""
    source = _repo(tmp_path / "repo")
    coder = WorktreeEnvironment(str(source), branch_name="after-coder")
    tester = WorktreeEnvironment(str(source), branch_name="after-tester")
    try:
        await coder.setup()
        await coder.write_file("f1.txt", "coder work\n")
        assert (await coder.exec_cmd("git add -A && git commit -qm coder")).returncode == 0
        handoff_sha = _git(coder.workspace, "rev-parse", "HEAD")

        await tester.setup()
        assert (await tester.exec_cmd(f"git checkout -q {handoff_sha}")).returncode == 0
        await tester.write_file("f2.txt", "tester work\n")
        assert (await tester.exec_cmd("git add -A && git commit -qm tester")).returncode == 0
        await tester.write_file("f3.txt", "still working\n")

        diff = await tester.get_diff()

        assert _changed_files(diff) == ["f2.txt", "f3.txt"]
        assert tester.diff_base == handoff_sha
    finally:
        await tester.cleanup()
        await coder.cleanup()

async def test_a_worktree_that_never_takes_a_handoff_is_measured_where_it_was(
    tmp_path,
) -> None:
    """No checkout: the base is the creation base, exactly as before."""
    source = _repo(tmp_path / "repo")
    env = WorktreeEnvironment(str(source), branch_name="diff-base-plain")
    try:
        await env.setup()
        creation_base = _git(env.workspace, "rev-parse", "HEAD")
        await env.write_file("committed.txt", "committed\n")
        assert (await env.exec_cmd("git add -A && git commit -qm plain")).returncode == 0
        await env.write_file("untracked.txt", "untracked\n")
        await env.write_file("tracked.txt", "edited\n")

        diff = await env.get_diff()

        assert env.diff_base == creation_base
        assert _changed_files(diff) == ["committed.txt", "tracked.txt", "untracked.txt"]
    finally:
        await env.cleanup()

async def test_a_reset_onto_a_teammate_commit_moves_the_base_too(tmp_path) -> None:
    """``checkout`` is not the only way to adopt another agent's commit."""
    source = _repo(tmp_path / "repo")
    coder = WorktreeEnvironment(str(source), branch_name="reset-coder")
    tester = WorktreeEnvironment(str(source), branch_name="reset-tester")
    try:
        await coder.setup()
        await coder.write_file("f1.txt", "coder work\n")
        assert (await coder.exec_cmd("git add -A && git commit -qm coder")).returncode == 0
        handoff_sha = _git(coder.workspace, "rev-parse", "HEAD")

        await tester.setup()
        assert (await tester.exec_cmd(f"git reset -q --hard {handoff_sha}")).returncode == 0
        await tester.write_file("f2.txt", "tester work\n")

        assert _changed_files(await tester.get_diff()) == ["f2.txt"]
        assert tester.diff_base == handoff_sha
    finally:
        await tester.cleanup()
        await coder.cleanup()

async def test_a_non_git_worktree_names_no_base_and_diffs_as_before(tmp_path) -> None:
    """The directory-copy fallback has no revisions; it must be untouched."""
    source = tmp_path / "plain"
    source.mkdir()
    (source / "tracked.txt").write_text("base\n", encoding="utf-8")
    env = WorktreeEnvironment(str(source), branch_name="diff-base-copy")
    try:
        await env.setup()
        await env.write_file("added.txt", "added\n")

        diff = await env.get_diff()

        assert "added.txt" in diff
        assert env.diff_base is None
    finally:
        await env.cleanup()

async def test_the_commits_an_agent_made_are_reported_beside_the_base(tmp_path) -> None:
    """Commits reachable from HEAD beyond diff_base identify the payload available for handoff."""
    source = _repo(tmp_path / "repo")
    env = WorktreeEnvironment(str(source), branch_name="own-commits-coder")
    try:
        await env.setup()
        creation_base = _git(env.workspace, "rev-parse", "HEAD")

        await env.get_diff()
        assert env.head_commit == creation_base
        assert env.own_commits == ()
        assert env.own_commit_count == 0

        await env.write_file("f1.txt", "first\n")
        assert (await env.exec_cmd("git add -A && git commit -qm one")).returncode == 0
        first = _git(env.workspace, "rev-parse", "HEAD")
        await env.write_file("f2.txt", "second\n")
        assert (await env.exec_cmd("git add -A && git commit -qm two")).returncode == 0
        second = _git(env.workspace, "rev-parse", "HEAD")

        await env.get_diff()

        assert env.diff_base == creation_base
        assert env.head_commit == second
        assert env.own_commits == (second, first)
        assert env.own_commit_count == 2
    finally:
        await env.cleanup()

async def test_a_commit_adopted_from_a_teammate_is_not_counted_as_the_takers_own(
    tmp_path,
) -> None:
    """The join the handoff evidence rests on, from both ends."""
    source = _repo(tmp_path / "repo")
    coder = WorktreeEnvironment(str(source), branch_name="own-commits-handoff-coder")
    tester = WorktreeEnvironment(str(source), branch_name="own-commits-handoff-tester")
    try:
        await coder.setup()
        await coder.write_file("f1.txt", "coder work\n")
        assert (await coder.exec_cmd("git add -A && git commit -qm coder")).returncode == 0
        await coder.get_diff()
        handoff_sha = coder.head_commit

        await tester.setup()
        assert (await tester.exec_cmd(f"git checkout -q {handoff_sha}")).returncode == 0
        await tester.write_file("f2.txt", "tester work\n")
        await tester.get_diff()

        assert coder.own_commits == (handoff_sha,)
        assert tester.diff_base == handoff_sha
        assert tester.own_commits == ()
        assert tester.own_commit_count == 0
        assert tester.head_commit == handoff_sha
    finally:
        await tester.cleanup()
        await coder.cleanup()

async def test_a_non_git_worktree_names_no_head_and_no_commits(tmp_path) -> None:
    """The directory-copy fallback has no revisions to report, and invents none."""
    source = tmp_path / "plain"
    source.mkdir()
    (source / "tracked.txt").write_text("base\n", encoding="utf-8")
    env = WorktreeEnvironment(str(source), branch_name="own-commits-copy")
    try:
        await env.setup()
        await env.write_file("added.txt", "added\n")

        await env.get_diff()

        assert env.head_commit is None
        assert env.own_commits == ()
        assert env.own_commit_count is None
    finally:
        await env.cleanup()

async def test_a_handoff_is_read_in_a_sha256_repository_too(tmp_path) -> None:
    """SHA-256 reflogs use 64-character IDs.

    A 40-character-only parser falls back to the creation base and misattributes handoffs.
    """
    source = _repo(tmp_path / "repo", object_format="sha256")
    coder = WorktreeEnvironment(str(source), branch_name="sha256-coder")
    tester = WorktreeEnvironment(str(source), branch_name="sha256-tester")
    try:
        await coder.setup()
        await coder.write_file("f1.txt", "coder work\n")
        assert (await coder.exec_cmd("git add -A && git commit -qm coder")).returncode == 0
        handoff_sha = _git(coder.workspace, "rev-parse", "HEAD")
        assert len(handoff_sha) == 64
        # Populate commit evidence through the same diff read used by the scheduler.
        assert _changed_files(await coder.get_diff()) == ["f1.txt"]

        await tester.setup()
        assert (await tester.exec_cmd(f"git checkout -q {handoff_sha}")).returncode == 0
        await tester.write_file("f2.txt", "tester work\n")

        diff = await tester.get_diff()

        assert _changed_files(diff) == ["f2.txt"]
        assert tester.diff_base == handoff_sha
        assert handoff_sha in coder.own_commits
    finally:
        await tester.cleanup()
        await coder.cleanup()
