"""Candidate adoption preserves directory subtrees with component boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.environments import local_environment


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True)

def _source(tmp_path: Path, monkeypatch) -> Path:
    config = tmp_path / "gitconfig"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("tests", "tests-other", "test data"):
        (root / name).mkdir()
        (root / name / "check.py").write_text("protected = True\n")
    (root / "app.py").write_text("initial = 1\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", ".")
    _git(
        root, "-c", "user.name=Kai", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm",
        "test: \u521b\u5efa\u8def\u5f84\u4fdd\u62a4\u6848\u4f8b",
    )
    (root / "app.py").write_text("user_change = 2\n")
    _git(root, "add", "app.py")
    (root / "notes.txt").write_text("user notes\n")
    return root

@pytest.mark.parametrize(("preserved", "target"), [
    ("tests", "tests/check.py"),
    ("./tests/", "tests/check.py"),
    ("tests/inner/..", "tests/check.py"),
    ("tests/check.py", "tests/check.py"),
    ("test data", "test data/check.py"),
    (".", "app.py"),
])
async def test_candidate_rejects_changes_under_preserved_paths(
    tmp_path, monkeypatch, preserved, target,
):
    root = _source(tmp_path, monkeypatch)
    environment = local_environment(root)
    await environment.setup()
    workspace = EnvCandidateWorkspace(environment)
    index_before = _git(root, "diff", "--cached")
    status_before = _git(root, "status", "--porcelain")
    original = (root / target).read_bytes()
    lease = await workspace.acquire("candidate")
    try:
        await lease.environment.write_file(target, "candidate_change = 3\n")
        patch = await lease.diff()
        assert patch.strip()
        with pytest.raises(ValueError, match="preserved path"):
            await workspace.adopt(patch, preserve_paths=[preserved])
        assert (root / target).read_bytes() == original
        assert _git(root, "diff", "--cached") == index_before
        assert _git(root, "status", "--porcelain") == status_before
        assert (root / "notes.txt").read_text() == "user notes\n"
    finally:
        await lease.cleanup()
        await environment.cleanup()

@pytest.mark.parametrize(("preserved", "target"), [
    ("tests", "tests-other/check.py"),
    ("tests/check.py", "tests/check.py.extra"),
    ("test data", "test data-other.py"),
])
async def test_candidate_allows_distinct_path_components(
    tmp_path, monkeypatch, preserved, target,
):
    root = _source(tmp_path, monkeypatch)
    environment = local_environment(root)
    await environment.setup()
    workspace = EnvCandidateWorkspace(environment)
    index_before = _git(root, "diff", "--cached")
    lease = await workspace.acquire("candidate")
    try:
        await lease.environment.write_file(target, "allowed_change = 4\n")
        await workspace.adopt(await lease.diff(), preserve_paths=[preserved])
        assert (root / target).read_text() == "allowed_change = 4\n"
        assert (root / "tests/check.py").read_text() == "protected = True\n"
        assert _git(root, "diff", "--cached") == index_before
        assert (root / "notes.txt").read_text() == "user notes\n"
    finally:
        await lease.cleanup()
        await environment.cleanup()
