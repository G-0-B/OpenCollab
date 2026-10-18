"""Exclude injected support files when probing for agent source edits.

Real Git integration checks untracked files as well as the scripted command path.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters.working_tree import WORKING_TREE_GIT_TIMEOUT_SECONDS, EnvWorkingTreeProbe
from opencollab.patches import patch_paths


def run(coro):
    return asyncio.run(coro)

class RecordingEnv:
    """Fake Environment recording every exec_cmd; returns a scripted stdout."""

    def __init__(self, *, stdout: str = "", workspace: str = "/ws") -> None:
        self.workspace = workspace
        self._stdout = stdout
        self.commands: list[str] = []

    async def exec_cmd(self, cmd: str, timeout: float = 120.0):
        self.commands.append(cmd)
        return SimpleNamespace(stdout=self._stdout)

class ScriptedEnv:
    def __init__(self, results, *, workspace: str = "/ws") -> None:
        self.workspace = workspace
        self._results = list(results)
        self.commands: list[str] = []
        self.timeouts: list[float] = []

    async def exec_cmd(self, cmd: str, timeout: float = 120.0):
        self.commands.append(cmd)
        self.timeouts.append(timeout)
        return self._results.pop(0)

def test_diff_uses_pre_model_workspace_timeout():
    env = ScriptedEnv(
        [
            ExecResult(0, "", ""),
            ExecResult(0, "/ws\n", ""),
            ExecResult(0, "", ""),
            ExecResult(0, "", ""),
        ]
    )

    assert run(EnvWorkingTreeProbe(env, workspace="/ws").diff()) == (
        "[Working tree status]\n(clean)"
    )
    assert env.timeouts == [WORKING_TREE_GIT_TIMEOUT_SECONDS] * 4

def test_changed_excluding_issues_exclude_pathspec_and_returns_bool():
    env = RecordingEnv(stdout=" M opencollab/foo.py\n", workspace="/ws")
    probe = EnvWorkingTreeProbe(env, workspace="/ws")

    result = run(probe.changed_excluding(["tests/test_inj.py"]))

    assert result is True
    assert env.commands[-1] == (
        "git --no-optional-locks -C /ws status --porcelain --untracked-files=all "
        "-- ':(top)' ':(exclude,literal)tests/test_inj.py'"
    )

def test_changed_excluding_returns_false_on_empty_status():
    env = RecordingEnv(stdout="", workspace="/ws")
    probe = EnvWorkingTreeProbe(env, workspace="/ws")

    assert run(probe.changed_excluding(["tests/test_inj.py"])) is False

def test_changed_excluding_quotes_each_exclude_token_separately():
    env = RecordingEnv(stdout="x\n", workspace="/ws")
    probe = EnvWorkingTreeProbe(env, workspace="/ws")

    run(probe.changed_excluding(["a/b.py", "c/d.py"]))

    assert env.commands[-1] == (
        "git --no-optional-locks -C /ws status --porcelain --untracked-files=all "
        "-- ':(top)' ':(exclude,literal)a/b.py' ':(exclude,literal)c/d.py'"
    )

def test_changed_excluding_empty_falls_back_to_plain_status():
    env = RecordingEnv(stdout=" M x.py\n", workspace="/ws")
    probe = EnvWorkingTreeProbe(env, workspace="/ws")

    excl = run(probe.changed_excluding([]))
    plain = run(probe.changed())

    assert excl is True and plain is True
    assert env.commands == [
        "git --no-optional-locks -C /ws status --porcelain",
        "git --no-optional-locks -C /ws status --porcelain",
    ]

@pytest.mark.parametrize(
    "result",
    [
        ExecResult(1, "partial", "failed"),
        ExecResult(0, "partial", "", stdout_truncated=True),
        ExecResult(0, "partial", "", stdout_dropped_bytes=9),
        ExecResult(0, "", "partial", stderr_truncated=True),
    ],
)
def test_diff_rejects_incomplete_status_evidence(result):
    probe = EnvWorkingTreeProbe(ScriptedEnv([result]), workspace="/ws")

    with pytest.raises(RuntimeError, match="working-tree status"):
        run(probe.diff())

def test_diff_rejects_truncated_tracked_patch():
    env = ScriptedEnv(
        [
            ExecResult(0, " M source.py\n", ""),
            ExecResult(0, "/ws\n", ""),
            ExecResult(
                0,
                "diff --git a/source.py b/source.py\n",
                "",
                stdout_truncated=True,
                stdout_dropped_bytes=128,
            ),
        ]
    )
    probe = EnvWorkingTreeProbe(env, workspace="/ws")

    with pytest.raises(RuntimeError, match="tracked diff exceeded capture limit"):
        run(probe.diff())

@pytest.mark.parametrize("result", [
    ExecResult(1, "", "missing repository"),
    ExecResult(0, "/partial", "", stdout_truncated=True),
    ExecResult(0, "/partial", "", stdout_dropped_bytes=3),
])
def test_diff_rejects_incomplete_repository_root(result):
    env = ScriptedEnv([ExecResult(0, "", ""), result])

    with pytest.raises(RuntimeError, match="working-tree repository root"):
        run(EnvWorkingTreeProbe(env).diff())

class _RealEnv:
    """Minimal env that shells out to real git via subprocess (no opencollab env)."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace

    async def exec_cmd(self, cmd: str, timeout: float = 120.0):
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_real_git_untracked_injected_test_is_excluded(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "src.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")

    (repo / "tests").mkdir()
    inj = "tests/test_inj.py"
    (repo / inj).write_text("def test_x():\n    assert True\n")

    probe = EnvWorkingTreeProbe(_RealEnv(str(repo)), workspace=str(repo))

    assert run(probe.changed()) is True
    assert run(probe.changed_excluding([inj])) is False

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_real_git_source_edit_alongside_injected_is_detected(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "src.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")

    (repo / "tests" / "nested").mkdir(parents=True)
    inj = "tests/nested/test_inj.py"
    (repo / inj).write_text("def test_x():\n    assert True\n")
    (repo / "src.py").write_text("x = 2\n")

    probe = EnvWorkingTreeProbe(_RealEnv(str(repo)), workspace=str(repo))

    assert run(probe.changed()) is True
    assert run(probe.changed_excluding([inj])) is True

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
@pytest.mark.parametrize("tracked", [False, True])
def test_literal_dynamic_route_exclusion_keeps_other_source_changes(tmp_path: Path, tracked):
    repo = tmp_path / "repo"
    excluded = repo / "app/[id]/page.tsx"
    other = repo / "app/i/page.tsx"
    excluded.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    excluded.write_text("initial route\n")
    if tracked:
        other.write_text("initial source\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    excluded.write_text("generated route\n")
    probe = EnvWorkingTreeProbe(_RealEnv(str(repo)), workspace=str(repo))
    paths = ["app/[id]/page.tsx"]
    assert run(probe.changed_excluding(paths)) is False
    other.write_text("source edit\n")
    assert run(probe.changed_excluding(paths)) is True

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_real_git_diff_covers_staged_unstaged_and_untracked_changes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "unstaged.py").write_text("unstaged = 1\n")
    (repo / "staged.py").write_text("staged = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")

    (repo / "unstaged.py").write_text("unstaged = 2\n")
    (repo / "staged.py").write_text("staged = 2\n")
    _git(repo, "add", "staged.py")
    (repo / "new file.py").write_text("untracked = 3\n")

    probe = EnvWorkingTreeProbe(_RealEnv(str(repo)), workspace=str(repo))
    evidence = run(probe.diff())

    assert run(probe.changed()) is True
    assert " M unstaged.py" in evidence
    assert "M  staged.py" in evidence
    assert "?? \"new file.py\"" in evidence
    assert "-unstaged = 1" in evidence
    assert "+unstaged = 2" in evidence
    assert "-staged = 1" in evidence
    assert "+staged = 2" in evidence
    assert "+untracked = 3" in evidence

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_real_git_untracked_empty_file_remains_visible_in_diff_evidence(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "tracked.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    (repo / "empty.txt").touch()

    evidence = run(
        EnvWorkingTreeProbe(_RealEnv(str(repo)), workspace=str(repo)).diff()
    )

    assert "?? empty.txt" in evidence
    assert "[Untracked file: empty.txt]" in evidence

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_real_git_untracked_diff_disables_textconv(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitattributes").write_text("*.raw diff=hostile\n")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-qm", "base")
    sentinel = tmp_path / "textconv-ran"
    converter = tmp_path / "convert.sh"
    converter.write_text(f"#!/bin/sh\nprintf ran > {sentinel}\nprintf forged\n")
    converter.chmod(0o755)
    _git(repo, "config", "diff.hostile.textconv", str(converter))
    (repo / "payload.raw").write_bytes(b"actual bytes\n")

    evidence = run(EnvWorkingTreeProbe(_RealEnv(str(repo)), workspace=str(repo)).diff())

    assert not sentinel.exists()
    assert "+actual bytes" in evidence
    assert "forged" not in evidence

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
@pytest.mark.parametrize("subdirectory", [False, True])
def test_real_git_binary_repository_evidence_keeps_aggregate_limit(tmp_path, monkeypatch, subdirectory):
    repo = tmp_path / "repo with spaces"
    workspace = repo / "package" if subdirectory else repo
    workspace.mkdir(parents=True)
    tracked = workspace / "tracked [1]*? \u96ea.bin"
    tracked.write_bytes(b"\0initial\xff")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "add", ".")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    tracked.write_bytes(b"\0modified\xff")
    (repo / "root [1]*? \u96ea.bin").write_bytes(b"\0created\xff")
    probe = EnvWorkingTreeProbe(_RealEnv(str(workspace)))

    evidence = run(probe.diff())

    prefix = "package/" if subdirectory else ""
    assert set(patch_paths(evidence)) == {prefix + tracked.name, "root [1]*? \u96ea.bin"}
    assert evidence.count("GIT binary patch") == 2
    monkeypatch.setattr("opencollab.adapters.working_tree.MAX_WORKING_TREE_DIFF_CHARS", len(evidence) - 1)
    with pytest.raises(RuntimeError, match="aggregate evidence limit"):
        run(probe.diff())
