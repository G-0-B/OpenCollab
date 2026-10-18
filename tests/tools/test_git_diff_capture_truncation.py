"""Git tools disclose incomplete environment captures within output limits."""

from __future__ import annotations

import subprocess

import pytest

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.git_diff import GitDiffTool
from opencollab.application.tool_execution import ToolRuntime


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True)

@pytest.mark.parametrize("mode", ["unstaged", "staged", "untracked"])
@pytest.mark.parametrize("max_chars", [8000, 2_000_000])
async def test_large_git_diff_discloses_capture_loss(tmp_path, mode, max_chars):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    target = tmp_path / "migration.sql"
    target.write_text("-- migration\n", encoding="utf-8")
    _git(tmp_path, "add", "migration.sql")
    _git(tmp_path, "commit", "-qm", "initial fixture")
    if mode == "untracked":
        target = tmp_path / "new.sql"
    rows = [f"INSERT INTO items VALUES ({i}, 'record-{i:05d}-description');" for i in range(22_000)]
    rows[0] += " -- CAPTURE_HEAD"
    rows[len(rows) // 2] += " -- CAPTURE_MIDDLE"
    rows[-1] += " -- CAPTURE_TAIL"
    target.write_text("-- migration\n" + "\n".join(rows) + "\n", encoding="utf-8")
    if mode == "staged":
        _git(tmp_path, "add", "migration.sql")
    environment = LocalEnvironment(str(tmp_path))
    runtime = ToolRuntime(environment=environment, safety_policy=None, permission_policy=None)
    try:
        result = await GitDiffTool(max_diff_chars=max_chars).execute_with_runtime(
            {"staged": mode == "staged", "include_status": False}, runtime,
        )
    finally:
        await environment.cleanup()
    assert "CAPTURE_HEAD" in result and "CAPTURE_TAIL" in result
    assert "CAPTURE_MIDDLE" not in result
    assert "truncated" in result.lower()
    if max_chars == 2_000_000 and mode != "untracked":
        assert "bytes omitted" in result
    assert len(result.partition(":\n")[2]) <= max_chars

@pytest.mark.parametrize("dropped_bytes", [0, 271])
async def test_status_capture_loss_is_visible_and_bounded(dropped_bytes):
    class Environment:
        async def exec_cmd(self, command, **kwargs):
            if command.endswith("status --short"):
                return ExecResult(
                    0, " M first.py\n M last.py\n", "", stdout_truncated=True,
                    stdout_dropped_bytes=dropped_bytes,
                )
            if command.endswith("rev-parse --verify HEAD"):
                return ExecResult(0, "fixture-head\n", "")
            return ExecResult(0, "", "")

    result = await GitDiffTool(max_status_chars=300).execute_with_runtime(
        {}, ToolRuntime(environment=Environment(), safety_policy=None, permission_policy=None),
    )
    status = result.partition("Status (git status --short):\n")[2].partition("\n\ndiff vs HEAD:")[0]
    assert "truncated" in status.lower()
    assert "first.py" in status and "last.py" in status
    assert len(status) <= 300
    if dropped_bytes:
        assert f"{dropped_bytes} bytes omitted" in status
