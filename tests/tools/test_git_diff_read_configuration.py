"""Read-only Git operations use the existing configuration restrictions."""

import pytest

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters.tools import git_diff
from opencollab.application.tool_execution import ToolRuntime


class RecordingGitEnvironment:
    def __init__(self, *, denied_call=None, unborn=False):
        self.calls = []
        self.denied_call = denied_call
        self.unborn = unborn

    async def exec_cmd(self, command, timeout):
        self.calls.append(command)
        if len(self.calls) == self.denied_call:
            return ExecResult(125, "", "repository configuration rejected")
        if "rev-parse --verify HEAD" in command:
            return ExecResult(128 if self.unborn else 0, "", "")
        if "--is-inside-work-tree" in command:
            return ExecResult(0, "true\n", "")
        if "hash-object" in command:
            return ExecResult(0, "empty-tree\n", "")
        if "--porcelain=v1" in command:
            return ExecResult(0, "?? new file.txt\0", "")
        if "--show-toplevel" in command:
            return ExecResult(0, "/workspace\n", "")
        if "status --short" in command:
            return ExecResult(0, " M app.py\n", "")
        return ExecResult(1 if "--no-index" in command else 0, "+updated\n", "")

@pytest.fixture
def recorded_guard(monkeypatch):
    calls = []

    def guard():
        calls.append(True)
        return "safe-git", "validate-repository; "

    monkeypatch.setattr(git_diff, "_git_command_and_config_guard", guard, raising=False)
    return calls

@pytest.mark.parametrize("params,unborn", [({}, False), ({}, True), ({"staged": True}, False),
                                         ({"stat_only": True}, False), ({"path": "app.py"}, False)])
async def test_all_git_reads_use_existing_configuration_guard(recorded_guard, params, unborn):
    env = RecordingGitEnvironment(unborn=unborn)
    result = await git_diff.GitDiffTool().execute_with_runtime(
        params, ToolRuntime(environment=env, safety_policy=None, permission_policy=None)
    )

    assert "+updated" in result
    assert recorded_guard
    assert env.calls
    assert all(command.startswith("validate-repository; safe-git --no-pager ") for command in env.calls)
    diffs = [command for command in env.calls if " diff " in command]
    assert diffs
    assert all("--no-ext-diff" in command and "--no-textconv" in command for command in diffs)

@pytest.mark.parametrize("denied_call", range(1, 7))
async def test_configuration_refusal_stops_at_the_rejected_read(recorded_guard, denied_call):
    env = RecordingGitEnvironment(denied_call=denied_call)
    result = await git_diff.GitDiffTool().execute_with_runtime(
        {}, ToolRuntime(environment=env, safety_policy=None, permission_policy=None)
    )

    assert "repository configuration rejected" in result
    assert len(env.calls) == denied_call
    assert "working tree clean" not in result
