"""Search command construction and backend failures through the tool runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencollab.adapters.tools.fs import GrepTool
from opencollab.application.tool_execution import ToolRuntime
from tests.support.tool_runtime_test_support import FakeEnv, SpySafetyPolicy, run


def test_grep_tool_requires_environment():
    runtime = ToolRuntime(environment=None, safety_policy=None, permission_policy=None)

    result = run(GrepTool().execute_with_runtime({"pattern": "needle"}, runtime))

    assert result == "Error: no execution environment available."

def test_grep_tool_routes_search_path_through_safety_policy():
    env = FakeEnv(stdout="src/app.py:1:needle\n")
    safety = SpySafetyPolicy()
    runtime = ToolRuntime(environment=env, safety_policy=safety, permission_policy=None)

    result = run(
        GrepTool().execute_with_runtime(
            {"pattern": "needle", "path": "../outside", "glob": "*.py", "max_results": 3},
            runtime,
        )
    )

    assert result == "src/app.py:1:needle"
    assert safety.path_calls == ["../outside"]
    assert len(env.exec_calls) == 1
    cmd, timeout = env.exec_calls[0]
    assert "needle" in cmd
    assert "../outside" in cmd
    assert timeout == 30

def test_grep_tool_rejects_non_integer_max_results_before_exec():
    env = FakeEnv(stdout="src/app.py:1:needle\n")
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    result = run(
        GrepTool().execute_with_runtime(
            {"pattern": "needle", "max_results": "1; touch pwned"},
            runtime,
        )
    )

    assert result == "Error: max_results must be an integer."
    assert env.exec_calls == []

def test_grep_tool_applies_max_results_globally():
    env = FakeEnv(
        stdout="\n".join(f"src/file_{index}.py:1:needle" for index in range(6))
    )
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    result = run(
        GrepTool().execute_with_runtime(
            {"pattern": "needle", "max_results": 2},
            runtime,
        )
    )

    lines = [line for line in result.splitlines() if line]
    assert lines[:2] == [
        "src/file_0.py:1:needle",
        "src/file_1.py:1:needle",
    ]
    # A visible cap distinguishes exact match counts from truncated results.
    assert lines[2:] == [
        "... [4 more matching lines not shown; raise max_results, or narrow "
        "the search with path or glob] ..."
    ]

def test_grep_tool_says_nothing_extra_when_nothing_was_capped():
    env = FakeEnv(
        stdout="\n".join(f"src/file_{index}.py:1:needle" for index in range(2))
    )
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    result = run(
        GrepTool().execute_with_runtime(
            {"pattern": "needle", "max_results": 2},
            runtime,
        )
    )

    assert result.splitlines() == [
        "src/file_0.py:1:needle",
        "src/file_1.py:1:needle",
    ]

def test_grep_tool_does_not_fallback_after_normal_rg_no_match():
    class NoMatchEnv(FakeEnv):
        async def exec_cmd(self, cmd: str, timeout: float = 120.0):
            self.exec_calls.append((cmd, timeout))
            return SimpleNamespace(returncode=1, stdout="", stderr="")

    env = NoMatchEnv()
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    result = run(GrepTool().execute_with_runtime({"pattern": "absent"}, runtime))

    assert result == "No matches found for pattern: absent"
    assert len(env.exec_calls) == 1
    assert "grep -r" not in env.exec_calls[0][0]

@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (2, "regex parse error", "regex parse error"),
        (2, "path does not exist", "path does not exist"),
        (-1, "Command timed out after 30s", "timed out"),
    ],
)
def test_grep_tool_reports_backend_failures(returncode, stderr, expected):
    class FailedSearchEnv(FakeEnv):
        async def exec_cmd(self, cmd: str, timeout: float = 120.0):
            self.exec_calls.append((cmd, timeout))
            return SimpleNamespace(
                returncode=returncode,
                stdout="",
                stderr=stderr,
                stdout_truncated=False,
                stderr_truncated=False,
            )

    env = FailedSearchEnv()
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    result = run(GrepTool().execute_with_runtime({"pattern": "["}, runtime))

    assert result.startswith("Error: rg search failed")
    assert expected in result
    assert "No matches" not in result
    assert "2>/dev/null" not in env.exec_calls[0][0]

def test_grep_tool_includes_hidden_project_paths_with_noise_exclusions():
    env = FakeEnv(stdout=".github/workflows/ci.yml:1:needle\n")
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    result = run(GrepTool().execute_with_runtime({"pattern": "needle"}, runtime))

    assert result == ".github/workflows/ci.yml:1:needle"
    command = env.exec_calls[0][0]
    assert "rg -n --hidden " in command
    assert "-g '!.git/**'" in command
    assert "-g '!.venv/**'" in command
    assert "-g '!.opencollab/**'" in command

def test_grep_tool_terminates_options_before_pattern_and_path():
    env = FakeEnv(stdout="")
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    run(
        GrepTool().execute_with_runtime(
            {"pattern": "--pre=touch pwned", "path": "--debug"},
            runtime,
        )
    )

    cmd, _ = env.exec_calls[0]
    assert "rg -n --hidden" in cmd
    assert "-- '--pre=touch pwned' --debug" in cmd

def test_grep_tool_fallback_uses_ere_and_skips_session_dir():
    # The grep fallback needs ERE alternation and must exclude logged patterns in .opencollab.
    class MissingRgEnv(FakeEnv):
        async def exec_cmd(self, cmd: str, timeout: float = 120.0):
            self.exec_calls.append((cmd, timeout))
            if len(self.exec_calls) == 1:
                return SimpleNamespace(returncode=127, stdout="", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="")

    env = MissingRgEnv(stdout="")
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    run(
        GrepTool().execute_with_runtime(
            {"pattern": "Scheduler|RowStatus"}, runtime
        )
    )

    assert len(env.exec_calls) == 2
    cmd, _ = env.exec_calls[1]
    assert cmd.startswith("grep -rEn")
    assert " grep -rn " not in cmd
    assert "--exclude-dir=.opencollab" in cmd

