
import pytest

from opencollab.adapters.safety import SandboxInterceptor
from opencollab.adapters.tools import human
from opencollab.adapters.tools.base import Tool
from opencollab.adapters.tools.bash import BashTool
from opencollab.adapters.tools.fs import FileReadTool, GrepTool
from opencollab.adapters.tools.git_diff import GitDiffTool
from opencollab.adapters.tools.human import AskUserTool
from opencollab.application.tool_execution import ToolRuntime
from tests.support.paths import REPO_ROOT
from tests.support.tool_execution_test_support import AlwaysAllowPermissionPolicy as FakePermissionPolicy
from tests.support.tool_runtime_test_support import FakeEnv, FalseyFakeEnv, SpySafetyPolicy, run


def test_tool_runtime_constructs_directly_with_env_safety_and_permission():
    env = FakeEnv()
    safety = SpySafetyPolicy()
    permission_policy = FakePermissionPolicy()

    runtime = ToolRuntime(
        environment=env,
        safety_policy=safety,
        permission_policy=permission_policy,
    )

    assert runtime.environment is env
    assert runtime.safety_policy is safety
    assert runtime.permission_policy is permission_policy
    assert runtime.confirm_fn() == permission_policy.confirm

def test_tool_runtime_confirm_fn_returns_none_without_permission_policy():
    runtime = ToolRuntime(environment=None, safety_policy=None, permission_policy=None)
    assert runtime.confirm_fn() is None

@pytest.mark.parametrize("value", [0, -1, False, 1.5])
def test_bash_rejects_invalid_output_limits_at_construction(value):
    with pytest.raises(ValueError, match="positive integer"):
        BashTool(max_output_chars=value)

@pytest.mark.parametrize(
    ("tool_type", "parameter"),
    [
        (FileReadTool, "max_read_chars"),
        (GrepTool, "max_grep_chars"),
        (GitDiffTool, "max_diff_chars"),
        (GitDiffTool, "max_status_chars"),
    ],
)
@pytest.mark.parametrize("value", [0, -1, False, 1.5])
def test_public_tools_reject_invalid_output_limits_at_construction(
    tool_type,
    parameter,
    value,
):
    with pytest.raises(ValueError, match="positive integer"):
        tool_type(**{parameter: value})

def test_bash_tool_without_env_returns_existing_error():
    runtime = ToolRuntime(environment=None, safety_policy=None, permission_policy=None)

    result = run(BashTool().execute_with_runtime({"command": "pwd"}, runtime))

    assert result == "Error: no execution environment available."

def test_bash_tool_accepts_falsey_environment():
    env = FalseyFakeEnv(stdout="ok\n")
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)

    result = run(BashTool().execute_with_runtime({"command": "echo ok"}, runtime))

    assert env.exec_calls == [("echo ok", 120.0)]
    assert "stdout:\nok" in result

def test_bash_tool_passes_permission_confirm_fn_to_safety_check():
    env = FakeEnv(stdout="ok\n")
    safety = SpySafetyPolicy()
    permission_policy = FakePermissionPolicy()
    runtime = ToolRuntime(environment=env, safety_policy=safety, permission_policy=permission_policy)

    result = run(BashTool().execute_with_runtime({"command": "echo ok"}, runtime))

    assert safety.cmd_calls == [("echo ok", permission_policy.confirm)]
    assert env.exec_calls == [("echo ok", 120.0)]
    assert "stdout:\nok" in result

def test_bash_tool_respects_custom_timeout():
    env = FakeEnv(stdout="ok\n")
    safety = SpySafetyPolicy()
    runtime = ToolRuntime(environment=env, safety_policy=safety, permission_policy=None)

    run(BashTool().execute_with_runtime({"command": "echo ok", "timeout": 7}, runtime))

    assert env.exec_calls == [("echo ok", 7)]

def test_bash_tool_preserves_blocked_command_behavior(tmp_path):
    env = FakeEnv()
    sandbox = SandboxInterceptor(str(tmp_path))
    runtime = ToolRuntime(environment=env, safety_policy=sandbox, permission_policy=None)

    with pytest.raises(PermissionError):
        run(BashTool().execute_with_runtime({"command": "rm -rf /"}, runtime))

    assert env.exec_calls == []

class FakeAskPolicy:
    """Records the question and returns a canned answer (no terminal I/O)."""

    def __init__(self, answer: str = "Use the smaller patch."):
        self.answer = answer
        self.questions: list[str] = []

    async def ask(self, question: str) -> str:
        self.questions.append(question)
        return self.answer

def test_ask_user_tool_non_interactive_fallback():
    runtime = ToolRuntime(environment=None, safety_policy=None, permission_policy=None)

    result = run(AskUserTool().execute_with_runtime({"question": "Proceed?"}, runtime))

    assert result == "Running in non-interactive mode. Make your own best judgment and proceed."

def test_ask_user_tool_routes_through_ask_port():
    ask_policy = FakeAskPolicy()
    runtime = ToolRuntime(
        environment=None,
        safety_policy=None,
        permission_policy=None,
        ask_policy=ask_policy,
    )

    result = run(AskUserTool().execute_with_runtime({"question": "Proceed?"}, runtime))

    assert ask_policy.questions == ["Proceed?"]
    assert result == "Use the smaller patch."

def test_ask_user_tool_declines_on_ask_port_eof():
    class EofAskPolicy:
        async def ask(self, question: str) -> str:
            raise EOFError

    runtime = ToolRuntime(
        environment=None,
        safety_policy=None,
        permission_policy=None,
        ask_policy=EofAskPolicy(),
    )

    result = run(AskUserTool().execute_with_runtime({"question": "Proceed?"}, runtime))

    assert result == "User declined to answer."

def test_ask_user_tool_uses_prompt_when_no_ask_port_but_permission_exists(monkeypatch):
    runtime = ToolRuntime(
        environment=None,
        safety_policy=None,
        permission_policy=FakePermissionPolicy(),
    )
    prompts = []

    def fake_prompt(question: str) -> str:
        prompts.append(question)
        return "Use the smaller patch."

    monkeypatch.setattr(human, "_prompt_user", fake_prompt)

    result = run(AskUserTool().execute_with_runtime({"question": "Proceed?"}, runtime))

    assert prompts == ["Proceed?"]
    assert result == "Use the smaller patch."

def test_base_tool_default_execute_with_runtime_raises_not_implemented():
    runtime = ToolRuntime(environment=None, safety_policy=None, permission_policy=None)

    with pytest.raises(NotImplementedError):
        run(Tool().execute_with_runtime({}, runtime))

def test_base_tool_does_not_define_legacy_execute():
    assert "execute" not in Tool.__dict__

def test_tool_modules_do_not_import_inner_layers_or_concrete_sandbox():
    package_root = REPO_ROOT
    tool_files = [
        package_root / "opencollab/adapters/tools/base.py",
        package_root / "opencollab/adapters/tools/bash.py",
        package_root / "opencollab/adapters/tools/apply_patch.py",
        package_root / "opencollab/adapters/tools/fs.py",
        package_root / "opencollab/adapters/tools/git_diff.py",
        package_root / "opencollab/adapters/tools/human.py",
    ]

    for path in tool_files:
        source = path.read_text(encoding="utf-8")
        assert "opencollab.core.session" not in source
        assert "opencollab.bootstrap" not in source
        assert "opencollab.adapters.safety" not in source
        assert "SandboxInterceptor" not in source
