"""Git global options retain the existing destructive-command confirmation."""

import shlex
import subprocess

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.safety import SandboxInterceptor
from opencollab.adapters.tools.bash import BashTool
from opencollab.application.tool_execution import ToolRuntime


@pytest.mark.parametrize("command", [
    "git -C repo reset --hard HEAD",
    "git -Crepo -c core.quotePath=false reset --hard",
    "git --no-pager -c 'core.commentChar=#' -C 'repo with spaces' reset --hard",
    "git --git-dir=.git --work-tree=. clean -fd",
    "git --config-env=core.editor=EDITOR push --force-with-lease",
    "git --namespace=test --no-optional-locks push origin --force",
    "cd repo && git -C . reset --hard HEAD",
    "git clean --force",
    "git -C repo clean --force",
    "git clean -d -f",
    "git clean -d --force",
    "git reset --quiet --hard HEAD",
    "git reset -q --hard HEAD",
    "git -C repo reset --quiet --hard HEAD",
])
def test_git_global_options_keep_risky_subcommand_visible(tmp_path, command):
    assert SandboxInterceptor(str(tmp_path)).is_risky(command)

@pytest.mark.parametrize("command", [
    "git -C repo status --short",
    "git -c core.quotePath=false diff",
    "git --git-dir=.git --work-tree=. log --oneline",
    "printf '%s' 'git -C repo reset --hard'",
    "printf '%s' 'git clean --force'",
    "printf '%s' 'git reset --quiet --hard HEAD'",
    "git clean --dry-run -- --force",
    "git -C repo clean --dry-run -- --force",
    "git reset --quiet --soft HEAD",
    "git -C repo status --porcelain=v1",
])
def test_ordinary_git_global_options_need_no_confirmation(tmp_path, command):
    assert not SandboxInterceptor(str(tmp_path)).is_risky(command)

@pytest.mark.parametrize("options, operation", [
    ("-C .", "reset --hard HEAD"),
    ("-c core.quotePath=false -C .", "reset --hard HEAD"),
    ("--git-dir=.git --work-tree=.", "clean -fd"),
    ("", "clean --force"),
    ("-C .", "clean --force"),
    ("", "clean -d -f"),
    ("-C .", "clean -d --force"),
    ("", "reset --quiet --hard HEAD"),
    ("", "reset -q --hard HEAD"),
    ("-C .", "reset --quiet --hard HEAD"),
])
async def test_denied_git_operation_preserves_real_workspace(tmp_path, options, operation):
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    for args in [
        ["init", "-q"],
        ["config", "user.name", "OpenCollab Tests"],
        ["config", "user.email", "tests@example.invalid"],
    ]:
        subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True)
    tracked = workspace / "tracked.txt"
    tracked.write_text("committed\n")
    subprocess.run(["git", "-C", str(workspace), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(workspace), "-c", "commit.gpgsign=false", "commit", "-qm",
         "\u521d\u59cb\u5316\u6d4b\u8bd5\u4ed3\u5e93"],
        check=True, capture_output=True,
    )
    tracked.write_text("uncommitted work\n")
    untracked = workspace / "draft.txt"
    untracked.write_text("user draft\n")

    class Deny:
        calls = 0

        async def confirm(self, prompt):
            self.calls += 1
            return False

    permission = Deny()
    environment = LocalEnvironment(str(workspace))
    runtime = ToolRuntime(
        environment=environment, safety_policy=SandboxInterceptor(str(workspace)),
        permission_policy=permission,
    )
    try:
        with pytest.raises(PermissionError, match="User denied risky command"):
            await BashTool().execute_with_runtime(
                {"command": f"git {options} {operation}"}, runtime,
            )
        assert permission.calls == 1
        assert tracked.read_text() == "uncommitted work\n"
        assert untracked.read_text() == "user draft\n"
        status = await BashTool().execute_with_runtime(
            {"command": f"git -C {shlex.quote(str(workspace))} status --short"}, runtime,
        )
        assert "tracked.txt" in status
        assert permission.calls == 1
    finally:
        await environment.cleanup()
