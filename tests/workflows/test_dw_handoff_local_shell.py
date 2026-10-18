"""The scripted handoff example connects explicit host shell permission."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencollab.bootstrap.workflow_runtime import WorkflowSessionFactory
from tests.support.paths import REPO_ROOT


@pytest.fixture
def handoff():
    source = REPO_ROOT / "examples/dw-handoff/workflows/handoff.py"
    spec = importlib.util.spec_from_file_location("handoff_shell_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class RecordingContext:
    def __init__(self):
        self.shell_requirements = []
        self.isolation = []

    async def phase(self, title):
        pass

    async def agent(self, prompt, *, label, tools, isolation, **kwargs):
        self.isolation.append(isolation)
        shell = next(tool for tool in tools if tool.name == "bash")
        self.shell_requirements.append(shell.require_process_isolation)
        if label == "analyst":
            return {"root_cause": "fixture"}
        if label.startswith("coder"):
            return {"commit": "fixture-commit", "summary": "fixture"}
        return {"adopted_commit": "fixture-commit", "verdict": "PASS", "findings": "fixture"}

    def tokens_spent(self):
        return 0

@pytest.mark.parametrize("permission", [None, False, True])
async def test_workflow_passes_explicit_permission_to_every_role(handoff, permission):
    ctx = RecordingContext()
    args = {"goal": "offline fixture"}
    if permission is not None:
        args["allow_unisolated_shell"] = permission

    result = await handoff.dw_handoff(ctx, args)

    assert result["status"] == "ok"
    assert ctx.shell_requirements == [permission is not True] * 3
    assert ctx.isolation == [True] * 3

@pytest.mark.parametrize("permission", ["true", "false", 1, None, []])
async def test_workflow_requires_a_boolean_permission(handoff, permission):
    ctx = RecordingContext()

    result = await handoff.dw_handoff(ctx, {"goal": "fixture", "allow_unisolated_shell": permission})

    assert result == {"status": "error", "error": "allow_unisolated_shell must be a boolean"}
    assert ctx.shell_requirements == []

async def test_explicit_permission_executes_commit_and_checkout_in_real_worktrees(handoff, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "base"],
        cwd=workspace, check=True,
    )
    factory = WorkflowSessionFactory(
        model="offline-model", provider="openai", api_key=None, base_url=None, workspace=str(workspace),
    )

    class LocalHandoffContext(RecordingContext):
        def __init__(self):
            super().__init__()
            self.environments = []
            self.commit = None

        async def agent(self, prompt, *, label, tools, isolation, **kwargs):
            assert isolation is True
            env = await factory.acquire_isolated_env(label=label)
            self.environments.append(env)
            assert env.process_isolated is False
            shell = next(tool for tool in tools if tool.name == "bash")
            runtime = SimpleNamespace(environment=env, safety_policy=None)
            if label == "analyst":
                observed = await shell.execute_with_runtime({"command": "git rev-parse --is-inside-work-tree"}, runtime)
                assert "true" in observed, observed
                return {"root_cause": "fixture"}
            if label.startswith("coder"):
                await env.write_file("patch.txt", "delivered\n")
                observed = await shell.execute_with_runtime(
                    {"command": "git add patch.txt && git -c user.name=Fixture "
                     "-c user.email=fixture@example.test commit -qm patch && git rev-parse HEAD"},
                    runtime,
                )
                self.commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=env.workspace, text=True,
                ).strip()
                assert self.commit in observed, observed
                return {"commit": self.commit, "summary": "fixture patch"}
            observed = await shell.execute_with_runtime({"command": f"git checkout -q {self.commit}"}, runtime)
            assert not observed.startswith("Error"), observed
            assert await env.read_file("patch.txt") == "delivered\n"
            adopted = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=env.workspace, text=True).strip()
            return {"adopted_commit": adopted, "verdict": "PASS", "findings": "fixture verified"}

    ctx = LocalHandoffContext()
    try:
        result = await handoff.dw_handoff(ctx, {"goal": "fixture", "allow_unisolated_shell": True})
        assert result["rounds"][0]["handoff_taken"] is True
        assert len({env.workspace for env in ctx.environments}) == 3
        assert not (workspace / "patch.txt").exists()
    finally:
        await factory.release_isolated_envs()
    assert all(not Path(env.workspace).exists() for env in ctx.environments)
