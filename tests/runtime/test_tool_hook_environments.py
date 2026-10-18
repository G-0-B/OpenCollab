"""Tool hook routing preserves environment execution and process ownership."""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from opencollab.adapters import hooks as hooks_adapter
from opencollab.adapters._env_process import ProcessCleanupError, ProcessResult
from opencollab.adapters.env import DockerEnvironment, LocalEnvironment
from opencollab.adapters.hooks import ShellHookRunner
from opencollab.bootstrap import build_runtime_context, build_scheduler
from opencollab.domain.hooks import HookSpec
from tests.support.docker_env_test_support import CONTAINER_ID, FakeDocker, _patch, _result


def _runner(environment, spec, workspace):
    scheduler = SimpleNamespace(agent_environment=lambda aid: environment if aid == 1 else None)
    return ShellHookRunner((spec,), scheduler=scheduler, workspace=workspace)

async def test_tool_hook_uses_container_exec_and_delivers_stdin_env(monkeypatch, tmp_path):
    remote_files = tmp_path / "container files"
    remote_files.mkdir()
    recorded = remote_files / "received.json"
    program = (
        "import json,os,sys; from pathlib import Path; "
        f"Path({str(recorded)!r}).write_text(json.dumps({{"
        "'payload': json.load(sys.stdin), "
        "'variables': {name: os.environ[name] for name in "
        "('OPENCOLLAB_HOOK_EVENT','OPENCOLLAB_TOOL','OPENCOLLAB_AID')}, "
        "'prefix': os.environ['HOOK_PREFIX'], 'cwd': os.getcwd()}))"
    )
    spec = HookSpec(event="PreToolUse", action_type="command",
                    command=f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}", timeout=7)
    payload = {"hook_event_name": "PreToolUse", "tool": "file_write", "aid": 1,
               "args": {"content": "literal '$(touch unexpected)'\n\u4e2d\u6587"}}

    def respond(command, kwargs):
        if command[1] == "inspect":
            return _result(stdout=f"{CONTAINER_ID}\t/probe\ttrue\n".encode())
        assert command[1] == "exec"
        assert command[command.index("-w") + 1] == "/container repo"
        assert kwargs["timeout"] == 7
        completed = subprocess.run(["bash", "-c", command[-1]], cwd=remote_files, capture_output=True, check=True)
        return ProcessResult(completed.returncode, completed.stdout, completed.stderr)

    async def reject_host_execution(*args, **kwargs):
        raise AssertionError("container tool hook was routed to the host")

    fake = FakeDocker(respond)
    _patch(monkeypatch, fake)
    monkeypatch.setattr(hooks_adapter, "run_process", reject_host_execution)
    environment = DockerEnvironment(container_id=CONTAINER_ID, workspace="/container repo",
                                    exec_workdir="/container repo", command_prefix="export HOOK_PREFIX=activated")
    assert (await _runner(environment, spec, tmp_path).fire("PreToolUse", payload)).allow
    receipt = json.loads(recorded.read_text())
    assert receipt == {
        "payload": payload,
        "variables": {"OPENCOLLAB_HOOK_EVENT": "PreToolUse", "OPENCOLLAB_TOOL": "file_write", "OPENCOLLAB_AID": "1"},
        "prefix": "activated", "cwd": str(remote_files),
    }
    assert [call[0][1] for call in fake.calls] == ["inspect", "exec"]
    assert not (remote_files / "unexpected").exists()

async def test_container_hook_timeout_recovers_inner_group_before_next_hook(monkeypatch, tmp_path, caplog):
    calls = 0

    def respond(command, kwargs):
        nonlocal calls
        if command[1] == "inspect":
            return _result(stdout=f"{CONTAINER_ID}\t/probe\ttrue\n".encode())
        calls += 1
        if calls == 1:
            assert kwargs["timeout"] == 0.5
            return asyncio.TimeoutError()
        return _result()

    fake = FakeDocker(respond)
    _patch(monkeypatch, fake)
    environment = DockerEnvironment(container_id=CONTAINER_ID, exec_workdir="/repo")
    first = HookSpec(event="PostToolUse", action_type="command", command="sleep 60", timeout=0.5)
    second = HookSpec(event="PostToolUse", action_type="command", command="printf next")
    runner = ShellHookRunner((first, second), scheduler=SimpleNamespace(agent_environment=lambda aid: environment))
    await runner.fire("PostToolUse", {"hook_event_name": "PostToolUse", "tool": "file_write", "aid": 1})
    assert calls == 3
    assert "opencollab-cancel" in fake.calls[2][0]
    assert "printf next" in fake.calls[3][0][-1]
    assert "timed out after 0.50s" in caplog.text
    assert runner.cleanup_quiesced and not environment.revoked

async def test_tool_hook_cleanup_failure_is_retained(tmp_path):
    async def fail(command, timeout):
        raise ProcessCleanupError("command group survived cleanup")

    environment = SimpleNamespace(exec_cmd=fail)
    spec = HookSpec(event="PostToolUse", action_type="command", command="sleep 60")
    runner = _runner(environment, spec, tmp_path)
    with pytest.raises(ProcessCleanupError, match="survived cleanup"):
        await runner.fire("PostToolUse", {"hook_event_name": "PostToolUse", "tool": "file_write", "aid": 1})
    assert not runner.cleanup_quiesced
    assert isinstance(runner.cleanup_error, ProcessCleanupError)

@pytest.mark.parametrize("ending", ["exit 3", "sleep 60; printf unexpected >> sequence.txt"])
async def test_local_hook_exit_and_timeout_finish_before_next_hook(tmp_path, caplog, ending):
    environment = LocalEnvironment(str(tmp_path))
    first = HookSpec(event="PostToolUse", action_type="command",
                     command="printf 'first\\n' >> sequence.txt; " + ending, timeout=0.2)
    second = HookSpec(event="PostToolUse", action_type="command", command="printf 'second\\n' >> sequence.txt")
    runner = ShellHookRunner((first, second), scheduler=SimpleNamespace(agent_environment=lambda aid: environment))
    try:
        payload = {"hook_event_name": "PostToolUse", "tool": "file_write", "aid": 1}
        assert (await runner.fire("PostToolUse", payload)).allow
        assert (tmp_path / "sequence.txt").read_text().splitlines() == ["first", "second"]
        assert "exited 3" in caplog.text if ending == "exit 3" else "timed out after 0.20s" in caplog.text
        assert runner.cleanup_quiesced and not environment.revoked
    finally:
        await environment.cleanup()

async def test_unknown_tool_source_skips_command_and_logs(tmp_path, caplog):
    spec = HookSpec(event="PostToolUse", action_type="command", command="touch unexpected")
    runner = _runner(None, spec, tmp_path)
    assert (await runner.fire("PostToolUse", {"hook_event_name": "PostToolUse", "tool": "file_write", "aid": 5})).allow
    assert "environment unavailable for aid 5" in caplog.text
    assert not (tmp_path / "unexpected").exists()

async def test_registered_environment_is_unavailable_after_cleanup(tmp_path):
    team = tmp_path / "team.yaml"
    team.write_text("roles:\n  lead:\n    prompt: Complete a task.\n    tools: []\ntopology:\n  lead: []\n")
    context = build_runtime_context(str(tmp_path), {
        "model": "fixture-model", "provider": "openai", "api_key": "controlled-test",  # pragma: allowlist secret
        "base_url": "https://controlled.invalid/v1", "budget": 10000,
    }, trace=False)
    scheduler = build_scheduler(context, use_worktrees=False, interactive=False, auto_save=False, team_config_path=team)
    environment = scheduler.agent_environment(0)
    assert isinstance(environment, LocalEnvironment)
    assert scheduler.agent_environment(1) is None
    assert scheduler.agent_environment(True) is None
    await scheduler.cleanup()
    assert scheduler.agent_environment(0) is None
