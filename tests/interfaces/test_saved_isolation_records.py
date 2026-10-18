"""Saved runs isolate source code without copying their live records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from opencollab import OpenCollab
from opencollab.adapters import worktree_pool
from opencollab.adapters.cli import workflow as workflow_cli
from opencollab.bootstrap.workflow_runtime import WorkflowSessionFactory
from opencollab.environments import attach_container
from tests.support.provider_sdk_http import http_response, install_sdk_transport
from tests.support.worktree_delivery_support import CONTAINER_ID
from tests.support.worktree_delivery_support import worktree_transport as worktree_transport


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True)

def repository(path):
    path.mkdir()
    git(path, "init", "-q")
    (path / "app.py").write_text("answer = 4\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "commit", "-qm", "\u521d\u59cb\u5316\u9694\u79bb\u6d4b\u8bd5\u4ed3\u5e93")
    return path

def response(request, calls, *, tool_calls=None, text=None):
    body = json.loads(request.content)
    calls.append(body)
    return http_response(request, 200, json={
        "id": f"saved-{len(calls)}", "object": "chat.completion", "created": 1,
        "model": body["model"], "choices": [{"index": 0,
            "message": {"role": "assistant", "content": text,
                        **({"tool_calls": tool_calls} if tool_calls else {})},
            "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })

def tool(name, arguments, identity="result"):
    return {"id": identity, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}

def structured_transport(monkeypatch):
    calls = []
    install_sdk_transport(monkeypatch, "openai", lambda request: response(
        request, calls, tool_calls=[tool("structured_output", {"answer": 4})],
    ))
    return calls

def client(workspace):
    return OpenCollab(
        workspace, model="offline-model", provider="openai",
        api_key="local-fixture",  # pragma: allowlist secret
        base_url="https://offline.example.invalid/v1",
    )

async def isolated_answer(ctx, _args):
    return await ctx.agent("Return answer four", isolation=True, tools=[], schema={
        "type": "object", "properties": {"answer": {"type": "integer"}},
        "required": ["answer"], "additionalProperties": False,
    })

@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("nested", [False, True])
async def test_public_sdk_saved_workflow_uses_real_git_worktree(
    tmp_path, monkeypatch, trace, nested,
):
    repo = repository(tmp_path / "repo")
    workspace = repo / "package" if nested else repo
    if nested:
        workspace.mkdir()
        (workspace / "marker.txt").write_text("source\n")
        git(repo, "add", ".")
        git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
            "commit", "-qm", "\u6dfb\u52a0\u5b50\u76ee\u5f55\u6d4b\u8bd5\u6e90\u7801")
    artifacts = workspace / ".opencollab" / "sessions" / "current"
    calls = structured_transport(monkeypatch)
    result = await client(workspace).workflow(
        isolated_answer, artifacts=artifacts, trace=trace, budget=1000,
        max_steps=2, limit_mode="explicit",
    )
    assert result.output == {"answer": 4}
    assert len(calls) == 1
    assert result.tokens == 15
    assert (artifacts / "workflow.json").is_file()
    assert (artifacts / "orchestration.jsonl").exists() is trace
    assert len(git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2
    assert (repo / "app.py").read_text() == "answer = 4\n"

@pytest.mark.parametrize("dirty", ["staged", "unstaged", "untracked", "old-run"])
async def test_public_sdk_saved_workflow_preserves_dirty_source(
    tmp_path, monkeypatch, dirty,
):
    workspace = repository(tmp_path / "repo")
    user_file = workspace / "app.py"
    if dirty in {"staged", "unstaged"}:
        user_file.write_text("answer = 5\n")
        if dirty == "staged":
            git(workspace, "add", "app.py")
    else:
        user_file = workspace / (
            ".opencollab/sessions/older/user.json" if dirty == "old-run" else "user.txt"
        )
        user_file.parent.mkdir(parents=True, exist_ok=True)
        user_file.write_text("user data\n")
    before = git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    contents = user_file.read_bytes()
    calls = structured_transport(monkeypatch)
    artifacts = workspace / ".opencollab" / "sessions" / "current"
    result = await client(workspace).workflow(
        isolated_answer, artifacts=artifacts, budget=1000, max_steps=2,
        limit_mode="explicit",
    )
    assert result.output is None
    assert calls == []
    assert user_file.read_bytes() == contents
    after = git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    assert all(line in after.splitlines() for line in before.splitlines())

async def test_second_saved_workflow_checks_previous_run_records(tmp_path, monkeypatch):
    workspace = repository(tmp_path / "repo")
    calls = structured_transport(monkeypatch)
    first = await client(workspace).workflow(
        isolated_answer, artifacts=workspace / ".opencollab/sessions/first",
        budget=1000, max_steps=2, limit_mode="explicit",
    )
    assert first.output == {"answer": 4}
    previous = workspace / ".opencollab/sessions/first/workflow.json"
    content = previous.read_bytes()
    second = await client(workspace).workflow(
        isolated_answer, artifacts=workspace / ".opencollab/sessions/second",
        budget=1000, max_steps=2, limit_mode="explicit",
    )
    assert second.output is None
    assert len(calls) == 1
    assert previous.read_bytes() == content

@pytest.mark.parametrize("dirty", [False, True])
async def test_saved_workflow_in_subdirectory_owns_same_repository_records(tmp_path, monkeypatch, dirty):
    repo = repository(tmp_path / "repo")
    workspace = repo / "package"
    workspace.mkdir()
    (workspace / "marker.txt").write_text("source\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "commit", "-qm", "\u6dfb\u52a0\u5b50\u76ee\u5f55\u6d4b\u8bd5\u6e90\u7801")
    if dirty:
        (repo / "app.py").write_text("answer = 5\n")
    calls = structured_transport(monkeypatch)
    result = await client(workspace).workflow(
        isolated_answer, artifacts=repo / ".opencollab/sessions/current",
        budget=1000, max_steps=2, limit_mode="explicit",
    )
    assert result.output == (None if dirty else {"answer": 4})
    assert len(calls) == (0 if dirty else 1)
    assert (repo / "app.py").read_text() == ("answer = 5\n" if dirty else "answer = 4\n")

async def test_saved_workflow_keeps_container_isolation_on_container_side(
    worktree_transport, tmp_path, monkeypatch,
):
    workspace = repository(tmp_path / "repo")
    artifacts = tmp_path / "records"
    calls = structured_transport(monkeypatch)
    container = attach_container(container_id=CONTAINER_ID, workspace=str(workspace))
    monkeypatch.setattr(worktree_pool, "CONTAINER_WORKTREE_ROOT", str(tmp_path / "worktrees"))
    result = await OpenCollab(
        workspace, model="offline-model", provider="openai",
        api_key="local-fixture",  # pragma: allowlist secret
        base_url="https://offline.example.invalid/v1", environment=container,
    ).workflow(
        isolated_answer, artifacts=artifacts, budget=1000, max_steps=2,
        limit_mode="explicit",
    )
    assert result.output == {"answer": 4}
    assert len(calls) == 1
    assert any("worktree" in command and "add" in command for command in worktree_transport)
    assert (artifacts / "workflow.json").exists()

def test_cli_default_save_runs_discovered_workflow_in_real_git_worktree(tmp_path, monkeypatch):
    workspace = repository(tmp_path / "repo")
    workflows = workspace / "workflows"
    workflows.mkdir()
    (workflows / "saved.py").write_text(
        "from opencollab.workflows import workflow\n"
        "@workflow(name='saved-isolation', description='Read isolated source')\n"
        "async def run(ctx, args):\n"
        "    return await ctx.agent('Return answer four', isolation=True, tools=[], schema={\n"
        "        'type': 'object', 'properties': {'answer': {'type': 'integer'}},\n"
        "        'required': ['answer'], 'additionalProperties': False})\n",
        encoding="utf-8",
    )
    git(workspace, "add", ".")
    git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "commit", "-qm", "\u6dfb\u52a0\u547d\u4ee4\u884c\u6d4b\u8bd5\u5de5\u4f5c\u6d41")
    assert git(workspace, "status", "--porcelain") == ""
    calls = structured_transport(monkeypatch)
    monkeypatch.setenv("OPENCOLLAB_WORKFLOWS_DIR", "workflows")
    result = CliRunner().invoke(workflow_cli.app, [
        "run", "saved-isolation", "--workspace", str(workspace),
        "--provider", "openai", "--model", "offline-model", "--budget", "1000",
        "--api-key", "local-fixture",  # pragma: allowlist secret
        "--base-url", "https://offline.example.invalid/v1",
    ])
    assert result.exit_code == 0, result.stdout
    assert '"answer": 4' in result.stdout
    assert len(calls) == 1
    runs = list((workspace / ".opencollab" / "sessions").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "orchestration.jsonl").is_file()
    assert (runs[0] / "workflow.json").is_file()

@pytest.mark.parametrize("mutation", ["modified", "staged", "deleted", "renamed"])
async def test_owned_save_directory_still_checks_tracked_changes(tmp_path, mutation):
    workspace = repository(tmp_path / "repo")
    artifacts = workspace / ".opencollab" / "sessions" / "current"
    artifacts.mkdir(parents=True)
    tracked = artifacts / "tracked.txt"
    tracked.write_text("tracked data\n")
    git(workspace, "add", ".")
    git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "commit", "-qm", "\u6dfb\u52a0\u4fdd\u5b58\u76ee\u5f55\u4e2d\u7684\u5df2\u8ddf\u8e2a\u6587\u4ef6")
    if mutation in {"modified", "staged"}:
        tracked.write_text("user edit\n")
        if mutation == "staged":
            git(workspace, "add", ".")
    elif mutation == "deleted":
        tracked.unlink()
    else:
        git(workspace, "mv", str(tracked), str(artifacts / "renamed.txt"))
    (artifacts / "orchestration.jsonl").write_text("current record\n")
    status = git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    factory = WorkflowSessionFactory(
        model="offline-model", provider="openai", api_key=None, base_url=None,
        workspace=str(workspace), save_dir=str(artifacts),
    )
    try:
        with pytest.raises(RuntimeError, match="uncommitted changes"):
            await factory.acquire_isolated_env(label="reader")
    finally:
        await factory.release_isolated_envs()
    assert git(workspace, "status", "--porcelain=v1", "--untracked-files=all") == status

@pytest.mark.parametrize("git_mode", [False, True])
@pytest.mark.parametrize("save_location", ["source", "ancestor"])
async def test_save_directory_covering_source_preserves_source(tmp_path, git_mode, save_location):
    workspace = repository(tmp_path / "repo") if git_mode else tmp_path / "plain"
    if not git_mode:
        workspace.mkdir()
        (workspace / "app.py").write_text("answer = 4\n")
    (workspace / "user.txt").write_text("user data\n")
    factory = WorkflowSessionFactory(
        model="offline-model", provider="openai", api_key=None, base_url=None,
        workspace=str(workspace),
        save_dir=str(workspace if save_location == "source" else workspace.parent),
    )
    try:
        if git_mode:
            with pytest.raises(RuntimeError, match="uncommitted changes"):
                await factory.acquire_isolated_env(label="reader")
        else:
            env = await factory.acquire_isolated_env(label="reader")
            assert await env.read_file("app.py") == "answer = 4\n"
            assert await env.read_file("user.txt") == "user data\n"
    finally:
        await factory.release_isolated_envs()

@pytest.mark.parametrize("git_mode", [False, True])
async def test_saved_worktree_uses_exact_directory_names(tmp_path, git_mode):
    workspace = repository(tmp_path / "repo") if git_mode else tmp_path / "plain"
    if not git_mode:
        workspace.mkdir()
    owned = workspace / "records [1]"
    sibling = workspace / "records [1]-older"
    owned.mkdir()
    sibling.mkdir()
    (owned / "current.json").write_text("live record\n")
    (sibling / "user.json").write_text("user record\n")
    factory = WorkflowSessionFactory(
        model="offline-model", provider="openai", api_key=None, base_url=None,
        workspace=str(workspace), save_dir=str(owned),
    )
    try:
        if git_mode:
            with pytest.raises(RuntimeError, match="records "):
                await factory.acquire_isolated_env(label="reader")
            (sibling / "user.json").unlink()
            sibling.rmdir()
            env = await factory.acquire_isolated_env(label="reader")
            assert await env.read_file("app.py") == "answer = 4\n"
        else:
            env = await factory.acquire_isolated_env(label="reader")
            assert await env.read_file("records [1]-older/user.json") == "user record\n"
        assert not (Path(env.workspace) / owned.name).exists()
    finally:
        await factory.release_isolated_envs()

async def test_public_team_copies_source_and_older_records_without_live_run(tmp_path, monkeypatch):
    workspace = tmp_path / "plain"
    workspace.mkdir()
    (workspace / "app.py").write_text("answer = 4\n")
    older = workspace / ".opencollab" / "sessions" / "older"
    older.mkdir(parents=True)
    (older / "user.json").write_text("older user data\n")
    artifacts = workspace / ".opencollab" / "sessions" / "current"
    team = tmp_path / "team.yaml"
    team.write_text(
        "entry: lead\nroles:\n"
        "  lead:\n    prompt: ROLE_LEAD\n    tools: [spawn_agent]\n"
        "  war:\n    prompt: ROLE_WAR\n    tools: []\n"
        "  justice:\n    prompt: ROLE_JUSTICE\n    tools: []\n"
        "topology:\n  lead: [war, justice]\n",
    )
    environments = []
    snapshots = []
    acquire = worktree_pool.WorktreePool.acquire

    async def capture(pool, role):
        env = await acquire(pool, role)
        environments.append(env)
        snapshots.append({
            "source": await env.read_file("app.py"),
            "older": await env.read_file(".opencollab/sessions/older/user.json"),
            "live_copy": (Path(env.workspace) / ".opencollab/sessions/current").exists(),
            "live_baseline": (Path(env._copy_baseline_dir) / ".opencollab/sessions/current").exists(),
        })
        return env

    monkeypatch.setattr(worktree_pool.WorktreePool, "acquire", capture)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        system = str(body["messages"][0].get("content"))
        if "ROLE_LEAD" in system and not any(m["role"] == "tool" for m in body["messages"]):
            return response(request, calls, tool_calls=[
                tool("spawn_agent", {"role": "war", "task": "Read source"}, "war"),
                tool("spawn_agent", {"role": "justice", "task": "Read source"}, "justice"),
            ])
        return response(request, calls, text="Source read successfully")

    install_sdk_transport(monkeypatch, "openai", handler)
    result = await client(workspace).team(
        "Read source in two roles", config=team, artifacts=artifacts,
        trace=True, use_worktrees=True, budget=100000, max_steps=10,
    )
    assert result.status == "completed", (result, len(calls))
    assert not result.agent_failures
    assert len(calls) == 4
    assert len(environments) == 2
    assert snapshots == [{
        "source": "answer = 4\n", "older": "older user data\n",
        "live_copy": False, "live_baseline": False,
    }] * 2
    manifest = json.loads((artifacts / "team.json").read_text())
    assert len(manifest["agents"]) == 3
