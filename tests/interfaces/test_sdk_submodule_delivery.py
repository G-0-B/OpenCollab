"""Public Team delivery keeps submodule edits through SDK shutdown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters import worktree_pool
from opencollab.environments import attach_container, local_environment
from tests.support.provider_sdk_http import http_response, install_sdk_transport
from tests.support.worktree_delivery_support import CONTAINER_ID, git
from tests.support.worktree_delivery_support import worktree_transport as worktree_transport
from tests.workflows.test_candidate_submodule_delivery import _source_with_dependency


def _call(name, arguments):
    return {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

@pytest.mark.parametrize("backend", ["local", "container"])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("target", ["tracked", "untracked", "parent"])
async def test_public_team_reports_unpatchable_modules_and_retains_files(
    worktree_transport, tmp_path, monkeypatch, backend, trace, target,
):
    source, module_path = _source_with_dependency(tmp_path, False)
    team = tmp_path / "team.yaml"
    team.write_text(
        "entry: lead\nroles:\n"
        "  lead:\n    prompt: ROLE_LEAD\n    tools: [spawn_agent]\n"
        "  coder:\n    prompt: ROLE_CODER\n    tools: [file_write]\n"
        "topology:\n  lead: [coder]\n",
    )
    relative_file = {
        "tracked": f"{module_path}/source.py", "untracked": f"{module_path}/new_source.py", "parent": "source.py",
    }[target]
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    monkeypatch.delenv("OPENCOLLAB_UNBOUNDED_LIMITS", raising=False)
    monkeypatch.setattr(worktree_pool, "CONTAINER_WORKTREE_ROOT", str(tmp_path / "worktrees"))
    observed = {}
    steps = {}
    acquire = worktree_pool.WorktreePool.acquire

    async def capture(pool, role):
        env = await acquire(pool, role)
        initialized = await env.exec_cmd("git -c protocol.file.allow=always submodule update --init --recursive")
        assert initialized.returncode == 0, initialized.stderr
        observed.update({"env": env, "pool": pool, "workspace": Path(env.workspace)})
        return env

    monkeypatch.setattr(worktree_pool.WorktreePool, "acquire", capture)

    def handler(request):
        messages = json.loads(request.content)["messages"]
        role = "lead" if "ROLE_LEAD" in str(messages[0].get("content")) else "coder"
        steps[role] = steps.get(role, 0) + 1
        calls = []
        if role == "lead" and steps[role] == 1:
            calls = [_call("spawn_agent", {"role": "coder", "task": "Update the source file."})]
        elif role == "coder" and steps[role] == 1:
            calls = [_call("file_write", {"path": relative_file, "mode": "create", "content": "value = 2\n"})]
        elif role == "coder" and steps[role] == 2:
            observed["written"] = (observed["workspace"] / relative_file).read_text()
        elif role == "lead" and steps[role] == 2:
            observed["delivery"] = [message["content"] for message in messages if message["role"] == "tool"][-1]
        return http_response(request, 200, json={
            "id": f"{role}-{steps[role]}", "object": "chat.completion", "created": 0, "model": "offline",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": None if calls else "done",
                **({"tool_calls": calls} if calls else {}),
            }, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    install_sdk_transport(monkeypatch, "openai", handler)
    environment = (
        attach_container(container_id=CONTAINER_ID, workspace=str(source))
        if backend == "container" else local_environment(source)
    )
    result = await OpenCollab(
        source, model="offline", api_key="fixture-key",  # pragma: allowlist secret
        base_url="https://offline.example.invalid/v1", environment=environment,
    ).team(
        "Update the source file.", config=team, use_worktrees=True, budget=20000,
        timeout=15, trace=trace, artifacts=tmp_path / "artifacts",
    )
    env = observed["env"]
    workspace = observed["workspace"]
    try:
        assert observed["written"] == "value = 2\n"
        if target == "parent":
            assert result.status == "completed"
            assert "+value = 2" in observed["delivery"]
            assert env.recovery_location is None
            assert not workspace.exists()
        else:
            assert result.status == "failed"
            assert result.agent_failures
            assert "submodule changes" in observed["delivery"]
            assert "```diff" not in observed["delivery"]
            assert str(workspace) in observed["delivery"]
            assert str(workspace) in env.recovery_location
            for _ in range(2):
                with pytest.raises(OSError, match="retry state retained"):
                    await observed["pool"].release()
                assert (workspace / relative_file).read_text() == "value = 2\n"
            with pytest.raises(RuntimeError, match="submodule changes"):
                await env.get_diff()
        assert (source / "source.py").read_text() == "value = 1\n"
        assert (source / module_path / "source.py").read_text() == "value = 1\n"
        assert not (source / module_path / "new_source.py").exists()
    finally:
        if workspace.exists():
            git(source, "worktree", "remove", "--force", "--force", str(workspace))
            git(source, "update-ref", "-d", f"refs/heads/{env._branch}", env._base_commit)
