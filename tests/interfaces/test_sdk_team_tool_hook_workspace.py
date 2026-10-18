"""Configured tool hooks execute beside the caller's actual files."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

import pytest
import yaml

from opencollab import OpenCollab
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport

_RECORD_HOOK = """\
import json
import os
import sys
import time
from pathlib import Path

records = Path(sys.argv[1])
expected = int(sys.argv[2])
payload = json.load(sys.stdin)
event = os.environ['OPENCOLLAB_HOOK_EVENT']
aid = os.environ['OPENCOLLAB_AID']
started = time.monotonic()
if event == 'PostToolUse' and expected > 1:
    (records / ('ready-' + aid)).touch()
    deadline = started + 10
    while len(list(records.glob('ready-*'))) < expected:
        if time.monotonic() >= deadline:
            raise RuntimeError('concurrent hooks did not overlap')
        time.sleep(0.02)
(records / (event + '-' + aid + '.json')).write_text(json.dumps({
    'payload': payload,
    'cwd': os.getcwd(),
    'variables': {name: os.environ[name] for name in (
        'OPENCOLLAB_HOOK_EVENT', 'OPENCOLLAB_TOOL', 'OPENCOLLAB_AID')},
    'started': started,
    'finished': time.monotonic(),
}))
"""

def _repository(workspace):
    workspace.mkdir()
    for args in (
        ("init", "-q"), ("config", "user.name", "Tests"),
        ("config", "user.email", "tests@example.invalid"), ("config", "commit.gpgsign", "false"),
    ):
        subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)
    (workspace / "app.py").write_text("value=0\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "\u6d4b\u8bd5\u521d\u59cb\u5316"],
        cwd=workspace, check=True, capture_output=True,
    )

async def _run_probe(tmp_path, monkeypatch, *, writers, isolated):
    workspace = tmp_path / "repo"
    _repository(workspace)
    records = tmp_path / "hook records"
    records.mkdir()
    script = tmp_path / "record hook.py"
    script.write_text(_RECORD_HOOK, encoding="utf-8")
    record = " ".join(shlex.quote(str(value)) for value in (sys.executable, script, records, len(writers)))
    formatter = f"{shlex.quote(sys.executable)} -m ruff format --no-cache app.py"
    roles = {
        role: {"prompt": "Edit and inspect app.py.", "model": role + "-model", "tools": ["file_write", "file_read"]}
        for role in writers
    }
    if writers != ("lead",):
        roles["lead"] = {"prompt": "Delegate the edits.", "model": "lead-model", "tools": ["spawn_agent"]}
    team = tmp_path / "team.yaml"
    team.write_text(yaml.safe_dump({
        "entry": "lead", "roles": roles,
        "topology": {"lead": [role for role in writers if role != "lead"],
                     **{role: [] for role in writers if role != "lead"}},
        "hooks": {
            "PreToolUse": [{"matcher": "file_write", "command": record, "timeout": 20}],
            "PostToolUse": [{"matcher": "file_write", "command": record + "\n" + formatter, "timeout": 20}],
            **{event: [{"command": record, "timeout": 20}]
               for event in ("SessionStart", "SubagentStop", "Stop")},
        },
    }), encoding="utf-8")
    turns = {}
    reads = {}
    requests = []

    def handler(request):
        body = json.loads(request.content)
        role = body["model"].removesuffix("-model")
        turns[role] = turns.get(role, 0) + 1
        turn = turns[role]
        requests.append(body)
        calls = []
        if role == "lead" and role not in writers and turn == 1:
            calls = [("spawn_agent", {"role": writer, "task": "Edit app.py."}, "spawn-" + writer)
                     for writer in writers]
        elif role in writers and turn == 1:
            value = writers.index(role) + 1
            calls = [("file_write", {"path": "app.py", "mode": "str_replace", "old_str": "value=0",
                                      "new_str": f"value={value}"}, "write-" + role)]
        elif role in writers and turn == 2:
            calls = [("file_read", {"path": "app.py"}, "read-" + role)]
        elif role in writers:
            reads[role] = next(message["content"] for message in body["messages"]
                               if message.get("tool_call_id") == "read-" + role)
        if not calls:
            return completion_http_response(request, text="Finished.")
        return http_response(request, 200, json={
            "id": f"chat-{role}-{turn}", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [
                    {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    for name, args, identity in calls
                ],
            }}], "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
        })

    install_sdk_transport(monkeypatch, "openai", handler)
    result = await OpenCollab(
        workspace, model="fixture-model", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"budget": 100_000, "llm_max_retries": 0},
    ).team("Perform the edits.", config=team, use_worktrees=isolated, artifacts=tmp_path / "artifacts",
           trace=False, timeout=30, max_steps=5)
    assert result.ok, (result.reason, result.error)
    assert len(requests) == 3 * len(writers) + (0 if writers == ("lead",) else 2)
    assert set(reads) == set(writers)
    for index, writer in enumerate(writers, 1):
        assert f"value = {index}" in reads[writer]
    assert (workspace / "app.py").read_text() == (
        "value=0\n" if isolated and writers != ("lead",) else "value = 1\n"
    )
    pre = [json.loads(path.read_text()) for path in records.glob("PreToolUse-*.json")]
    post = [json.loads(path.read_text()) for path in records.glob("PostToolUse-*.json")]
    assert len(pre) == len(post) == len(writers)
    for item in pre + post:
        variables = item["variables"]
        payload = item["payload"]
        assert variables == {
            "OPENCOLLAB_HOOK_EVENT": payload["hook_event_name"], "OPENCOLLAB_TOOL": "file_write",
            "OPENCOLLAB_AID": str(payload["aid"]),
        }
        if isolated and writers != ("lead",):
            assert item["cwd"] != str(workspace)
        else:
            assert item["cwd"] == str(workspace)
    assert {item["payload"]["aid"]: item["cwd"] for item in pre} == {
        item["payload"]["aid"]: item["cwd"] for item in post
    }
    starts = {item["payload"]["aid"]: item for item in (
        json.loads(path.read_text()) for path in records.glob("SessionStart-*.json")
    )}
    endings = {item["payload"]["aid"]: item for item in (
        json.loads(path.read_text()) for pattern in ("Stop-*.json", "SubagentStop-*.json")
        for path in records.glob(pattern)
    )}
    assert len(starts) == len(endings) == len(writers) + (0 if writers == ("lead",) else 1)
    assert set(starts) == set(endings)
    assert all(item["cwd"] == str(workspace) for item in (*starts.values(), *endings.values()))
    for before, after in zip(sorted(pre, key=lambda item: item["payload"]["aid"]),
                             sorted(post, key=lambda item: item["payload"]["aid"])):
        aid = before["payload"]["aid"]
        assert starts[aid]["finished"] <= before["started"] <= after["started"] <= endings[aid]["started"]
    if len(writers) > 1:
        assert len({item["cwd"] for item in post}) == len(writers)
        assert max(item["started"] for item in post) <= min(item["finished"] for item in post)

@pytest.mark.parametrize(("writers", "isolated"), [(("lead",), True), (("worker",), False), (("worker",), True)])
async def test_formatter_and_tool_hook_payload_follow_writer(tmp_path, monkeypatch, writers, isolated):
    await _run_probe(tmp_path, monkeypatch, writers=writers, isolated=isolated)

async def test_concurrent_child_hooks_keep_each_worktree(tmp_path, monkeypatch):
    await _run_probe(tmp_path, monkeypatch, writers=("first", "second"), isolated=True)
