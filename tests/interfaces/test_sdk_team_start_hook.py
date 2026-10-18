"""Entry sessions initialize through the same hook bridge as child sessions."""

from __future__ import annotations

import copy
import json

import pytest

from opencollab import OpenCollab
from opencollab.bootstrap import build_runtime_context, build_scheduler, container
from tests.support.session_characterization_test_support import llm_response


class _Model:
    requests = []

    def __init__(self, **kwargs):
        self.calls = 0

    def context_window(self):
        return 128_000

    async def close(self):
        pass

    async def complete(self, messages, **kwargs):
        self.calls += 1
        self.requests.append(copy.deepcopy(messages))
        if "Worker role" in messages[0]["content"]:
            return llm_response(content="worker done")
        if self.calls == 1:
            return llm_response(tool_calls=[{
                "id": "read-startup", "type": "function",
                "function": {"name": "file_read", "arguments": '{"path":"startup.txt"}'},
            }], finish_reason="tool_calls")
        if self.calls == 2 and any(message.get("content", "").startswith("Delegate") for message in messages):
            return llm_response(tool_calls=[{
                "id": "spawn", "type": "function",
                "function": {"name": "spawn_agent", "arguments": '{"role":"worker","task":"Do the work"}'},
            }], finish_reason="tool_calls")
        results = [message.get("content", "") for message in messages if message["role"] == "tool"]
        return llm_response(content="startup marker available" if any("INITIALIZED" in result for result in results)
                            else "startup marker missing")

def _team(tmp_path, *, start_ending="success", child=False):
    record = 'printf "%s %s\\n" "$OPENCOLLAB_HOOK_EVENT" "$OPENCOLLAB_AID" >> lifecycle.txt'
    first = record
    if start_ending == "error":
        first += "\nexit 3"
    elif start_ending == "timeout":
        first += "\nsleep 60"
    start_timeout = 0.02 if start_ending == "timeout" else 2
    child_role = "  worker:\n    prompt: Worker role.\n    tools: []\n" if child else ""
    lead_tools = "[file_read, spawn_agent]" if child else "[file_read]"
    topology = "  lead: [worker]\n  worker: []\n" if child else "  lead: []\n"
    path = tmp_path / "team.yaml"
    path.write_text(
        "entry: lead\nroles:\n  lead:\n    prompt: Read the startup marker.\n"
        f"    tools: {lead_tools}\n{child_role}topology:\n{topology}hooks:\n"
        f"  SessionStart:\n    - command: {json.dumps(first)}\n      timeout: {start_timeout}\n"
        "    - command: 'printf INITIALIZED > startup.txt'\n"
        f"  Stop:\n    - command: {json.dumps(record)}\n"
        f"  SubagentStop:\n    - command: {json.dumps(record)}\n",
    )
    return path

@pytest.mark.parametrize("start_ending", ["success", "error", "timeout"])
async def test_public_team_start_initializes_before_file_read(tmp_path, monkeypatch, start_ending):
    monkeypatch.setattr(container, "LLMClient", _Model)
    _Model.requests = []
    team = _team(tmp_path, start_ending=start_ending)
    result = await OpenCollab(tmp_path, model="local-model").team(
        "Check initialization", config=team, use_worktrees=False,
        artifacts=tmp_path / "artifacts", trace=False, timeout=5,
    )
    assert result.status == "completed"
    assert result.output == "startup marker available"
    assert (tmp_path / "startup.txt").read_text() == "INITIALIZED"
    assert (tmp_path / "lifecycle.txt").read_text().splitlines() == ["SessionStart 0", "Stop 0"]
    assert any("INITIALIZED" in message.get("content", "") for message in _Model.requests[-1]
               if message["role"] == "tool")

async def test_new_and_restored_lead_instances_fire_once_at_actual_start(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "LLMClient", _Model)
    team = _team(tmp_path)
    config = {
        "model": "local-model", "provider": "openai",
        "api_key": "test", "base_url": None, "budget": 100_000,  # pragma: allowlist secret
    }

    def scheduler_for(session_file=None):
        return build_scheduler(
            build_runtime_context(str(tmp_path), config, trace=False),
            use_worktrees=False, interactive=False, auto_save=False,
            session_file=session_file, team_config_path=team,
        )

    scheduler = scheduler_for()
    path = tmp_path / "saved.json"
    assert not (tmp_path / "startup.txt").exists()
    try:
        assert await scheduler.run("First request") == "startup marker available"
        assert await scheduler.run("Second request") == "startup marker available"
        scheduler._lead_session.save(str(path))
    finally:
        await scheduler.cleanup()
    assert (tmp_path / "lifecycle.txt").read_text().splitlines() == ["SessionStart 0", "Stop 0", "Stop 0"]

    (tmp_path / "startup.txt").unlink()
    restored = scheduler_for(str(path))
    assert not (tmp_path / "startup.txt").exists()
    try:
        assert await restored.run("Restored instance request") == "startup marker available"
        assert await restored.run("Another restored request") == "startup marker available"
    finally:
        await restored.cleanup()
    assert (tmp_path / "lifecycle.txt").read_text().splitlines() == [
        "SessionStart 0", "Stop 0", "Stop 0", "SessionStart 0", "Stop 0", "Stop 0",
    ]

async def test_child_start_and_subagent_stop_keep_their_existing_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "LLMClient", _Model)
    result = await OpenCollab(tmp_path, model="local-model").team(
        "Delegate the work", config=_team(tmp_path, child=True), use_worktrees=False,
        artifacts=tmp_path / "artifacts", trace=False, timeout=5,
    )
    assert result.status == "completed"
    assert result.output == "startup marker available"
    assert (tmp_path / "lifecycle.txt").read_text().splitlines() == [
        "SessionStart 0", "SessionStart 1", "SubagentStop 1", "Stop 0",
    ]
