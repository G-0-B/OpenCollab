"""A saved parent retains completed child results while siblings still run."""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from typer.testing import CliRunner

from opencollab import OpenCollab
from opencollab.adapters.cli import main as cli_main
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport


@pytest.fixture
def controlled_team(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("OPENCOLLAB_", "OPENAI_", "ANTHROPIC_", "DASHSCOPE_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_LLM_MAX_RETRIES", "0")
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "team.yaml"
    config.write_text(
        "entry: lead\nroles:\n  lead:\n    prompt: Combine the section reviews.\n"
        "    model: lead-demo\n    tools: [spawn_agent]\n  fast:\n"
        "    prompt: Read the section facts and report the result.\n    model: fast-demo\n"
        "    tools: [file_read]\n  slow:\n    prompt: Calculate the other section.\n"
        "    model: slow-demo\n    tools: []\ntopology:\n  lead: [fast, slow]\n"
        "  fast: []\n  slow: []\n", encoding="utf-8",
    )
    (tmp_path / "facts.txt").write_text("Section A verified total 731\n", encoding="utf-8")
    requests = []
    slow_started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        model = body["model"]
        results = [message for message in body["messages"] if message["role"] == "tool"]
        if model == "slow-demo":
            slow_started.set()
            await release.wait()
            return completion_http_response(request, text="SLOW_COMPLETED_RESULT")
        if model == "fast-demo" and results:
            assert "731" in results[-1]["content"]
            return completion_http_response(request, text="FAST_COMPLETED_RESULT")
        if model == "lead-demo" and results:
            return completion_http_response(request, text="\n".join(message["content"] for message in results))
        calls = (
            [("read-facts", "file_read", {"path": "facts.txt"})] if model == "fast-demo" else
            [(f"{role}-call", "spawn_agent", {"role": role, "task": f"Review section {role}."})
             for role in ("fast", "slow")]
        )
        return http_response(request, 200, json={
            "id": "chat-delegation", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [
                    {"id": ident, "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments),
                    }} for ident, name, arguments in calls
                ],
            }}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        })

    install_sdk_transport(monkeypatch, "openai", handler)
    client = OpenCollab(
        tmp_path, model="lead-demo", provider="openai",
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"wire_protocol": "chat_completions", "llm_max_retries": 0},
    )
    return client, config, requests, slow_started, release

async def test_public_parent_partial_autosave_restores_each_child_outcome(tmp_path, controlled_team):
    client, config, requests, slow_started, release = controlled_team
    artifacts = tmp_path / "artifacts"
    active = asyncio.create_task(client.team(
        "Review both sections.", config=config, artifacts=artifacts,
        use_worktrees=False, trace=False, timeout=10, budget=100_000,
    ))

    async def wait_for_partial_snapshot():
        while True:
            for path in artifacts.glob("agent_0_*.json"):
                snapshot = OpenCollab.read_session_snapshot(path)
                rows = snapshot["session_state"]["pending_events"]
                if len(rows) == 2 and rows[0]["status"] == "done":
                    return snapshot
            await asyncio.sleep(0.005)

    try:
        await asyncio.wait_for(slow_started.wait(), 5)
        snapshot = await asyncio.wait_for(wait_for_partial_snapshot(), 5)
        assert snapshot["session_state"]["phase"] == "awaiting_events"
        fast, slow = snapshot["session_state"]["pending_events"]
        assert fast["tool_call_id"] == "fast-call"
        assert fast["result"] == "FAST_COMPLETED_RESULT"
        assert slow["tool_call_id"] == "slow-call"
        assert slow["status"] == "pending"
        assert slow["result"] is None
        assert not active.done()

        # Retain the durable state before any normal cancellation or cleanup.
        checkpoint = tmp_path / "partial.json"
        checkpoint.write_text(json.dumps(snapshot), encoding="utf-8")
        before_resume = len(requests)
        result = await asyncio.to_thread(CliRunner().invoke, cli_main.app, [
            "--workspace", str(tmp_path), "--team-config", str(config), "--no-worktrees",
            "--provider", "openai", "--model", "lead-demo", "--api-key", "controlled-test",
            "--base-url", "https://controlled.invalid/v1", "--budget", "100000",
            "--session", str(checkpoint), "--prompt", "Continue the section report.",
        ])
        assert result.exit_code == 0, (result.output, result.exception)
        restored_requests = requests[before_resume:]
        assert [request["model"] for request in restored_requests] == ["lead-demo", "lead-demo"]
        restored_results = [message for message in restored_requests[0]["messages"] if message["role"] == "tool"]
        assert [(message["tool_call_id"], message["content"]) for message in restored_results] == [
            ("fast-call", "FAST_COMPLETED_RESULT"),
            ("slow-call", "deferred child interrupted by session restore"),
        ]
    finally:
        release.set()
        completed = await asyncio.wait_for(active, 5)
    assert completed.status == "completed", completed.reason
    assert completed.output == "FAST_COMPLETED_RESULT\nSLOW_COMPLETED_RESULT"

@pytest.mark.parametrize("serialized", [False, True])
async def test_public_team_without_autosave_delivers_serial_and_parallel_children(
    tmp_path, controlled_team, serialized,
):
    client, config, requests, _slow_started, release = controlled_team
    release.set()
    result = await client.team(
        "Review both sections.", config=config, serialize_turns=serialized,
        use_worktrees=False, trace=False, timeout=5, budget=100_000,
    )
    assert result.status == "completed", result.reason
    assert result.output == "FAST_COMPLETED_RESULT\nSLOW_COMPLETED_RESULT"
    assert len([request for request in requests if request["model"] == "lead-demo"]) == 2
    assert not (tmp_path / ".opencollab/sessions").exists()
