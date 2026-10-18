"""Queued CLI turns preserve the user's live prompt selection and transcript."""

from __future__ import annotations

import asyncio
import json
import os
from io import StringIO

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from opencollab.adapters.cli import main as cli_main
from opencollab.adapters.cli.live_prompt import LivePrompt, _pin_status_row_under_input
from opencollab.adapters.tui import TUI
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport


async def _wait_for(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout=5)

def _snapshot(workspace, aid):
    paths = list((workspace / ".opencollab/sessions").glob(f"*/agent_{aid}_*.json"))
    return json.loads(paths[0].read_text(encoding="utf-8")) if paths else None

def _install_cli(monkeypatch, workspace):
    import opencollab.adapters.tui as tui_module

    for key in tuple(os.environ):
        if key.startswith(("OPENCOLLAB_", "OPENAI_", "ANTHROPIC_", "DASHSCOPE_")):
            monkeypatch.delenv(key)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("OPENCOLLAB_LLM_MAX_RETRIES", "0")
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_main.sys.stdout, "isatty", lambda: True)
    output = StringIO()
    console = Console(file=output, width=120, color_system=None)
    tui = TUI(console)
    monkeypatch.setattr(cli_main, "console", console)
    monkeypatch.setattr(tui_module, "TUI", lambda *args, **kwargs: tui)
    team = workspace / "team.yaml"
    team.write_text(
        "entry: lead\nroles:\n  lead:\n    prompt: OFFLINE_LEAD_ROLE\n"
        "    tools: [spawn_agent]\n  analyst:\n    prompt: OFFLINE_ANALYST_ROLE\n"
        "    tools: [file_read, file_write]\ntopology:\n  lead: [analyst]\n  analyst: []\n",
        encoding="utf-8",
    )
    cfg = cli_main.resolve_config(
        str(workspace), "offline-model", "openai", "offline-unused",  # pragma: allowlist secret
        "https://offline.invalid/v1", 100000,
    )
    return tui, output, team, cfg

@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["done", "error", "cancel", "dropped", "no_queue"])
async def test_real_prompt_keeps_lead_draft_when_analyst_turn_starts(monkeypatch, tmp_path, outcome):
    """Tab, queue submission, HTTP requests and saved sessions use production code."""
    tui, output, team, cfg = _install_cli(monkeypatch, tmp_path)
    lead_entered, child_entered = asyncio.Event(), asyncio.Event()
    release_lead, release_child = asyncio.Event(), asyncio.Event()
    requests = []

    async def respond(request):
        body = json.loads(request.content)
        users = [message["content"] for message in body["messages"] if message["role"] == "user"]
        role = "analyst" if "OFFLINE_ANALYST_ROLE" in body["messages"][0]["content"] else "lead"
        requests.append((role, users[-1]))
        message = {"role": "assistant", "content": f"Completed {role} turn"}
        if "setup_team" in users[-1] and body["messages"][-1]["role"] == "user":
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "offline_spawn", "type": "function", "function": {
                    "name": "spawn_agent",
                    "arguments": json.dumps({"role": "analyst", "task": "initial_child_task"}),
                },
            }]}
        elif "slow_lead_task" in users[-1]:
            lead_entered.set()
            await release_lead.wait()
            message["content"] = "Lead response retained while analyst runs"
        elif "queued_analyst_followup" in users[-1]:
            child_entered.set()
            await release_child.wait()
            if outcome == "error":
                return http_response(request, 400, json={"error": {
                    "message": "offline queued failure", "type": "invalid_request_error",
                }})
            message["content"] = "Queued analyst answer retained for Tab"
            if outcome == "cancel":
                message["tool_calls"] = [{
                    "id": "offline_write", "type": "function", "function": {
                        "name": "file_write", "arguments": json.dumps({
                            "path": "cancelled.txt", "mode": "create", "content": "cancelled write",
                        }),
                    },
                }]
        return http_response(request, 200, json={
            "id": "chatcmpl_offline", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        })

    install_sdk_transport(monkeypatch, "openai", respond)
    with create_pipe_input() as keyboard:
        session = PromptSession(input=keyboard, output=DummyOutput(), erase_when_done=True)
        _pin_status_row_under_input(session)
        monkeypatch.setattr(LivePrompt, "_ensure_session", lambda self: session)
        run = asyncio.create_task(cli_main._run(
            str(tmp_path), cfg, None, False, True, False, team_config_path=str(team),
        ))
        try:
            await _wait_for(lambda: session.app.is_running)
            keyboard.send_text("setup_team\r")
            await _wait_for(lambda: _snapshot(tmp_path, 0)
                            and _snapshot(tmp_path, 0)["session_state"]["phase"] == "done")
            keyboard.send_text("slow_lead_task\r")
            await asyncio.wait_for(lead_entered.wait(), timeout=5)
            if outcome != "no_queue":
                keyboard.send_text("\tqueued_analyst_followup\r")
                await _wait_for(lambda: tui._queued_turns == 1)
                keyboard.send_text("\t")
                await _wait_for(lambda: tui.selected_aid == 0)
            draft = "lead_draft_after_queue"
            keyboard.send_text(draft)
            await _wait_for(lambda: session.default_buffer.text == draft)
            if outcome == "dropped":
                keyboard.send_bytes(b"\x03")
                await _wait_for(lambda: session.app.is_running and session.default_buffer.text == "")
            release_lead.set()
            if outcome not in {"dropped", "no_queue"}:
                await asyncio.wait_for(child_entered.wait(), timeout=5)
            else:
                await _wait_for(lambda: _snapshot(tmp_path, 0)["session_state"]["phase"] in {"done", "stopped"})
            focus_at_background_start = tui.selected_aid
            if outcome == "cancel":
                assert session.default_buffer.text == draft
                keyboard.send_bytes(b"\x03")
                await _wait_for(lambda: session.app.is_running and session.default_buffer.text == "")
            if outcome in {"cancel", "dropped"}:
                keyboard.send_text(draft)
                await _wait_for(lambda: session.default_buffer.text == draft)
            else:
                assert session.default_buffer.text == draft
            keyboard.send_text("\r")
            release_child.set()
            await _wait_for(lambda: any(draft in text for _, text in requests))
            draft_role = next(role for role, text in requests if draft in text)
            draft_aid = 0 if draft_role == "lead" else 1
            await _wait_for(lambda: _snapshot(tmp_path, draft_aid)["session_state"]["phase"] == "done"
                            and any(draft in (message.get("content") or "")
                                    for message in _snapshot(tmp_path, draft_aid)["messages"]))
            before_tab = output.getvalue()
            keyboard.send_text("\t")
            await _wait_for(lambda: tui.selected_aid != focus_at_background_start)
            after_tab = output.getvalue()
            keyboard.send_text("/exit\r")
            await asyncio.wait_for(run, timeout=5)
        finally:
            release_lead.set()
            release_child.set()
            if not run.done():
                run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    assert focus_at_background_start == 0
    assert draft_role == "lead"
    lead_snapshot, child_snapshot = _snapshot(tmp_path, 0), _snapshot(tmp_path, 1)
    assert any(message["role"] == "user" and draft in message["content"] for message in lead_snapshot["messages"])
    assert all(draft not in (message.get("content") or "") for message in child_snapshot["messages"])
    if outcome == "done":
        assert "Lead response retained while analyst runs" in before_tab
        assert "Queued analyst answer retained for Tab" not in before_tab
        assert "Queued analyst answer retained for Tab" in after_tab
        assert "queued_analyst_followup" in after_tab
    if outcome in {"dropped", "no_queue"}:
        assert all("queued_analyst_followup" not in text for _, text in requests)
    if outcome in {"error", "cancel"}:
        assert child_snapshot["session_state"]["phase"] == ("error" if outcome == "error" else "stopped")
    assert not (tmp_path / "cancelled.txt").exists()

@pytest.mark.asyncio
async def test_one_shot_prints_final_lead_answer_after_user_tabs_to_analyst(monkeypatch, tmp_path):
    tui, output, team, cfg = _install_cli(monkeypatch, tmp_path)
    final_entered, release_final = asyncio.Event(), asyncio.Event()

    async def respond(request):
        body = json.loads(request.content)
        if "OFFLINE_ANALYST_ROLE" in body["messages"][0]["content"]:
            return completion_http_response(request, text="Initial analyst answer")
        if body["messages"][-1]["role"] == "user" and "setup_team" in body["messages"][-1]["content"]:
            return http_response(request, 200, json={
                "id": "chatcmpl_offline", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{
                        "id": "offline_spawn", "type": "function", "function": {
                            "name": "spawn_agent",
                            "arguments": json.dumps({"role": "analyst", "task": "initial_child_task"}),
                        },
                    }],
                }}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            })
        final_entered.set()
        await release_final.wait()
        return completion_http_response(request, text="Final lead answer after inspection")

    install_sdk_transport(monkeypatch, "openai", respond)
    with create_pipe_input() as keyboard:
        session = PromptSession(input=keyboard, output=DummyOutput(), erase_when_done=True)
        _pin_status_row_under_input(session)
        monkeypatch.setattr(LivePrompt, "_ensure_session", lambda self: session)
        run = asyncio.create_task(cli_main._run(
            str(tmp_path), cfg, None, False, True, False,
            team_config_path=str(team), one_shot_prompt="setup_team",
        ))
        try:
            await asyncio.wait_for(final_entered.wait(), timeout=5)
            await _wait_for(lambda: session.app.is_running)
            keyboard.send_text("\t")
            await _wait_for(lambda: tui.selected_aid == 1)
            release_final.set()
            await asyncio.wait_for(run, timeout=5)
        finally:
            release_final.set()
            if not run.done():
                run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    assert tui.selected_aid == 1
    assert output.getvalue().count("Final lead answer after inspection") == 1
