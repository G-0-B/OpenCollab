"""CLI terminal errors preserve output, cleanup, and interactive controls."""

import contextlib
import json
import os
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import opencollab.adapters.cli.main as cli_main
from opencollab.application.scheduler_types import SchedulerTurnError
from opencollab.domain.session import SessionPhase
from tests.interfaces.test_cli_main_lifecycle import FakeTracer, config, install_cli_fakes


@pytest.mark.parametrize("first_call_tokens,exit_code", [(10000, 1), (30, 0)])
def test_cli_budget_exit_matches_real_tool_completion(monkeypatch, tmp_path, first_call_tokens, exit_code):
    import httpx
    import openai

    for key in tuple(os.environ):
        if key.startswith(("OPENCOLLAB_", "OPENAI_", "ANTHROPIC_", "DASHSCOPE_")):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    team_file = tmp_path / "team.yaml"
    team_file.write_text(
        "entry: worker\nroles:\n  worker:\n    prompt: Complete the supplied task.\n"
        "    tools: [file_read, file_write]\ntopology:\n  worker: []\n",
        encoding="utf-8",
    )
    (tmp_path / "receipt.txt").write_text("workspace receipt\n", encoding="utf-8")
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        message = {"role": "assistant", "content": None}
        if len(calls) < 3:
            name = "file_read" if len(calls) == 1 else "file_write"
            arguments = (
                {"path": "receipt.txt"} if len(calls) == 1
                else {"path": "requested.txt", "mode": "create", "content": "completed task\n"}
            )
            message["tool_calls"] = [{
                "id": f"fixture_call_{len(calls)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]
        else:
            message["content"] = "Created requested.txt."
        tokens = first_call_tokens if len(calls) == 1 else 30
        return httpx.Response(200, json={
            "id": f"chatcmpl-offline-{len(calls)}", "object": "chat.completion",
            "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if len(calls) < 3 else "stop"}],
            "usage": {"prompt_tokens": tokens // 2, "completion_tokens": tokens - tokens // 2,
                      "total_tokens": tokens},
        })

    original_client = openai.AsyncOpenAI

    def offline_client(**kwargs):
        return original_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "AsyncOpenAI", offline_client)
    result = CliRunner().invoke(cli_main.app, [
        "--workspace", str(tmp_path), "--team-config", str(team_file), "--no-worktrees",
        "--provider", "openai", "--model", "offline-model", "--api-key", "offline-unused",
        "--base-url", "https://offline.invalid/v1", "--budget", "10000",
        "--prompt", "Read receipt then create requested.txt",
    ])

    assert result.exit_code == exit_code, result.stdout
    snapshot_path, = (tmp_path / ".opencollab/sessions").glob("*/agent_0*.json")
    state = json.loads(snapshot_path.read_text(encoding="utf-8"))["session_state"]
    if exit_code == 1:
        assert len(calls) == 1
        assert state["phase"] == "stopped"
        assert state["terminal_reason"] == "budget exceeded: 10000 tokens used"
        assert not (tmp_path / "requested.txt").exists()
    else:
        assert len(calls) == 3
        assert state["phase"] == "done"
        assert (tmp_path / "requested.txt").read_text(encoding="utf-8") == "completed task\n"

@pytest.mark.parametrize("prompt_source", ["prompt", "prompt-file"])
@pytest.mark.parametrize("phase,reason,exit_code", [
    (SessionPhase.ERROR, "fixture failure", 1),
    (SessionPhase.STOPPED, "budget exceeded: 10000 tokens used", 1),
    (SessionPhase.STOPPED, "budget exhausted before model call: no output headroom", 1),
    (SessionPhase.STOPPED, "step limit reached: 100 steps", 1),
    (SessionPhase.STOPPED, "interrupted by user", 0),
])
def test_cli_one_shot_terminal_exit_preserves_output_and_cleanup(
    monkeypatch, tmp_path, prompt_source, phase, reason, exit_code,
):
    events = []

    class Scheduler:
        used_tokens = 1
        lead_session = SimpleNamespace(auto_save_path=None, step_count=1)

        def team_roster(self):
            return []

        async def run_turn(self, aid, line, *, cancel_event=None):
            raise SchedulerTurnError(aid, phase, reason, "partial answer")

        def agent_step_count(self, aid):
            return 1

        async def cleanup(self):
            events.append("cleanup")

    tracer = FakeTracer()
    install_cli_fakes(monkeypatch, Scheduler(), tracer)
    monkeypatch.setattr(cli_main, "resolve_config", lambda *args: {**config(), "api_key": None, "base_url": None})
    monkeypatch.setattr(cli_main, "missing_api_key_for", lambda *args: False)
    monkeypatch.setattr(cli_main, "_report_turn_failure", lambda *args: events.append("report"))
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: False, raising=False)
    if prompt_source == "prompt-file":
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("do work", encoding="utf-8")
        arguments = ["--prompt-file", str(prompt_file)]
    else:
        arguments = ["--prompt", "do work"]

    result = CliRunner().invoke(cli_main.app, [*arguments, "--workspace", str(tmp_path), "--yolo"])

    assert events == ["report", "cleanup"]
    assert tracer.closed is True
    assert result.exit_code == exit_code

@pytest.mark.asyncio
@pytest.mark.parametrize("phase,reason", [
    (SessionPhase.ERROR, "fixture failure"),
    (SessionPhase.STOPPED, "budget exceeded: 10000 tokens used"),
    (SessionPhase.STOPPED, "interrupted by user"),
])
async def test_cli_interactive_failed_turn_accepts_a_followup(monkeypatch, tmp_path, phase, reason):
    calls = []

    class Scheduler:
        used_tokens = 1
        lead_session = SimpleNamespace(auto_save_path=None, step_count=1)

        def team_roster(self):
            return []

        async def run_turn(self, aid, line, *, cancel_event=None):
            calls.append(line)
            if len(calls) == 1:
                raise SchedulerTurnError(aid, phase, reason, None)

        def agent_step_count(self, aid):
            return 1

        async def cleanup(self):
            calls.append("cleanup")

    async def read_followup(prompt, queue, tui, lead, **kwargs):
        queue.submit("first", 0)
        await queue.drain()
        queue.submit("followup", 0)
        await queue.drain()

    tracer = FakeTracer()
    install_cli_fakes(monkeypatch, Scheduler(), tracer)
    monkeypatch.setattr(cli_main, "_read_loop", read_followup)
    await cli_main._run(str(tmp_path), config(), None, True, True, False)

    assert calls == ["first", "followup", "cleanup"]
    assert tracer.closed is True

@pytest.mark.asyncio
@pytest.mark.parametrize("phase,reason", [
    (SessionPhase.ERROR, "fixture failure"),
    (SessionPhase.STOPPED, "budget exceeded: 10000 tokens used"),
    (SessionPhase.STOPPED, "step limit reached: 100 steps"),
])
async def test_cli_hold_preserves_inspection_before_failed_exit(monkeypatch, tmp_path, phase, reason):
    events = []

    class Scheduler:
        used_tokens = 1
        lead_session = SimpleNamespace(auto_save_path=None, step_count=1)

        def team_roster(self):
            return []

        async def run_turn(self, aid, line, *, cancel_event=None):
            raise SchedulerTurnError(aid, phase, reason, None)

        def agent_step_count(self, aid):
            return 1

        async def cleanup(self):
            events.append("cleanup")

    class Prompt:
        def __init__(self, *args, **kwargs):
            pass

        def attached(self):
            return contextlib.nullcontext()

        async def ask(self, *args, **kwargs):
            raise AssertionError("no prompt input needed")

    async def inspect_then_exit(prompt, queue, tui, lead, **kwargs):
        await queue.drain()
        events.append("inspect")

    tracer = FakeTracer()
    install_cli_fakes(monkeypatch, Scheduler(), tracer)
    monkeypatch.setattr(cli_main, "LivePrompt", Prompt)
    monkeypatch.setattr(cli_main, "_read_loop", inspect_then_exit)
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_main.sys.stdout, "isatty", lambda: True, raising=False)

    with pytest.raises(typer.Exit) as error:
        await cli_main._run(
            str(tmp_path), config(), None, True, True, False,
            one_shot_prompt="do work", hold_after_run=True,
        )

    assert error.value.exit_code == 1
    assert events == ["inspect", "cleanup"]
    assert tracer.closed is True
