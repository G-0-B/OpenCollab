"""A failed CLI turn preserves streamed output exactly once."""

from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from opencollab.adapters.cli import main as cli_main
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.adapters.tui import TUI
from opencollab.bootstrap import container
from opencollab.domain.events import SessionRuntimeEvent


@pytest.mark.parametrize("variant", ["normal", "length", "after_tool"])
def test_real_cli_prints_partial_output_once(monkeypatch, tmp_path, variant):
    """Tool and error events settle the text before the turn finishes."""
    (tmp_path / "read.txt").write_text("offline fixture\n", encoding="utf-8")

    class OfflineLLM:
        def __init__(self, **kwargs):
            self.calls = 0

        async def complete(self, messages, tools=None, **kwargs):
            self.calls += 1
            if variant == "after_tool" and self.calls > 1:
                raise RuntimeError("offline followup failure")
            calls = [
                {
                    "id": "offline-read",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": '{"path":"read.txt"}'},
                }
            ] if variant == "after_tool" else []
            return LLMResponse(
                content="Retained partial output",
                tool_calls=calls,
                finish_reason="tool_calls" if calls else ("length" if variant == "length" else "stop"),
                usage=Usage(input_tokens=4, output_tokens=2),
            )

    monkeypatch.setattr(container, "LLMClient", OfflineLLM)
    monkeypatch.setenv("OPENCOLLAB_API_KEY", "offline-unused")
    monkeypatch.delenv("OPENCOLLAB_TEAM_FILE", raising=False)
    monkeypatch.delenv("OPENCOLLAB_CONFIG_FILE", raising=False)
    result = CliRunner().invoke(
        cli_main.app,
        ["--workspace", str(tmp_path), "--prompt", "offline read", "--no-worktrees", "--yolo"],
    )

    assert result.output.count("Retained partial output") == 1, result.output
    if variant == "after_tool":
        assert result.exit_code == 1
        assert "offline followup failure" in result.output

@pytest.mark.parametrize("event", ["tool_start", "error"])
def test_settle_remembers_text_flushed_by_events_and_resets_for_next_turn(event):
    tui = TUI(Console(file=StringIO(), width=100, color_system=None))
    tui.start_turn(0)
    tui.event_handler(SessionRuntimeEvent("text_delta", {"aid": 0, "content": "partial answer"}))
    tui.event_handler(SessionRuntimeEvent(event, {"aid": 0, "tool": "bash", "reason": "failure"}))
    assert tui._state_for(0).current_text == ""

    tui.settle_turn()

    assert tui.drained_partial_answer(0) is True
    assert tui.console.file.getvalue().count("partial answer") == 1
    tui.start_turn(0)
    tui.settle_turn()
    assert tui.drained_partial_answer(0) is False

def test_hidden_text_flushed_by_tool_still_needs_salvage_until_printed():
    tui = TUI(Console(file=StringIO(), width=100, color_system=None))
    tui.event_handler(SessionRuntimeEvent("text_delta", {"aid": 1, "content": "hidden partial"}))
    tui.event_handler(SessionRuntimeEvent("tool_start", {"aid": 1, "tool": "bash"}))

    tui.settle_turn()

    assert tui.drained_partial_answer(1) is False
    assert "hidden partial" not in tui.console.file.getvalue()
