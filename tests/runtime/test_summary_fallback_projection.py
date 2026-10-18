"""Failed read-time summaries retain the messages needed by the next decision."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.application.compaction_summary import ReadTimeSummarizer
from opencollab.application.shaping import AutoCompactShaper
from opencollab.bootstrap import build_session
from opencollab.bootstrap.context_policy import ContextPolicy
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase
from opencollab.domain.token_estimation import estimate_request_tokens
from opencollab.tools import builtin_tools


def projection_history():
    return [
        {"role": "system", "content": "Follow the latest user request."},
        {"role": "user", "content": "Earlier discussion.\n" + "x" * 8_000},
        {"role": "user", "content": "Save the final report to report-green.txt."},
        {"role": "assistant", "tool_calls": [
            {"id": "read", "type": "function", "function": {"name": "file_read", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "read", "content": "recent source"},
    ]

def compactor(summarizer, cache_size=8):
    return AutoCompactShaper(
        summarizer=summarizer, summary_cache_size=cache_size,
        trigger_tokens=1_000, target_tokens=500, keep_recent_groups=1,
    )

async def shape(shaper, messages, asynchronous):
    return await shaper.ashape(messages) if asynchronous else shaper.shape(messages)

@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cache_size", [0, 8])
@pytest.mark.parametrize("failure", ["exception", "empty"])
async def test_failed_summary_retains_original_projection_then_retries(asynchronous, cache_size, failure):
    calls = []

    async def complete(request):
        calls.append(request)
        if len(calls) == 1:
            if failure == "exception":
                raise RuntimeError("temporary provider failure")
            return SimpleNamespace(content="<summary> </summary>")
        return SimpleNamespace(content="<summary>Save report-green.txt.</summary>")

    summarizer = ReadTimeSummarizer(complete)
    shaper = compactor(summarizer, cache_size)
    messages = projection_history()
    original = copy.deepcopy(messages)

    failed = await shape(shaper, messages, asynchronous)

    assert failed is messages
    assert all(result is source for result, source in zip(failed, messages))
    assert messages == original
    assert not any(message.get("compacted") for message in failed)
    assert not summarizer.last_call_cacheable
    assert shaper._summary_cache == {}

    successful = await shape(shaper, messages, asynchronous)
    repeated = await shape(shaper, messages, asynchronous)

    assert successful == repeated
    assert successful[1]["compacted"]
    assert "report-green.txt" in successful[1]["content"]
    assert successful[-2:] == messages[-2:]
    assert messages == original
    assert len(calls) == (3 if cache_size == 0 else 2)

@pytest.mark.parametrize("asynchronous", [False, True])
async def test_valid_cached_summary_survives_an_intervening_failure(asynchronous):
    calls = []

    async def complete(request):
        calls.append(request)
        if len(calls) == 2:
            raise RuntimeError("temporary provider failure")
        return SimpleNamespace(content="<summary>Save report-green.txt.</summary>")

    summarizer = ReadTimeSummarizer(complete)
    shaper = compactor(summarizer)
    messages = projection_history()
    first = await shape(shaper, messages, asynchronous)
    changed = copy.deepcopy(messages)
    changed[1]["content"] += "A different old discussion."

    assert await shape(shaper, changed, asynchronous) is changed
    assert not summarizer.last_call_cacheable
    assert await shape(shaper, copy.deepcopy(messages), asynchronous) == first
    assert len(calls) == 2
    assert len(shaper._summary_cache) == 1

@pytest.mark.parametrize("asynchronous", [False, True])
async def test_custom_summarizer_without_failure_flag_remains_usable(asynchronous):
    shaper = compactor(lambda _: "Save report-green.txt.", cache_size=0)
    messages = projection_history()

    result = await shape(shaper, messages, asynchronous)

    assert result[1]["compacted"]
    assert "report-green.txt" in result[1]["content"]

OLD_DISCUSSION = (
    "Previous plan used report-blue.txt. Architecture notes review components, tests, and error handling. " * 100
)[:8_000]

class ReportModel:
    def __init__(self, fail_summary):
        self.fail_summary = fail_summary
        self.active = False
        self.reads = 0
        self.selected = None
        self.requests = []
        self.summaries = []

    def context_window(self):
        return 128_000

    async def complete(self, messages, tools=None, **kwargs):
        usage = Usage(estimate_request_tokens(messages), 3, estimated=True)
        if tools is None:
            self.summaries.append(copy.deepcopy(messages))
            if self.fail_summary:
                raise RuntimeError("temporary provider failure")
            filename = "report-green.txt" if "report-green.txt" in json.dumps(messages) else "report-blue.txt"
            return LLMResponse(
                content=f"<summary>The latest user requires {filename} after reading the four files.</summary>",
                usage=usage, finish_reason="stop",
            )
        if not self.active:
            return LLMResponse(content=OLD_DISCUSSION, usage=usage, finish_reason="stop")
        self.requests.append(copy.deepcopy(messages))
        if self.reads < 4:
            name, arguments = "file_read", {"path": f"module{self.reads}.txt"}
            self.reads += 1
        elif self.selected is None:
            self.selected = "report-green.txt" if "report-green.txt" in json.dumps(messages) else "report-blue.txt"
            name, arguments = "file_write", {"path": self.selected, "mode": "create", "content": "Report saved.\n"}
        else:
            return LLMResponse(content="Report saved.", usage=usage, finish_reason="stop")
        return LLMResponse(
            content="Inspect the requested source and record the result.",
            tool_calls=[{"id": f"call-{len(self.requests)}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}], usage=usage, finish_reason="tool_calls",
        )

@pytest.mark.parametrize("mode", ["failure", "success", "uncompacted"])
async def test_default_session_keeps_latest_filename_after_four_real_reads(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    for index in range(4):
        (tmp_path / f"module{index}.txt").write_text("\n".join(
            f"entry_{line} = process_record(value_{line}, enabled=True, retry_count=3)" for line in range(400)
        ))
    model = ReportModel(fail_summary=mode != "success")
    saved = tmp_path / "session.json"
    session = build_session(
        agent=Agent(name="reader", system_prompt="Follow the latest user request.",
                    tools=builtin_tools("file_read", "file_write", headless=False)),
        llm=model, env=LocalEnvironment(str(tmp_path)), max_budget_tokens=None,
        max_steps=100, auto_save_path=str(saved),
        context_policy=ContextPolicy("no_history_compaction" if mode == "uncompacted" else "default"),
    )
    try:
        for index in range(30):
            if estimate_request_tokens(session.messages) >= 77_000:
                break
            await session.add_user_message(f"Historical discussion {index}.\n{OLD_DISCUSSION}")
            await session.run_loop()
        assert 77_000 <= estimate_request_tokens(session.messages) < 95_000
        model.active = True
        current_reference = ""
        if mode == "success":
            # Cross the trigger again even when the earlier summary is reused.
            current_reference = "\nCurrent project reference.\n" + (
                "Preserve current module behavior and caller-owned inputs.\n" * 4_500
            )
        await session.add_user_message(
            "The old blue filename is replaced. Read module0.txt through module3.txt and write the report "
            "to report-green.txt. Preserve this exact output name."
            + current_reference
        )

        assert await session.run_loop() == "Report saved."

        assert (tmp_path / "report-green.txt").read_text() == "Report saved.\n"
        assert not (tmp_path / "report-blue.txt").exists()
        assert all("report-green.txt" in json.dumps(request) for request in model.requests)
        assert all(estimate_request_tokens(request) + 8_192 < 128_000 for request in model.requests)
        assert "report-green.txt" in saved.read_text()
        tool_results = [message for message in session.messages if message.get("role") == "tool"]
        assert len(tool_results) == 5
        assert all("entry_399" in result["content"] for result in tool_results[:4])
        assert bool(model.summaries) == (mode != "uncompacted")
        if mode != "uncompacted":
            assert any("report-green.txt" in json.dumps(request) for request in model.summaries)
        if mode == "failure":
            assert not any(message.get("compacted") for request in model.requests for message in request)
    finally:
        await session.aclose()

async def test_failed_summary_leaves_real_overflow_to_the_existing_controlled_stop(monkeypatch):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")

    class OverflowError(RuntimeError):
        status_code = 400

    class OverflowModel:
        def __init__(self):
            self.summaries = []
            self.requests = []

        def context_window(self):
            return 128_000

        async def complete(self, messages, tools=None, **kwargs):
            if "detailed summary of the conversation" in messages[-1].get("content", ""):
                assert estimate_request_tokens(messages) + kwargs["max_output_tokens"] < 128_000
                self.summaries.append(copy.deepcopy(messages))
                raise RuntimeError("temporary provider failure")
            assert estimate_request_tokens(messages) > 128_000
            self.requests.append(copy.deepcopy(messages))
            raise OverflowError("maximum context length exceeded")

    model = OverflowModel()
    session = build_session(agent=Agent(name="overflow", system_prompt="Follow the user."),
                            llm=model, max_budget_tokens=None)
    try:
        session.messages = [
            {"role": "system", "content": "Follow the user."},
            *[{"role": "user", "content": "Earlier discussion.\n" + "x" * 12_000} for _ in range(4)],
            *[{"role": "user", "content": "Current required context.\n" + "y" * 100_000} for _ in range(4)],
        ]
        original = copy.deepcopy(session.messages)

        await session.run_loop()

        assert session.state.phase is SessionPhase.STOPPED
        assert "context overflow" in session.state.terminal_reason
        assert len(model.requests) == len(model.summaries) == 2
        assert model.requests == [original, original]
        assert session.messages[:-1] == original
    finally:
        await session.aclose()
