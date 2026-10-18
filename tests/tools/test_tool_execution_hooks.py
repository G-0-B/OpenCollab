"""Real configured hooks follow the resolved tool while calls keep provider names."""

from __future__ import annotations

import copy
import json

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.hooks import ShellHookRunner
from opencollab.adapters.tools.fs import FileWriteTool
from opencollab.application.event_bus import EventBus
from opencollab.application.hooks import HookEventSubscriber
from opencollab.bootstrap import build_session
from opencollab.bootstrap.team_config import load_team_config
from opencollab.domain.agent import Agent
from tests.support.session_characterization_test_support import FakeLLMClient, llm_response
from tests.support.tool_execution_test_support import FakeTracer, build_use_case, tool_call


def _hook_bus(workspace):
    config_path = workspace / "team.yaml"
    config_path.write_text(
        "roles:\n"
        "  lead:\n"
        "    tools: [file_write]\n"
        "    prompt: Edit a file.\n"
        "hooks:\n"
        "  PreToolUse:\n"
        "    - matcher: file_write\n"
        "      command: cat >> pre.jsonl && printf '\\n' >> pre.jsonl\n"
        "    - matcher: bash\n"
        "      command: printf unexpected > mismatched-pre.txt\n"
        "  PostToolUse:\n"
        "    - matcher: file_write\n"
        "      command: cat >> post.jsonl && printf '\\n' >> post.jsonl\n"
        "    - matcher: bash\n"
        "      command: printf unexpected > mismatched-post.txt\n",
        encoding="utf-8",
    )
    config = load_team_config(str(workspace), path=str(config_path))
    events = []
    bus = EventBus(events.append)
    bus.subscribe(HookEventSubscriber(ShellHookRunner(config.hooks, workspace=workspace)))
    return bus, events

def _write_call(name, index=0):
    return tool_call(
        name=name,
        arguments=json.dumps(
            {"path": f"edited-{index}.txt", "mode": "create", "content": f"written {index}"}
        ),
        call_id=f"write-{index}",
    )

def _assert_hooks(workspace, events, call_ids):
    for filename, hook_event in (("pre.jsonl", "PreToolUse"), ("post.jsonl", "PostToolUse")):
        payloads = [json.loads(line) for line in (workspace / filename).read_text().splitlines()]
        assert [payload["hook_event_name"] for payload in payloads] == [hook_event] * len(call_ids)
        assert [payload["tool"] for payload in payloads] == ["file_write"] * len(call_ids)
        assert [payload["tool_call_id"] for payload in payloads] == call_ids
    assert not (workspace / "mismatched-pre.txt").exists()
    assert not (workspace / "mismatched-post.txt").exists()
    assert [
        (event.type, event.data["tool"], event.data["tool_call_id"])
        for event in events
        if event.type in {"tool_start", "tool_end"}
    ] == [
        (event_type, "file_write", call_id)
        for call_id in call_ids
        for event_type in ("tool_start", "tool_end")
    ]

@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["file_write", "File_Write", "FILE_WRITE"])
@pytest.mark.parametrize("call_count", [1, 2])
async def test_session_hooks_use_resolved_name_and_keep_original_transcript(tmp_path, name, call_count):
    bus, events = _hook_bus(tmp_path)
    calls = [_write_call(name, index) for index in range(call_count)]
    original_calls = copy.deepcopy(calls)
    llm = FakeLLMClient([
        llm_response(tool_calls=calls, finish_reason="tool_calls"),
        llm_response(content="done"),
    ])
    session = build_session(
        agent=Agent(name="lead", system_prompt="Edit a file.", tools=[FileWriteTool()]),
        env=LocalEnvironment(str(tmp_path)),
        llm=llm,
        event_sink=bus,
    )
    try:
        await session.add_user_message("Write the files.")
        assert await session.run_loop() == "done"
    finally:
        await session.aclose()

    for index in range(call_count):
        assert (tmp_path / f"edited-{index}.txt").read_text() == f"written {index}"
    assert calls == original_calls
    assistant = next(message for message in session.state.messages if message.get("tool_calls"))
    assert assistant["tool_calls"] == original_calls
    provider_assistant = next(message for message in llm.calls[1]["messages"] if message.get("tool_calls"))
    assert provider_assistant["tool_calls"] == original_calls
    results = [message for message in session.state.messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == [call["id"] for call in calls]
    assert all(message["content"].startswith("Created/wrote") for message in results)
    _assert_hooks(tmp_path, events, [call["id"] for call in calls])

@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["file_write", "File_Write"])
async def test_deferred_entry_runs_real_tool_and_configured_hooks(tmp_path, name):
    bus, events = _hook_bus(tmp_path)
    use_case, _ = build_use_case(
        agent=Agent(name="lead", system_prompt="Edit a file.", tools=[FileWriteTool()]),
        environment=LocalEnvironment(str(tmp_path)),
        event_publisher=bus,
    )
    call = _write_call(name)
    original_call = copy.deepcopy(call)

    ref, result = await use_case.execute_deferred(call)

    assert ref is None
    assert result.startswith("Created/wrote")
    assert (tmp_path / "edited-0.txt").read_text() == "written 0"
    assert call == original_call
    _assert_hooks(tmp_path, events, [call["id"]])

@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["process", "preflight", "deferred"])
async def test_unknown_tool_keeps_original_diagnostic_without_execution_events(tmp_path, entry):
    bus, events = _hook_bus(tmp_path)
    tracer = FakeTracer()
    use_case, _ = build_use_case(
        agent=Agent(name="lead", system_prompt="Edit a file.", tools=[FileWriteTool()]),
        environment=LocalEnvironment(str(tmp_path)),
        event_publisher=bus,
        tracer=tracer,
    )
    call = _write_call("Missing_Tool")
    original_call = copy.deepcopy(call)
    expected = "Error: unknown tool 'Missing_Tool'."
    if entry == "deferred":
        assert await use_case.execute_deferred(call) == (None, expected)
    else:
        expected += " Available: ['file_write']"
        if entry == "preflight":
            errors = use_case.preflight_tool_batch([call])
            assert errors == [expected]
            result = use_case.preflight_rejection_result([call], errors)
        else:
            result = await use_case.process([call])
        assert result.messages_to_append == [
            {"role": "tool", "tool_call_id": call["id"], "content": expected}
        ]
        assert tracer.steps[0]["payload"]["tool"] == "Missing_Tool"
    assert call == original_call
    assert events == []
    assert not (tmp_path / "edited-0.txt").exists()
    assert not (tmp_path / "pre.jsonl").exists()
    assert not (tmp_path / "post.jsonl").exists()
