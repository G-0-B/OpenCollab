"""Resolved native tools retain one execution policy across accepted spellings."""

from __future__ import annotations

import copy
import json
import subprocess

import pytest

from opencollab import OpenCollab
from opencollab.adapters.env import LocalEnvironment
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionState
from opencollab.tools import builtin_tools
from tests.support.session_characterization_test_support import FakeLLMClient, llm_response
from tests.support.tool_execution_test_support import build_use_case, tool_call


def _call(name, args, call_id):
    return tool_call(name=name, arguments=json.dumps(args), call_id=call_id)

@pytest.mark.parametrize(("observer", "writer"), [
    ("git_diff", "file_write"), ("Git_Diff", "File_Write"), ("GIT_DIFF", "FILE_WRITE"),
    ("file_read", "file_write"), ("File_Read", "File_Write"), ("FILE_READ", "FILE_WRITE"),
    ("grep", "file_write"), ("Grep", "File_Write"), ("GREP", "FILE_WRITE"),
])
async def test_public_sdk_observes_each_native_edit_through_accepted_names(tmp_path, observer, writer):
    for args in (
        ("init", "-q"), ("config", "user.email", "fixture@example.invalid"),
        ("config", "user.name", "Test"),
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "app.py").write_text("value = 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "\u6d4b\u8bd5\u521d\u59cb\u5316"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    observe_args = {"path": "app.py"}
    if observer.lower() == "grep":
        observe_args["pattern"] = "value"
    calls = [_call(observer, observe_args, "review-0")]
    for index in (1, 2):
        calls.extend([
            _call(writer, {
                "path": "app.py", "mode": "str_replace",
                "old_str": f"value = {index - 1}", "new_str": f"value = {index}",
            }, f"edit-{index}"),
            _call(observer, observe_args, f"review-{index}"),
        ])
    original_calls = copy.deepcopy(calls)
    llm = FakeLLMClient([
        *(llm_response(tool_calls=[call], finish_reason="tool_calls") for call in calls),
        llm_response(content="done"),
    ])

    result = await OpenCollab(tmp_path, model="fixture-model").agent(
        "Observe the current file after each edit.",
        tools=builtin_tools("file_write", "file_read", "git_diff", "grep"),
        system_prompt="Execute the tools and inspect each change.", llm=llm, trace=False,
    )

    assert result.ok and result.output == "done"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 2\n"
    assert calls == original_calls
    request_messages = llm.calls[-1]["messages"]
    recorded_calls = [call for message in request_messages for call in message.get("tool_calls", [])]
    assert recorded_calls == original_calls
    tool_messages = [message for message in request_messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [call["id"] for call in calls]
    for index in (1, 2):
        assert f"value = {index}" in tool_messages[2 * index]["content"]

@pytest.mark.parametrize("name", ["file_read", "File_Read", "FILE_READ"])
async def test_native_empty_read_alias_keeps_read_and_low_yield_observations(tmp_path, name):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    environment = LocalEnvironment(str(tmp_path))
    state = SessionState(messages=[])
    execution, _ = build_use_case(
        agent=Agent(name="lead", system_prompt="Read a file.", tools=list(builtin_tools("file_read"))),
        state=state, environment=environment,
    )
    try:
        result = await execution.process([_call(name, {"path": "empty.txt"}, "read")])
        result.apply_to(state)
    finally:
        await environment.cleanup()

    assert result.messages_to_append[0]["tool_call_id"] == "read"
    assert "0 lines total" in result.messages_to_append[0]["content"]
    assert result.reads_executed == state.turn.reads_since_last_edit == 1
    assert state.turn.distinct_evidence_count == 0
    assert state.turn.low_yield_since_progress == 1

async def test_native_read_alias_and_explicit_defaults_share_existing_repeat_limit(tmp_path):
    (tmp_path / "read.txt").write_text("content\n", encoding="utf-8")
    environment = LocalEnvironment(str(tmp_path))
    state = SessionState(messages=[])
    execution, _ = build_use_case(
        agent=Agent(name="lead", system_prompt="Read a file.", tools=list(builtin_tools("file_read"))),
        state=state, environment=environment,
    )
    results = []
    names = ("file_read", "File_Read", "FILE_READ")
    expected_key = execution.tool_call_hash("file_read", {"path": "read.txt"})
    try:
        for index in range(8):
            args = {"path": "read.txt"}
            if index % 2:
                args.update(offset=1, limit=500)
            result = await execution.process([_call(names[index % 3], args, f"read-{index}")])
            assert result.recent_hash_updates == [execution.tool_call_hash(names[index % 3], args)] == [expected_key]
            result.apply_to(state)
            results.append(result)
    finally:
        await environment.cleanup()

    assert all("1\tcontent" in result.messages_to_append[0]["content"] for result in results[:7])
    assert results[-1].messages_to_append[0]["content"].startswith("[Loop detected")
    assert results[-1].loop_detections[0].count == 8
    assert state.turn.reads_since_last_edit == 7
    assert len(set(state.turn.recent_call_hashes)) == 1
