"""Size Responses history after native conversion, before usage is measured."""

from __future__ import annotations

import uuid
from copy import deepcopy

import pytest

from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.responses_messages import messages_to_input
from opencollab.adapters.llm.types import Usage
from opencollab.application.context_pressure import ContextPressure
from opencollab.application.shaping.pipeline import context_token_estimator
from opencollab.bootstrap.container import _build_default_shaper
from opencollab.domain.session import SessionState
from tests.support.session_run_loop_test_support import FakeLLM, build_runner, llm_response


@pytest.fixture
async def client():
    client = LLMClient(model="gpt-fake", api_key=uuid.uuid4().hex,
                       wire_protocol="responses", context_window=34_000)
    try:
        yield client
    finally:
        await client.close()

def _history():
    return [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "I located the failing assertion."},
        {"role": "user", "content": "Continue."},
        {"role": "assistant", "content": "I will edit the implementation."},
    ]

@pytest.mark.parametrize("part_type", ["text", "input_text", "output_text", "image_url"])
async def test_restored_block_metadata_does_not_change_responses_pressure(client, part_type):
    history = _history()
    block = {"type": part_type, "text": "Visible answer."}
    if part_type == "image_url":
        block = {"type": part_type, "image_url": {"url": "https://example.test/image.png", "detail": "high"}}
    history[2]["content"] = [block]
    sent = messages_to_input(history)
    reserved = client.estimate_request_tokens(history)
    pressure = ContextPressure(lambda message: client.estimate_context_tokens([message]))
    before = pressure(history)
    block["annotations"] = "LOCAL_DISPLAY_ANNOTATION " * 10_000
    if part_type == "image_url":
        block["image_url"]["local_metadata"] = block["annotations"]
    original = deepcopy(history)
    summaries = []
    shaper = _build_default_shaper(client, lambda segment: summaries.append(segment) or "Summary.")

    with context_token_estimator(pressure):
        shaped, reports = shaper.shape_with_report(history)

    assert messages_to_input(history) == sent
    assert client.estimate_request_tokens(history) == reserved
    assert pressure(history) == before < 1_000
    assert summaries == []
    assert [report["rung"] for report in reports] == ["none"]
    assert reports[0]["tokens_before"] == reports[0]["tokens_after"] == before
    assert shaped == history == original

async def test_legacy_reasoning_is_excluded_before_output_measurement(client):
    history = _history()
    sent = messages_to_input(history)
    reserved = client.estimate_request_tokens(history)
    history[2]["reasoning_content"] = "reasoning " * 20_000
    history[2]["provider_state"] = {"local_display_copy": "reasoning " * 20_000}
    original = deepcopy(history)
    pressure = ContextPressure(lambda message: client.estimate_context_tokens([message]))

    assert messages_to_input(history) == sent
    assert client.estimate_request_tokens(history) == reserved
    assert pressure(history) < 1_000
    pressure.observe_output(history[2], Usage(output_tokens=10_000, reasoning_tokens=9_900))
    assert pressure([history[2]]) == client.estimate_context_tokens([history[2]]) < 1_000
    assert history == original

async def test_tool_fragments_count_native_fields_additively_and_keep_real_output(client):
    history = [
        {"role": "user", "content": "Inspect the test."},
        {"role": "assistant", "tool_calls": [{
            "id": "call_one", "type": "function",
            "function": {"name": "file_read", "arguments": '{"path":"test.py"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_one", "content": [{"type": "text", "text": "Assertion."}]},
    ]
    sent = messages_to_input(history)
    reserved = client.estimate_request_tokens(history)
    before = client.estimate_context_tokens(history)
    history[1]["tool_calls"][0]["function"]["local_metadata"] = "local data " * 20_000
    history[2]["content"][0]["annotations"] = "local data " * 20_000
    history[2]["name"] = "local tool name " * 20_000
    original = deepcopy(history)

    assert messages_to_input(history) == sent
    assert client.estimate_request_tokens(history) == reserved
    assert client.estimate_context_tokens(history) == before
    assert before == sum(client.estimate_context_tokens([message]) for message in history)
    assert history == original
    history[2]["content"][0]["text"] = "real tool output " * 20_000
    assert client.estimate_context_tokens(history) > before + 1_000

async def test_unsent_annotations_do_not_stop_a_session_before_calling_the_model(client):
    history = _history()
    budget = client.estimate_request_tokens(history) + 2_000
    history[2]["content"] = [{"type": "output_text", "text": "Found the assertion.",
                              "annotations": "LOCAL_DISPLAY_ANNOTATION " * 10_000}]
    llm = FakeLLM([llm_response(content="done", input_tokens=10, total_tokens=20)])
    llm.estimate_request_tokens = client.estimate_request_tokens
    llm.estimate_context_tokens = client.estimate_context_tokens
    runner = build_runner(state=SessionState(messages=history), llm=llm, max_budget_tokens=budget)

    assert await runner.run_loop() == "done"
    assert len(llm.calls) == 1

async def test_measured_usage_still_calibrates_the_converted_history(client):
    history = _history()
    history[2]["content"] = "real model input " * 20_000
    pressure = ContextPressure(lambda message: client.estimate_context_tokens([message]))
    original = deepcopy(history)

    assert pressure(history) > 20_000
    pressure.observe_request(history, Usage(input_tokens=2_000, output_tokens=100))
    assert 2_000 <= pressure(history) <= 2_000 + len(history)
    assert pressure(history) == sum(pressure([message]) for message in history)
    assert pressure(deepcopy(history)) == pressure(history)
    assert history == original
