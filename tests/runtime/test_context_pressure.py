"""Provider measurements control compaction without changing native replay."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid

import httpx
import openai
import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm._attempt_usage import _combine_attempt_usage
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.adapters.tools.fs import FileReadTool
from opencollab.application.context_pressure import ContextPressure
from opencollab.application.shaping import ShaperPipeline, ToolOutputClearShaper
from opencollab.application.shaping.pipeline import approx_messages_tokens, context_token_estimator
from opencollab.bootstrap.session_factory import build_session
from opencollab.domain.agent import Agent
from opencollab.domain.token_estimation import estimate_request_message_tokens
from tests.runtime.test_responses_http_contract import _response
from tests.support.responses_provider_test_support import function_item, message_item
from tests.support.session_run_loop_test_support import build_runner


def pressure():
    return ContextPressure(lambda message: estimate_request_message_tokens([message]))

def encrypted_message(index=0):
    return {
        "role": "assistant", "content": "Inspect the next source file.",
        "response_items": [{
            "type": "reasoning", "id": f"rs_{index}", "summary": [],
            "encrypted_content": "opaque" * 20_000,
        }],
    }

def test_measured_prefix_and_new_output_are_additive_and_preserve_history():
    history = [{"role": "user", "content": "fix the task"}, encrypted_message()]
    original = copy.deepcopy(history)
    estimator = pressure()
    assert estimator(history) > 40_000
    estimator.observe_request(history, Usage(input_tokens=2_000, output_tokens=100))
    assert 2_000 <= estimator(history) <= 2_002
    assert estimator(history) == sum(estimator([message]) for message in history)
    assert estimator(copy.deepcopy(history)) == estimator(history)

    new_assistant = encrypted_message(1)
    estimator.observe_output(new_assistant, Usage(input_tokens=2_000, output_tokens=900,
                                                 reasoning_tokens=800))
    tool = {"role": "tool", "tool_call_id": "new", "content": "x" * 3_000}
    assert estimator([new_assistant]) == 964
    assert estimator([tool]) == estimate_request_message_tokens([tool])
    assert estimator(history + [new_assistant, tool]) < 5_000
    assert history == original

def test_retries_use_final_input_and_estimated_usage_does_not_calibrate():
    history = [encrypted_message()]
    estimator = pressure()
    fallback = estimator(history)
    estimator.observe_request(history, Usage(input_tokens=10, estimated=True))
    assert estimator(history) == fallback
    estimator.observe_request(history, Usage(input_tokens=30_000, context_tokens=1_000))
    assert estimator(history) == 1_000
    changed = copy.deepcopy(history)
    changed[0]["response_items"][0]["encrypted_content"] += "new"
    assert estimator(changed) == estimate_request_message_tokens(changed)
    estimator.clear()
    assert estimator(history) == fallback

@pytest.mark.parametrize("failed", [None, Usage(input_tokens=500, output_tokens=7_000),
                                    Usage(input_tokens=500, output_tokens=7_000, estimated=True)])
def test_actual_retry_combination_keeps_final_measurement_separate_from_cost(failed):
    usage = _combine_attempt_usage([failed, Usage(input_tokens=1_000, output_tokens=100)])
    estimator = pressure()
    message = encrypted_message()
    estimator.observe_request([message], usage)
    assert estimator([message]) == 1_000
    new_message = encrypted_message(1)
    estimator.observe_output(new_message, usage)
    assert estimator([new_message]) == 164
    assert usage.estimated == (failed is None or failed.estimated)
    assert usage.total_tokens == 1_100 + (failed.total_tokens if failed is not None else 0)

async def test_estimator_scope_isolated_between_concurrent_sessions():
    messages = [encrypted_message()]

    async def size(tokens):
        estimator = pressure()
        estimator.observe_request(messages, Usage(input_tokens=tokens))
        with context_token_estimator(estimator):
            await asyncio.sleep(0)
            return approx_messages_tokens(messages)

    assert await asyncio.gather(size(100), size(200)) == [100, 200]
    assert approx_messages_tokens(messages) > 40_000

async def test_summary_usage_is_accounted_but_does_not_calibrate_main_history():
    runner = build_runner()
    history = [encrypted_message()]
    runner._context_pressure.observe_request(history, Usage(input_tokens=2_000))

    async def summarize(messages, *, on_response, **kwargs):
        response = LLMResponse(content="<summary>state</summary>",
                               usage=Usage(input_tokens=90_000, output_tokens=1_000))
        on_response(response)
        return response

    await runner._invoke_summary(summarize, [{"role": "user", "content": "summarize"}])
    assert runner.state.used_tokens == 91_000
    assert runner._context_pressure(history) == 2_000
    runner.reset_for_restore()
    assert runner._context_pressure(history) == estimate_request_message_tokens(history)

def test_real_pressure_still_clears_old_tools_and_reestimates_edited_content():
    history = [{"role": "system", "content": "system"}]
    for i in range(8):
        history.extend([
            {"role": "assistant", "tool_calls": [{"id": str(i), "type": "function",
             "function": {"name": "file_read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": "x" * 9_000},
        ])
    original = copy.deepcopy(history)
    estimator = pressure()
    estimator.observe_request(history, Usage(input_tokens=100_000))
    layer = ToolOutputClearShaper(trigger_tokens=95_000, target_tokens=71_250, keep_recent=2)
    with context_token_estimator(estimator):
        shaped, reports = ShaperPipeline((layer,)).shape_with_report(history)
    assert reports[0]["rung"] == "tool_output_clear"
    assert reports[0]["tokens_before"] >= 100_000
    assert reports[0]["tokens_after"] < 95_000
    assert history == original
    assert shaped[-1] == history[-1]

@pytest.mark.parametrize("protocol", ["responses", "chat_completions"])
async def test_provider_projection_avoids_aliases_and_follows_thinking_policy(protocol):
    client = LLMClient(model="gpt-fake", api_key=uuid.uuid4().hex, wire_protocol=protocol)
    try:
        message = encrypted_message()
        message["reasoning_content"] = "visible reasoning" * 1_000
        projected = {"role": "assistant", "response_items": message["response_items"]}
        if protocol == "responses":
            assert client.estimate_context_tokens([message]) == client.estimate_context_tokens([projected])
        else:
            assert client.estimate_context_tokens([message]) == client.estimate_context_tokens([
                {"role": "assistant", "content": message["content"]},
            ])
            assert client.estimate_context_tokens([message], thinking=True) > 5_000
            estimator = ContextPressure(lambda m: client.estimate_context_tokens([m]))
            estimator.observe_output(message, Usage(output_tokens=10_000, reasoning_tokens=9_990))
            assert estimator([message]) == client.estimate_context_tokens([message])
    finally:
        await client.close()

async def test_anthropic_projection_counts_native_blocks_once():
    client = LLMClient(model="claude-sonnet-4-6", api_key=uuid.uuid4().hex, provider="anthropic")
    blocks = [{"type": "thinking", "thinking": "reasoning" * 100, "signature": "opaque" * 100},
              {"type": "text", "text": "answer"}]
    message = {"role": "assistant", "content": "answer", "reasoning_content": "reasoning" * 100,
               "provider_state": {"anthropic_content": blocks}}
    try:
        assert client.estimate_context_tokens([message]) == client.estimate_context_tokens([
            {"role": "assistant", "provider_state": {"anthropic_content": blocks}},
        ])
    finally:
        await client.close()

async def test_responses_session_keeps_encrypted_history_on_real_sdk_wire(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    requests = []
    reasoning_items = []
    for i in range(8):
        (tmp_path / f"source_{i}.py").write_text(f"# file {i}\n" + "x = 1\n" * 800)

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        step = len(requests) - 1
        assert [item for item in body["input"] if item.get("type") == "reasoning"] == reasoning_items
        assert body["reasoning"]["effort"] == "max"
        assert len([item for item in body["input"] if item.get("type") == "function_call_output"]) == step
        output = [message_item("done")]
        if step < 8:
            native = encrypted_message(step)["response_items"][0]
            reasoning_items.append(native)
            output = [native, function_item(f"call_{step}", "file_read",
                                           json.dumps({"path": f"source_{step}.py"}))]
        response = {**_response(f"resp_{step}"), "output": output}
        response["usage"].update(input_tokens=500 + step * 2_000,
                                 output_tokens=1_000, total_tokens=1_500 + step * 2_000)
        events = [{"type": "response.created", "response": {**response, "output": []}}]
        events.extend({"type": "response.output_item.done", "output_index": i, "item": item}
                      for i, item in enumerate(output))
        events.append({"type": "response.completed", "response": response})
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                              content="".join(f"data: {json.dumps(event)}\n\n" for event in events))

    client = LLMClient(model="gpt-5.6-luna", api_key=uuid.uuid4().hex, wire_protocol="responses",
                       context_window=128_000, max_retries=0)
    await client._openai.close()
    client._openai = openai.AsyncOpenAI(
        api_key=client._openai.api_key, base_url="https://context.invalid/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    agent = Agent(name="coder", system_prompt="inspect the files", model="gpt-5.6-luna",
                  tools=[FileReadTool()], reasoning_effort="max")
    session = build_session(agent=agent, llm=client, env=LocalEnvironment(str(tmp_path)),
                            max_steps=12, max_budget_tokens=None)
    try:
        await session.add_user_message("Inspect source_0.py through source_7.py.")
        assert await session.run_loop() == "done"
        assert len(requests) == 9
        assert session.used_tokens == sum(1_500 + step * 2_000 for step in range(9))
        assert len([m for m in session.messages if m.get("response_items")]) == 9
    finally:
        await client.close()
