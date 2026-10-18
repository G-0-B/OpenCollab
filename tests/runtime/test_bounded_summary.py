"""Window-bounded summaries preserve chronological requests and tool groups."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from opencollab.application._session_run_shared import GenerationTimeoutError, _TeamBudgetStop, _TokenBudgetStop
from opencollab.application.compaction_prompt import build_summary_request
from opencollab.application.compaction_summary import ReadTimeSummarizer
from opencollab.application.run_control import _RunCancelled, _RunControlPolicyError
from opencollab.domain.token_estimation import estimate_request_tokens

WINDOW = 6_000
OUTPUT = 800
REVISION = re.compile(r"USER_REVISION \d+ \w+")

def revisions():
    return [
        {"role": "user", "content": f"USER_REVISION {index} {color}\n" + "x" * 6_000}
        for index, color in enumerate(("blue", "green", "red", "yellow", "purple"))
    ]

def assert_paired(messages):
    pending = set()
    for message in messages:
        if message["role"] == "tool":
            pending.remove(message["tool_call_id"])
        else:
            assert not pending
            pending.update(call["id"] for call in message.get("tool_calls", ()))
    assert not pending

class RevisionModel:
    def __init__(self):
        self.requests = []

    async def complete(self, request):
        assert estimate_request_tokens(request) + OUTPUT <= WINDOW
        assert_paired(request)
        self.requests.append(request)
        changes = [match.group() for message in request for match in REVISION.finditer(message.get("content") or "")]
        return SimpleNamespace(content="<summary>" + "\n".join(changes) + "</summary>")

async def test_chunked_summary_carries_all_revisions_in_order_and_estimates_the_full_request():
    model = RevisionModel()
    measured = []

    def estimate(request):
        assert "detailed summary of the conversation" in request[-1]["content"]
        measured.append(request)
        return estimate_request_tokens(request)

    summarizer = ReadTimeSummarizer(
        model.complete, context_window=WINDOW, max_output_tokens=OUTPUT,
        estimate_tokens=estimate, transcript_path="session.json",
        custom_instructions="Keep the entire chronological revision record.",
    )
    segment = revisions()
    original = [message.copy() for message in segment]
    summary = await summarizer.asummarize(segment)

    assert len(model.requests) > 1
    assert [message for request in model.requests for message in request if message in segment] == segment
    assert REVISION.findall(summary) == [f"USER_REVISION {i} {color}" for i, color in enumerate(
        ("blue", "green", "red", "yellow", "purple"),
    )]
    assert "USER_REVISION 0 blue" in model.requests[1][0]["content"]
    assert "applying later revisions in chronological order" in model.requests[1][0]["content"]
    assert all(request in measured for request in model.requests)
    assert all("Keep the entire chronological revision record." in request[-1]["content"] for request in model.requests)
    assert summary.count("Transcript location: session.json") == 1
    assert summarizer.last_call_cacheable
    assert segment == original

async def test_chunking_keeps_multi_call_tool_exchange_together():
    model = RevisionModel()
    tool_group = [
        {"role": "assistant", "content": "inspect files", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            for call_id in ("first", "second")
        ]},
        {"role": "tool", "tool_call_id": "first", "content": "a" * 2_800},
        {"role": "tool", "tool_call_id": "second", "content": "b" * 2_800},
    ]
    segment = [revisions()[0], *tool_group, *revisions()[1:]]
    summarizer = ReadTimeSummarizer(model.complete, context_window=WINDOW, max_output_tokens=OUTPUT)

    summary = await summarizer.asummarize(segment)

    containing = [request for request in model.requests if tool_group[0] in request]
    assert len(containing) == 1
    position = containing[0].index(tool_group[0])
    assert containing[0][position:position + 3] == tool_group
    assert REVISION.findall(summary)[-1] == "USER_REVISION 4 purple"

@pytest.mark.parametrize("window", [None, 0, WINDOW, 128_000])
async def test_short_summary_still_uses_one_original_request(window):
    requests = []

    async def complete(request):
        requests.append(request)
        return SimpleNamespace(content="<summary>green</summary>")

    segment = [{"role": "user", "content": "Keep the field green."}]
    summarizer = ReadTimeSummarizer(complete, context_window=window, max_output_tokens=OUTPUT)

    assert await summarizer.asummarize(segment) == "Summary:\ngreen"
    assert requests == [build_summary_request(segment)]
    assert summarizer.last_call_cacheable

@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), "<summary> </summary>"])
async def test_chunk_failure_stops_before_the_tail_and_never_returns_a_partial_summary(failure):
    requests = []

    async def complete(request):
        requests.append(request)
        if len(requests) == 1:
            return SimpleNamespace(content="<summary>earlier user changes</summary>")
        if isinstance(failure, Exception):
            raise failure
        return SimpleNamespace(content=failure)

    summarizer = ReadTimeSummarizer(
        complete, context_window=WINDOW, max_output_tokens=OUTPUT, transcript_path="session.json",
    )

    with pytest.raises(ValueError, match="could not include all earlier messages") as caught:
        await summarizer.asummarize(revisions())

    assert len(requests) == 2
    assert "earlier user changes" in requests[1][0]["content"]
    assert "Transcript location: session.json" in str(caught.value)
    assert not summarizer.last_call_cacheable

async def test_oversized_tool_group_reports_the_unprocessed_group_without_a_model_call():
    requests = []

    async def complete(request):
        requests.append(request)

    segment = [
        {"role": "assistant", "tool_calls": [
            {"id": "large", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "large", "content": "x" * 30_000},
    ]
    summarizer = ReadTimeSummarizer(
        complete, context_window=WINDOW, max_output_tokens=OUTPUT, transcript_path="session.json",
    )

    with pytest.raises(ValueError, match="Message group 1") as caught:
        await summarizer.asummarize(segment)

    assert "Transcript location: session.json" in str(caught.value)
    assert requests == []
    assert not summarizer.last_call_cacheable

async def test_a_summary_that_leaves_no_room_for_the_next_group_stops_with_its_transcript():
    requests = []

    async def complete(request):
        requests.append(request)
        return SimpleNamespace(content="<summary>" + "previous summary " * 1_500 + "</summary>")

    summarizer = ReadTimeSummarizer(
        complete, context_window=WINDOW, max_output_tokens=OUTPUT, transcript_path="session.json",
    )

    with pytest.raises(ValueError, match="Message group") as caught:
        await summarizer.asummarize(revisions())

    assert len(requests) == 1
    assert "Transcript location: session.json" in str(caught.value)
    assert not summarizer.last_call_cacheable

@pytest.mark.parametrize("failure", [
    GenerationTimeoutError("deadline"), _TeamBudgetStop(),
    _TokenBudgetStop(reserved_input_tokens=10, remaining_budget=9),
    _RunCancelled(), _RunControlPolicyError("policy"), asyncio.CancelledError(),
])
async def test_later_chunk_preserves_budget_policy_timeout_and_cancellation(failure):
    requests = []

    async def complete(request):
        requests.append(request)
        if len(requests) == 2:
            raise failure
        return SimpleNamespace(content="<summary>first chunk</summary>")

    summarizer = ReadTimeSummarizer(complete, context_window=WINDOW, max_output_tokens=OUTPUT)

    with pytest.raises(type(failure)) as caught:
        await summarizer.asummarize(revisions())

    assert caught.value is failure
    assert len(requests) == 2
    assert not summarizer.last_call_cacheable

async def test_budget_sizing_stop_leaves_the_real_request_to_prepare_its_budget():
    calls = []
    failure = _TokenBudgetStop(reserved_input_tokens=10, remaining_budget=9)

    def output_limit(_request):
        raise failure

    async def complete(request):
        calls.append(request)
        raise failure

    summarizer = ReadTimeSummarizer(
        complete, context_window=WINDOW, max_output_tokens=OUTPUT, output_limit=output_limit,
    )

    with pytest.raises(_TokenBudgetStop) as caught:
        await summarizer.asummarize(revisions())

    assert caught.value is failure
    assert len(calls) == 1
    assert calls[0][:-1] == revisions()[:1]
    assert not summarizer.last_call_cacheable

async def test_window_limit_is_recomputed_after_budget_extension_adds_a_final_prompt(tmp_path):
    from opencollab.adapters.env import LocalEnvironment
    from opencollab.application.run_control import BudgetDecision, RunControl
    from opencollab.bootstrap import build_session
    from opencollab.domain.agent import Agent
    from tests.support.session_run_test_support import llm_response

    requests = []
    final_prompt = "Complete the edits with this final context.\n" + "x" * 6_000

    class Model:
        def context_window(self):
            return WINDOW

        async def complete(self, messages, **kwargs):
            requests.append((messages, kwargs))
            assert "_summary_output_limit" not in kwargs
            assert estimate_request_tokens(messages) + kwargs["max_output_tokens"] <= WINDOW
            return llm_response(content="<summary>green</summary>")

    environment = LocalEnvironment(str(tmp_path))
    session = build_session(
        agent=Agent(name="bounded", system_prompt="Keep the active request.", max_tokens_per_step=5_000),
        llm=Model(), env=environment, max_budget_tokens=20_000,
        run_control=RunControl(
            initial_soft_budget_tokens=3_500,
            decide_budget=lambda _: BudgetDecision(20_000, final_prompt=final_prompt),
        ),
    )
    try:
        summary = await session._context_summarizer.asummarize([{"role": "user", "content": "Make it green."}])
        assert summary == "Summary:\ngreen"
        assert len(requests) == 1
        messages, kwargs = requests[0]
        assert {"role": "system", "content": final_prompt} in messages
        assert kwargs["max_output_tokens"] == WINDOW - estimate_request_tokens(messages)
        assert session.runner.max_budget_tokens == 20_000
    finally:
        await session.aclose()
        await environment.cleanup()

async def test_standalone_summary_uses_native_provider_requests_with_configured_output(monkeypatch, tmp_path):
    from opencollab.adapters.llm.client import LLMClient
    from opencollab.bootstrap.container import _build_summarizer
    from opencollab.domain.agent import Agent
    from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport

    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    bodies = []

    async def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert estimate_request_tokens(body["messages"]) + body["max_tokens"] <= WINDOW
        assert body["max_tokens"] == OUTPUT
        changes = [match.group() for message in body["messages"] for match in REVISION.finditer(
            message.get("content") or "",
        )]
        return completion_http_response(request, text="<summary>" + "\n".join(changes) + "</summary>")

    install_sdk_transport(monkeypatch, "openai", handler)
    async with LLMClient(
        model="gpt-4o", provider="openai", context_window=WINDOW, max_retries=0,
        api_key="controlled-test",  # pragma: allowlist secret
    ) as client:
        summarizer = _build_summarizer(
            Agent(name="standalone", system_prompt="Keep user requests.", max_tokens_per_step=OUTPUT),
            client, client, 30.0, None,
        )
        summary = await summarizer.asummarize(revisions())

    assert len(bodies) > 1
    assert REVISION.findall(summary) == [f"USER_REVISION {i} {color}" for i, color in enumerate(
        ("blue", "green", "red", "yellow", "purple"),
    )]
    assert summarizer.last_call_cacheable
