"""Recover a provider-rejected legacy prefix through the real SDK transport."""

from __future__ import annotations

import copy
import json

import anthropic
import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.client import LLMClient
from opencollab.application.run_control import RunControl
from opencollab.bootstrap import build_session, load_session
from tests.runtime.test_anthropic_preserved_history import (
    _agent,
    _BoundThinkingProvider,
    _client,
    _signatures,
)
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport
from tests.support.session_runtime_test_support import FakeTracer

_PREFIX_ERROR = (
    "messages.3.content.0: Invalid `signature` in `thinking` block. "
    "The block is bound to a different conversation. Remove the block, or set "
    'thinking.block_binding.prefix_mismatch_behavior to "drop_block".'
)

class _RejectingProvider(_BoundThinkingProvider):
    def __init__(self):
        super().__init__(reads=2)
        self.failure = None
        self.reject_plain = False
        self.rejected_signatures = {f"{kind}-{number}" for kind in ("signed", "redacted") for number in range(1, 4)}

    def __call__(self, request):
        body = json.loads(request.content)
        if self.failure is not None and (self.rejected_signatures.intersection(_signatures(body)) or self.reject_plain):
            self.requests.append(body)
            status, kind, message = self.failure
            return http_response(request, status, json={"type": "error", "error": {
                "type": kind, "message": message,
            }})
        return super().__call__(request)

@pytest.fixture(autouse=True)
def _local_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    (tmp_path / "source.txt").write_text("original tool evidence\n")

async def _snapshot(tmp_path, client, *, with_native_view=False):
    session = build_session(agent=_agent(), llm=client, env=LocalEnvironment(str(tmp_path)),
                            max_budget_tokens=100_000)
    try:
        await session.add_user_message("Read the evidence twice")
        assert await session.run_loop() == "done"
        path = tmp_path / "legacy.json"
        session.save(str(path))
        raw = json.loads(path.read_text())
        if not with_native_view:
            raw["session_state"].pop("anthropic_history")
        path.write_text(json.dumps(raw))
        return path, copy.deepcopy(session.messages)
    finally:
        await session.aclose()

@pytest.mark.parametrize("with_native_view", [False, True], ids=["legacy", "saved-after-old-failure"])
async def test_rejected_old_history_recovers_once_and_stays_recovered(monkeypatch, tmp_path, with_native_view):
    peer = _RejectingProvider()
    install_sdk_transport(monkeypatch, "anthropic", peer)
    tracer = FakeTracer()
    events = []
    async with _client() as client:
        path, original = await _snapshot(tmp_path, client, with_native_view=with_native_view)
        peer.failure = (400, "invalid_request_error", _PREFIX_ERROR)
        agent = _agent()
        agent.max_tokens_per_step = 100_000
        session = load_session(str(path), agent=agent, llm=client, tracer=tracer,
                               env=LocalEnvironment(str(tmp_path)), max_budget_tokens=10_000,
                               run_control=RunControl(on_event=events.append))
        try:
            before_tokens = session.state.used_tokens
            await session.add_user_message("Continue using the previous evidence")
            assert await session.run_loop() == "done"
            rejected, recovered = peer.requests[-2:]
            assert len(_signatures(rejected)) == 6
            assert _signatures(recovered) == []
            assert "original tool evidence" in json.dumps(recovered)
            assert "Read the evidence twice" in json.dumps(recovered)
            assert session.messages[:len(original)] == original
            assert session.state.used_tokens == before_tokens + 140
            view = session.state.anthropic_history
            reserved = client.estimate_request_tokens(view["messages"], view["tools"])
            assert recovered["max_tokens"] == 10_000 - before_tokens - reserved
            assert recovered["max_tokens"] > rejected["max_tokens"]
            assert len(peer.requests) == 5
            records = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
            assert [row["status"] for row in records[-2:]] == ["error", "success"]
            assert records[-2]["usage"]["total_tokens"] == 0
            assert records[-1]["usage"]["total_tokens"] == 140
            session.save(str(path))
        finally:
            await session.aclose()
        session = load_session(str(path), agent=agent, llm=client, env=LocalEnvironment(str(tmp_path)),
                               max_budget_tokens=100_000)
        try:
            await session.add_user_message("Continue after restoring")
            assert await session.run_loop() == "done"
            assert _signatures(peer.requests[-1]) == ["signed-5", "redacted-5"]
            assert len(peer.requests) == 6
        finally:
            await session.aclose()
    recoveries = [step for step in tracer.steps if step["step_type"] == "thinking_prefix_recovery"]
    assert len(recoveries) == 1
    assert recoveries[0]["payload"]["removed_blocks"] == 6
    assert recoveries[0]["payload"]["reason"] == "prefix_binding_mismatch"
    assert recoveries[0]["payload"]["error_type"] == "BadRequestError"
    assert len([event for event in events if event.type == "thinking_prefix_recovery"]) == 1
    usage_events = [event for event in events if event.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].data["total_tokens"] == 140
    assert peer.retired == {f"{kind}-{number}" for kind in ("signed", "redacted") for number in range(1, 4)}

async def test_valid_legacy_snapshot_keeps_all_native_thinking(monkeypatch, tmp_path):
    peer = _RejectingProvider()
    install_sdk_transport(monkeypatch, "anthropic", peer)
    tracer = FakeTracer()
    async with _client() as client:
        path, original = await _snapshot(tmp_path, client)
        session = load_session(str(path), agent=_agent(), llm=client, tracer=tracer,
                               env=LocalEnvironment(str(tmp_path)), max_budget_tokens=100_000)
        try:
            await session.add_user_message("Continue")
            assert await session.run_loop() == "done"
            assert len(peer.requests) == 4
            assert len(_signatures(peer.requests[-1])) == 6
            assert session.messages[:len(original)] == original
            assert peer.retired == set()
        finally:
            await session.aclose()
    assert not any(step["step_type"] == "thinking_prefix_recovery" for step in tracer.steps)

@pytest.mark.parametrize("status,kind,message", [
    (400, "invalid_request_error", "messages.1.content.0: Invalid `signature` in `thinking` block"),
    (400, "invalid_request_error", "tool schema is invalid"),
    (400, "invalid_request_error", "The block is bound to a different conversation."),
    (400, "other_error", _PREFIX_ERROR),
    (401, "authentication_error", _PREFIX_ERROR),
    (500, "api_error", _PREFIX_ERROR),
])
async def test_other_provider_errors_keep_their_original_failure(monkeypatch, tmp_path, status, kind, message):
    peer = _RejectingProvider()
    install_sdk_transport(monkeypatch, "anthropic", peer)
    async with _client() as client:
        path, original = await _snapshot(tmp_path, client)
        peer.failure = (status, kind, message)
        session = load_session(str(path), agent=_agent(), llm=client, env=LocalEnvironment(str(tmp_path)),
                               max_budget_tokens=100_000)
        try:
            await session.add_user_message("Continue")
            with pytest.raises(anthropic.APIStatusError) as caught:
                await session.run_loop()
            assert caught.value.status_code == status
            assert len(peer.requests) == 4
            assert len(_signatures(peer.requests[-1])) == 6
            assert session.messages[:len(original)] == original
        finally:
            await session.aclose()

async def test_recovery_failure_propagates_after_the_single_retry(monkeypatch, tmp_path):
    peer = _RejectingProvider()
    install_sdk_transport(monkeypatch, "anthropic", peer)
    async with _client() as client:
        path, original = await _snapshot(tmp_path, client)
        peer.failure = (400, "invalid_request_error", _PREFIX_ERROR)
        peer.reject_plain = True
        session = load_session(str(path), agent=_agent(), llm=client, env=LocalEnvironment(str(tmp_path)),
                               max_budget_tokens=100_000)
        try:
            await session.add_user_message("Continue")
            with pytest.raises(anthropic.BadRequestError):
                await session.run_loop()
            assert len(peer.requests) == 5
            assert _signatures(peer.requests[-1]) == []
            assert session.messages[:len(original)] == original
        finally:
            await session.aclose()

async def test_prefix_error_without_replayed_thinking_keeps_the_original_failure(monkeypatch, tmp_path):
    peer = _RejectingProvider()
    peer.failure = (400, "invalid_request_error", _PREFIX_ERROR)
    peer.reject_plain = True
    install_sdk_transport(monkeypatch, "anthropic", peer)
    tracer = FakeTracer()
    async with _client() as client:
        session = build_session(agent=_agent(), llm=client, tracer=tracer, max_budget_tokens=100_000)
        try:
            await session.add_user_message("Answer the first question")
            with pytest.raises(anthropic.BadRequestError):
                await session.run_loop()
            assert len(peer.requests) == 1
        finally:
            await session.aclose()
    assert not any(step["step_type"] == "thinking_prefix_recovery" for step in tracer.steps)

@pytest.mark.parametrize("model,provider,wire_protocol", [
    ("claude-sonnet-4-6", "anthropic", "chat_completions"),
    ("claude-opus-5-5", "openai", "chat_completions"),
    ("claude-opus-5-5", "openai", "responses"),
])
async def test_other_protocols_do_not_attempt_prefix_recovery(monkeypatch, tmp_path, model, provider, wire_protocol):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return http_response(request, 400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": _PREFIX_ERROR,
            }})
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, provider, handle)
    async with LLMClient(model=model, provider=provider, wire_protocol=wire_protocol,
                         api_key="controlled-test",  # pragma: allowlist secret
                         base_url="https://controlled.invalid/v1", max_retries=0) as client:
        session = build_session(agent=_agent(model=model), llm=client, max_budget_tokens=100_000)
        session.state.append_message({"role": "assistant", "content": "old answer", "provider_state": {
            "anthropic_content": [{"type": "thinking", "thinking": "", "signature": "old-signed"},
                                  {"type": "text", "text": "old answer"}],
        }})
        try:
            await session.add_user_message("Continue")
            with pytest.raises(Exception) as caught:
                await session.run_loop()
            assert getattr(caught.value, "status_code", None) == 400
            assert len(requests) == 1
            assert session.state.anthropic_history is None
        finally:
            await session.aclose()
