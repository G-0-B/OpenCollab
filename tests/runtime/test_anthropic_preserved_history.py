"""Native SDK sessions retain valid thinking and retire edited-prefix blocks."""

from __future__ import annotations

import copy
import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.tools.fs import FileReadTool, FileWriteTool
from opencollab.application.shaping import AutoCompactShaper, PerToolResultBudgetShaper, ToolOutputClearShaper
from opencollab.application.steering import READS_NUDGE_HARD
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from tests.support.provider_sdk_http import http_response, install_sdk_transport

_THINKING = {"thinking", "redacted_thinking"}

def _blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else content

def _plain_messages(messages):
    result = []
    for message in messages:
        blocks = _blocks(message["content"])
        kept = [block for block in blocks if block["type"] not in _THINKING]
        if not kept:
            continue
        if result and result[-1]["role"] == message["role"]:
            result[-1]["content"].extend(copy.deepcopy(kept))
        else:
            result.append({"role": message["role"], "content": copy.deepcopy(kept)})
    return result

def _prefix(body, messages):
    return {"system": body.get("system"), "tools": body.get("tools"), "messages": _plain_messages(messages)}

def _signatures(body):
    return [block.get("signature") or block.get("data") for message in body["messages"]
            for block in _blocks(message["content"]) if block["type"] in _THINKING]

class _BoundThinkingProvider:
    """Compare the documented prefixes through the real Anthropic SDK wire."""

    def __init__(self, reads=2, *, thinking=True, correction=False):
        self.reads = reads
        self.thinking = thinking
        self.correction = correction
        self.requests = []
        self.origins = {}
        self.retired = set()

    def __call__(self, request):
        body = json.loads(request.content)
        signatures = _signatures(body)
        assert not self.retired.intersection(signatures)
        previous = None
        preceding = []
        for message in body["messages"]:
            blocks = []
            for block in _blocks(message["content"]):
                if block["type"] in _THINKING:
                    key = block.get("signature") or block.get("data")
                    origin = self.origins[key]
                    prefix = preceding + ([{"role": message["role"], "content": blocks}] if blocks else [])
                    assert block == origin["block"]
                    assert _prefix(body, prefix) == origin["prefix"]
                    if previous is not None:
                        assert origin["previous"] == previous
                    previous = key
                blocks.append(block)
            preceding.append(message)
        self.retired.update(set(self.origins) - set(signatures))
        self.requests.append(body)
        number = len(self.requests)
        if self.correction and body.get("tool_choice", {}).get("type") == "tool":
            return http_response(request, 400, json={"type": "error", "error": {
                "type": "invalid_request_error",
                "message": 'tool_choice: type "tool" and "any" are not supported for this model.',
            }})
        content = []
        if self.thinking:
            content = [
                {"type": "thinking", "thinking": "", "signature": f"signed-{number}"},
                {"type": "redacted_thinking", "data": f"redacted-{number}"},
            ]
        if number <= self.reads:
            content.append({"type": "tool_use", "id": f"read-{number}", "name": "file_read",
                            "input": {"path": "source.txt"}})
        elif self.correction and [tool["name"] for tool in body.get("tools", [])] == ["structured_output"]:
            content.append({"type": "tool_use", "id": "capture", "name": "structured_output", "input": {"answer": 42}})
        else:
            content.append({"type": "text", "text": "done"})
        for index, block in enumerate(content):
            if block["type"] in _THINKING:
                key = block.get("signature") or block.get("data")
                prefix = body["messages"] + ([{"role": "assistant", "content": content[:index]}] if index else [])
                self.origins[key] = {"block": copy.deepcopy(block), "prefix": _prefix(body, prefix),
                                     "previous": previous}
                previous = key
        return http_response(request, 200, json={
            "id": f"message-{number}", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": "tool_use" if content[-1]["type"] == "tool_use" else "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 40},
        })

def _agent(*, thinking=True, model="claude-opus-5-5"):
    return Agent(name="reader", system_prompt="Read files and answer.", tools=[FileReadTool()],
                 model=model, provider="anthropic", thinking=thinking,
                 thinking_params={"thinking": {"type": "adaptive"}}, max_tokens_per_step=1024)

def _client(model="claude-opus-5-5"):
    return LLMClient(model=model, provider="anthropic",
                     api_key="controlled-test",  # pragma: allowlist secret
                     base_url="https://controlled.invalid/v1", max_retries=0)

@pytest.fixture(autouse=True)
def _local_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    (tmp_path / "source.txt").write_text("file evidence\n" * 100)

@pytest.mark.parametrize("restore", [False, True], ids=["same-session", "restore"])
@pytest.mark.parametrize("thinking", [False, True], ids=["plain", "thinking"])
async def test_native_steering_prefix_survives_tool_rounds_and_new_user_turns(monkeypatch, tmp_path, restore, thinking):
    peer = _BoundThinkingProvider(thinking=thinking)
    install_sdk_transport(monkeypatch, "anthropic", peer)
    async with _client() as client:
        session = build_session(agent=_agent(thinking=thinking), llm=client, env=LocalEnvironment(str(tmp_path)),
                                max_budget_tokens=100_000, max_steps=20)
        try:
            await session.add_user_message("Read twice and answer")
            assert await session.run_loop() == "done"
            original = copy.deepcopy(session.messages)
            session.save(str(tmp_path / "session.json"))
            if restore:
                await session.aclose()
                session = load_session(str(tmp_path / "session.json"), agent=_agent(thinking=thinking), llm=client,
                                       env=LocalEnvironment(str(tmp_path)), max_budget_tokens=100_000, max_steps=20)
                assert session.messages == original
            await session.add_user_message("Continue with the same evidence")
            assert await session.run_loop() == "done"
        finally:
            await session.aclose()
    assert len(peer.requests) == 4
    assert peer.retired == set()
    if thinking:
        assert _signatures(peer.requests[2]) == ["signed-1", "redacted-1", "signed-2", "redacted-2"]

@pytest.mark.parametrize("mode", ["clear", "summary", "first-trim"])
async def test_compacted_native_view_survives_restore_and_disabling_the_shaper(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    peer = _BoundThinkingProvider(reads=4)
    install_sdk_transport(monkeypatch, "anthropic", peer)
    if mode == "clear":
        shaper = ToolOutputClearShaper(trigger_tokens=2, target_tokens=1, keep_recent=1)
    elif mode == "summary":
        shaper = AutoCompactShaper(trigger_tokens=2, target_tokens=1, keep_recent_groups=0,
                                  summarizer=lambda _: "Earlier work")
    else:
        shaper = PerToolResultBudgetShaper(max_chars=200)
    async with _client() as client:
        session = build_session(agent=_agent(), llm=client, env=LocalEnvironment(str(tmp_path)), shaper=shaper,
                                max_budget_tokens=100_000, max_steps=20)
        try:
            await session.add_user_message("Inspect four files")
            assert await session.run_loop() == "done"
            raw = copy.deepcopy(session.messages)
            assert len([message for message in raw if message["role"] == "tool"]) == 4
            assert all("file evidence" in message["content"] for message in raw if message["role"] == "tool")
            before = copy.deepcopy(session.state.anthropic_history)
            session.save(str(tmp_path / "session.json"))
            await session.aclose()
            session = load_session(str(tmp_path / "session.json"), agent=_agent(), llm=client,
                                   env=LocalEnvironment(str(tmp_path)), max_budget_tokens=100_000, max_steps=20)
            assert session.state.anthropic_history == before
            assert session.messages == raw
            session.runner.shaper = None
            await session.add_user_message("Continue using the saved context")
            assert await session.run_loop() == "done"
        finally:
            await session.aclose()
    assert len(peer.requests) == 6
    if mode == "first-trim":
        assert peer.retired == set()
        assert "signed-1" in _signatures(peer.requests[-1])
    else:
        assert peer.retired
    assert _signatures(peer.requests[-1])[-2:] == ["signed-5", "redacted-5"]

async def test_appended_system_and_changed_tools_retire_old_native_thinking(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    peer = _BoundThinkingProvider(reads=0)
    install_sdk_transport(monkeypatch, "anthropic", peer)
    async with _client() as client:
        session = build_session(agent=_agent(), llm=client, env=LocalEnvironment(str(tmp_path)),
                                max_budget_tokens=100_000, max_steps=20)
        try:
            await session.add_user_message("Answer")
            assert await session.run_loop() == "done"
            session.state.append_message({"role": "system", "content": "A new system instruction"})
            await session.add_user_message("Continue")
            assert await session.run_loop() == "done"
            assert _signatures(peer.requests[-1]) == []
            await session.add_user_message("Finish")
            await session.runner.call_llm(None)
            assert _signatures(peer.requests[-1]) == []
            assert peer.requests[-1].get("tools") is None
        finally:
            await session.aclose()

async def test_chronological_compaction_marker_retains_earlier_valid_thinking(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    peer = _BoundThinkingProvider(reads=0)
    install_sdk_transport(monkeypatch, "anthropic", peer)

    class SummarizeLastUser:
        def shape(self, messages):
            return [*messages[:-1], {"role": "system", "compacted": True, "content": "Earlier requirements"}]

    async with _client() as client:
        session = build_session(agent=_agent(), llm=client, max_budget_tokens=100_000)
        try:
            await session.add_user_message("Answer the first question")
            assert await session.run_loop() == "done"
            session.runner.shaper = SummarizeLastUser()
            await session.add_user_message("The next lengthy requirements " * 200)
            assert await session.run_loop() == "done"
            assert _signatures(peer.requests[-1]) == ["signed-1", "redacted-1"]
            assert peer.requests[-1]["system"] == peer.requests[0]["system"]
            session.runner.shaper = None
            await session.add_user_message("Continue")
            assert await session.run_loop() == "done"
            assert _signatures(peer.requests[-1]) == ["signed-1", "redacted-1", "signed-2", "redacted-2"]
        finally:
            await session.aclose()

async def test_hard_steering_restricts_tools_and_removes_old_prefix_thinking(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "on")
    peer = _BoundThinkingProvider(reads=0)
    install_sdk_transport(monkeypatch, "anthropic", peer)
    agent = _agent()
    agent.tools.append(FileWriteTool())
    async with _client() as client:
        session = build_session(agent=agent, llm=client, env=LocalEnvironment(str(tmp_path)),
                                max_budget_tokens=100_000, max_steps=20)
        try:
            await session.add_user_message("Answer")
            assert await session.run_loop() == "done"
            await session.add_user_message("Make the requested edit")
            session.state.turn.reads_since_last_edit = READS_NUDGE_HARD
            await session.runner.call_llm(session.runner.build_tool_schemas())
            assert [tool["name"] for tool in peer.requests[-1]["tools"]] == ["file_write"]
            assert _signatures(peer.requests[-1]) == []
            assert session.runner._pending_tool_allowlist == frozenset({"file_write", "apply_patch"})
            assert any(message.get("provider_state") for message in session.messages)
        finally:
            await session.aclose()

async def test_auxiliary_summary_omits_bound_thinking_and_preserves_the_transcript(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    peer = _BoundThinkingProvider(reads=0)
    install_sdk_transport(monkeypatch, "anthropic", peer)
    async with _client("claude-fable-5-1") as client:
        session = build_session(agent=_agent(model="claude-fable-5-1"), llm=client, max_budget_tokens=100_000)
        try:
            await session.add_user_message("Retain this evidence")
            assert await session.run_loop() == "done"
            original = copy.deepcopy(session.messages)
            await session._context_summarizer.asummarize(session.messages[1:])
            assert session.messages == original
        finally:
            await session.aclose()
    assert _signatures(peer.requests[-1]) == []
    assert "Retain this evidence" in json.dumps(peer.requests[-1])
    assert peer.requests[-1].get("system") is None
    assert peer.requests[-1].get("tools") is None

async def test_structured_correction_keeps_evidence_and_capture_only_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    peer = _BoundThinkingProvider(correction=True)
    install_sdk_transport(monkeypatch, "anthropic", peer)
    app = OpenCollab(tmp_path, model="claude-fable-5-1", provider="anthropic",
                     api_key="controlled-test", base_url="https://controlled.invalid/v1",  # pragma: allowlist secret
                     config={"thinking": True, "thinking_params": {"thinking": {"type": "adaptive"}},
                             "max_output_tokens": 4096, "llm_max_retries": 0})

    async def flow(ctx, _args):
        return await ctx.agent("Read twice and return the sum", tools=[FileReadTool()], budget=80_000,
                               schema={"type": "object", "properties": {"answer": {"type": "integer"}},
                                       "required": ["answer"]})

    result = await app.workflow(flow, budget=100_000, max_steps=20, timeout=20, concurrency=1,
                                trace=False, artifacts=tmp_path / "artifacts")
    assert result.ok
    assert result.output == {"answer": 42}
    assert len(peer.requests) == 5
    assert {tool["name"] for tool in peer.requests[2]["tools"]} == {"file_read", "structured_output"}
    for body in peer.requests[3:]:
        assert [tool["name"] for tool in body["tools"]] == ["structured_output"]
        assert _signatures(body) == []
        assert "file evidence" in json.dumps(body)
    assert peer.requests[-1]["tool_choice"] == {"type": "auto"}

@pytest.mark.parametrize("model,provider,expected", [
    ("claude-fable-5-1", "anthropic", True),
    ("cloud/claude-opus-5-5-20261001", " Anthropic ", True),
    ("claude-sonnet-5-5", "anthropic", True),
    ("claude-haiku-5-5", "anthropic", True),
    ("claude-sonnet-4-6", "anthropic", False),
    ("claude-fable-5", "anthropic", False),
    ("claude-mythos-5-1", "anthropic", False),
    ("deepseek-v4-pro", "anthropic", False),
    ("claude-opus-5-5", "openai", False),
])
async def test_prefix_constraint_uses_native_transport_and_documented_models(model, provider, expected):
    async with LLMClient(model=model, provider=provider, api_key="controlled-test",  # pragma: allowlist secret
                         base_url="https://controlled.invalid/v1") as client:
        assert client.requires_thinking_prefix is expected
