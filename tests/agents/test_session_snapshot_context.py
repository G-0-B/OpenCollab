"""Runnable snapshots retain the effective context view and independent owners."""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.fs import FileReadTool
from opencollab.application.shaping import AutoCompactShaper, PerToolResultBudgetShaper
from opencollab.bootstrap import SnapshotSessionError, build_session, snapshot_session
from opencollab.bootstrap.agent_profiles import resolve_agent_profile
from opencollab.bootstrap.context_policy import ContextPolicy
from opencollab.domain.agent import Agent
from tests.runtime.test_async_compaction import history
from tests.runtime.test_compaction_accounting import AccountingModel
from tests.support.session_characterization_test_support import llm_response

_MARKER = "END_OF_FIXTURE"

class _ForkableEnvironment(LocalEnvironment):
    def fork_snapshot(self):
        directory = tempfile.mkdtemp(prefix="snapshot-", dir=Path(self.workspace).parent)
        shutil.copytree(self.workspace, directory, dirs_exist_ok=True)
        return _ForkableEnvironment(directory)

class _MarkerProvider:
    def __init__(self):
        self.calls = []

    def context_window(self):
        return 128_000

    async def complete(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        results = [message["content"] for message in messages if message["role"] == "tool"]
        if not results:
            return llm_response(tool_calls=[{
                "id": "read", "type": "function",
                "function": {"name": "file_read", "arguments": '{"path":"fixture.txt"}'},
            }], finish_reason="tool_calls")
        return llm_response(content="FOUND" if any(_MARKER in result for result in results) else "MISSING")

@pytest.mark.parametrize("configuration", ["policy", "injected", "profile"])
async def test_snapshot_keeps_the_configured_tool_result_view(tmp_path, configuration):
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "fixture.txt").write_text("x" * 19_800 + "\n" + _MARKER)
    options = {
        "policy": {"context_policy": ContextPolicy(name="no_history_compaction", tool_result_budget=32_000)},
        "injected": {"shaper": PerToolResultBudgetShaper(16_000, preserve_tail=True)},
        "profile": {"agent_profile": resolve_agent_profile("single2")},
    }[configuration]
    provider = _MarkerProvider()
    source = build_session(
        agent=Agent(name="reader", system_prompt="Inspect the file.", tools=[FileReadTool()]),
        env=_ForkableEnvironment(str(workspace)), llm=provider, **options,
    )
    snapshot = None
    try:
        await source.add_user_message("Read the fixture")
        assert await source.run_loop() == "FOUND"
        snapshot = snapshot_session(source)
        assert snapshot.runner.shaper is not source.runner.shaper
        for branch in (source, snapshot):
            await branch.add_user_message("Inspect the final marker in the saved result")
            assert await branch.run_loop() == "FOUND"
            assert any(_MARKER in message.get("content", "") for message in provider.calls[-1]
                       if message["role"] == "tool")
            assert any(_MARKER in message.get("content", "") for message in branch.messages
                       if message["role"] == "tool")
        if configuration == "injected":
            snapshot.runner.shaper.max_chars = 123
            assert source.runner.shaper.max_chars == 16_000
        else:
            snapshot.runner.shaper._shapers[0].max_chars = 123
            assert source.runner.shaper._shapers[0].max_chars == (32_000 if configuration == "policy" else 16_000)
    finally:
        if snapshot is not None:
            await snapshot.aclose()
        await source.aclose()

async def test_snapshot_summary_charges_its_own_budget_and_has_an_independent_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    workspace = tmp_path / "source"
    workspace.mkdir()
    provider = AccountingModel()
    source = build_session(
        agent=Agent(name="summary", system_prompt="Help the user."),
        env=_ForkableEnvironment(str(workspace)), llm=provider,
        max_budget_tokens=30_000, llm_timeout=0.5,
        auto_save_path=str(workspace / "session.json"),
    )
    source.messages = history()
    snapshot = None
    try:
        await source.runner._shape_and_trace(source.messages)
        assert source.used_tokens == 7_600
        snapshot = snapshot_session(source)
        source_compact = next(layer for layer in source.runner.shaper._shapers if isinstance(layer, AutoCompactShaper))
        branch_compact = next(
            layer for layer in snapshot.runner.shaper._shapers if isinstance(layer, AutoCompactShaper)
        )
        assert branch_compact is not source_compact
        assert branch_compact.summarizer is not source_compact.summarizer
        assert branch_compact.summarizer._transcript_path is None
        assert source_compact.summarizer._transcript_path == source.auto_save_path
        assert snapshot.runner._per_call_timeout == 0.5
        source_cache = copy.deepcopy(source_compact._summary_cache)
        assert await snapshot.run_loop() == "answer"
        assert snapshot.used_tokens == 15_210
        assert source.used_tokens == 7_600
        assert [call["summary"] for call in provider.calls] == [True, True, False]
        assert source_compact._summary_cache == source_cache
        branch_compact._summary_cache.clear()
        assert source_compact._summary_cache == source_cache
    finally:
        if snapshot is not None:
            await snapshot.aclose()
        await source.aclose()

class _OwnedShaper:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def shape(self, messages):
        self.calls.append(len(messages))
        return messages

async def test_injected_shaper_reuses_the_model_and_rebinds_session_owners(tmp_path):
    class Model(AccountingModel):
        def __deepcopy__(self, memo):
            raise AssertionError("the live model client must retain its owner")

    provider = Model()
    workspace = tmp_path / "source"
    workspace.mkdir()
    shaper = _OwnedShaper(provider)
    source = build_session(
        agent=Agent(name="owner", system_prompt="Help the user."),
        env=_ForkableEnvironment(str(workspace)), llm=provider, shaper=shaper,
    )
    shaper.runner = source.runner
    shaper.events = source.event_bus
    shaper.session = source
    snapshot = None
    try:
        await source.add_user_message("First request")
        assert await source.run_loop() == "answer"
        snapshot = snapshot_session(source)
        branch_shaper = snapshot.runner.shaper
        assert branch_shaper.provider is provider
        assert branch_shaper.runner is snapshot.runner
        assert branch_shaper.events is snapshot.event_bus
        assert branch_shaper.session is snapshot
        assert branch_shaper.calls == shaper.calls
        assert branch_shaper.calls is not shaper.calls
        await snapshot.add_user_message("Branch request")
        assert await snapshot.run_loop() == "answer"
        assert len(branch_shaper.calls) == 2
        assert len(shaper.calls) == 1
        assert source.used_tokens == 10
        assert snapshot.used_tokens == 20
    finally:
        if snapshot is not None:
            await snapshot.aclose()
        await source.aclose()

@pytest.mark.parametrize("copy_failure", ["exception", "same_object"])
async def test_snapshot_reports_an_uncloneable_injected_shaper(tmp_path, copy_failure):
    class Shaper(_OwnedShaper):
        def __deepcopy__(self, memo):
            if copy_failure == "exception":
                raise TypeError("this context owner cannot be copied")
            return self

    provider = AccountingModel()
    workspace = tmp_path / "source"
    workspace.mkdir()
    source = build_session(
        agent=Agent(name="owner", system_prompt="Help the user."),
        env=_ForkableEnvironment(str(workspace)), llm=provider, shaper=Shaper(provider),
    )
    try:
        with pytest.raises(SnapshotSessionError, match="context shaper"):
            snapshot_session(source)
        await source.add_user_message("Continue the source")
        assert await source.run_loop() == "answer"
    finally:
        await source.aclose()
