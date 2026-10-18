"""JSONL record boundaries preserve legal Unicode characters in strings."""

from __future__ import annotations

import asyncio
import json

import pytest

from opencollab.adapters.storage import SessionStore
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from tests.support.session_run_test_support import FakeLLM


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085", "\n", "\r", "\t"])
def test_autosave_restores_unicode_string_content(tmp_path, separator):
    path = str(tmp_path / "autosave.json")
    content = f"first{separator}second"
    session = build_session(
        agent=Agent(name="unicode", system_prompt="system"),
        llm=FakeLLM(),
        auto_save_path=path,
    )

    async def scenario():
        await session.add_user_message(content)
        await asyncio.gather(*session.pending_cleanup_tasks)

    asyncio.run(scenario())
    assert session.persistence_errors == ()
    assert SessionStore().load_snapshot(path, "system")["messages"][-1]["content"] == content
    session.save(path)
    assert SessionStore().load_snapshot(path, "system")["messages"][-1]["content"] == content

@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085", "\n", "\r", "\t"])
@pytest.mark.parametrize("record_separator", ["\n", "\r\n"])
def test_legacy_jsonl_restores_only_lf_delimited_records(tmp_path, separator, record_separator):
    messages = [
        {"role": "system", "content": f"first{separator}second"},
        {"role": "user", "content": "next record"},
    ]
    path = tmp_path / "legacy.jsonl"
    path.write_bytes(record_separator.join(json.dumps(row, ensure_ascii=False) for row in messages).encode("utf-8"))
    assert SessionStore().load_messages(str(path), "system") == messages
