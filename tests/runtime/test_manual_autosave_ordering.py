"""Manual checkpoints retain their ordering against frozen autosaves."""

from __future__ import annotations

import asyncio
import threading

import pytest

from opencollab.adapters.storage import SessionStore
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from tests.support.session_run_test_support import FakeLLM


class PausedAppendStore(SessionStore):
    def __init__(self):
        self.pause_next = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail_checkpoint = False

    def append_snapshot_delta(self, *args, **kwargs):
        if self.pause_next:
            self.pause_next = False
            self.started.set()
            if not self.release.wait(2):
                raise TimeoutError("bounded append pause expired")
        return super().append_snapshot_delta(*args, **kwargs)

    def checkpoint_snapshot(self, *args, **kwargs):
        if self.fail_checkpoint:
            raise OSError("manual checkpoint failed")
        return super().checkpoint_snapshot(*args, **kwargs)

def new_session(path, store):
    return build_session(
        agent=Agent(name="save-order", system_prompt="system"),
        llm=FakeLLM(), store=store, auto_save_path=str(path),
    )

@pytest.mark.parametrize("initial_saves", [0, 1, 2])
@pytest.mark.parametrize("replace_messages", [False, True])
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_manual_checkpoint_survives_older_autosave_and_next_delta(
    tmp_path, monkeypatch, initial_saves, replace_messages, cancel_owner,
):
    path = tmp_path / "session.json"
    store = PausedAppendStore()
    session = new_session(path, store)
    session.messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    session.state.turn.seen_result_hashes = {"old-result"}
    for _ in range(initial_saves):
        await session.enqueue_auto_save()
    store.pause_next = True
    owner = session.enqueue_auto_save()
    try:
        assert await asyncio.to_thread(store.started.wait, 1)
        if replace_messages:
            session.messages = [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "latest"},
            ]
        else:
            session.messages.append({"role": "assistant", "content": "latest"})
        session.used_tokens = 17
        session.state.turn.seen_result_hashes = {"manual-result"}
        monkeypatch.chdir(tmp_path)
        session.save("session.json")
        saved = store.load_snapshot(str(path), "system")
        assert saved["messages"][-1]["content"] == "latest"
        if cancel_owner:
            owner.cancel()
        store.release.set()
        if cancel_owner:
            with pytest.raises(asyncio.CancelledError):
                await owner
        else:
            await owner
        restored = load_session(str(path), agent=session.agent, llm=FakeLLM(), store=store)
        assert restored.messages == session.messages
        assert restored.used_tokens == 17
        assert restored.state.turn.seen_result_hashes == {"manual-result"}
        assert session.persistence_errors == ()

        session.messages.append({"role": "assistant", "content": "after manual"})
        session.state.turn.seen_result_hashes.add("next-result")
        await session.enqueue_auto_save()
        restored = load_session(str(path), agent=session.agent, llm=FakeLLM(), store=store)
        assert restored.messages == session.messages
        assert restored.state.turn.seen_result_hashes == {"manual-result", "next-result"}
        assert session.persistence_errors == ()
    finally:
        store.release.set()
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)

async def test_failed_manual_checkpoint_retains_the_older_durable_autosave(tmp_path):
    path = tmp_path / "session.json"
    store = PausedAppendStore()
    session = new_session(path, store)
    session.messages = [{"role": "user", "content": "old"}]
    store.pause_next = True
    owner = session.enqueue_auto_save()
    try:
        assert await asyncio.to_thread(store.started.wait, 1)
        session.messages = [{"role": "user", "content": "latest"}]
        store.fail_checkpoint = True
        with pytest.raises(OSError, match="manual checkpoint failed"):
            session.save(str(path))
        store.fail_checkpoint = False
        store.release.set()
        await owner
        assert store.load_messages(str(path), "system")[-1]["content"] == "old"
        assert session.persistence_errors == ()

        await session.enqueue_auto_save()
        assert store.load_messages(str(path), "system")[-1]["content"] == "latest"
        assert session.persistence_errors == ()
    finally:
        store.fail_checkpoint = False
        store.release.set()
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)
