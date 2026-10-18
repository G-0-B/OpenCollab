"""Slow persistence keeps snapshot preparation and event-loop timers runnable."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from opencollab.adapters.storage import SessionStore
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from tests.support.session_run_test_support import FakeLLM


class SlowCheckpointStore(SessionStore):
    def __init__(self):
        self.pause_next = False
        self.fail_on_release = False
        self.started = threading.Event()
        self.release = threading.Event()

    def checkpoint_snapshot(self, *args, **kwargs):
        if self.pause_next:
            self.pause_next = False
            self.started.set()
            if not self.release.wait(2):
                raise TimeoutError("bounded checkpoint pause expired")
            if self.fail_on_release:
                raise OSError("manual checkpoint failed")
        return super().checkpoint_snapshot(*args, **kwargs)

@pytest.mark.parametrize("manual_failure", [False, True])
async def test_manual_checkpoint_io_keeps_the_loop_runnable_and_latest_snapshot_recoverable(
    tmp_path, manual_failure,
):
    path = str(tmp_path / "session.json")
    store = SlowCheckpointStore()
    session = build_session(
        agent=Agent(name="loop-progress", system_prompt="system"),
        llm=FakeLLM(), store=store, auto_save_path=path,
    )
    session.messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": f"old-{index}"} for index in range(5)
    ]
    session.state.turn.seen_result_hashes = {"old-result"}
    await session.enqueue_auto_save()
    # A full manual save accepts a shorter public message list. The later
    # frozen delta must recover against that short base as well as the old one.
    session.messages[:] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "manual"},
    ]
    session.state.turn.seen_result_hashes = {"manual-result"}
    store.pause_next = True
    store.fail_on_release = manual_failure
    save = asyncio.create_task(asyncio.to_thread(session.save, path))
    assert await asyncio.to_thread(store.started.wait, 1)
    # A thread release also bounds the broken implementation, whose blocking
    # lock prevents the event loop from running its own release or timeout.
    fallback_release = threading.Timer(0.5, store.release.set)
    fallback_release.start()
    timer = None
    try:
        session.messages.extend(
            {"role": "assistant", "content": f"latest-{index}"} for index in range(5)
        )
        session.state.turn.seen_result_hashes = {"latest-result"}
        session.used_tokens = 29
        ticked = asyncio.Event()
        timer = asyncio.get_running_loop().call_later(0.02, ticked.set)
        before = time.monotonic()
        owner = session.enqueue_auto_save()
        await ticked.wait()
        delay = time.monotonic() - before
        assert delay < 0.2
        assert not save.done()
        assert not store.release.is_set()
        store.release.set()
        if manual_failure:
            with pytest.raises(OSError, match="manual checkpoint failed"):
                await save
        else:
            await save
        await owner
        restored = load_session(path, agent=session.agent, llm=FakeLLM(), store=store)
        assert restored.messages == session.messages
        assert restored.used_tokens == 29
        assert restored.state.turn.seen_result_hashes == {"latest-result"}
        assert session.persistence_errors == ()
        session.messages.append({"role": "assistant", "content": "after slow save"})
        await session.enqueue_auto_save()
        restored = load_session(path, agent=session.agent, llm=FakeLLM(), store=store)
        assert restored.messages == session.messages
    finally:
        store.release.set()
        fallback_release.cancel()
        if timer is not None:
            timer.cancel()
        await asyncio.gather(save, *session.pending_cleanup_tasks, return_exceptions=True)

async def test_slow_autosave_compaction_keeps_message_replacement_and_timers_runnable(tmp_path):
    path = str(tmp_path / "session.json")
    store = SlowCheckpointStore()
    session = build_session(
        agent=Agent(name="compaction-progress", system_prompt="system"),
        llm=FakeLLM(), store=store, auto_save_path=path,
    )
    session.messages = [{"role": "user", "content": "old"}]
    store.pause_next = True
    owner = session.enqueue_auto_save()
    assert await asyncio.to_thread(store.started.wait, 1)
    fallback_release = threading.Timer(0.5, store.release.set)
    fallback_release.start()
    timer = None
    try:
        ticked = asyncio.Event()
        timer = asyncio.get_running_loop().call_later(0.02, ticked.set)
        before = time.monotonic()
        session.messages = [{"role": "user", "content": "latest"}]
        assert session.enqueue_auto_save() is owner
        await ticked.wait()
        assert time.monotonic() - before < 0.2
        assert not store.release.is_set()
        store.release.set()
        await owner
        assert store.load_messages(path, "system")[-1]["content"] == "latest"
        assert session.persistence_errors == ()
    finally:
        store.release.set()
        fallback_release.cancel()
        if timer is not None:
            timer.cancel()
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)
