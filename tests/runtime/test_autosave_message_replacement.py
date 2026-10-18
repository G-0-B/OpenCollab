"""Message replacement survives a frozen autosave finishing later."""

from __future__ import annotations

import asyncio
import threading

import pytest

from opencollab.adapters.storage import SessionStore
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from tests.support.session_run_test_support import FakeLLM


class PausedStore(SessionStore):
    def __init__(self):
        self.block_next = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.replace_from = []
        self.active = 0
        self.max_active = 0

    def append_snapshot_delta(self, *args, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.replace_from.append(kwargs["replace_from"])
        try:
            if self.block_next:
                self.block_next = False
                self.started.set()
                if not self.release.wait(2):
                    raise TimeoutError("bounded test append pause expired")
            return super().append_snapshot_delta(*args, **kwargs)
        finally:
            self.active -= 1

@pytest.mark.parametrize("replace_during_append", [False, True])
@pytest.mark.parametrize("initial_saves", [0, 2])
def test_message_replacement_survives_the_single_autosave_owner(
    tmp_path, replace_during_append, initial_saves,
):
    path = str(tmp_path / "autosave.json")
    store = PausedStore()
    session = build_session(
        agent=Agent(name="replacement", system_prompt="system"),
        llm=FakeLLM(), store=store, auto_save_path=path,
    )
    session.messages = [
        {"role": "system", "content": "old system"},
        {"role": "user", "content": "old user"},
    ]

    async def scenario():
        for _ in range(initial_saves):
            await session.enqueue_auto_save()
        store.block_next = True
        owner = session.enqueue_auto_save()
        try:
            assert await asyncio.to_thread(store.started.wait, 1)
            if not replace_during_append:
                store.release.set()
                await owner
            session.messages = [
                {"role": "system", "content": "new system"},
                {"role": "user", "content": "new user"},
            ]
            pending = session.enqueue_auto_save()
            if replace_during_append:
                assert pending is owner
            store.release.set()
            await pending
        finally:
            store.release.set()
            await asyncio.gather(*session.pending_cleanup_tasks)

    asyncio.run(scenario())
    assert store.max_active == 1
    assert store.replace_from[-1] == 0
    assert session.persistence_errors == ()
    restored = SessionStore().load_snapshot(path, "system")
    assert [message["content"] for message in restored["messages"]] == ["new system", "new user"]
    session.save(path)
    assert [message["content"] for message in SessionStore().load_messages(path, "system")] == [
        "new system", "new user"
    ]
