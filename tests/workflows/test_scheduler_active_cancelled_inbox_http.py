"""Accepted teammate work survives cancellation at ordinary step boundaries."""

from __future__ import annotations

import asyncio
import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.tools.base import Tool
from opencollab.adapters.tools.message import MessageAgentTool
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler, SchedulerTurnError
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport


def tool_response(request, name, arguments):
    return http_response(request, 200, json={
        "id": "chatcmpl_test", "object": "chat.completion", "created": 1,
        "model": json.loads(request.content)["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    })

class WriteQueued(Tool):
    name = "write_queued"
    description = "Write a local receipt for the accepted teammate request."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, path):
        self.path = path
        self.calls = 0

    async def execute_with_runtime(self, params, runtime):
        self.calls += 1
        self.path.write_text("queued work completed\n", encoding="utf-8")
        return "queue work recorded"

class HeldWork(Tool):
    name = "held_work"
    description = "Finish the current local operation at its ordinary step boundary."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, path, started, release):
        self.path = path
        self.started = started
        self.release = release
        self.calls = 0

    async def execute_with_runtime(self, params, runtime):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        self.path.write_text("original work completed\n", encoding="utf-8")
        return "original work recorded"

class Scenario:
    def __init__(self, root, boundary, *, independent_recipient=False):
        self.root = root
        self.boundary = boundary
        self.independent_recipient = independent_recipient
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.sent = asyncio.Event()
        self.send_independent = asyncio.Event()
        self.independent_done = asyncio.Event()
        self.lead_requests = []
        self.sender_requests = []
        self.observer_requests = []
        self.writer = WriteQueued(root / "queued.txt")
        self.held = HeldWork(root / "original.txt", self.started, self.release)

    async def handler(self, request):
        body = json.loads(request.content)
        role = body["messages"][0]["content"]
        if "ROLE_LEAD" in role:
            self.lead_requests.append(body)
            if len(self.lead_requests) == 1:
                if self.boundary == "tool":
                    return tool_response(request, "held_work", {})
                self.started.set()
                await self.release.wait()
                return completion_http_response(request, text="original answer")
            last_user = next(message["content"] for message in reversed(body["messages"]) if message["role"] == "user")
            if last_user.startswith("new request"):
                return completion_http_response(request, text="new answer")
            if "QUEUE_PAYLOAD" in last_user:
                if not self.writer.calls:
                    return tool_response(request, "write_queued", {})
                return completion_http_response(request, text="queued answer")
            return completion_http_response(request, text="original answer")
        if "ROLE_OBSERVER" in role:
            self.observer_requests.append(body)
            self.independent_done.set()
            return completion_http_response(request, text="independent answer")
        self.sender_requests.append(body)
        if len(self.sender_requests) == 1:
            await self.started.wait()
            return tool_response(request, "message_agent", {
                "to_aid": 0, "summary": "continue work", "content": "QUEUE_PAYLOAD execute write_queued",
            })
        self.sent.set()
        if len(self.sender_requests) == 2 and self.independent_recipient:
            await self.send_independent.wait()
            return tool_response(request, "message_agent", {
                "to_aid": 2, "summary": "independent work", "content": "INDEPENDENT_PAYLOAD",
            })
        return completion_http_response(request, text="sender answer")

    def build_spawn_session(self, **kwargs):
        return build_session(
            agent=Agent(
                name=kwargs["role"], system_prompt="ROLE_SENDER", tools=[MessageAgentTool(kwargs["scheduler"])],
            ),
            llm=self.client, env=kwargs["env"], aid=kwargs["aid"], max_budget_tokens=kwargs["budget"],
            seed_user_messages=[{"role": "user", "content": kwargs["task"]}],
        )

    def scheduler(self):
        return Scheduler(
            session_factory=self, worktree_pool=WorktreePool(str(self.root), use_worktrees=False),
            event_sink=EventBus(), max_budget_tokens=100_000,
        )

    def lead_agent(self):
        return Agent(name="lead", system_prompt="ROLE_LEAD", tools=[self.writer, self.held])

    def register_peer(self, scheduler, peer):
        scheduler.table.add(SessionControlBlock(aid=peer.state.aid, parent_aid=0, agent=peer.agent, state=peer.state))
        scheduler._sessions[peer.state.aid] = peer
        peer.scheduler = scheduler

@pytest.mark.parametrize("boundary", ["generation", "tool"])
@pytest.mark.parametrize("continuation", ["ordinary", "cancelled", "restored"])
async def test_accepted_teammate_turn_survives_ordinary_cancel_boundary(monkeypatch, tmp_path, boundary, continuation):
    scenario = Scenario(tmp_path, boundary)
    install_sdk_transport(monkeypatch, "openai", scenario.handler)
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    async with LLMClient(
        model="gpt-4o", provider="openai", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0,
    ) as client:
        scenario.client = client
        scheduler = scenario.scheduler()
        lead = build_session(agent=scenario.lead_agent(), llm=client)
        scheduler.register_lead(lead)
        sender_aid = await scheduler.spawn(0, "sender", "send update")
        cancel_event = asyncio.Event()
        active = asyncio.create_task(scheduler.run("original request", cancel_event=cancel_event))
        try:
            await asyncio.wait_for(scenario.sent.wait(), 2)
            assert len(scheduler._message_inbox[0]) == 1
            if continuation != "ordinary":
                cancel_event.set()
                assert not active.done()
            scenario.release.set()
            if boundary == "tool" and continuation != "ordinary":
                with pytest.raises(SchedulerTurnError, match="interrupted by user"):
                    await asyncio.wait_for(active, 2)
            else:
                await asyncio.wait_for(active, 2)

            path = tmp_path / "lead.json"
            lead.save(str(path))
            snapshot = OpenCollab.read_session_snapshot(path)
            pending_count = int(continuation != "ordinary")
            assert len(scheduler._message_inbox.get(0, [])) == pending_count
            assert len(lead.state.pending_user_messages) == pending_count
            assert len(snapshot.get("pending_messages", [])) == pending_count
            assert snapshot["session_state"]["pending_external_user_turn"] is None
            assert scenario.writer.calls == 1 - pending_count
            assert scenario.writer.path.exists() is (continuation == "ordinary")
            assert scenario.held.calls == int(boundary == "tool")
            if boundary == "tool":
                assert scenario.held.path.read_text(encoding="utf-8") == "original work completed\n"
            if pending_count:
                assert snapshot["pending_messages"][0]["delivery_status"] == "held_after_cancel"
                assert len(scenario.lead_requests) == 1
                assert 0 in scheduler._cancelled_turn_inbox_holds
                assert cancel_event.is_set()
                assert not any("QUEUE_PAYLOAD" in str(message) for message in lead.messages)
            assert not scheduler._turn_lease

            if continuation == "restored":
                sender_path = tmp_path / "sender.json"
                scheduler._sessions[sender_aid].save(str(sender_path))
                await scheduler.cleanup()
                scheduler = scenario.scheduler()
                lead = load_session(str(path), agent=scenario.lead_agent(), llm=client)
                scheduler.register_lead(lead)
                sender = load_session(
                    str(sender_path),
                    agent=Agent(name="sender", system_prompt="ROLE_SENDER", tools=[MessageAgentTool(scheduler)]),
                    llm=client,
                )
                scenario.register_peer(scheduler, sender)
                assert len(scheduler._message_inbox[0]) == 1
                assert 0 in scheduler._cancelled_turn_inbox_holds

            retry_start = len(scenario.lead_requests)
            assert await asyncio.wait_for(scheduler.run("new request"), 2) == "new answer"
            assert scenario.writer.calls == 1
            assert scenario.writer.path.read_text(encoding="utf-8") == "queued work completed\n"
            assert scenario.held.calls == int(boundary == "tool")
            if pending_count:
                recovery_requests = scenario.lead_requests[retry_start:]
                assert len(recovery_requests) == 3
                assert "QUEUE_PAYLOAD" in str(recovery_requests[0]["messages"][-1])
                assert all("new request" not in str(body["messages"]) for body in recovery_requests[:2])
                assert recovery_requests[-1]["messages"][-1]["content"].startswith("new request")
            assert sum("QUEUE_PAYLOAD" in str(message) for message in lead.messages) == 1
            assert lead.phase is SessionPhase.DONE
            assert lead.state.pending_user_messages == []
            assert scheduler._message_inbox.get(0, []) == []
            assert 0 not in scheduler._cancelled_turn_inbox_holds
            assert not scheduler._turn_lease
        finally:
            scenario.release.set()
            await scheduler.cleanup()
            await asyncio.gather(active, return_exceptions=True)

async def test_cancelled_public_turn_allows_other_recipient_messages(monkeypatch, tmp_path):
    scenario = Scenario(tmp_path, "generation", independent_recipient=True)
    install_sdk_transport(monkeypatch, "openai", scenario.handler)
    async with LLMClient(
        model="gpt-4o", provider="openai", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0,
    ) as client:
        scenario.client = client
        scheduler = scenario.scheduler()
        lead = build_session(agent=scenario.lead_agent(), llm=client)
        scheduler.register_lead(lead)
        await scheduler.spawn(0, "sender", "send update")
        observer = build_session(agent=Agent(name="observer", system_prompt="ROLE_OBSERVER"), llm=client, aid=2)
        scenario.register_peer(scheduler, observer)
        cancel_event = asyncio.Event()
        active = asyncio.create_task(scheduler.run("original request", cancel_event=cancel_event))
        try:
            await asyncio.wait_for(scenario.sent.wait(), 2)
            cancel_event.set()
            scenario.send_independent.set()
            await asyncio.wait_for(scenario.independent_done.wait(), 2)
            assert len(scenario.observer_requests) == 1
            assert "INDEPENDENT_PAYLOAD" in str(scenario.observer_requests[0]["messages"])
            assert 2 not in scheduler._cancelled_turn_inbox_holds
            scenario.release.set()
            await asyncio.wait_for(active, 2)
            assert len(scheduler._message_inbox[0]) == 1
            assert scenario.writer.calls == 0
            assert await asyncio.wait_for(scheduler.run("new request"), 2) == "new answer"
            assert scenario.writer.calls == 1
        finally:
            scenario.release.set()
            scenario.send_independent.set()
            await scheduler.cleanup()
            await asyncio.gather(active, return_exceptions=True)
