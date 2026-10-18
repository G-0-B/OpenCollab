"""Exercise cancellation holds using public runs and real SDK HTTP calls."""

from __future__ import annotations

import asyncio
import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.tools.base import Tool
from opencollab.adapters.tools.message import MessageAgentTool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler, SchedulerTurnError
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport


def tool_reply(request, name, arguments):
    return http_response(request, 200, json={
        "id": "chatcmpl_local", "object": "chat.completion", "created": 1,
        "model": json.loads(request.content)["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": name, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    })

class ReceiptTool(Tool):
    name = "write_receipt"
    description = "Record the accepted teammate request in a local file."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, path):
        self.path = path
        self.calls = 0

    async def execute_with_runtime(self, params, runtime):
        self.calls += 1
        self.path.write_text("accepted teammate work executed\n", encoding="utf-8")
        return "receipt written"

class BoundaryTool(Tool):
    name = "finish_operation"
    description = "Finish a bounded local operation."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, started, release):
        self.started = started
        self.release = release

    async def execute_with_runtime(self, params, runtime):
        self.started.set()
        await self.release.wait()
        return "operation finished"

class Factory:
    def __init__(self, client):
        self.client = client

    def build_spawn_session(self, **kwargs):
        return build_session(
            agent=Agent(name=kwargs["role"], system_prompt="ROLE_SENDER",
                tools=[MessageAgentTool(kwargs["scheduler"])]),
            llm=self.client, aid=kwargs["aid"], max_budget_tokens=kwargs["budget"],
            seed_user_messages=[{"role": "user", "content": kwargs["task"]}],
        )

class Pool:
    async def acquire(self, role):
        return None

    async def release(self):
        pass

@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["generation", "tool"])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("followup", ["other_agent", "already_cancelled"])
async def test_cancelled_inbox_remains_held_across_restore_and_unrelated_runs(
    monkeypatch, tmp_path, boundary, restore, followup,
):
    started = asyncio.Event()
    release = asyncio.Event()
    queued = asyncio.Event()
    writer = ReceiptTool(tmp_path / "receipt.txt")
    operation = BoundaryTool(started, release)
    lead_calls = []
    sender_calls = []

    async def handler(request):
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        if "ROLE_LEAD" in system:
            lead_calls.append(body)
            if len(lead_calls) == 1:
                if boundary == "tool":
                    return tool_reply(request, "finish_operation", {})
                started.set()
                await release.wait()
                return completion_http_response(request, text="original answer")
            latest_user = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
            if "QUEUED_WORK" in latest_user:
                if writer.calls == 0:
                    return tool_reply(request, "write_receipt", {})
                return completion_http_response(request, text="queued answer")
            return completion_http_response(request, text="new answer")
        sender_calls.append(body)
        if len(sender_calls) == 1:
            await started.wait()
            return tool_reply(request, "message_agent", {
                "to_aid": 0, "summary": "accepted work", "content": "QUEUED_WORK write the receipt",
            })
        queued.set()
        return completion_http_response(request, text="sender answer")

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")
    async with LLMClient(model="gpt-4o", provider="openai", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0) as client:
        def make_scheduler():
            return Scheduler(session_factory=Factory(client), worktree_pool=Pool(),
                event_sink=EventBus(), max_budget_tokens=100_000)

        def lead_agent():
            return Agent(name="lead", system_prompt="ROLE_LEAD", tools=[writer, operation])

        scheduler = make_scheduler()
        lead = build_session(
            agent=lead_agent(), llm=client, auto_save_path=str(tmp_path / "lead-auto.json"),
        )
        scheduler.register_lead(lead)
        sender_aid = await scheduler.spawn(0, "sender", "send accepted work")
        cancel = asyncio.Event()
        active = asyncio.create_task(scheduler.run("original request", cancel_event=cancel))
        try:
            await asyncio.wait_for(queued.wait(), 2)
            cancel.set()
            release.set()
            if boundary == "tool":
                with pytest.raises(SchedulerTurnError, match="interrupted by user"):
                    await asyncio.wait_for(active, 2)
            else:
                assert await asyncio.wait_for(active, 2) == "original answer"
            assert writer.calls == 0
            assert len(lead.state.pending_user_messages) == 1
            assert 0 in scheduler._cancelled_turn_inbox_holds
            assert cancel.is_set()
            expected_phase = SessionPhase.STOPPED if boundary == "tool" else SessionPhase.DONE
            expected_reason = "interrupted by user" if boundary == "tool" else "completed"
            assert lead.phase is expected_phase
            assert lead.state.terminal_reason == expected_reason
            assert lead.state.pending_user_messages[0]["delivery_status"] == "held_after_cancel"
            await asyncio.gather(*lead.pending_cleanup_tasks)
            autosaved = OpenCollab.read_session_snapshot(lead.auto_save_path)
            assert autosaved["pending_messages"][0]["delivery_status"] == "held_after_cancel"

            lead_path = tmp_path / "lead.json"
            sender_path = tmp_path / "sender.json"
            lead.save(str(lead_path))
            scheduler._sessions[sender_aid].save(str(sender_path))
            snapshot = OpenCollab.read_session_snapshot(lead_path)
            if restore:
                await scheduler.cleanup()
                scheduler = make_scheduler()
                lead = load_session(
                    str(lead_path), agent=lead_agent(), llm=client,
                    auto_save_path=str(tmp_path / "restored-auto.json"),
                )
                scheduler.register_lead(lead)
                sender = load_session(str(sender_path), agent=Agent(name="sender",
                    system_prompt="ROLE_SENDER", tools=[MessageAgentTool(scheduler)]), llm=client)
                scheduler.table.add(SessionControlBlock(aid=sender_aid, parent_aid=0,
                    agent=sender.agent, state=sender.state))
                scheduler._sessions[sender_aid] = sender

            assert 0 in scheduler._cancelled_turn_inbox_holds
            assert lead.phase is expected_phase
            assert lead.state.terminal_reason == expected_reason
            if boundary == "generation":
                tokens_before_query = lead.used_tokens
                assert await lead.run_loop() == "original answer"
                assert lead.phase is SessionPhase.DONE
                assert lead.state.terminal_reason == "completed"
                assert lead.used_tokens == tokens_before_query
                assert len(lead_calls) == 1

            if followup == "other_agent":
                assert await asyncio.wait_for(scheduler.run_turn(sender_aid, "independent task"), 2) == "sender answer"
            else:
                next_cancel = asyncio.Event()
                next_cancel.set()
                with pytest.raises(SchedulerTurnError, match="interrupted by user"):
                    await asyncio.wait_for(scheduler.run("cancelled next request", cancel_event=next_cancel), 2)

            observation = {
                "boundary": boundary, "restore": restore, "followup": followup,
                "snapshot_phase": snapshot["session_state"]["phase"],
                "snapshot_reason": snapshot["session_state"]["terminal_reason"],
                "holds_after_followup": sorted(scheduler._cancelled_turn_inbox_holds),
                "queued_after_followup": len(lead.state.pending_user_messages),
                "lead_http_calls": len(lead_calls), "receipt_calls": writer.calls,
                "receipt_exists": writer.path.exists(),
            }
            print(json.dumps(observation, sort_keys=True))
            (tmp_path / "observation.json").write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
            assert writer.calls == 0
            assert not writer.path.exists()
            assert len(lead.state.pending_user_messages) == 1
            assert len(lead_calls) == 1
            assert lead.state.pending_user_messages[0]["delivery_status"] == "held_after_cancel"

            original_autosave = scheduler._autosave_session
            saved_statuses = []

            def observe_autosave(aid):
                if aid == 0:
                    saved_statuses.append(tuple(
                        message.get("delivery_status") for message in lead.state.pending_user_messages
                    ))
                return original_autosave(aid)

            monkeypatch.setattr(scheduler, "_autosave_session", observe_autosave)
            for _ in range(3):
                await scheduler._drain_message_inbox(0)
            assert saved_statuses == []

            assert await asyncio.wait_for(scheduler.run("new request"), 2) == "new answer"
            assert saved_statuses[0] == ("pending",)
            assert writer.calls == 1
            assert writer.path.read_text(encoding="utf-8") == "accepted teammate work executed\n"
            assert lead.state.pending_user_messages == []
            assert 0 not in scheduler._cancelled_turn_inbox_holds
            assert len(lead_calls) == 4
            assert "QUEUED_WORK" in lead_calls[1]["messages"][-1]["content"]
            assert all("new request" not in str(call["messages"]) for call in lead_calls[1:3])
            assert lead_calls[3]["messages"][-1]["content"].startswith("new request")
            assert await asyncio.wait_for(scheduler.run("another request"), 2) == "new answer"
            assert len(lead_calls) == 5
            assert writer.calls == 1
            await asyncio.gather(*lead.pending_cleanup_tasks)
            completed = OpenCollab.read_session_snapshot(lead.auto_save_path)
            assert completed["session_state"]["phase"] == "done"
            assert not completed.get("pending_messages")
            after_resume = load_session(str(lead.auto_save_path), agent=lead_agent(), llm=client)
            later = make_scheduler()
            later.register_lead(after_resume)
            assert 0 not in later._cancelled_turn_inbox_holds
            await later.cleanup()
            (tmp_path / "requests.json").write_text(json.dumps({
                "lead": lead_calls, "sender": sender_calls, "saved_delivery_statuses": saved_statuses,
                "receipt_calls": writer.calls,
            }, indent=2) + "\n", encoding="utf-8")
        finally:
            release.set()
            await scheduler.cleanup()
            await asyncio.gather(active, return_exceptions=True)
