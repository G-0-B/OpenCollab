"""Public waiting-turn persistence with real tools and local fake completions."""

from __future__ import annotations

import asyncio
import json

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.storage import SessionStore
from opencollab.adapters.tools.spawn import SpawnAgentTool
from opencollab.adapters.tools.submit import SubmitTool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from opencollab.application.scheduler_types import SchedulerTurnError
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase, SessionState
from tests.support.session_characterization_test_support import FakeLLMClient, llm_response, tool_call


def agent(tools=()):
    return Agent(name="lead", system_prompt="Finish the task.", tools=list(tools))

class DelayedClient(FakeLLMClient):
    async def complete(self, messages, tools=None, temperature=0.0):
        await asyncio.sleep(0.03)
        return await super().complete(messages, tools=tools, temperature=temperature)

class ChildClient:
    def __init__(self):
        self.release = asyncio.Event()

    async def complete(self, **kwargs):
        await self.release.wait()
        return llm_response(content="child done")

class Pool:
    def __init__(self, workspace):
        self.env = LocalEnvironment(str(workspace))

    async def acquire(self, role):
        return self.env

    async def release(self):
        pass

class Factory:
    def __init__(self, workspace):
        self.workspace = workspace
        self.client = ChildClient()

    def build_spawn_session(self, **kwargs):
        return build_session(
            agent=agent(), llm=self.client, env=LocalEnvironment(str(self.workspace)),
            max_budget_tokens=kwargs["budget"], aid=kwargs["aid"],
        )

def make_team(workspace, *, extra_calls=(), extra_tools=(), autosave=True, response_content=None):
    factory = Factory(workspace)
    scheduler = Scheduler(session_factory=factory, worktree_pool=Pool(workspace), event_sink=EventBus())
    calls = [tool_call("child", "spawn_agent", '{"role":"worker","task":"old child"}'), *extra_calls]
    model = DelayedClient([
        llm_response(content=response_content, tool_calls=calls), llm_response(content="old task answer"),
    ])
    events = []
    path = workspace / "waiting.json"
    session = build_session(
        agent=agent([SpawnAgentTool(scheduler), *extra_tools]), llm=model,
        env=LocalEnvironment(str(workspace)), auto_save_path=str(path) if autosave else None,
        event_sink=EventBus(events.append),
    )
    scheduler.register_lead(session)
    return scheduler, factory, session, model, events, path

async def wait_for_suspension(session):
    async def wait():
        while session.phase is not SessionPhase.AWAITING_EVENTS:
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 2)
    await asyncio.gather(*session.pending_cleanup_tasks)

async def finish_team(scheduler, factory, active):
    factory.client.release.set()
    if not active.done():
        await asyncio.wait_for(asyncio.shield(active), 2)
    await scheduler.cleanup()

@pytest.mark.asyncio
async def test_awaiting_autosave_restores_old_turn_before_accepting_new_request(tmp_path):
    scheduler, factory, session, model, events, path = make_team(tmp_path)
    active = asyncio.create_task(scheduler.run("old user task"))
    resumed_scheduler = None
    try:
        await wait_for_suspension(session)
        snapshot = SessionStore().load_snapshot(str(path), session.agent.system_prompt)
        assert snapshot["session_state"]["phase"] == "awaiting_events"
        assert snapshot["session_state"]["pending_events"][0]["ref"] == 1
        latency = snapshot["session_state"]["pending_step_latency"]
        assert latency >= 0.03
        assert not [event for event in events if event.type == "step_end"]

        resumed_model = FakeLLMClient([
            llm_response(content="old resumed answer"), llm_response(content="new task answer"),
        ])
        restored = load_session(
            str(path), agent=agent(), llm=resumed_model, env=LocalEnvironment(str(tmp_path)),
        )
        assert restored.phase is SessionPhase.AWAITING_EVENTS
        resumed_scheduler = Scheduler(
            session_factory=factory, worktree_pool=Pool(tmp_path), event_sink=EventBus(),
        )
        resumed_scheduler.register_lead(restored)

        assert await resumed_scheduler.run("new user task") == "new task answer"
        assert len(resumed_model.calls) == 2
        assert all("new user task" not in str(message.get("content"))
                   for message in resumed_model.calls[0]["messages"])
        assert any(message.get("content") == "old resumed answer"
                   for message in resumed_model.calls[1]["messages"])

        factory.client.release.set()
        assert await asyncio.wait_for(active, 2) == "old task answer"
        completed = [event for event in events if event.type == "step_end" and event.data["step"] == 1]
        assert len(completed) == 1
        assert completed[0].data["latency"] == latency
        assert len(model.calls) == 2
    finally:
        await finish_team(scheduler, factory, active)
        if resumed_scheduler is not None:
            await resumed_scheduler.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("autosave", [False, True])
@pytest.mark.parametrize("content", [None, "accepted summary", "accepted summary\n", "I am waiting for the worker."])
async def test_restored_waiting_turn_retains_accepted_submit_without_another_model_call(tmp_path, autosave, content):
    scheduler, factory, session, model, events, path = make_team(
        tmp_path, extra_calls=[tool_call("submit", "submit", '{"summary":"accepted summary"}')],
        extra_tools=[SubmitTool()], autosave=autosave, response_content=content,
    )
    active = asyncio.create_task(scheduler.run("old user task"))
    try:
        await wait_for_suspension(session)
        if not autosave:
            session.save(str(path))
        snapshot = SessionStore().load_snapshot(str(path), session.agent.system_prompt)
        assert snapshot["session_state"]["phase"] == "awaiting_events"
        assert snapshot["session_state"]["pending_events"][0]["ref"] == 1
        resumed_model = FakeLLMClient([llm_response(content="new task answer")])
        restored_events = []
        restored = load_session(
            str(path), agent=agent(), llm=resumed_model, env=LocalEnvironment(str(tmp_path)),
            event_sink=EventBus(restored_events.append),
        )
        used_tokens = restored.used_tokens

        assert await restored.run_loop() == "accepted summary"
        assert restored.state.terminal_reason == "submitted"
        assert resumed_model.calls == []
        assert restored.used_tokens == used_tokens
        assert restored.state.pending_events.is_empty()
        assert restored.messages[-1]["content"] == "accepted summary"
        assert snapshot["session_state"]["submitted_summary"] == "accepted summary"
        assert [event.data["content"] for event in restored_events if event.type == "text_delta"] == (
            [] if content and content.strip() == "accepted summary" else ["accepted summary"]
        )

        await restored.add_user_message("new user task")
        assert await restored.run_loop() == "new task answer"
        assert restored.state.terminal_reason == "completed"
        assert len(resumed_model.calls) == 1

        factory.client.release.set()
        assert await asyncio.wait_for(active, 2) == "accepted summary"
        assert len(model.calls) == 1
        expected_text = [content] if content else []
        if content is None or content.strip() != "accepted summary":
            expected_text.append("accepted summary")
        assert [event.data["content"] for event in events if event.type == "text_delta"] == expected_text
    finally:
        await finish_team(scheduler, factory, active)

@pytest.mark.asyncio
@pytest.mark.parametrize("stored_summary", [None, False, {"summary": "unaccepted"}])
async def test_legacy_or_malformed_waiting_submission_does_not_skip_the_model(tmp_path, stored_summary):
    scheduler, factory, session, _model, _events, path = make_team(tmp_path, autosave=False)
    active = asyncio.create_task(scheduler.run("old user task"))
    try:
        await wait_for_suspension(session)
        session.save(str(path))
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if stored_summary is None:
            snapshot["session_state"].pop("submitted_summary", None)
        else:
            snapshot["session_state"]["submitted_summary"] = stored_summary
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        resumed_model = FakeLLMClient([llm_response(content="old resumed answer")])
        restored = load_session(str(path), agent=agent(), llm=resumed_model, env=LocalEnvironment(str(tmp_path)))

        assert await restored.run_loop() == "old resumed answer"
        assert restored.state.terminal_reason == "completed"
        assert len(resumed_model.calls) == 1
    finally:
        await finish_team(scheduler, factory, active)

def test_accepted_submission_resets_with_fresh_turn_and_rolls_back_with_failed_acceptance():
    state = SessionState(messages=[])
    state.submitted_summary = "accepted summary"
    checkpoint = state.checkpoint_user_turn()

    state.reset_for_user_turn()
    assert state.submitted_summary is None
    state.restore_user_turn(checkpoint)
    assert state.submitted_summary == "accepted summary"
    state.clear_active_turn()
    assert state.submitted_summary is None
    state.restore_user_turn(checkpoint)
    assert state.submitted_summary == "accepted summary"

@pytest.mark.asyncio
async def test_cancelling_a_waiting_submitted_turn_clears_the_durable_submission(tmp_path):
    scheduler, factory, session, _model, _events, path = make_team(
        tmp_path, extra_calls=[tool_call("submit", "submit", '{"summary":"accepted summary"}')],
        extra_tools=[SubmitTool()],
    )
    cancel_event = asyncio.Event()
    active = asyncio.create_task(scheduler.run("old user task", cancel_event=cancel_event))
    try:
        await wait_for_suspension(session)
        assert session.state.submitted_summary == "accepted summary"
        cancel_event.set()
        with pytest.raises(SchedulerTurnError, match="interrupted by user"):
            await asyncio.wait_for(active, 2)
        await asyncio.gather(*session.pending_cleanup_tasks)

        assert session.phase is SessionPhase.STOPPED
        assert session.state.submitted_summary is None
        assert session.state.pending_events.is_empty()
        snapshot = SessionStore().load_snapshot(str(path), session.agent.system_prompt)
        assert snapshot["session_state"]["submitted_summary"] is None

        resumed_model = FakeLLMClient([llm_response(content="new task answer")])
        restored = load_session(str(path), agent=agent(), llm=resumed_model, env=LocalEnvironment(str(tmp_path)))
        await restored.add_user_message("new user task")
        assert await restored.run_loop() == "new task answer"
        assert restored.state.terminal_reason == "completed"
        assert len(resumed_model.calls) == 1
    finally:
        await finish_team(scheduler, factory, active)
