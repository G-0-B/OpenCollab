from __future__ import annotations

import json

import pytest

from opencollab import OpenCollab
from opencollab.bootstrap import build_scheduler
from opencollab.bootstrap.config import build_config
from opencollab.bootstrap.runtime_context import build_runtime_context
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport


@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
async def test_team_native_members_carry_the_public_run_identity(tmp_path, monkeypatch, trace, explicit, prebuilt):
    team = tmp_path / "team.yaml"
    team.write_text(
        "entry: lead\nroles:\n  lead:\n    model: record-lead\n"
        "    prompt: Delegate a short review and report.\n    tools: [spawn_agent]\n"
        "  reviewer:\n    model: record-reviewer\n    prompt: Return a review.\n"
        "    tools: []\ntopology:\n  lead: [reviewer]\n"
    )
    if prebuilt:
        team.write_text(
            team.read_text().replace("tools: [spawn_agent]", "tools: []")
            .replace("  lead: [reviewer]", "  lead: []")
        )
    requests = []
    lead_requests = 0

    def handler(request):
        nonlocal lead_requests
        body = json.loads(request.content)
        requests.append(body)
        if body["model"] == "record-reviewer":
            return completion_http_response(request, text="Review completed.")
        lead_requests += 1
        if lead_requests == 1 and not prebuilt:
            return http_response(request, 200, json={
                "id": "chat-delegate", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{
                        "id": "review-call", "type": "function", "function": {"name": "spawn_agent",
                        "arguments": json.dumps({"role": "reviewer", "task": "Review the note."})},
                    }],
                }}], "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            })
        return completion_http_response(request, text="Final reviewed note.")

    install_sdk_transport(monkeypatch, "openai", handler)
    app = OpenCollab(
        tmp_path, model="record-lead", provider="openai", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://fixture.invalid/v1", config={"llm_max_retries": 0},
    )
    artifacts = tmp_path / "artifacts"
    supplied = {"run_id": "recorded-team-call"} if explicit else {}
    result = await app.team("Produce a reviewed note.", config=team, trace=trace,
                            artifacts=artifacts, use_worktrees=False, prebuild_team=prebuilt, **supplied)
    expected = result.metrics["run_id"]
    snapshots = {p.name: app.read_session_snapshot(p) for p in artifacts.glob("agent_*.json")}
    assert result.status == "completed"
    assert result.output == "Final reviewed note."
    assert json.loads((artifacts / "team.json").read_text())["run_id"] == expected
    if trace:
        rows = [json.loads(x) for x in (artifacts / "trajectory.jsonl").read_text().splitlines()]
        assert {x["run_id"] for x in rows} == {expected}
    assert len(snapshots) == 2
    assert {data["run_id"] for data in snapshots.values()} == {expected}
    assert [body["model"] for body in requests] == (
        ["record-lead"] if prebuilt else ["record-lead", "record-reviewer", "record-lead"]
    )

@pytest.mark.parametrize("trace", [False, True])
async def test_solo_record_identity_is_already_complete(tmp_path, monkeypatch, trace):
    install_sdk_transport(monkeypatch, "openai", lambda request: completion_http_response(request, text="Finished."))
    app = OpenCollab(tmp_path, model="gpt-4o", provider="openai", api_key="controlled-test",  # pragma: allowlist secret
                     base_url="https://fixture.invalid/v1", config={"llm_max_retries": 0})
    artifacts = tmp_path / "artifacts"
    result = await app.agent("Reply once.", tools=[], trace=trace, artifacts=artifacts, run_id="solo-control")
    assert result.status == "completed"
    assert result.metrics["run_id"] == app.read_session_snapshot(artifacts / "agent.json")["run_id"] == "solo-control"

@pytest.mark.parametrize("trace", [False, True])
async def test_composition_uses_the_existing_manifest_fallback_for_native_records(tmp_path, monkeypatch, trace):
    monkeypatch.chdir(tmp_path)
    install_sdk_transport(monkeypatch, "openai", lambda request: completion_http_response(request, text="Finished."))
    team = tmp_path / "team.yaml"
    team.write_text("entry: lead\nroles:\n  lead:\n    prompt: Reply once.\n    tools: []\n")
    config = build_config(str(tmp_path), overrides={
        "model": "gpt-4o", "provider": "openai", "api_key": "controlled-test",  # pragma: allowlist secret
        "base_url": "https://fixture.invalid/v1", "llm_max_retries": 0,
    })
    context = build_runtime_context(str(tmp_path), config.model_dump(), trace=trace)
    artifacts = tmp_path / "reserved-run"
    scheduler = build_scheduler(context, use_worktrees=False, interactive=False,
                                team_config_path=team, save_dir=artifacts)
    try:
        assert await scheduler.run("Reply once.") == "Finished."
    finally:
        await scheduler.cleanup()
        if context.tracer is not None:
            context.tracer.close()
    manifest = json.loads((artifacts / "team.json").read_text())
    expected = context.tracer.run_id if trace else artifacts.name
    assert manifest["run_id"] == expected
    saved = next(artifacts.glob("agent_*.json"))
    assert OpenCollab.read_session_snapshot(saved)["run_id"] == expected

async def test_composition_without_artifacts_keeps_lightweight_tracer_support(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_sdk_transport(monkeypatch, "openai", lambda request: completion_http_response(request, text="Finished."))
    team = tmp_path / "team.yaml"
    team.write_text("entry: lead\nroles:\n  lead:\n    prompt: Reply once.\n    tools: []\n")
    config = build_config(str(tmp_path), overrides={
        "model": "gpt-4o", "provider": "openai", "api_key": "controlled-test",  # pragma: allowlist secret
        "base_url": "https://fixture.invalid/v1", "llm_max_retries": 0,
    })
    context = build_runtime_context(str(tmp_path), config.model_dump(), trace=False)

    class LightweightTrace:
        def log_step(self, **kwargs):
            pass

        def flush(self):
            pass

    context.tracer = LightweightTrace()
    scheduler = build_scheduler(context, use_worktrees=False, interactive=False,
                                team_config_path=team, auto_save=False)
    try:
        assert await scheduler.run("Reply once.") == "Finished."
        assert scheduler.lead_session.run_id is None
    finally:
        await scheduler.cleanup()
