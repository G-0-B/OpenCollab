"""Reconstruct assignment, commit, checkout, and reply from persisted trace fields.

Run the real scheduler, Git worktrees, messaging, and topology checks with a
scripted provider. The control omits only the Tester checkout.
PYTHONDONTWRITEBYTECODE keeps the probe from dirtying its measured worktree.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.adapters.trace import Tracer
from opencollab.bootstrap import build_runtime_context, build_scheduler, container
from opencollab.bootstrap.team_config import load_team_config
from tests.support.paths import REPO_ROOT

TEAM_FILE = REPO_ROOT / "configs" / "team.handoff.experiment.yaml"

CONFIG = {
    "model": "gpt-4o",
    "provider": "openai",
    "api_key": "test-key",  # pragma: allowlist secret
    "base_url": None,
    "budget": 2_000_000,
}

ANALYST_AID, CODER_AID, TESTER_AID = 0, 1, 2
FULL_SHA = re.compile(r"\b[0-9a-f]{40}\b")

BROKEN = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"

def _git(cwd, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()

def _repo(path: Path) -> Path:
    """A committed repository — the precondition for linked worktrees."""
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "OpenCollab Tests")
    (path / "app.py").write_text(BROKEN, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "base")
    return path

def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }

def _acts(tool_calls: list[dict]) -> LLMResponse:
    return LLMResponse(
        tool_calls=tool_calls,
        usage=Usage(input_tokens=10, output_tokens=5),
        finish_reason="tool_calls",
    )

def _says(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )

def _last_tool_result(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return str(message.get("content") or "")
    raise AssertionError("the script expected a tool result and found none")

def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content)
    raise AssertionError("the script expected a teammate message and found none")

def _scripted_llm(*, tester_checks_out: bool):
    """Read roles from system prompts and commit IDs from Git output for each scripted client."""

    class ScriptedLLM:
        def __init__(self, **_kwargs) -> None:
            self.role = "?"
            self.turn = 0

        def context_window(self) -> int:
            return 200_000

        async def close(self) -> None:
            return None

        async def complete(self, messages, **_kwargs) -> LLMResponse:
            system = messages[0].get("content") or ""
            for role in ("Analyst", "Coder", "Tester"):
                if f"You are the {role}" in str(system):
                    self.role = role.lower()
                    break
            self.turn += 1
            return getattr(self, f"_{self.role}")(messages)

        def _analyst(self, _messages: list[dict]) -> LLMResponse:
            if self.turn == 1:
                return _acts([
                    _call("a1", "message_agent", {
                        "to_aid": CODER_AID,
                        "summary": "fix add()",
                        "content": (
                            "app.py's add() subtracts. Fix it, commit it, and "
                            f"hand the commit to the tester (aid {TESTER_AID})."
                        ),
                    })
                ])
            return _says("handed the work to the coder")

        def _coder(self, messages: list[dict]) -> LLMResponse:
            if self.turn == 1:
                return _acts([
                    _call("c1", "bash", {
                        "command": (
                            f"printf '{FIXED}' > app.py && git add -A && "
                            "git commit -qm 'fix add' && git rev-parse HEAD"
                        )
                    })
                ])
            if self.turn == 2:
                sha = FULL_SHA.search(_last_tool_result(messages))
                assert sha is not None, "the coder's commit produced no sha"
                return _acts([
                    _call("c2", "message_agent", {
                        "to_aid": TESTER_AID,
                        "summary": "committed the fix",
                        "content": (
                            f"Fixed app.py and committed it as {sha.group(0)}. "
                            "Check it out and run the probe."
                        ),
                    })
                ])
            return _says("fix committed and handed to the tester")

        def _tester(self, messages: list[dict]) -> LLMResponse:
            probe = (
                "PYTHONDONTWRITEBYTECODE=1 python3 -c "
                "\"import app; print('add(1,2) =', app.add(1, 2))\""
            )
            if self.turn == 1 and tester_checks_out:
                sha = FULL_SHA.search(_last_user_text(messages))
                assert sha is not None, "the coder's message carried no sha"
                return _acts([
                    _call("t1", "bash", {"command": f"git checkout -q {sha.group(0)}"})
                ])
            if self.turn == (2 if tester_checks_out else 1):
                return _acts([_call("t2", "bash", {"command": probe})])
            if self.turn == (3 if tester_checks_out else 2):
                return _acts([
                    _call("t3", "message_agent", {
                        "to_aid": CODER_AID,
                        "summary": "ran the probe",
                        "content": _last_tool_result(messages),
                    })
                ])
            return _says("reported the probe result to the coder")

    return ScriptedLLM

async def _run_handoff(tmp_path, monkeypatch, *, tester_checks_out: bool) -> list[dict]:
    """Run the whole team once; return every trace record, in order."""
    workspace = _repo(tmp_path / "repo")
    traces = tmp_path / "traces"
    traces.mkdir()
    monkeypatch.setattr(
        container, "LLMClient", _scripted_llm(tester_checks_out=tester_checks_out)
    )
    tracer = Tracer(run_id="handoff", output_dir=str(traces))
    ctx = build_runtime_context(str(workspace), dict(CONFIG), trace=False)
    ctx.tracer = tracer
    scheduler = build_scheduler(
        ctx,
        use_worktrees=True,
        interactive=True,
        auto_save=False,
        enable_hooks=False,
        prebuild_team=True,
        resolved_team_config=load_team_config(path=str(TEAM_FILE)),
    )
    try:
        await scheduler.ensure_team_prebuilt()
        await scheduler.run_turn(
            ANALYST_AID, "app.py's add() is wrong. Get it fixed and verified."
        )
        tracer.flush()
    finally:
        tracer.close()
        await scheduler.cleanup()

    with open(tracer.path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

# Reconstruct each handoff from recorded fields and compare commit ids directly.

def _payloads(records: list[dict], step_type: str) -> list[dict]:
    return [record["payload"] for record in records if record["type"] == step_type]

def _messages(records: list[dict], *, sender: str, recipient: str) -> list[dict]:
    return [
        payload
        for payload in _payloads(records, "message_sent")
        if payload["from_role"] == sender and payload["to_role"] == recipient
    ]

def _worktree_rows(records: list[dict], role: str) -> list[dict]:
    return [
        payload
        for payload in _payloads(records, "worktree_changes")
        if payload["role"] == role
    ]

def _commits_made_by(records: list[dict], role: str) -> set[str]:
    """Every commit that role put on top of the base it started from."""
    return {
        commit for row in _worktree_rows(records, role) for commit in row["commits"]
    }

def _bases_adopted_by(records: list[dict], role: str) -> set[str]:
    return {row["diff_base"] for row in _worktree_rows(records, role)}

async def test_every_step_of_the_handoff_is_recoverable_from_recorded_fields(
    tmp_path, monkeypatch
) -> None:
    records = await _run_handoff(tmp_path, monkeypatch, tester_checks_out=True)

    edges = _payloads(records, "assigned.topology_edges")[0]["edges"]
    assert {"from_role": "analyst", "to_role": "coder"} in edges
    assert {"from_role": "coder", "to_role": "tester"} in edges

    assert len(_messages(records, sender="analyst", recipient="coder")) == 1

    coder_rows = _worktree_rows(records, "coder")
    assert [entry["path"] for entry in coder_rows[0]["files"]] == ["app.py"]
    assert coder_rows[0]["commit_count"] == 1
    assert coder_rows[0]["head_commit"] == coder_rows[0]["commits"][0]
    assert coder_rows[0]["head_commit"] != coder_rows[0]["diff_base"]

    committed = _commits_made_by(records, "coder")
    handoff = _messages(records, sender="coder", recipient="tester")
    assert len(handoff) == 1
    assert set(handoff[0]["commit_refs"]) & committed == committed

    assert _bases_adopted_by(records, "tester") & committed == committed

    assert len(_messages(records, sender="tester", recipient="coder")) == 1

async def test_a_tester_that_never_checked_out_reconstructs_as_not_having(
    tmp_path, monkeypatch
) -> None:
    """Omit only checkout while preserving commit, message, command, and reply steps."""
    records = await _run_handoff(tmp_path, monkeypatch, tester_checks_out=False)

    committed = _commits_made_by(records, "coder")
    assert committed, "the control is only a control if the Coder still committed"
    assert _messages(records, sender="coder", recipient="tester")[0]["commit_refs"]
    assert _messages(records, sender="tester", recipient="coder")

    assert _bases_adopted_by(records, "tester") & committed == set()
    assert _bases_adopted_by(records, "tester") == _bases_adopted_by(records, "coder")
