"""The capability cell removes Analyst edit tools while preserving the primary stance."""

from __future__ import annotations

import pytest

from opencollab.bootstrap.team_config import (
    ANALYST_TOOL_NAMES,
    load_team_config,
)
from tests.support.paths import REPO_ROOT

PROMPTS = REPO_ROOT / "configs" / "handoff-experiment"
JUDGE = "## What is yours to judge"
# Only capability changes, so starved shares the facts-v2 stance file.
JUDGE_BLOCK = PROMPTS / "blocks" / "judge.md"
LEGACY = PROMPTS / "legacy"
WRITE_TOOLS = ("apply_patch", "file_write")

def _card(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")

@pytest.fixture(scope="module")
def starved():
    return load_team_config(path=str(REPO_ROOT / "configs" / "team.handoff.starved.yaml"))

@pytest.fixture(scope="module")
def baseline():
    return load_team_config(path=str(REPO_ROOT / "configs" / "team.handoff.experiment.yaml"))

def test_the_analyst_holds_no_tool_that_edits_a_file(starved) -> None:
    for tool in WRITE_TOOLS:
        assert tool not in starved.roles["analyst"].tools, tool

def test_the_analyst_keeps_the_collaboration_channel(starved) -> None:
    """Starving the bundle must not also seat the Analyst mute."""
    assert "message_agent" in starved.roles["analyst"].tools
    assert "team_status" in starved.roles["analyst"].tools

def test_the_analyst_keeps_bash_so_a_handoff_can_reach_the_graded_tree(starved) -> None:
    """The Analyst needs bash to check a teammate commit into the graded tree."""
    assert "bash" in starved.roles["analyst"].tools

def test_the_coder_can_still_do_the_work_that_was_taken_away(starved) -> None:
    for tool in WRITE_TOOLS:
        assert tool in starved.roles["coder"].tools, tool
    assert "bash" in starved.roles["coder"].tools

def test_the_stance_block_is_primarys_byte_for_byte() -> None:
    """Capability is the variable. The stance must not move with it."""
    block = JUDGE_BLOCK.read_text(encoding="utf-8")
    assert block in _card("analyst.starved.md")
    primary = (LEGACY / "analyst.md").read_text(encoding="utf-8")
    assert primary.partition(JUDGE)[2].rstrip("\n") == (
        block.partition(JUDGE)[2].rstrip("\n")
        + "\n\nDo not report a change as verified unless you have the evidence for it."
    )

def test_the_card_states_the_bundle_it_actually_has() -> None:
    card = " ".join(_card("analyst.starved.md").split())
    assert "No tool in your bundle edits a file" in card
    assert "`apply_patch` and `file_write` edit" not in card

def test_the_card_does_not_ask_the_analyst_to_delegate() -> None:
    card = _card("analyst.starved.md").lower()
    for phrase in ("delegate", "you must", "hand it over to the coder"):
        assert phrase not in card, phrase

def test_only_the_analyst_differs_from_the_baseline_team(starved, baseline) -> None:
    assert starved.entry == baseline.entry
    assert sorted(starved.roles) == sorted(baseline.roles)
    for role in ("coder", "tester"):
        assert starved.roles[role].tools == baseline.roles[role].tools, role
        assert starved.roles[role].prompt == baseline.roles[role].prompt, role
    roles = ("analyst", "coder", "tester")
    walked = {(s, d) for s in roles for d in roles if starved.topology.allows(s, d)}
    assert walked == {(s, d) for s in roles for d in roles if s != d}

def test_this_is_not_the_shipped_analyst_bundle() -> None:
    """The shipped team also commands delegation and blocks on spawn results."""
    assert "spawn_agent" in ANALYST_TOOL_NAMES
    assert "bash" not in ANALYST_TOOL_NAMES
    starved_tools = load_team_config(
        path=str(REPO_ROOT / "configs" / "team.handoff.starved.yaml")
    ).roles["analyst"].tools
    assert "spawn_agent" not in starved_tools
