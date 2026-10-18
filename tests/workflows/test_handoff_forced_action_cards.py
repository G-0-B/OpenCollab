"""Forced-action cards vary the closing section while preserving facts-v2's team and body."""

from __future__ import annotations

import pytest

from opencollab.bootstrap.team_config import load_team_config
from tests.support.paths import REPO_ROOT

CONFIGS = REPO_ROOT / "configs"
PROMPTS = CONFIGS / "handoff-experiment"
BLOCKS = PROMPTS / "blocks"
BASE_CARD = "analyst.facts-v2.md"
#: cell -> the block file its card is assembled from. ``facts-v2`` carries the
#: ``judge`` block, which is what these three replace.
BLOCK_OF = {
    "decide-first": "decide-first",
    "opt-out-message": "opt-out-message",
    "role-identity": "role-identity",
    BASE_CARD: "judge",
}
CLOSING_LINE = "Do not report a change as verified unless you have the evidence for it."
ROLES = ("analyst", "coder", "tester")

# Each named card and team file defines a separate experimental cell.
CELLS = {
    "decide-first": ("analyst.decide-first.md", "team.handoff.decide-first.yaml"),
    "opt-out-message": ("analyst.opt-out-message.md", "team.handoff.opt-out-message.yaml"),
    "role-identity": ("analyst.role-identity.md", "team.handoff.role-identity.yaml"),
}

# Delegation instructions belong to the instructed cell, outside the forced-action axis.
DELEGATION_ORDERS = ("delegate", "you must", "do not apply the fix yourself")

def _card(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")

def _flat(text: str) -> str:
    """Line wrapping is not a variable; compare on the words."""
    return " ".join(text.split())

def _block(cell_or_card: str) -> str:
    """Read each declared block because the cards use different closing headings."""
    return (BLOCKS / f"{BLOCK_OF[cell_or_card]}.md").read_text(encoding="utf-8")

def _shared_body_of(card_name: str, cell: str) -> str:
    """The card with its block removed -- everything the cell may not vary."""
    block = _block(cell)
    assert card_name == BASE_CARD or _card(card_name).count(block) == 1, card_name
    return _card(card_name).replace(block, "")

def _closing(card_name: str) -> str:
    cell = next(c for c, (name, _) in CELLS.items() if name == card_name) \
        if card_name != BASE_CARD else BASE_CARD
    return _block(cell)

@pytest.fixture(scope="module")
def baseline():
    return load_team_config(path=str(CONFIGS / "team.handoff.experiment.yaml"))

def _team(cell: str):
    return load_team_config(path=str(CONFIGS / CELLS[cell][1]))

@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_card_is_facts_v2_byte_for_byte_up_to_its_closing_section(cell: str) -> None:
    assert _shared_body_of(CELLS[cell][0], cell) == _shared_body_of(BASE_CARD, BASE_CARD)
    assert _block(cell) in _card(CELLS[cell][0]), "the card does not carry its declared block"

@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_closing_section_differs_from_the_one_it_replaces(cell: str) -> None:
    assert _closing(CELLS[cell][0]) != _closing(BASE_CARD)

def test_the_three_closing_sections_are_three_different_sections() -> None:
    """Three files that say the same thing are one cell, not three."""
    blocks = {cell: _closing(CELLS[cell][0]) for cell in CELLS}
    assert len(set(blocks.values())) == len(CELLS), sorted(blocks)

@pytest.mark.parametrize("cell", sorted(CELLS))
def test_no_card_on_this_axis_orders_the_handoff(cell: str) -> None:
    """These cells compel an action while leaving delegation as a choice."""
    card = _card(CELLS[cell][0]).lower()
    for order in DELEGATION_ORDERS:
        assert order not in card, order

def test_the_decide_first_card_names_both_answers() -> None:
    """Naming both branches keeps the compelled choice from prescribing its answer."""
    block = _flat(_closing(CELLS["decide-first"][0]))
    assert "keeping the implementation" in block
    assert "handing it to the Coder" in block
    assert "Both are legitimate answers and neither one is expected of you" in block
    assert "What is not an answer is starting without having chosen" in block

def test_the_opt_out_card_asks_for_a_message_on_the_keeping_branch() -> None:
    """A message on the keeping branch distinguishes an explicit choice from silence."""
    block = _flat(_closing(CELLS["opt-out-message"][0]))
    assert "`message_agent`" in block
    assert "If you are keeping the implementation, that message says so" in block
    assert "so the Coder can stop waiting" in block
    assert "The choice between them is yours. Sending one of them is not." in block

def test_the_role_identity_card_says_the_tools_are_in_hand_and_permitted() -> None:
    """Explicit tool permission prevents role wording from implying a capability restriction."""
    block = _flat(_closing(CELLS["role-identity"][0]))
    assert "You also hold the Coder's tools" in block
    assert "nothing here forbids using them" in block
    assert "no rule is broken and nothing refuses it" in block

@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_cell_seats_the_baseline_team(cell: str, baseline) -> None:
    """Missing message tools or return edges can suppress handoff while the run still completes."""
    variant = _team(cell)
    assert variant.entry == baseline.entry
    assert sorted(variant.roles) == sorted(baseline.roles)
    for role in ROLES:
        assert variant.roles[role].tools == baseline.roles[role].tools, role
        if role != "analyst":
            assert variant.roles[role].prompt == baseline.roles[role].prompt, role
    walked = {(s, d) for s in ROLES for d in ROLES if variant.topology.allows(s, d)}
    assert walked == {(s, d) for s in ROLES for d in ROLES if s != d}

@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_cell_loads_its_own_analyst_card(cell: str, baseline) -> None:
    """A prompt_file left on facts-v2 silently duplicates the base cell."""
    variant = _team(cell)
    assert variant.roles["analyst"].prompt == _card(CELLS[cell][0])
    assert variant.roles["analyst"].prompt != baseline.roles["analyst"].prompt
    assert variant.roles["analyst"].prompt != _card(BASE_CARD)

@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_team_file_declares_that_its_runs_are_not_poolable(cell: str) -> None:
    text = (CONFIGS / CELLS[cell][1]).read_text(encoding="utf-8").lower()
    assert "poolable" in text
    assert "forced-action axis" in text
