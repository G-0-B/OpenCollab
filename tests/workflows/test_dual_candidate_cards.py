"""Dual-candidate cards share an Adopter, two independent Coders, and four directed edges."""

from __future__ import annotations

import difflib
import itertools
import re
from pathlib import Path

import pytest

from opencollab.bootstrap.team_config import load_team_config
from scripts.analyst_cards import (
    BLOCK_SLOT,
    CAPABILITIES_SLOT,
    CLOSING_LINE,
    DUAL_CARDS,
    load_registry,
    render,
    shared_body,
)
from tests.support.paths import REPO_ROOT

CONFIGS = REPO_ROOT / "configs"
PROMPTS = CONFIGS / "dual-candidate"
HANDOFF = CONFIGS / "handoff-experiment"
# Each card preserves its handoff counterpart outside the replaced division paragraph.
COUNTERPARTS = {
    "judge": "judge",
    "prohibit": "cmd-prohibit",
    "plain": "cmd-plain",
    "bare": "cmd-bare",
    "optout": "cmd-optout",
}
DIVISION_MARKER = "Implementation is the Coder"
# Role-independent ladder sentences survive replacement of the division paragraph.
CARRIED_SENTENCES = ("Do not apply the fix yourself.",)
ROLES = ("adopter", "coder_a", "coder_b")
ENTRY = "adopter"
# Four directed edges keep the two coders disconnected.
EDGES = {
    ("adopter", "coder_a"),
    ("adopter", "coder_b"),
    ("coder_a", "adopter"),
    ("coder_b", "adopter"),
}
TEAM_TOOLS = (
    "apply_patch", "bash", "file_read", "file_write", "grep",
    "message_agent", "submit", "team_status",
)

VARIANTS, LADDERS = load_registry(DUAL_CARDS)
NAMES = sorted(VARIANTS)

def _block(name: str) -> str:
    return (PROMPTS / "blocks" / f"{name}.md").read_text(encoding="utf-8")

def _capabilities(name: str) -> str:
    return (PROMPTS / "capabilities" / f"{name}.md").read_text(encoding="utf-8")

def _card_on_disk(name: str) -> str:
    return VARIANTS[name].card_path.read_text(encoding="utf-8")

def _unwrapped(text: str) -> str:
    """The text with every run of whitespace collapsed, for move counting."""
    return re.sub(r"\s+", " ", text).strip()

def _paragraphs(text: str) -> list[str]:
    return text.split("\n\n")

@pytest.mark.parametrize("ours,theirs", sorted(COUNTERPARTS.items()))
def test_each_block_is_its_handoff_counterpart_outside_the_division_paragraph(
    ours: str, theirs: str
) -> None:
    mine = _paragraphs(_block(ours))
    theirs_paragraphs = _paragraphs((HANDOFF / "blocks" / f"{theirs}.md").read_text(encoding="utf-8"))
    kept = [p for p in mine if DIVISION_MARKER not in p]
    their_kept = [p for p in theirs_paragraphs if DIVISION_MARKER not in p]
    assert kept == their_kept, ours
    replaced, their_replaced = len(mine) - len(kept), len(theirs_paragraphs) - len(their_kept)
    assert replaced == their_replaced <= 1, (ours, replaced, their_replaced)

@pytest.mark.parametrize("ours,theirs", sorted(COUNTERPARTS.items()))
def test_the_role_free_sentences_of_a_division_paragraph_cross_unedited(
    ours: str, theirs: str
) -> None:
    """A move the handoff ladder makes, this ladder makes in the same words."""
    mine = _unwrapped(_block(ours))
    theirs_text = _unwrapped((HANDOFF / "blocks" / f"{theirs}.md").read_text(encoding="utf-8"))
    for sentence in CARRIED_SENTENCES:
        assert (sentence in mine) == (sentence in theirs_text), (ours, sentence)

def test_the_two_families_state_the_same_tools_in_the_same_words() -> None:
    """The bundle paragraph matches while teammate descriptions reflect the different roster."""
    ours = _paragraphs(_capabilities("full"))[0]
    theirs = _paragraphs((HANDOFF / "capabilities" / "full.md").read_text(encoding="utf-8"))[0]
    assert ours == theirs

@pytest.mark.parametrize("name", NAMES)
def test_every_checked_in_card_is_exactly_what_its_declaration_assembles_to(name: str) -> None:
    """The team files load the checked-in card, not the assembly."""
    assert _card_on_disk(name) == render(VARIANTS[name])

@pytest.mark.parametrize("name", NAMES)
def test_a_card_is_the_shared_body_everywhere_outside_its_block(name: str) -> None:
    """Whatever surrounds the block is not this cell's to vary."""
    variant = VARIANTS[name]
    card = _card_on_disk(name)
    block = _block(variant.block)
    assert card.count(block) == 1, name
    assert card.replace(block, BLOCK_SLOT) == shared_body(DUAL_CARDS).replace(
        CAPABILITIES_SLOT, _capabilities(variant.capabilities)
    )

@pytest.mark.parametrize("name", NAMES)
def test_every_card_ends_on_the_shared_evidence_rule(name: str) -> None:
    """The one rule both instruments share: what counts as a verified change."""
    assert _card_on_disk(name).endswith("\n" + CLOSING_LINE)
    assert CLOSING_LINE not in _block(VARIANTS[name].block)

def test_the_two_card_sets_do_not_share_a_body() -> None:
    """The handoff shared body names a Tester, which this family does not seat."""
    from scripts.analyst_cards import HANDOFF_CARDS

    assert shared_body(DUAL_CARDS) != shared_body(HANDOFF_CARDS)
    assert DUAL_CARDS.directory != HANDOFF_CARDS.directory

def test_no_two_cells_assemble_to_the_same_card() -> None:
    cards = {name: _card_on_disk(name) for name in NAMES}
    duplicates = [
        (a, b) for a, b in itertools.combinations(NAMES, 2) if cards[a] == cards[b]
    ]
    assert not duplicates, duplicates

def test_no_two_block_files_hold_the_same_text() -> None:
    blocks = sorted({VARIANTS[name].block for name in NAMES})
    texts = {name: _block(name) for name in blocks}
    duplicates = [
        (a, b) for a, b in itertools.combinations(blocks, 2) if texts[a] == texts[b]
    ]
    assert not duplicates, duplicates

def test_every_registered_block_and_bundle_file_exists_and_is_used() -> None:
    used_blocks = {VARIANTS[name].block for name in NAMES}
    assert used_blocks == {p.stem for p in (PROMPTS / "blocks").glob("*.md")}
    used_capabilities = {VARIANTS[name].capabilities for name in NAMES}
    assert used_capabilities == {p.stem for p in (PROMPTS / "capabilities").glob("*.md")}

def test_each_ladder_step_is_exactly_one_contiguous_move() -> None:
    """Collapse wrapping changes before checking for a single contiguous insertion."""
    for ladder, rungs in LADDERS.items():
        for upper, lower in zip(rungs, rungs[1:]):
            a = _unwrapped(_block(VARIANTS[upper].block))
            b = _unwrapped(_block(VARIANTS[lower].block))
            moves = [
                op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                if op[0] != "equal"
            ]
            assert len(moves) == 1, f"{ladder}: {upper} -> {lower} is {len(moves)} moves, not one"
            assert moves[0][0] in {"insert", "delete"}, (
                f"{ladder}: {upper} -> {lower} replaces text rather than adding or removing it"
            )

@pytest.mark.parametrize("name", NAMES)
def test_each_cell_seats_the_dual_candidate_team(name: str) -> None:
    """Extra Coder edges break candidate independence, and missing return edges prevent delivery."""
    variant = VARIANTS[name]
    team = load_team_config(path=str(CONFIGS / variant.team_file))
    assert team.entry == ENTRY
    assert sorted(team.roles) == sorted(ROLES)
    for role in ROLES:
        assert tuple(team.roles[role].tools) == TEAM_TOOLS, role
    assert team.roles[ENTRY].prompt == _card_on_disk(name)
    for role, card in (("coder_a", "coder-a.md"), ("coder_b", "coder-b.md")):
        assert team.roles[role].prompt == (PROMPTS / card).read_text(encoding="utf-8")
    walked = {(s, d) for s in ROLES for d in ROLES if team.topology.allows(s, d)}
    assert walked == EDGES
    assert not team.topology.allows("coder_a", "coder_b")
    assert not team.topology.allows("coder_b", "coder_a")

def test_the_two_coder_cards_differ_only_in_identity_and_the_answer_asked_for() -> None:
    heading = "## The answer you are asked for"
    a, b = (
        (PROMPTS / f"coder-{side}.md").read_text(encoding="utf-8") for side in ("a", "b")
    )
    body_a, stance_a = a.split(heading)
    body_b, stance_b = b.split(heading)
    assert stance_a != stance_b
    identity = str.maketrans({})
    normalize = lambda text: (  # noqa: E731
        text.translate(identity)
        .replace("Coder A", "<peer>").replace("Coder B", "<peer>")
        .replace("`coder_a`", "<token>").replace("`coder_b`", "<token>")
    )
    assert normalize(body_a) == normalize(body_b)

ENTRY_PROMPT_LINE = "    prompt_file: dual-candidate/adopter."

def _team_file_body(path: Path) -> str:
    """The configuration below the header comment, with the Adopter's line masked."""
    text = path.read_text(encoding="utf-8")
    body = text[text.index("entry:"):]
    return "".join(
        "PROMPT_FILE\n" if line.startswith(ENTRY_PROMPT_LINE) else line
        for line in body.splitlines(keepends=True)
    )

def test_every_team_file_differs_only_in_the_adopters_prompt_line() -> None:
    """Four files, one instrument. A cell is selected by naming its team file."""
    paths = sorted(CONFIGS.glob("team.dual-*.yaml"))
    bodies = {path.name: _team_file_body(path) for path in paths}
    assert len(bodies) == len(VARIANTS)
    distinct = set(bodies.values())
    assert len(distinct) == 1, sorted(
        name for name, body in bodies.items() if body != next(iter(distinct))
    )

def test_this_family_stays_out_of_the_ladders_namespace() -> None:
    """The handoff family discovers its configs with the team.handoff.*.yaml glob."""
    assert not list(CONFIGS.glob("team.handoff.dual-*.yaml"))
    for name in NAMES:
        assert VARIANTS[name].team_file.startswith("team.dual-")

@pytest.mark.parametrize("name", NAMES)
def test_every_cell_carries_a_note(name: str) -> None:
    assert len(VARIANTS[name].note.strip()) > 40, name
