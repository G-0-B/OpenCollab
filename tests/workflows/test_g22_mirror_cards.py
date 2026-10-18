"""Mirror G22 prompt wording and candidate capabilities in the team runtime.

Expected strings come from OpenCollab-Eval's validation_council_dual_coder_contract
and _validation_council_solve_defs.SHARED_RULES in the separate repository.
"""

from __future__ import annotations

import re

import pytest

from opencollab.bootstrap.team_config import load_team_config
from opencollab.bootstrap.tool_registry import KNOWN_TOOL_NAMES
from tests.support.paths import REPO_ROOT

TEAM_FILE = REPO_ROOT / "configs" / "team.g22-mirror.yaml"
CARDS = REPO_ROOT / "configs" / "g22-mirror"

ENTRY = "adopter"
ROLES = ("adopter", "coder_a", "coder_b")
# The four directed edges leave candidates disconnected.
EDGES = {
    ("adopter", "coder_a"),
    ("adopter", "coder_b"),
    ("coder_a", "adopter"),
    ("coder_b", "adopter"),
}
# Match G22 _coder_tools() to preserve the intended tool bundle.
G22_CANDIDATE_TOOLS = frozenset(
    {"bash", "file_read", "file_write", "apply_patch", "grep", "git_diff"}
)
# Team coordination adds tools absent from Python-sequenced G22.
COLLABORATION_TOOLS = frozenset({"message_agent", "submit", "team_status"})
EDIT_TOOLS = frozenset({"apply_patch", "file_write"})
#: Named in every card only to say the team does not have it ("there is no
#: ``spawn_agent`` on this team"), so a mention is not a claim to hold it.
NAMED_AS_ABSENT = frozenset({"spawn_agent"})

CARD_FILES = {
    "adopter": "adopter.md",
    "coder_a": "coder-a.md",
    "coder_b": "coder-b.md",
}

MINIMAL_CODER_TASK = """\
Find the narrowest root cause that fully explains the issue. Preserve backward
compatibility and existing public behavior outside the requested change. Trace
the immediate callers and consumers needed to verify the fix, then implement a
minimal complete source patch. Avoid broad refactors, speculative cleanup, test
edits, generated files, caches, and logs. Run the nearest relevant public tests
with bash, inspect the final diff, and finish with a non-empty source diff.
Do not use official results, hidden tests, FAIL_TO_PASS ids, grader patches, or
historical outcomes."""

CROSS_COMPONENT_CODER_TASK = """\
Solve the issue end to end with emphasis on cross-component completeness.
Trace every producer and producing state or data path, every direct consumer, public API and
serialization contract, error propagation, lifecycle boundary, and relevant
edge cases. Implement the smallest patch that covers the whole contract while
preserving unrelated behavior. Avoid test edits, generated files, caches, and
logs. Run relevant public tests through Bash using the project's native test command. When the shared command is
available, execute the same native command without replacing
it with an easier test. Inspect the final diff and finish with a non-empty source
patch. Do not use official results, hidden tests, FAIL_TO_PASS ids, grader
patches, or historical outcomes."""

CONTRACT_OPENING = """\
Candidate A was instructed to make the narrowest compatible root-cause repair.
Candidate B was instructed to cover producer, consumer, API, lifecycle, and
edge-case contracts. You cannot edit, merge, or rerun either candidate."""

CONTRACT_JUDGING = """\
Enumerate every explicit behavior requirement in the public issue. For each
requirement, compare the actual A and B diffs against the relevant producer,
consumer, and public API behavior. Cite concrete changed paths and diff details.
Public test records are comparable only when target, runner, and command are
identical. Do not reward larger diffs, stylistic changes, or unsupported claims.
Choose B only when public evidence shows B covers at least one requirement A
does not cover and no requirement is better covered by A."""

#: ``SHARED_RULES`` minus its last line, which is the one documented inversion.
SHARED_RULES_HEAD = """\
Rules:
- Use public issue, repository, test, and documentation evidence only.
- Never use hidden grader data, official hidden tests, grader patches, or FAIL_TO_PASS IDs.
- Obey this role and its tools.
- Keep probes under /tmp/opencollab-validation-* and out of the patch.
- Report unavailable probes as not_run. Make the smallest source fix.
- Read-only roles do not search for write tools."""
NO_COMMIT_RULE = "- Do not run git commit."

@pytest.fixture(scope="module")
def team():
    return load_team_config(path=str(TEAM_FILE))

def _card(role: str) -> str:
    return (CARDS / CARD_FILES[role]).read_text(encoding="utf-8")

def test_the_cell_seats_g22_s_roster_over_g22_s_four_edges(team) -> None:
    assert team.entry == ENTRY
    assert sorted(team.roles) == sorted(ROLES)
    edges = {
        (sender, receiver)
        for sender, receivers in team.topology.edges.items()
        for receiver in receivers
    }
    assert edges == EDGES

@pytest.mark.parametrize("role", ["coder_a", "coder_b"])
def test_each_candidate_carries_g22_s_candidate_bundle_and_nothing_else(team, role) -> None:
    granted = frozenset(team.roles[role].tools)
    assert granted - COLLABORATION_TOOLS == G22_CANDIDATE_TOOLS
    assert "run_tests" not in granted

def test_the_adjudicator_holds_no_tool_that_edits_a_file(team) -> None:
    """The adjudicator needs bash to check the winning commit into its answer tree."""
    granted = frozenset(team.roles[ENTRY].tools)
    assert granted & EDIT_TOOLS == frozenset()
    assert "bash" in granted

@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("coder_a", MINIMAL_CODER_TASK),
        ("coder_b", CROSS_COMPONENT_CODER_TASK),
        ("adopter", CONTRACT_OPENING),
        ("adopter", CONTRACT_JUDGING),
    ],
)
def test_g22_s_own_paragraphs_crossed_unedited(role: str, expected: str) -> None:
    assert expected in _card(role)

@pytest.mark.parametrize("role", ROLES)
def test_the_shared_rules_cross_with_exactly_one_documented_inversion(role: str) -> None:
    """Worktree handoff requires candidate commits despite G22's original commit prohibition."""
    card = _card(role)
    assert SHARED_RULES_HEAD in card
    if role == ENTRY:
        assert NO_COMMIT_RULE in card
    else:
        assert NO_COMMIT_RULE not in card
        assert "Commit your finished work; that is how it reaches the Adopter." in card

@pytest.mark.parametrize("role", ROLES)
def test_a_card_names_only_tools_this_team_actually_has(team, role: str) -> None:
    """Cards can describe teammate tools, so validate mentions against the whole team bundle."""
    held_by_anyone = {tool for seat in team.roles.values() for tool in seat.tools}
    named = {
        name
        for name in re.findall(r"`([a-z_]+)`", _card(role))
        if name in KNOWN_TOOL_NAMES
    } - NAMED_AS_ABSENT
    assert named <= held_by_anyone, (
        f"{role} names {sorted(named - held_by_anyone)}, which no seat holds"
    )

def test_the_adjudicators_claim_about_the_candidates_bundles_is_true(team) -> None:
    adopter = set(team.roles[ENTRY].tools)
    claimed = EDIT_TOOLS
    assert "They hold the edit tools you do not:" in _card(ENTRY)
    assert claimed & adopter == set()
    for role in ("coder_a", "coder_b"):
        assert claimed <= set(team.roles[role].tools)

def test_the_two_candidate_cards_differ_only_where_g22_differs() -> None:
    """The task paragraph carries G22's distinction between candidates A and B."""
    a, b = _card("coder_a"), _card("coder_b")
    assert MINIMAL_CODER_TASK in a and MINIMAL_CODER_TASK not in b
    assert CROSS_COMPONENT_CODER_TASK in b and CROSS_COMPONENT_CODER_TASK not in a
    assert "The shared public command" in b and "The shared public command" not in a
