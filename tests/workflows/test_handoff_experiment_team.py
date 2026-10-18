"""The experimental team supports three-peer Git handoff with matching tools and edges."""

from __future__ import annotations

import pytest

from opencollab.bootstrap.team_config import load_team_config
from tests.support.paths import REPO_ROOT

TEAM_FILE = REPO_ROOT / "configs" / "team.handoff.experiment.yaml"
ROLES = ("analyst", "coder", "tester")

@pytest.fixture(scope="module")
def team():
    return load_team_config(path=str(TEAM_FILE))

def test_the_entry_is_the_analyst_and_the_roster_is_the_three_roles(team) -> None:
    assert team.entry == "analyst"
    assert sorted(team.roles) == sorted(ROLES)

def test_every_ordered_pair_is_a_declared_edge(team) -> None:
    """Prebuilt peers have no spawn join path and need return edges to send results."""
    walked = {
        (source, destination)
        for source in ROLES
        for destination in ROLES
        if team.topology.allows(source, destination)
    }
    assert walked == {
        (source, destination)
        for source in ROLES
        for destination in ROLES
        if source != destination
    }

def test_every_role_can_walk_its_edges(team) -> None:
    """``message_agent`` is the only channel a prebuilt team leaves open."""
    for role in ROLES:
        assert "message_agent" in team.roles[role].tools, role
        assert "team_status" in team.roles[role].tools, role

def test_the_two_halves_of_the_git_handoff_are_both_runnable(team) -> None:
    """Git commit and checkout both need bash on their respective seats."""
    assert "bash" in team.roles["coder"].tools
    assert "bash" in team.roles["tester"].tools

def test_no_role_carries_a_tool_that_can_only_be_refused(team) -> None:
    """Prebuilt teams refuse spawn_agent, leaving message_agent as the delegation channel."""
    for role in ROLES:
        assert "spawn_agent" not in team.roles[role].tools, role
        assert "spawn_with_review" not in team.roles[role].tools, role

def test_each_role_prompt_is_loaded_from_its_own_file(team) -> None:
    prompts = REPO_ROOT / "configs" / "handoff-experiment"
    for role in ROLES:
        body = (prompts / f"{role}.md").read_text(encoding="utf-8")
        assert team.roles[role].prompt == body
        assert "sha" in body
        assert "git checkout <sha>" in body

def test_analyst_prompt_describes_the_testers_editing_boundary(team) -> None:
    editing_tools = {"apply_patch", "file_write"}
    assert editing_tools <= set(team.roles["analyst"].tools)
    assert editing_tools.isdisjoint(team.roles["tester"].tools)
    card = " ".join(team.roles["analyst"].prompt.split())

    assert "The Coder and the Tester hold the same working tools you do" not in card

def test_the_shipped_default_team_is_not_this_team() -> None:
    from opencollab.bootstrap.team_config import default_team_config

    shipped = load_team_config()
    assert shipped == default_team_config()
    assert "message_agent" not in shipped.roles["analyst"].tools

# Match WORKING_TOOL_NAMES from OpenCollab-Eval generation/gen_prediction_constants.py.
SINGLE_AGENT_WORKING_TOOLS = frozenset(
    {"apply_patch", "bash", "file_read", "file_write", "grep", "submit"}
)

# Add coordination tools and git_diff for roles without edit access.
COLLABORATION_TOOLS = frozenset({"git_diff", "message_agent", "team_status"})

def test_the_analyst_holds_the_single_agent_s_tools_and_the_channel(team) -> None:
    """Equal tool bundles isolate organization from capability in the solo/team comparison."""
    analyst = set(team.roles["analyst"].tools)

    assert analyst - COLLABORATION_TOOLS == set(SINGLE_AGENT_WORKING_TOOLS)
    assert analyst & COLLABORATION_TOOLS == {"message_agent", "team_status"}

def test_no_role_holds_a_tool_that_cannot_succeed_here(team) -> None:
    """Unattended benchmark containers have neither a user to answer nor a shipped skill package."""
    for role in ROLES:
        tools = set(team.roles[role].tools)
        assert "ask_user" not in tools, role
        assert "use_skill" not in tools, role

def test_every_role_is_told_which_tree_is_read_as_the_answer() -> None:
    """The harness collects only the Analyst repository, excluding teammate worktrees."""
    prompts = REPO_ROOT / "configs" / "handoff-experiment"
    for role in ROLES:
        body = " ".join((prompts / f"{role}.md").read_text(encoding="utf-8").split())
        assert "read as the answer" in body, role

def test_every_role_is_told_that_finishing_is_how_you_wait() -> None:
    """Delivered replies reopen finished turns, so waiting can avoid repeated provider-backed polls."""
    prompts = REPO_ROOT / "configs" / "handoff-experiment"
    for role in ROLES:
        body = " ".join((prompts / f"{role}.md").read_text(encoding="utf-8").split())
        assert "finishing is how you wait" in body, role
