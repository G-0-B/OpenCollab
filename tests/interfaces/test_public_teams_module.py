"""Read team roster size so a caller can scale a per-seat budget correctly."""

from __future__ import annotations

from opencollab.domain.scheduler import per_agent_cap
from opencollab.teams import (
    declared_context_policy,
    declared_role_budgets,
    declared_role_names,
    declared_role_prompt_digests,
    declared_role_tools,
)
from tests.support.paths import REPO_ROOT

EXPERIMENT_TEAM = REPO_ROOT / "configs" / "team.handoff.experiment.yaml"

def test_the_declared_roles_are_the_file_s_roles_in_order() -> None:
    assert declared_role_names(str(EXPERIMENT_TEAM)) == ("analyst", "coder", "tester")

def test_a_pool_of_n_seats_gives_each_seat_one_solo_agent_s_budget() -> None:
    solo_budget = 1_000_000
    roles = declared_role_names(str(EXPERIMENT_TEAM))

    cap = per_agent_cap(solo_budget * len(roles), len(roles))

    # The pool size must preserve the solo budget per seat.
    assert cap == solo_budget
    assert per_agent_cap(solo_budget, len(roles)) == solo_budget // 3

ICLR_EXPERIMENT_TEAM = REPO_ROOT / "configs" / "team.handoff.primary-legacy.yaml"

def test_the_declared_tools_are_read_through_the_loader_a_run_uses() -> None:
    bundles = declared_role_tools(str(ICLR_EXPERIMENT_TEAM))

    assert set(bundles) == set(declared_role_names(str(ICLR_EXPERIMENT_TEAM)))
    assert "message_agent" in bundles["analyst"]
    assert "apply_patch" not in bundles["tester"]
    assert "file_write" not in bundles["tester"]

def test_the_bundles_are_keyed_by_the_same_identities_as_the_role_names(tmp_path) -> None:
    # load_team_config normalizes role names, matching the names used by the run.
    team_file = tmp_path / "team.yaml"
    team_file.write_text(
        "entry: Solo\nroles:\n  Solo:\n    prompt: do the work\n"
        "    tools: [file_read]\n",
        encoding="utf-8",
    )

    bundles = declared_role_tools(str(team_file))

    assert tuple(bundles) == declared_role_names(str(team_file))
    assert bundles[declared_role_names(str(team_file))[0]] == ("file_read",)

def test_the_prompt_digest_is_of_the_card_text_not_of_its_path(tmp_path) -> None:
    # Resolved card text determines the metric grouping key, even when its path stays the same.
    card = tmp_path / "analyst.md"
    team_file = tmp_path / "team.yaml"
    team_file.write_text(
        "entry: analyst\nroles:\n  analyst:\n    prompt_file: analyst.md\n",
        encoding="utf-8",
    )

    card.write_text("decide for yourself", encoding="utf-8")
    first = declared_role_prompt_digests(str(team_file))["analyst"]
    card.write_text("hand the work over", encoding="utf-8")
    second = declared_role_prompt_digests(str(team_file))["analyst"]

    assert first != second

def test_the_two_generations_of_the_analyst_card_do_not_pool() -> None:
    """The same stance block can accompany different shared bodies or a repointed path."""
    first_generation = declared_role_prompt_digests(str(ICLR_EXPERIMENT_TEAM))["analyst"]
    second_generation = declared_role_prompt_digests(
        str(REPO_ROOT / "configs" / "team.handoff.facts-v2.yaml")
    )["analyst"]

    assert first_generation != second_generation

def test_roles_that_share_a_card_share_a_digest() -> None:
    """The Coder and Tester are held fixed across cells, so they must not move."""
    primary = declared_role_prompt_digests(str(ICLR_EXPERIMENT_TEAM))
    facts_v2 = declared_role_prompt_digests(
        str(REPO_ROOT / "configs" / "team.handoff.facts-v2.yaml")
    )

    assert primary["coder"] == facts_v2["coder"]
    assert primary["tester"] == facts_v2["tester"]

def test_a_file_without_allowances_declares_no_role_budgets() -> None:
    assert declared_role_budgets(str(EXPERIMENT_TEAM)) == {}

def test_declared_allowances_are_read_per_role_with_overrides(tmp_path) -> None:
    team_file = tmp_path / "team.yaml"
    team_file.write_text(
        "entry: Lead\nbudget: {tokens: 2000000}\nroles:\n"
        "  Lead:\n    prompt: lead\n"
        "  coder:\n    prompt: code\n    budget: {tokens: 500000}\n",
        encoding="utf-8",
    )

    budgets = declared_role_budgets(str(team_file))

    assert budgets == {"Lead": 2_000_000, "coder": 500_000}
    assert tuple(budgets) == declared_role_names(str(team_file))

def test_the_context_policy_defaults_when_the_file_names_none() -> None:
    assert declared_context_policy(str(EXPERIMENT_TEAM)) == {
        "policy": "default",
        "tool_result_budget": None,
    }

def test_a_declared_context_policy_is_read_with_its_parameter(tmp_path) -> None:
    team_file = tmp_path / "team.yaml"
    team_file.write_text(
        "entry: lead\ncontext: {policy: no_history_compaction, tool_result_budget: 20000}\n"
        "roles:\n  lead:\n    prompt: lead\n",
        encoding="utf-8",
    )

    assert declared_context_policy(str(team_file)) == {
        "policy": "no_history_compaction",
        "tool_result_budget": 20000,
    }
