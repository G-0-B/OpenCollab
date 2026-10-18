"""Accepted default spellings must select the same team agent."""

import pytest

from opencollab.bootstrap.agent_profiles import resolve_agent_profile
from opencollab.bootstrap.context_builder import ContextBuilder
from opencollab.bootstrap.single2_prompt import SINGLE2_SYSTEM_PROMPT
from opencollab.bootstrap.team_config import load_team_config
from tests.support.bootstrap_test_support import spawn_config


@pytest.mark.parametrize("profile", ["default", "Default", "DEFAULT", " default "])
def test_default_profile_alias_keeps_the_custom_role_prompt(tmp_path, profile):
    path = tmp_path / "team.yaml"
    path.write_text(
        f'entry: solo\nroles:\n  solo:\n    prompt: custom role card\n    profile: "{profile}"\n',
        encoding="utf-8",
    )
    team = load_team_config(path=str(path))
    builder = ContextBuilder(team, spawn_config(api_key="fixture-key"))

    assert team.roles["solo"].profile is None
    assert builder._profile_for(team.roles["solo"]) is None
    assert builder.build_plan("solo").system_prompt() == "custom role card"

@pytest.mark.parametrize("profile", ["base", "BASE", "single2", "Single2"])
def test_named_profile_alias_keeps_its_base_prompt_and_custom_role(tmp_path, profile):
    path = tmp_path / "team.yaml"
    path.write_text(
        f'entry: solo\nroles:\n  solo:\n    prompt: custom role card\n    profile: "{profile}"\n',
        encoding="utf-8",
    )
    team = load_team_config(path=str(path))
    builder = ContextBuilder(team, spawn_config(api_key="fixture-key"))
    prompt = builder.build_plan("solo").system_prompt()

    assert resolve_agent_profile(team.roles["solo"].profile).name == "single2"
    assert prompt.startswith(SINGLE2_SYSTEM_PROMPT)
    assert prompt.endswith("custom role card")
