"""Exact production context assembly for a fully populated plan."""

from __future__ import annotations

from opencollab.application.event_bus import EventBus
from opencollab.bootstrap.container import ContextBuilder, SpawnConfig
from opencollab.bootstrap.team_config import RoleConfig, TeamConfig
from opencollab.domain.scheduler import DelegationTask
from opencollab.domain.skill import SkillManifest
from opencollab.domain.team import Topology


class _SkillStore:
    def __init__(self, skills: dict[str, str]):
        self._skills = dict(skills)

    def list_manifests(self) -> tuple[SkillManifest, ...]:
        return tuple(SkillManifest(name=n, description=d) for n, d in self._skills.items())

    def get_body(self, name: str) -> str | None:
        return self._skills.get(name)

def _cfg() -> SpawnConfig:
    return SpawnConfig(
        model="m",
        provider="openai",
        api_key="k",
        base_url=None,
        llm_timeout=600.0,
        tracer=None,
        event_bus=EventBus(None),
        permission_policy=None,
    )

def _team() -> TeamConfig:
    return TeamConfig(
        roles={
            "lead": RoleConfig(
                prompt="You are the lead.",
                model=None,
                tools=["bash", "spawn_agent", "message_agent", "team_status", "use_skill"],
            ),
            "coder": RoleConfig(prompt="Coder.", model=None, tools=["bash"]),
        },
        topology=Topology(edges={"lead": frozenset({"coder"})}),
    )

# Literal prompt bodies provide an expected value independent of the source objects.
_TEAM_SECTION = (
    "## Your team\n"
    "\n"
    "Roles you may spawn or message:\n"
    "- coder\n"
    "\n"
    "Use `team_status` to list live agents (with their ids) and "
    "`message_agent` to send an async message to one."
)
_SKILLS_SECTION = (
    "## Skills\n"
    "\n"
    "Specialized instruction sets you can load on demand. When one matches "
    "the task, call `use_skill(name)` to pull its full instructions into "
    "your context.\n"
    "\n"
    "- refactor: Refactor code safely."
)
_PROJECT_MAP = "# Repo Map\n\nsrc/app.py"

def _builder() -> ContextBuilder:
    return ContextBuilder(
        _team(), _cfg(),
        skill_store=_SkillStore({"refactor": "Refactor code safely."}),
        project_context=_PROJECT_MAP,
    )

def test_golden_system_prompt_is_identity_team_skills_project_in_order():
    plan = _builder().build_plan("lead", task="Fix the bug.", context="See ticket #42.")
    assert plan.system_prompt() == "\n\n".join(
        ["You are the lead.", _TEAM_SECTION, _SKILLS_SECTION, _PROJECT_MAP]
    )

def test_golden_startup_user_messages_is_the_task_alone_with_ctx_tag():
    plan = _builder().build_plan("lead", task="Fix the bug.", context="See ticket #42.")
    task_body = DelegationTask(
        role="lead", task="Fix the bug.", context="See ticket #42."
    ).render()
    assert plan.startup_user_messages() == [
        {
            "role": "user",
            "content": task_body,
            "_ctx": {"name": "task", "layer": "task", "priority": 80},
        }
    ]

def test_golden_minimal_plan_assembles_to_only_its_real_sources():
    # A bare role contributes identity and task without empty reserved layers.
    plan = ContextBuilder(_team(), _cfg()).build_plan("coder", task="do it")
    assert plan.system_prompt() == "Coder."
    assert [m["content"] for m in plan.startup_user_messages()] == [
        DelegationTask(role="coder", task="do it", context="").render()
    ]
