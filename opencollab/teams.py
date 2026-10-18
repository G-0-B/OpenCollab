"""Read team roles, tools, prompts and budgets before starting a run.

A caller-supplied team pool is the solo allowance times the declared role count.
Files with per-role ``budget.tokens`` instead set their own total and reject a
caller-supplied pool.
"""

import hashlib

from opencollab.bootstrap.team_config import load_team_config

__all__ = [
    "declared_context_policy",
    "declared_role_budgets",
    "declared_role_names",
    "declared_role_profiles",
    "declared_role_prompt_digests",
    "declared_role_tools",
]

def declared_role_names(path: str) -> tuple[str, ...]:
    """Return normalized role names, entry role first.

    The declared count is ``N`` in the per-agent cap ``c * total / N``.
    """
    config = load_team_config(path=path)
    return (config.entry, *(name for name in config.roles if name != config.entry))

def declared_role_tools(path: str) -> dict[str, tuple[str, ...]]:
    """Return each normalized role's declared tools in file order.

    Live capability filtering and fallback bundles are applied later at spawn.
    """
    config = load_team_config(path=path)
    return {name: tuple(role.tools) for name, role in config.roles.items()}

def declared_role_prompt_digests(path: str) -> dict[str, str]:
    """Return each normalized role's SHA-256 digest of its loaded prompt text.

    Digests include resolved ``prompt_file`` contents, allowing metrics to group
    identical wording independently of file paths.
    """
    config = load_team_config(path=path)
    return {
        name: hashlib.sha256(role.prompt.encode("utf-8")).hexdigest()
        for name, role in config.roles.items()
    }

def declared_role_profiles(path: str) -> dict[str, str | None]:
    """Return each normalized role's profile name, or ``None`` for the native agent.

    Profiles supply the base prompt, history shaping, safety and tool limits;
    role prompt digests cover the role's own card.
    """
    config = load_team_config(path=path)
    return {name: role.profile for name, role in config.roles.items()}

def declared_role_budgets(path: str) -> dict[str, int]:
    """Return each normalized role's declared token allowance, or ``{}``.

    Declared allowances set the total and reject a caller-supplied pool. An empty
    mapping leaves the caller to supply a pool using the module's per-seat rule.
    """
    config = load_team_config(path=path)
    budgets = config.role_budgets
    return {name: budgets[name] for name in config.roles if name in budgets}

def declared_context_policy(path: str) -> dict[str, object]:
    """Return ``{"policy": name, "tool_result_budget": chars or None}``.

    Files without a ``context`` entry use ``default`` and the built-in result cap.
    """
    policy = load_team_config(path=path).context
    return {"policy": policy.name, "tool_result_budget": policy.tool_result_budget}
