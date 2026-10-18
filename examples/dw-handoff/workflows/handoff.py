"""Script handoffs for the roles and tools in ``team.handoff.experiment.yaml``.

Peers use separate worktrees and share changes through Git commits. The script
controls handoff timing; the team configuration delegates timing to agents.
"""

from __future__ import annotations

import json
from typing import Any

from opencollab import workflow
from opencollab.tools import builtin_tools
from opencollab.workflows import WorkflowContext

# Match the scripted team comparison with one repair round.
MAX_REPAIR_ROUNDS = 1

ROLE_TOOLS = {
    # The analyst retains the single agent's working tools, minus coordination.
    "analyst": ("apply_patch", "bash", "file_read", "grep"),
    "coder": ("apply_patch", "bash", "file_read", "grep"),
    # Match the team's tester bundle, using git_diff instead of apply_patch.
    "tester": ("bash", "file_read", "git_diff", "grep"),
}

HANDOFF_RULES = """\
You are working in a git worktree of your own. Your teammates have their own,
and none of you can see another's files. Your worktree shares .git/objects with
theirs, so any commit you make is reachable from their trees immediately, by
sha. A sha is the only way your work can reach anyone else.
"""

BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["root_cause", "files", "implementation_task", "verification_task"],
    "properties": {
        "root_cause": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"}},
        "implementation_task": {"type": "string"},
        "verification_task": {"type": "string"},
    },
    "additionalProperties": False,
}

# Require a commit so isolated peers can access the coder's changes.
PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["commit", "summary"],
    "properties": {
        "commit": {"type": "string"},
        "summary": {"type": "string"},
    },
    "additionalProperties": False,
}

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["adopted_commit", "verdict", "findings"],
    "properties": {
        # Compare the reported checkout with the tester's worktree evidence.
        "adopted_commit": {"type": "string"},
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "findings": {"type": "string"},
    },
    "additionalProperties": False,
}

ANALYST_PROMPT = """\
{rules}
You are the analyst. Read the repository and work out what is actually broken.

Task: {goal}

Report the root cause, the files involved, what implementing the fix requires,
and what verifying it requires. You have the tools to fix it yourself; whether
you use them is your decision and does not change what is asked of you here.
"""

CODER_PROMPT = """\
{rules}
You are the coder. Implement the fix in your worktree.

Task: {goal}

The analyst's brief:
{brief}

{findings}

When the edit is done, commit it in your worktree and report the commit sha.
Your teammates cannot see your files; the sha is what reaches them.
"""

TESTER_PROMPT = """\
{rules}
You are the tester. The coder committed {commit} in its worktree.

Task: {goal}

The analyst's brief:
{brief}

The coder reported:
{summary}

That sha is reachable from your worktree. Check it out, then verify the fix
against the task. Report which sha you actually adopted, your verdict, and what
you found.
"""

def _tools(role: str, *, allow_unisolated_shell: bool = False) -> list[Any]:
    return list(builtin_tools(*ROLE_TOOLS[role], headless=not allow_unisolated_shell))

def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)

@workflow(
    name="dw-handoff",
    description="The handoff experiment with the exchange sequenced by a script.",
    phases=["analyze", "implement", "verify"],
)
async def dw_handoff(ctx: WorkflowContext, args: dict[str, Any]) -> dict[str, Any]:
    goal = str(args.get("goal") or args.get("description") or "").strip()
    if not goal:
        return {"status": "error", "error": 'missing "goal"'}
    allow_unisolated_shell = args.get("allow_unisolated_shell", False)
    if not isinstance(allow_unisolated_shell, bool):
        return {"status": "error", "error": "allow_unisolated_shell must be a boolean"}

    await ctx.phase("analyze")
    brief = await ctx.agent(
        ANALYST_PROMPT.format(rules=HANDOFF_RULES, goal=goal),
        schema=BRIEF_SCHEMA,
        label="analyst",
        tools=_tools("analyst", allow_unisolated_shell=allow_unisolated_shell),
        isolation=True,
    )
    if not isinstance(brief, dict):
        return {"status": "error", "error": "analyst produced no brief"}

    findings = "No tester findings yet."
    rounds: list[dict[str, Any]] = []
    for round_no in range(1, MAX_REPAIR_ROUNDS + 2):
        await ctx.phase("implement" if round_no == 1 else "verify")
        patch = await ctx.agent(
            CODER_PROMPT.format(
                rules=HANDOFF_RULES,
                goal=goal,
                brief=_dump(brief),
                findings=findings,
            ),
            schema=PATCH_SCHEMA,
            label=f"coder:{round_no}",
            tools=_tools("coder", allow_unisolated_shell=allow_unisolated_shell),
            isolation=True,
        )
        if not isinstance(patch, dict):
            rounds.append({"round": round_no, "status": "coder produced no commit"})
            break

        await ctx.phase("verify")
        verdict = await ctx.agent(
            TESTER_PROMPT.format(
                rules=HANDOFF_RULES,
                goal=goal,
                brief=_dump(brief),
                commit=patch["commit"],
                summary=patch["summary"],
            ),
            schema=VERDICT_SCHEMA,
            label=f"tester:{round_no}",
            tools=_tools("tester", allow_unisolated_shell=allow_unisolated_shell),
            isolation=True,
        )
        if not isinstance(verdict, dict):
            verdict = {
                "adopted_commit": "",
                "verdict": "FAIL",
                "findings": "Tester produced no verdict.",
            }
        rounds.append(
            {
                "round": round_no,
                "offered_commit": patch["commit"],
                "adopted_commit": verdict["adopted_commit"],
                # Compare the tester's reported commit with the offered handoff.
                "handoff_taken": verdict["adopted_commit"] == patch["commit"],
                "verdict": verdict["verdict"],
            }
        )
        if verdict["verdict"] == "PASS":
            break
        findings = f"The tester reported: {verdict['findings']}"

    return {
        "status": "ok",
        "rounds": rounds,
        "tokens_spent": ctx.tokens_spent(),
    }
