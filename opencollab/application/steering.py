"""Build budget/progress and action nudges from caller-supplied counters.

The runner owns tool vocabulary and persistence; this module returns values.
"""

from __future__ import annotations

import os
from typing import Any

# Reads-without-write escalation thresholds: at SOFT we advise a write, at HARD
# we demand it and force a tool call.
READS_NUDGE_SOFT = 8
READS_NUDGE_HARD = 16

# Treat edit and structured-submit pressure as one experimental intervention.
# Turning it off permits measuring delegation without that action demand;
# budget status remains independently controlled.
WRITE_NUDGE_ON = "on"
WRITE_NUDGE_OFF = "off"
WRITE_NUDGE_MODES = (WRITE_NUDGE_ON, WRITE_NUDGE_OFF)
WRITE_NUDGE_ENV_VAR = "OPENCOLLAB_WRITE_NUDGE_MODE"

# Status cadence is independent of action nudges. Thresholds fire only on
# crossing, once per band under monotone spend.
BUDGET_NUDGE_EVERY_STEP = "every-step"
BUDGET_NUDGE_THRESHOLDS = "thresholds"
BUDGET_NUDGE_OFF = "off"
BUDGET_NUDGE_MODES = (
    BUDGET_NUDGE_EVERY_STEP,
    BUDGET_NUDGE_THRESHOLDS,
    BUDGET_NUDGE_OFF,
)
BUDGET_NUDGE_ENV_VAR = "OPENCOLLAB_BUDGET_NUDGE_MODE"
BUDGET_NUDGE_THRESHOLDS_FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def resolve_budget_nudge_mode(env: Any = None) -> str:
    """Return the configured cadence; unset, blank, or invalid values use every-step."""
    source = os.environ if env is None else env
    raw = str(source.get(BUDGET_NUDGE_ENV_VAR, "") or "").strip().lower()
    return raw if raw in BUDGET_NUDGE_MODES else BUDGET_NUDGE_EVERY_STEP


def resolve_write_nudge_mode(env: Any = None) -> str:
    """Return the write-nudge mode; unset, blank, or invalid values preserve on."""
    source = os.environ if env is None else env
    raw = str(source.get(WRITE_NUDGE_ENV_VAR, "") or "").strip().lower()
    return raw if raw in WRITE_NUDGE_MODES else WRITE_NUDGE_ON


def _crosses_budget_threshold(
    used_tokens: int, prev_used_tokens: int, max_budget_tokens: int
) -> bool:
    """Fire once when spend crosses a threshold, not on every later step."""
    if max_budget_tokens <= 0:
        return False
    previous = min(max(prev_used_tokens, 0), used_tokens)
    frac = used_tokens / max_budget_tokens
    prev_frac = previous / max_budget_tokens
    return any(
        prev_frac < fraction <= frac for fraction in BUDGET_NUDGE_THRESHOLDS_FRACTIONS
    )


def build_steering_block(
    *,
    used_tokens: int,
    max_budget_tokens: int | None,
    step_count: int,
    max_steps: int | None,
    reads: int,
    has_write: bool,
    has_structured_output: bool,
    structured_override: Any,
    write_landed: bool = False,
    budget_nudge_mode: str = BUDGET_NUDGE_EVERY_STEP,
    write_nudge_mode: str = WRITE_NUDGE_ON,
    prev_used_tokens: int = 0,
) -> tuple[dict[str, Any] | None, Any | None, str | None]:
    """Return ``(message, tool_choice_override, level)`` for this step.

    ``level`` is hard, soft, or None for tracing. ``structured_override`` is a
    caller-supplied choice for the submission tool. Budget cadence controls only
    the status line; thresholds need ``prev_used_tokens`` and a finite budget.
    Write-nudge mode controls edit and structured-submit escalation together.
    Suppressing both status and action produces ``message=None``.
    A landed write permits submission; unknown write effects are resolved by the caller.
    """
    if max_budget_tokens is None:
        token_status = f"~{used_tokens // 1000}k tokens used"
        if max_steps is None:
            step_status = f"{step_count} steps completed"
        else:
            step_status = f"~{max(0, max_steps - step_count)} steps left"
        status = f"[Progress: {token_status}, {step_status}.]"
    else:
        remaining_k = max(0, max_budget_tokens - used_tokens) // 1000
        total_k = max_budget_tokens // 1000
        if max_steps is None:
            step_status = f"{step_count} steps completed"
        else:
            step_status = f"~{max(0, max_steps - step_count)} steps left"
        status = (
            f"[Budget: ~{remaining_k}k/{total_k}k tokens left, "
            f"{step_status}.]"
        )
    # Unbounded runs have no denominator for threshold crossings.
    if budget_nudge_mode == BUDGET_NUDGE_OFF:
        status = ""
    elif budget_nudge_mode == BUDGET_NUDGE_THRESHOLDS and not _crosses_budget_threshold(
        used_tokens, prev_used_tokens, max_budget_tokens or 0
    ):
        status = ""

    override: Any | None = None
    level: str | None = None
    extra = ""
    needs_write = has_write and not write_landed
    needs_structured_submit = has_structured_output and (
        write_landed or not has_write
    )
    if write_nudge_mode == WRITE_NUDGE_OFF:
        needs_write = False
        needs_structured_submit = False
    if needs_write and reads >= READS_NUDGE_HARD:
        extra = (
            f" You have read {reads} times without making an edit. STOP reading"
            " — your next action MUST be a file_write or apply_patch edit."
        )
        override = "required"
        level = "hard"
    elif needs_structured_submit and reads >= READS_NUDGE_SOFT:
        extra = (
            f" You have read {reads} times without submitting structured output."
            " STOP reading — your next action MUST be structured_output using"
            " the evidence you already have."
        )
        override = structured_override
        level = "hard"
    elif needs_write and reads >= READS_NUDGE_SOFT:
        extra = (
            f" You have read {reads} times without making an edit. If"
            " you can describe the fix, make it now with file_write or"
            " apply_patch before reading more."
        )
        level = "soft"
    content = status + extra if status else extra.lstrip()
    if not content:
        return None, override, level
    return {"role": "user", "content": content}, override, level


def fold_steering(last_user_msg: dict[str, Any], steering_text: str) -> dict[str, Any]:
    """Copy a user message, appending steering to text or content parts.

    Do not mutate the original dict.
    """
    content = last_user_msg.get("content")
    if isinstance(content, str):
        folded = f"{content}\n\n{steering_text}" if content else steering_text
    elif isinstance(content, list):
        folded = [*content, {"type": "text", "text": steering_text}]
    else:
        folded = steering_text
    return {**last_user_msg, "content": folded}
