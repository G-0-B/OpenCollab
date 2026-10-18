"""Record explicit completion and use the submitted summary as the turn answer.

A message with no tool calls still ends a turn. Workspace evaluation also
runs when the agent omits ``submit``.
"""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool
from opencollab.application.tool_execution import ToolRuntime

MAX_SUMMARY_CHARS = 4000

class SubmitTool(Tool):
    """End this agent's turn and record what it is handing over."""

    name = "submit"
    description = (
        "Declare that your work on this task is finished and end your turn. "
        "It is also how you wait. If you are waiting for a teammate's answer, "
        "submit what you have so far. The answer reopens your turn. "
        "Give a short summary of what you changed and why it answers the task. "
        "Call this instead of trailing off: it records that you stopped on "
        "purpose rather than ran out of things to say. Anything you have "
        "already written to the workspace is what gets read, whether or not "
        "you call this, so summarize rather than restating your work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "What you did and why it answers the task, in a "
                "few sentences.",
            },
        },
        "required": ["summary"],
    }

    def __init__(self) -> None:
        self.turn_submitted = False
        self.submitted_summary: str | None = None

    async def execute_with_runtime(
        self,
        params: dict[str, Any],
        runtime: ToolRuntime,
    ) -> str:
        # Reset first: one instance serves every step of a session, so a bad
        # call after a good one must not leave the turn marked as submitted.
        self.turn_submitted = False
        self.submitted_summary = None
        summary = params.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return "Error: submit needs a summary of what you are handing over."
        if len(summary) > MAX_SUMMARY_CHARS:
            return (
                f"Error: summary is {len(summary)} characters; keep it under "
                f"{MAX_SUMMARY_CHARS}. Summarize rather than restating your work."
            )
        self.submitted_summary = summary.strip()
        self.turn_submitted = True
        if runtime is not None and runtime.observations is not None:
            runtime.observations.record_submission(self.submitted_summary)
        return "Submitted. Your turn ends here."

__all__ = ["MAX_SUMMARY_CHARS", "SubmitTool"]
