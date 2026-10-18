"""Load a catalog skill's full instructions through ``use_skill(name)``.

The store is injected at construction. Its body becomes the tool result,
preserving the cached system prefix. The store owns description and body
size limits, and the dispatcher returns its body unchanged.
"""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool
from opencollab.application.ports import SkillStorePort
from opencollab.application.tool_execution import ToolRuntime


class UseSkillTool(Tool):
    """Loads a catalog skill's full instructions into the agent's context."""

    name = "use_skill"
    description = (
        "Load the full instructions for a named skill into your context. Call "
        "this when a skill from the catalog matches the task at hand; the skill's "
        "complete instructions are returned to you as this tool call's result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact skill name from the catalog.",
            },
        },
        "required": ["name"],
    }

    def __init__(self, store: SkillStorePort) -> None:
        self._store = store

    async def execute_with_runtime(
        self,
        params: dict[str, Any],
        runtime: ToolRuntime,
    ) -> str:
        name = params.get("name", "")
        body = self._store.get_body(name)
        if body is None:
            available = ", ".join(m.name for m in self._store.list_manifests())
            if not available:
                return (
                    f"Unknown skill '{name}'. No skills are available in the catalog."
                )
            return f"Unknown skill '{name}'. Available skills: {available}"
        return body

__all__ = ["UseSkillTool"]
