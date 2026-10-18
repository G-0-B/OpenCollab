"""JSON Schema binding for named tools on Responses-compatible endpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_STRUCTURED_OUTPUT_TOOL = "structured_output"

@dataclass(frozen=True)
class ForcedTextTool:
    """One named tool represented by a Responses JSON Schema text result."""

    name: str
    description: str | None
    schema: dict[str, Any]

def forced_text_tool(
    converted_tools: list[dict[str, Any]],
    tool_choice: Any,
    *,
    supports_forced_tool_choice: bool,
    supports_json_schema: bool,
) -> ForcedTextTool | None:
    """Return the single named tool that needs JSON Schema binding."""
    if supports_forced_tool_choice or not supports_json_schema or not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "function":
        return None
    name = tool_choice.get("name")
    if name != _STRUCTURED_OUTPUT_TOOL or len(converted_tools) != 1:
        return None
    tool = converted_tools[0]
    if tool.get("name") != name:
        raise ValueError("forced tool_choice does not match the available tool")
    description = tool.get("description")
    return ForcedTextTool(
        name=name,
        description=description if isinstance(description, str) else None,
        schema=tool["parameters"],
    )

def forced_text_format(tool: ForcedTextTool) -> dict[str, Any]:
    """Build the native Responses ``text.format`` JSON Schema payload."""
    result: dict[str, Any] = {
        "type": "json_schema",
        "name": tool.name,
        "schema": tool.schema,
        "strict": True,
    }
    if tool.description:
        result["description"] = tool.description
    return {"format": result}

def project_forced_text_tool(
    tool: ForcedTextTool,
    content: str,
    *,
    response_identity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert schema-bound response text into a replayable function call."""
    try:
        arguments_value = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise ValueError("JSON Schema tool response contained invalid JSON") from exc
    if not isinstance(arguments_value, dict):
        raise ValueError("JSON Schema tool response must be an object")
    arguments = json.dumps(
        arguments_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{response_identity}\0{tool.name}\0{arguments}".encode()).hexdigest()
    call_id = f"call_schema_{digest[:24]}"
    tool_call = {
        "id": call_id,
        "type": "function",
        "function": {"name": tool.name, "arguments": arguments},
    }
    provider_item = {
        "id": f"fc_{digest[:24]}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": tool.name,
        "arguments": arguments,
    }
    return tool_call, provider_item

__all__ = [
    "ForcedTextTool",
    "forced_text_format",
    "forced_text_tool",
    "project_forced_text_tool",
]
