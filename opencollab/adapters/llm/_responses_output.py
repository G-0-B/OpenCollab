"""Semantic projection and reconciliation of native Responses output."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.llm.responses_errors import ResponsesProtocolError
from opencollab.adapters.llm.responses_messages import function_call_identity as _function_call_identity


def _output_text(item: dict[str, Any]) -> str:
    if item.get("type") != "message":
        return ""
    parts: list[str] = []
    for content in item.get("content") or ():
        if not isinstance(content, dict):
            continue
        if content.get("type") == "output_text" and isinstance(content.get("text"), str):
            parts.append(content["text"])
        elif content.get("type") == "refusal" and isinstance(content.get("refusal"), str):
            parts.append(content["refusal"])
    return "".join(parts)

def _reasoning_text(item: dict[str, Any]) -> str:
    if item.get("type") != "reasoning":
        return ""
    parts: list[str] = []
    for summary in item.get("summary") or ():
        if isinstance(summary, dict) and isinstance(summary.get("text"), str):
            parts.append(summary["text"])
    return "\n".join(parts)

def _semantic_output_item(item: dict[str, Any]) -> tuple[Any, ...]:
    """Return the stable meaning shared by streamed and terminal output items."""
    item_type = item.get("type")
    if item_type == "function_call":
        return (
            item_type,
            *_function_call_identity(
                item.get("call_id"),
                item.get("name"),
                item.get("arguments"),
            ),
        )
    if item_type == "message":
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ResponsesProtocolError("message output item is missing its role")
        phase = item.get("phase")
        if phase is not None and phase not in {"commentary", "final_answer"}:
            raise ResponsesProtocolError(
                f"message output contains unsupported phase {phase!r}"
            )
        parts: list[tuple[str, str]] = []
        for raw_part in item.get("content") or ():
            if not isinstance(raw_part, dict):
                raise ResponsesProtocolError("message output contains an invalid content part")
            part_type = raw_part.get("type")
            if part_type == "output_text" and isinstance(raw_part.get("text"), str):
                parts.append((part_type, raw_part["text"]))
            elif part_type == "refusal" and isinstance(raw_part.get("refusal"), str):
                parts.append((part_type, raw_part["refusal"]))
            else:
                raise ResponsesProtocolError("message output contains an unsupported content part")
        return item_type, role, tuple(parts)
    if item_type == "reasoning":
        summaries: list[tuple[str | None, str]] = []
        for raw_summary in item.get("summary") or ():
            if not isinstance(raw_summary, dict) or not isinstance(raw_summary.get("text"), str):
                raise ResponsesProtocolError("reasoning output contains an invalid summary part")
            summaries.append((raw_summary.get("type"), raw_summary["text"]))
        return item_type, tuple(summaries)
    raise ResponsesProtocolError("response output contains an unsupported item")

def _output_items_agree(streamed: dict[str, Any], terminal: dict[str, Any]) -> bool:
    if _semantic_output_item(streamed) != _semantic_output_item(terminal):
        return False
    if streamed.get("type") == "message":
        streamed_phase = streamed.get("phase")
        terminal_phase = terminal.get("phase")
        if (
            streamed_phase is not None
            and terminal_phase is not None
            and streamed_phase != terminal_phase
        ):
            return False
        return True
    if streamed.get("type") != "reasoning":
        return True
    streamed_encrypted = streamed.get("encrypted_content")
    terminal_encrypted = terminal.get("encrypted_content")
    for value in (streamed_encrypted, terminal_encrypted):
        if value is not None and not isinstance(value, str):
            raise ResponsesProtocolError("reasoning output contains invalid encrypted content")
    return streamed_encrypted is None or terminal_encrypted is None or streamed_encrypted == terminal_encrypted

def _merge_terminal_projection(
    streamed: dict[str, Any],
    terminal: dict[str, Any],
) -> dict[str, Any]:
    if (
        streamed.get("type") == "reasoning"
        and streamed.get("encrypted_content") is None
        and terminal.get("encrypted_content") is not None
    ):
        streamed = {**streamed, "encrypted_content": terminal["encrypted_content"]}
    if (
        streamed.get("type") == "message"
        and streamed.get("phase") is None
        and terminal.get("phase") is not None
    ):
        streamed = {**streamed, "phase": terminal["phase"]}
    return streamed
