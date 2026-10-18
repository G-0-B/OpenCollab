"""Optional age-based tool-output clearing over a copy of history.

Keep the newest K compactable results and replace older content with stable
message-derived stubs. Pinned sources and call/result pairing survive.
The projection is deterministic and idempotent; a given result's full-to-stub
transition still changes the sent prefix. Bootstrap therefore omits this
always-on layer from the default pressure-triggered pipeline.
"""

from __future__ import annotations

from typing import Any

from opencollab.application.shaping.pipeline import (
    is_pinned,
    matched_tool_result_occurrences,
    require_positive_int,
)
from opencollab.application.shaping.reactive import DEFAULT_COMPACTABLE_TOOLS

# K most-recent compactable tool results kept verbatim; everything older is
# stubbed. Set ABOVE the reactive keep-recent (``DEFAULT_TOOL_CLEAR_KEEP_RECENT``)
# on purpose: this rung runs at LOW context, where the cost of dropping a still-
# needed read is a re-read loop (a multi-file task whose working set exceeds K
# can never hold all its files at once and thrashes), so it retains more. The
# reactive layer stays tighter for genuine 120k pressure. The informative stubs
# below (which name the exact slice already read) are what make a cleared result
# safe — the model can see it already read X without re-fetching it.
DEFAULT_EAGER_KEEP_RECENT = 12

# Marker prefix on every eager stub. A stub announces itself (no invisible
# compression) and names the source tool + target so a reader / the model can
# re-fetch deterministically. The full text lives in the transcript untouched.
EAGER_STUB_PREFIX = "[Old tool result cleared"


def _stub_for_call(tool_name: str, target: str | None) -> str:
    """Name the tool and already-read target with a stable, message-derived stub.

    Exclude counters and randomness so unchanged old calls retain identical text.
    """
    where = f" {target}" if target else ""
    return (
        f"{EAGER_STUB_PREFIX}: {tool_name}{where}]"
        " — you already ran this; the full output is in the transcript above."
        " Re-issue ONLY if you need the exact content again, not to check whether"
        " you read it."
    )


def _as_dict(arguments: Any) -> dict[str, Any] | None:
    """A tool call's arguments as a dict, or ``None``. Deterministic; never raises."""
    if isinstance(arguments, str):
        import json

        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return None
    return arguments if isinstance(arguments, dict) else None


def _range_label(arguments: dict[str, Any]) -> str:
    """`` lines A-B`` / `` from line A`` / `` first N lines`` / `` (whole file)``,
    derived only from integer ``offset``/``limit`` so it is byte-stable."""
    offset = arguments.get("offset")
    limit = arguments.get("limit")
    off = offset if isinstance(offset, int) else None
    lim = limit if isinstance(limit, int) else None
    if off is not None and lim is not None:
        return f" lines {off}-{off + lim - 1}"
    if off is not None:
        return f" from line {off}"
    if lim is not None:
        return f" first {lim} lines"
    return " (whole file)"


def _call_target(tool_name: str | None, arguments: Any) -> str | None:
    """Extract a deterministic target label, returning safely on malformed arguments.

    Include file-read line ranges so separate pages do not collapse to one stub.
    """
    parsed = _as_dict(arguments)
    if parsed is None:
        return None
    if tool_name == "file_read":
        for key in ("path", "file_path", "filename", "file"):
            value = parsed.get(key)
            if isinstance(value, str) and value:
                return f"{value}{_range_label(parsed)}"
    for key in ("path", "file_path", "filename", "file", "pattern", "command"):
        value = parsed.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class EagerToolOutputClearShaper:
    """Clear old compactable content on every call, without a pressure trigger.

    Keep recent results, pinned sources, and call/result pairing intact.
    """

    #: Frozen trajectory label for this rung (see ``ShaperPipeline``).
    rung = "eager_clear"

    def __init__(
        self,
        *,
        compactable_tools: frozenset[str] = DEFAULT_COMPACTABLE_TOOLS,
        keep_recent: int = DEFAULT_EAGER_KEEP_RECENT,
    ):
        require_positive_int(keep_recent, "keep_recent")
        self.compactable_tools = compactable_tools
        self.keep_recent = keep_recent

    def shape(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stubs = self._stubs_for_old_results(messages)
        if not stubs:
            return messages

        out: list[dict[str, Any]] = []
        changed = False
        for index, message in enumerate(messages):
            stub = stubs.get(index)
            if (
                stub is not None
                and message.get("role") == "tool"
                and not is_pinned(message)
                and message.get("content") != stub
            ):
                out.append({**message, "content": stub})
                changed = True
            else:
                out.append(message)
        return out if changed else messages

    def _stubs_for_old_results(
        self, messages: list[dict[str, Any]]
    ) -> dict[int, str]:
        """Map old compactable result indices to stubs, excluding pinned issuing turns.

        Age follows assistant-call order; tool answers do not carry the tool name.
        The newest ``keep_recent`` results remain verbatim.
        """
        # Collect raw (id, name, arguments) for every compactable call first, then
        # parse arguments + build stubs ONLY for the older calls that get cleared —
        # the last ``keep_recent`` are kept verbatim, so parsing their JSON every
        # turn would be pure waste on this always-on rung.
        compactable: list[tuple[int, str, Any]] = []
        for result_index, _call_id, name, arguments in matched_tool_result_occurrences(
            messages
        ):
            result = messages[result_index]
            if (
                name in self.compactable_tools
                and not is_pinned(result)
                and isinstance(result.get("content"), str)
            ):
                compactable.append((result_index, name, arguments))
        if len(compactable) <= self.keep_recent:
            return {}
        stubs: dict[int, str] = {}
        for result_index, name, arguments in compactable[: -self.keep_recent]:
            stub = _stub_for_call(name, _call_target(name, arguments))
            if len(stub) < len(messages[result_index]["content"]):
                stubs[result_index] = stub
        return stubs
