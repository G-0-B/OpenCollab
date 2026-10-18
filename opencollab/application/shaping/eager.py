"""Optional age-based tool-output clearing before each model call.

Keep the K most recent compactable results and replace older content with
deterministic, message-derived stubs. Pinned sources, other messages and
assistant/tool pairing remain intact. Full history and transcripts are kept.

The projection is deterministic and idempotent. As history grows, results
move from full content to stable stubs once, preserving the deep prefix for
caching. It uses age rather than token-pressure thresholds and does not
configure provider cache breakpoints.
"""

from __future__ import annotations

from typing import Any

from opencollab.application.shaping.pipeline import (
    is_pinned,
    matched_tool_result_occurrences,
    require_positive_int,
)
from opencollab.application.shaping.reactive import DEFAULT_COMPACTABLE_TOOLS

# Keep more recent results than reactive clearing at low context pressure.
# Dropping reads still in the working set causes repeated multi-file reads.
# Older results retain source slices in stubs so the model can identify prior reads.
DEFAULT_EAGER_KEEP_RECENT = 12

# Visible stubs name the source tool and target for deterministic re-reads.
# The full text remains in the transcript.
EAGER_STUB_PREFIX = "[Old tool result cleared"

def _stub_for_call(tool_name: str, target: str | None) -> str:
    """Deterministic, byte-stable stub naming the tool, its target and (for reads)
    the exact slice already seen.

    A pure function of ``(tool_name, target)`` — no token counts, no message
    index, no randomness — so the same old message yields an identical stub on
    every turn (the monotonic / cacheable property). The wording tells the model
    it ALREADY ran this call and the output is in the transcript, so it does not
    re-issue the call merely to re-confirm what it read — the failure mode that
    burns a whole step budget on identical re-reads.
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
    """Best-effort target label lifted from a tool call's arguments.

    For ``file_read`` the label carries the exact line range already seen
    (``foo.py lines 1-100``) so each page of a multi-page read gets a DISTINCT
    stub — otherwise every page of one file collapses to the same stub and the
    model cannot tell which slice it already holds. Other tools keep their plain
    path/pattern/command label. Deterministic; never raises.
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
    """Always-on, age-based, deterministic tool-output clearing.

    Keeps the ``keep_recent`` most-recent compactable tool results verbatim and
    replaces every older compactable result's content with a deterministic,
    message-derived stub. Pinned sources and non-compactable messages are never
    touched; the ``tool_call`` <-> result pairing skeleton is preserved. Runs on
    every call with no estimate / trigger dependency.
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
        """Map exact result index -> stub for every compactable result
        older than the last ``keep_recent``.

        Compactable calls are scanned in issue order from the assistant
        ``tool_calls`` (the tool *name* lives on the call, not the ``role:"tool"``
        answer), so age is positional and independent of any token estimate.
        Pinned issuing turns are skipped. The last ``keep_recent`` ids are
        excluded so they stay verbatim.
        """
        # Build stubs only for cleared calls to avoid parsing kept results every turn.
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
