"""Shared head-and-tail truncation for tool output.

An optional ``label`` identifies the truncated stream in the marker.
"""

from __future__ import annotations


def require_positive_int(value: int, name: str) -> int:
    """Validate one public output budget before a tool can perform work."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

def truncate(text: str, max_chars: int, label: str | None = None) -> str:
    """Keep head + tail, drop the middle to avoid context explosion."""
    require_positive_int(max_chars, "max_chars")
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    while True:
        marker = (
            f"\n\n... [{dropped} chars of {label} truncated] ...\n\n"
            if label is not None
            else f"\n\n... [{dropped} chars truncated] ...\n\n"
        )
        source_budget = max(0, max_chars - len(marker))
        actual_dropped = len(text) - source_budget
        if actual_dropped == dropped:
            break
        dropped = actual_dropped
    if len(marker) >= max_chars:
        return marker[:max_chars]
    head = (source_budget + 1) // 2
    tail = source_budget - head
    suffix = text[-tail:] if tail else ""
    result = text[:head] + marker + suffix
    assert len(result) <= max_chars
    return result

__all__ = ["require_positive_int", "truncate"]
