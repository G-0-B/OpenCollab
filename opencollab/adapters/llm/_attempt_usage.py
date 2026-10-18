"""Shared usage projection for completions with provider retries."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.llm.types import Usage


def _optional_usage_int(source: Any, key: str) -> int | None:
    if not isinstance(source, dict) or source.get(key) is None:
        return None
    if isinstance(source[key], bool):
        return None
    try:
        value = int(source[key])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None

def _combine_attempt_usage(attempts: list[Usage | None]) -> Usage | None:
    """Sum known attempts and retain omissions in the native usage record."""
    known = [usage for usage in attempts if usage is not None]
    if not known:
        return None
    if len(attempts) == 1:
        return known[0]

    def optional_total(name: str) -> int | None:
        values = [getattr(usage, name) if usage is not None else None for usage in attempts]
        return None if any(value is None for value in values) else sum(values)

    return Usage(
        input_tokens=sum(usage.input_tokens for usage in known),
        output_tokens=sum(usage.output_tokens for usage in known),
        cache_read_tokens=optional_total("cache_read_tokens"),
        cache_creation_tokens=optional_total("cache_creation_tokens"),
        reasoning_tokens=optional_total("reasoning_tokens"),
        estimated=any(usage is None or usage.estimated for usage in attempts),
        raw_usage={"attempts": [usage.raw_usage if usage is not None else None for usage in attempts]},
        markup_recovered=sum(usage.markup_recovered for usage in known),
        context_tokens=known[-1].context_tokens if known[-1].context_tokens is not None else known[-1].input_tokens,
        context_output_tokens=(known[-1].context_output_tokens
                               if known[-1].context_output_tokens is not None else known[-1].output_tokens),
        context_tokens_estimated=(known[-1].context_tokens_estimated
                                  if known[-1].context_tokens_estimated is not None else known[-1].estimated),
    )
