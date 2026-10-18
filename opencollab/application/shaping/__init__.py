"""Message-view transformations applied before model calls.

Shapers work on copies of history, preserving full transcripts for recovery.
The lazy-degradation design follows Liu et al. 2026, sections 3.1 and 4.3.
Per-result budgets cap individual tool outputs. Under context pressure,
reactive layers progressively clear old tool content, remove old exchanges,
and summarize through an injected summarizer.

The optional eager layer clears aging tool outputs on every call. It keeps
recent results and stable stubs for older ones while preserving tool pairing.
Default assembly is defined in ``bootstrap.container``.

``pipeline`` owns ordering, thresholds and span helpers, ``tool_budget`` owns
per-result limits, and ``reactive`` owns pressure-triggered history layers.
"""

from opencollab.application.shaping.eager import (
    DEFAULT_EAGER_KEEP_RECENT,
    EAGER_STUB_PREFIX,
    EagerToolOutputClearShaper,
)
from opencollab.application.shaping.pipeline import (
    DEFAULT_COMPACT_BUFFER_TOKENS,
    DEFAULT_HISTORY_KEEP_RECENT_GROUPS,
    DEFAULT_HISTORY_TARGET_TOKENS,
    DEFAULT_HISTORY_TRIGGER_TOKENS,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    HISTORY_TARGET_RATIO,
    PIN_FLOOR,
    ShaperPipeline,
    approx_messages_tokens,
    forced_shape,
    history_trigger_target,
)
from opencollab.application.shaping.reactive import (
    COMPACTED_MARKER_PREFIX,
    DEFAULT_CLEARED_TOOL_CONTENT,
    DEFAULT_COMPACTABLE_TOOLS,
    DEFAULT_TOOL_CLEAR_KEEP_RECENT,
    AutoCompactShaper,
    OldHistorySnipShaper,
    SummarizerPort,
    ToolOutputClearShaper,
)
from opencollab.application.shaping.tool_budget import (
    DEFAULT_TOOL_RESULT_BUDGET,
    PerToolResultBudgetShaper,
)

__all__ = [
    "DEFAULT_TOOL_RESULT_BUDGET",
    "DEFAULT_HISTORY_TRIGGER_TOKENS",
    "DEFAULT_HISTORY_TARGET_TOKENS",
    "DEFAULT_HISTORY_KEEP_RECENT_GROUPS",
    "DEFAULT_CLEARED_TOOL_CONTENT",
    "DEFAULT_TOOL_CLEAR_KEEP_RECENT",
    "DEFAULT_COMPACTABLE_TOOLS",
    "DEFAULT_EAGER_KEEP_RECENT",
    "EAGER_STUB_PREFIX",
    "DEFAULT_OUTPUT_RESERVE_TOKENS",
    "DEFAULT_COMPACT_BUFFER_TOKENS",
    "HISTORY_TARGET_RATIO",
    "COMPACTED_MARKER_PREFIX",
    "PIN_FLOOR",
    "SummarizerPort",
    "approx_messages_tokens",
    "forced_shape",
    "history_trigger_target",
    "ShaperPipeline",
    "PerToolResultBudgetShaper",
    "EagerToolOutputClearShaper",
    "ToolOutputClearShaper",
    "OldHistorySnipShaper",
    "AutoCompactShaper",
]
