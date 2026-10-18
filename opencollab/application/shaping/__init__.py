"""Read-time context shaping that preserves the stored conversation.

The default wiring in bootstrap.container caps each tool result, then clears,
snips, or summarizes eligible history only under context pressure. Each layer
sees the prior layer's output, avoiding costlier work after pressure eases.
EagerToolOutputClearShaper is exported for explicit composition; it is not a
default rung. Pipeline helpers preserve pinned sources and call/result pairing.
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
