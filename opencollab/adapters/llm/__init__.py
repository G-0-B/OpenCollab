"""LLM clients for OpenAI-compatible endpoints and native Anthropic requests.

``LLMClient`` dispatches to provider adapters using shared response types,
model metadata, token estimators, and retry handling.
"""

from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.errors import is_context_overflow_error
from opencollab.adapters.llm.types import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_CONTEXT_WINDOWS,
    LLMResponse,
    ModelCapabilities,
    Usage,
    estimate_messages_tokens,
    estimate_tokens,
    model_capabilities,
    model_context_window,
)

__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "MODEL_CONTEXT_WINDOWS",
    "LLMClient",
    "LLMResponse",
    "ModelCapabilities",
    "Usage",
    "estimate_messages_tokens",
    "estimate_tokens",
    "is_context_overflow_error",
    "model_capabilities",
    "model_context_window",
]
