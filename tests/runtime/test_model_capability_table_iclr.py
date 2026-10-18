"""Model-specific tool support and capability defaults."""

from __future__ import annotations

import dataclasses

from opencollab.adapters.llm.types import ModelCapabilities, model_capabilities


def test_qwen_flash_refuses_forced_tool_choice():
    capabilities = model_capabilities("qwen3.8-flash")

    assert capabilities.supports_forced_tool_choice is False

def test_qwen_flash_leaves_every_unprobed_dimension_at_its_default():
    """Four dimensions were never probed, so the entry must not assert them.

    Copying them from the neighbouring ``deepseek-v4-flash`` row would read as
    measurement and be invention; this fails if anyone does.
    """
    capabilities = model_capabilities("qwen3.8-flash")
    default = ModelCapabilities()

    for field in (
        "supports_responses_json_schema",
        "honors_workflow_thinking_override",
        "supports_responses_reasoning",
        "supports_responses_tools",
    ):
        assert getattr(capabilities, field) == getattr(default, field), field

def test_adding_qwen_leaves_deepseek_byte_for_byte_unchanged():
    """The recorded DeepSeek settings and its Chat replay protocol stay explicit.

    Every field is spelled out rather than compared to the live table, so the
    test fails if the row is edited as well as if the lookup is.
    """
    assert dataclasses.asdict(model_capabilities("deepseek-v4-flash")) == {
        "context_window": 1_048_576,
        "supports_forced_tool_choice": False,
        "supports_anthropic_thinking_forced_tool_choice": True,
        "supports_responses_json_schema": True,
        "honors_workflow_thinking_override": True,
        "supports_responses_streaming": True,
        "supports_responses_sampling": True,
        "supports_responses_reasoning": True,
        "supports_responses_tools": True,
        "requires_chat_reasoning_content": True,
        "chat_thinking_protocol": "thinking.type",
        "requires_anthropic_thinking_budget": False,
        "supports_anthropic_thinking_top_p": True,
    }
