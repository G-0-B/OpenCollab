"""Model-specific tool support and capability defaults."""

from __future__ import annotations

import dataclasses

from opencollab.adapters.llm.types import ModelCapabilities, model_capabilities


def test_qwen_flash_refuses_forced_tool_choice():
    capabilities = model_capabilities("qwen3.8-flash")

    assert capabilities.supports_forced_tool_choice is False

def test_qwen_flash_leaves_every_unprobed_dimension_at_its_default():
    """Unprobed dimensions must retain defaults instead of inheriting a neighbouring row."""
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
    """Explicit field values detect changes to the stored row as well as its lookup."""
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
