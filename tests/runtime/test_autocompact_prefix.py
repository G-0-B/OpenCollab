"""Stable summary prefixes survive ordinary history growth."""

from __future__ import annotations

import copy

import pytest

from opencollab.application.shaping import AutoCompactShaper, ShaperPipeline, forced_shape


def _text(content, role="assistant"):
    return {"role": role, "content": content}

def _chars(messages):
    return sum(len(message.get("content", "")) for message in messages)

def _history():
    return [
        _text("system", "system"),
        _text("repair the project", "user"),
        _text("reference A " * 150),
        _text("reference B " * 150),
        _text("inspect recent code"),
        _text("retain the public behavior", "user"),
    ]

class _Summarizer:
    cache_key = "model-a:prompt-v1"
    last_call_cacheable = True

    def __init__(self):
        self.segments = []
        self.routes = []
        self.summary = "Preserve the project requirements and inspect the next file."

    def _record(self, segment, route):
        self.segments.append(copy.deepcopy(segment))
        self.routes.append(route)
        return self.summary

    def __call__(self, segment):
        return self._record(segment, "sync")

    async def asummarize(self, segment):
        return self._record(segment, "async")

def _shaper(summarizer, **kwargs):
    return AutoCompactShaper(
        summarizer=summarizer,
        estimate_tokens=_chars,
        trigger_tokens=1_500,
        target_tokens=800,
        keep_recent_groups=2,
        **kwargs,
    )

@pytest.fixture(params=["sync", "async"])
def shape(request):
    async def run(shaper, messages, *, force=False):
        if request.param == "async":
            return await ShaperPipeline((shaper,)).ashape(messages, force=force)
        return forced_shape(shaper, messages) if force else shaper.shape(messages)

    run.route = request.param
    return run

async def test_successful_prefix_keeps_short_appends_verbatim_without_new_summary(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    history = _history()
    first = await shape(shaper, history)
    assert _chars(first) < shaper.target_tokens

    for step in range(5):
        history.append(_text(f"inspect the next relevant project location {step}"))
        snapshot = copy.deepcopy(history)
        output = await shape(shaper, history)
        assert output == [*first[:2], *history[4:]]
        assert history == snapshot
        assert await shape(shaper, copy.deepcopy(history)) == output

    assert summarizer.routes == [shape.route]
    assert summarizer.segments == [_history()[1:4]]

@pytest.mark.parametrize("change", ["content", "metadata", "namespace", "summarizer"])
async def test_changed_original_prefix_or_model_cannot_reuse_old_summary(shape, change):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    history = _history()
    await shape(shaper, history)
    history.append(_text("continue"))
    if change == "content":
        history[2]["content"] += "corrected requirement"
    elif change == "metadata":
        history[2]["provider_state"] = {"version": "changed"}
    elif change == "namespace":
        summarizer.cache_key = "model-b:prompt-v1"
    else:
        shaper.summarizer = _Summarizer()

    await shape(shaper, history)

    active = shaper.summarizer
    assert active.segments[-1] == history[1:-2]
    assert len(summarizer.segments) == (1 if change == "summarizer" else 2)

async def test_pressure_after_prefix_reuse_summarizes_full_originals(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    history = _history()
    first = await shape(shaper, history)
    history.extend([_text("new reference " * 180), _text("recent one"), _text("recent two")])
    snapshot = copy.deepcopy(history)

    output = await shape(shaper, history)

    assert _chars([*first[:2], *history[4:]]) > shaper.trigger_tokens
    assert len(summarizer.segments) == 2
    assert summarizer.segments[-1] == history[1:-2]
    assert all(not message.get("compacted") for message in summarizer.segments[-1])
    assert _chars(output) < shaper.target_tokens
    assert history == snapshot
    history.append(_text("continue with the next small step"))
    reused = await shape(shaper, history)
    assert reused == [*output[:2], *history[len(snapshot) - 2:]]
    assert len(summarizer.segments) == 2

async def test_forced_compaction_refreshes_growing_region_from_originals(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    history = _history()
    first = await shape(shaper, history)
    history.append(_text("continue"))
    assert _chars([*first[:2], *history[4:]]) < shaper.trigger_tokens

    await shape(shaper, history, force=True)

    assert len(summarizer.segments) == 2
    assert summarizer.segments[-1] == history[1:-2]
    assert shaper._forced is False

async def test_new_pin_and_pending_tool_group_survive_prefix_reuse(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    history = _history()
    first = await shape(shaper, history)
    pin = {**_text("retain the new task instruction", "user"), "_ctx": {"priority": 70}}
    pending = {
        "role": "assistant",
        "tool_calls": [{"id": "pending", "function": {"name": "file_read", "arguments": "{}"}}],
    }
    history.extend([pin, pending, _text("recent one"), _text("recent two")])
    snapshot = copy.deepcopy(history)

    output = await shape(shaper, history)

    assert output == [*first[:2], *history[4:]]
    assert pin in output and pending in output
    assert len(summarizer.segments) == 1
    assert history == snapshot

async def test_new_pin_inside_cached_prefix_stays_verbatim(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    history = _history()
    first = await shape(shaper, history)
    history[2]["_ctx"] = {"priority": 70}
    history.append(_text("continue"))

    output = await shape(shaper, history)

    assert history[2] in output
    assert first[1] not in output
    assert all(history[2] not in segment for segment in summarizer.segments[1:])

async def test_expanded_recent_window_cannot_hide_messages_with_cached_prefix(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    history = _history()
    first = await shape(shaper, history)
    shaper.keep_recent_groups = 4

    output = await shape(shaper, history)

    assert output[-4:] == history[-4:]
    assert first[1] not in output

async def test_changed_tool_group_cannot_be_partly_replaced_by_cached_prefix(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    call = {
        "role": "assistant",
        "tool_calls": [{"id": "read", "function": {"name": "file_read", "arguments": "{}"}}],
    }
    result = {"role": "tool", "tool_call_id": "read", "content": "reference " * 300}
    history = [_text("system", "system"), call, result, _text("recent one"), _text("recent two")]
    first = await shape(shaper, history)
    call["tool_calls"].append({"id": "pending", "function": {"name": "file_read", "arguments": "{}"}})
    history.append(_text("continue"))

    output = await shape(shaper, history)

    assert output is history
    assert first[1] not in output
    assert all(call not in segment and result not in segment for segment in summarizer.segments[1:])

async def test_complete_tool_group_prefix_reuses_without_orphaning_results(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer)
    call = {
        "role": "assistant",
        "tool_calls": [{"id": "read", "function": {"name": "file_read", "arguments": "{}"}}],
    }
    result = {"role": "tool", "tool_call_id": "read", "content": "reference " * 300}
    history = [_text("system", "system"), call, result, _text("recent one"), _text("recent two")]
    first = await shape(shaper, history)
    history.append(_text("continue"))

    output = await shape(shaper, history)

    assert output == [*first[:2], *history[3:]]
    assert len(summarizer.segments) == 1
    assert all(message.get("role") != "tool" for message in output)

async def test_reusing_prefix_refreshes_its_position_in_bounded_lru(shape):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer, summary_cache_size=2)
    first_history = _history()
    second_history = _history()
    second_history[1]["content"] = "second project"
    third_history = _history()
    third_history[1]["content"] = "third project"
    first = await shape(shaper, first_history)
    await shape(shaper, second_history)
    first_history.append(_text("continue first project"))
    await shape(shaper, first_history)
    await shape(shaper, third_history)
    first_history.append(_text("continue first project again"))

    output = await shape(shaper, first_history)

    assert output == [*first[:2], *first_history[4:]]
    assert len(summarizer.segments) == 3
    assert len(shaper._summary_cache) == 2

@pytest.mark.parametrize("cache_size", [0, 1])
async def test_disabled_or_evicted_cache_summarizes_current_originals(shape, cache_size):
    summarizer = _Summarizer()
    shaper = _shaper(summarizer, summary_cache_size=cache_size)
    history = _history()
    await shape(shaper, history)
    if cache_size:
        unrelated = copy.deepcopy(history)
        unrelated[1]["content"] = "different project"
        await shape(shaper, unrelated)
    history.append(_text("continue"))

    await shape(shaper, history)

    assert len(summarizer.segments) == 2 + cache_size
    assert summarizer.segments[-1] == history[1:-2]
    assert len(shaper._summary_cache) <= cache_size

async def test_fallback_cannot_supply_a_reusable_prefix(shape):
    summarizer = _Summarizer()
    summarizer.last_call_cacheable = False
    shaper = _shaper(summarizer)
    history = _history()
    assert await shape(shaper, history) is history
    summarizer.last_call_cacheable = True
    history.append(_text("continue"))

    output = await shape(shaper, history)

    assert len(summarizer.segments) == 2
    assert summarizer.segments[-1] == history[1:-2]
    assert output is not history

async def test_nonreducing_cached_summary_remains_rejected_after_append(shape):
    summarizer = _Summarizer()
    summarizer.summary = "oversized summary " * 400
    shaper = _shaper(summarizer)
    history = _history()
    assert await shape(shaper, history) is history
    history.append(_text("continue"))

    assert await shape(shaper, history) is history
    assert len(summarizer.segments) == 2
