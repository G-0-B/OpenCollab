"""Deterministic, monotonic, and idempotent eager tool-result clearing."""

from __future__ import annotations

import copy

import pytest

from opencollab.application.shaping import (
    EAGER_STUB_PREFIX,
    EagerToolOutputClearShaper,
)
from opencollab.application.shaping.pipeline import PIN_FLOOR
from tests.support.shaping_test_support import _orphaned_tool_ids


def _sys():
    return {"role": "system", "content": "s"}

def _call(tid, name="bash", arguments="{}"):
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": tid, "function": {"name": name, "arguments": arguments}}
        ],
    }

def _tool(tid, content):
    return {"role": "tool", "tool_call_id": tid, "content": content}

def _exchange(tid, name="bash", arguments="{}", body="x" * 500):
    return [_call(tid, name=name, arguments=arguments), _tool(tid, body)]

def _history(n, keep_names=None):
    """A system seed plus ``n`` compactable bash tool exchanges t1..tn."""
    out = [_sys()]
    for i in range(1, n + 1):
        out += _exchange(f"t{i}")
    return out

def _shaper(**kw):
    return EagerToolOutputClearShaper(keep_recent=3, **kw)

def _tool_content(messages, tid):
    return next(m["content"] for m in messages if m.get("tool_call_id") == tid)

def _is_stub(content):
    return content.startswith(EAGER_STUB_PREFIX)

def test_runs_with_no_trigger_clears_old_without_expanding_short_results():
    # Replacing one-byte results with the stub would grow the history.
    messages = [_sys()]
    for i in range(1, 6):
        messages += _exchange(f"t{i}", body="s")
    out = _shaper().shape(messages)
    assert _tool_content(out, "t1") == "s"
    assert _tool_content(out, "t2") == "s"
    assert _tool_content(out, "t3") == "s"
    assert _tool_content(out, "t5") == "s"

def test_reused_call_id_only_clears_the_old_occurrence():
    messages = [
        _sys(),
        *_exchange("call_1", body="old" * 300),
        *_exchange("call_1", body="new result"),
    ]
    out = EagerToolOutputClearShaper(keep_recent=1).shape(messages)
    results = [message["content"] for message in out if message.get("role") == "tool"]
    assert _is_stub(results[0])
    assert results[1] == "new result"

def test_keeps_recent_k_verbatim_stubs_older():
    out = _shaper().shape(_history(7))
    stubbed = {
        m["tool_call_id"]
        for m in out
        if m.get("role") == "tool" and _is_stub(m["content"])
    }
    verbatim = {
        m["tool_call_id"]
        for m in out
        if m.get("role") == "tool" and not _is_stub(m["content"])
    }
    assert stubbed == {"t1", "t2", "t3", "t4"}
    assert verbatim == {"t5", "t6", "t7"}

def test_noop_when_within_keep_recent():
    msgs = _history(3)
    assert _shaper().shape(msgs) is msgs

@pytest.mark.parametrize("keep_recent", [0, -1, True, 1.5])
def test_rejects_invalid_keep_recent(keep_recent):
    with pytest.raises(ValueError, match="keep_recent"):
        EagerToolOutputClearShaper(keep_recent=keep_recent)

def test_deterministic_byte_identical_across_calls():
    messages = _history(8)
    first = _shaper().shape(messages)
    second = _shaper().shape(copy.deepcopy(messages))
    assert first == second
    assert _tool_content(first, "t1") == _tool_content(second, "t1")

def test_clear_decision_never_expands_an_aged_result():
    # Only results larger than the deterministic stub are replaced.
    small = [_sys(), *_exchange("a", body="x"), *_exchange("b", body="x"),
             *_exchange("c", body="x"), *_exchange("d", body="x")]
    big = [_sys(), *_exchange("a", body="x" * 9000),
           *_exchange("b", body="x" * 9000), *_exchange("c", body="x" * 9000),
           *_exchange("d", body="x" * 9000)]
    sh = EagerToolOutputClearShaper(keep_recent=3)
    assert _tool_content(sh.shape(small), "a") == "x"
    assert _is_stub(_tool_content(sh.shape(big), "a"))

def test_stub_names_tool_and_target():
    messages = [
        _sys(),
        *_exchange("r", name="file_read",
                   arguments='{"path": "src/foo.py"}'),
        *_exchange("a"), *_exchange("b"), *_exchange("c"),
    ]
    out = EagerToolOutputClearShaper(keep_recent=3).shape(messages)
    stub = _tool_content(out, "r")
    assert "file_read" in stub and "src/foo.py" in stub

def test_default_keep_recent_retains_more_than_legacy_five():
    # The default retains the working set of a multi-file task to avoid repeated reads.
    out = EagerToolOutputClearShaper().shape(_history(13))
    stubbed = {
        m["tool_call_id"]
        for m in out
        if m.get("role") == "tool" and _is_stub(m["content"])
    }
    assert stubbed == {"t1"}
    assert not _is_stub(_tool_content(out, "t6"))

def test_stub_includes_read_range_so_pages_get_distinct_stubs():
    # Paged reads need distinct line-range stubs to avoid repeated reads of the same slice.
    messages = [
        _sys(),
        *_exchange("p1", name="file_read",
                   arguments='{"path": "a.py", "offset": 1, "limit": 100}'),
        *_exchange("p2", name="file_read",
                   arguments='{"path": "a.py", "offset": 101, "limit": 100}'),
        *_exchange("x"), *_exchange("y"), *_exchange("z"),
    ]
    out = EagerToolOutputClearShaper(keep_recent=3).shape(messages)
    s1 = _tool_content(out, "p1")
    s2 = _tool_content(out, "p2")
    assert "a.py lines 1-100" in s1
    assert "a.py lines 101-200" in s2
    assert s1 != s2
    assert "already ran this" in s1

def test_monotonic_one_more_exchange_flips_exactly_one_and_keeps_old_stubs():
    sh = EagerToolOutputClearShaper(keep_recent=3)
    turn_n = sh.shape(_history(6))
    turn_n1 = sh.shape(_history(7))

    assert not _is_stub(_tool_content(turn_n, "t4"))
    assert _is_stub(_tool_content(turn_n1, "t4"))

    for tid in ("t1", "t2", "t3"):
        assert _tool_content(turn_n, tid) == _tool_content(turn_n1, tid)
        assert _is_stub(_tool_content(turn_n1, tid))

def test_monotonic_never_unstubs():
    sh = EagerToolOutputClearShaper(keep_recent=3)
    once = sh.shape(_history(8))
    again = sh.shape(once)
    for tid in ("t1", "t2", "t3", "t4", "t5"):
        assert _is_stub(_tool_content(again, tid))

def test_idempotent_shape_of_shape_equals_shape():
    messages = _history(9)
    once = _shaper().shape(messages)
    twice = _shaper().shape(once)
    assert twice == once
    assert twice is once

def test_non_compactable_tools_untouched():
    messages = [
        _sys(),
        *_exchange("p1", name="apply_patch", body="p" * 500),
        *_exchange("p2", name="apply_patch", body="p" * 500),
        *_exchange("b1"), *_exchange("b2"), *_exchange("b3"), *_exchange("b4"),
    ]
    out = EagerToolOutputClearShaper(keep_recent=1).shape(messages)
    assert _tool_content(out, "p1") == "p" * 500
    assert _tool_content(out, "p2") == "p" * 500

def test_pinned_sources_never_touched():
    pinned_call = {
        **_call("pin", name="bash"),
        "_ctx": {"priority": PIN_FLOOR + 5},
    }
    pinned_result = {**_tool("pin", "secret" * 100), "_ctx": {"priority": PIN_FLOOR + 5}}
    messages = [
        _sys(),
        pinned_call, pinned_result,
        *_exchange("a"), *_exchange("b"), *_exchange("c"), *_exchange("d"),
    ]
    out = EagerToolOutputClearShaper(keep_recent=3).shape(messages)
    assert _tool_content(out, "pin") == "secret" * 100

def test_preserves_tool_call_pairing_no_orphans():
    out = _shaper().shape(_history(8))
    assert _orphaned_tool_ids(out) == set()
    assert sum(1 for m in out if m.get("role") == "tool") == 8

def test_input_messages_not_mutated():
    messages = _history(8)
    snapshot = copy.deepcopy(messages)
    out = _shaper().shape(messages)
    assert out is not messages
    assert messages == snapshot
