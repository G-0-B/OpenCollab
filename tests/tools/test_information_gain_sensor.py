"""Observe novelty and low-yield tool results without changing session control flow."""

from __future__ import annotations

import asyncio

from opencollab.application.tool_execution import (
    _intrinsic_low_yield,
)
from opencollab.domain.session import MAX_SCOUT_LEDGER_CARDS, SessionState
from tests.support.tool_execution_test_support import build_sensor_use_case as _use_case


def run(coro):
    return asyncio.run(coro)

class ScriptedTool:
    """Return queued outputs for repeated paths and record invocation arguments."""

    def __init__(self, name, outputs):
        self.name = name
        self._outputs = list(outputs)
        self.calls = []

    async def execute_with_runtime(self, args, runtime):
        self.calls.append(args)
        return self._outputs.pop(0) if self._outputs else ""

def _call(name, args, cid="c1"):
    return {"id": cid, "function": {"name": name, "arguments": args}}

def _run_one(state, tool, name, args):
    """Process a single tool call and fold the result into ``state``."""
    result = run(_use_case(state, tool).process([_call(name, args)]))
    result.apply_to(state)
    return result

def test_intrinsic_low_yield_flags_no_match_empty_read_and_zero_lines():
    assert _intrinsic_low_yield("grep", "No matches found for pattern: zzz") is True
    assert _intrinsic_low_yield("file_read", "") is True
    assert _intrinsic_low_yield("file_read", "   \n  ") is True
    assert _intrinsic_low_yield("file_read", "File: e.py (0 lines total, showing 1-0)") is True
    assert _intrinsic_low_yield("file_read", "File: a.py (3 lines)\n1\tdef f(): ...") is False
    assert _intrinsic_low_yield("grep", "a.py:1: def f(): ...") is False
    assert _intrinsic_low_yield("bash", "") is False

def test_t1_low_yield_increments_and_novel_resets_and_counts():
    state = SessionState(messages=[])

    _run_one(state, ScriptedTool("grep", ["fs.py:42: end = start + n"]), "grep", '{"pattern":"end"}')
    assert state.turn.distinct_evidence_count == 1
    assert state.turn.low_yield_since_progress == 0

    _run_one(state, ScriptedTool("grep", ["fs.py:42: end = start + n"]), "grep", '{"pattern":"start"}')
    assert state.turn.distinct_evidence_count == 1
    assert state.turn.low_yield_since_progress == 1

    _run_one(state, ScriptedTool("file_read", [""]), "file_read", '{"path":"empty.py"}')
    assert state.turn.low_yield_since_progress == 2

    _run_one(state, ScriptedTool("grep", ["No matches found for pattern: zzz"]), "grep", '{"pattern":"zzz"}')
    assert state.turn.low_yield_since_progress == 3

    _run_one(
        state,
        ScriptedTool("file_read", ["File: b.py (2 lines total)\n1\tdef f(): pass"]),
        "file_read",
        '{"path":"b.py"}',
    )
    assert state.turn.low_yield_since_progress == 0
    assert state.turn.distinct_evidence_count == 2

def test_t1_new_read_range_counts_as_new_evidence():
    state = SessionState(messages=[])
    _run_one(
        state,
        ScriptedTool("file_read", ["File: ccode.py (...)\n1\talpha"]),
        "file_read",
        '{"path":"ccode.py","offset":1}',
    )
    assert state.turn.distinct_evidence_count == 1
    assert state.turn.low_yield_since_progress == 0

    _run_one(
        state,
        ScriptedTool("file_read", ["File: ccode.py (...)\n50\tbeta"]),
        "file_read",
        '{"path":"ccode.py","offset":50}',
    )
    assert state.turn.distinct_evidence_count == 2
    assert state.turn.low_yield_since_progress == 0

    _run_one(
        state,
        ScriptedTool("file_read", ["File: ccode.py (...)\n50\tupdated beta"]),
        "file_read",
        '{"path":"ccode.py","offset":50}',
    )
    assert state.turn.distinct_evidence_count == 2
    assert state.turn.low_yield_since_progress == 1

def test_t1_within_batch_duplicate_is_low_yield():
    # Two calls in ONE batch returning identical content: the first is informative,
    # the second collides on the content hash within the same apply -> low-yield.
    state = SessionState(messages=[])
    tool = ScriptedTool("grep", ["hit X", "hit X"])
    batch = [_call("grep", '{"pattern":"a"}', cid="c1"), _call("grep", '{"pattern":"b"}', cid="c2")]
    run(_use_case(state, tool).process(batch)).apply_to(state)
    assert state.turn.distinct_evidence_count == 1
    assert state.turn.low_yield_since_progress == 1

def test_t1_counters_reset_on_a_fresh_user_turn():
    state = SessionState(messages=[])
    _run_one(state, ScriptedTool("grep", ["a hit"]), "grep", '{"pattern":"a"}')
    _run_one(state, ScriptedTool("grep", ["No matches found for pattern: z"]), "grep", '{"pattern":"z"}')
    assert state.turn.distinct_evidence_count == 1 and state.turn.low_yield_since_progress == 1

    state.reset_for_user_turn()
    assert state.turn.low_yield_since_progress == 0
    assert state.turn.distinct_evidence_count == 0
    assert state.turn.seen_result_hashes == set()

def test_evidence_ledger_retains_only_bounded_latest_cards():
    state = SessionState(messages=[])
    for index in range(MAX_SCOUT_LEDGER_CARDS + 5):
        state.record_evidence_signal(
            f"content-{index}",
            f"call-{index}",
            False,
            card={"tool": "grep", "target": str(index), "snippet": "hit"},
        )

    assert len(state.turn.scout_ledger) == MAX_SCOUT_LEDGER_CARDS
    assert state.turn.scout_ledger[0]["target"] == "5"
