"""Short histories retain their provider-cache prefix across context shaping.

Rewriting an earlier message invalidates the cached tail on subsequent calls.
Per-result truncation remains stable because it depends on each result's own size.
"""

from __future__ import annotations

import copy

from opencollab.bootstrap.container import _build_default_shaper


class _Llm:
    """Just the one thing the pipeline asks a resolved LLM for."""

    def context_window(self) -> int:
        return 200_000

class _Summarizer:
    """Held by the auto-compact rung; never reached at low context."""

    async def summarize(self, messages):  # pragma: no cover - must not run
        raise AssertionError("auto-compact fired on a small history")

# Results exceed the clearing stub size while staying below per-result and reactive compaction limits.
_RESULT_BODY = "".join(f"    value_{n} = compute(n) + {n}\n" for n in range(60))

def _small_history(exchanges: int) -> list[dict]:
    """A history of ``exchanges`` tool exchanges, all well under every bound."""
    messages: list[dict] = [{"role": "system", "content": "you are an agent"}]
    for index in range(exchanges):
        call_id = f"call-{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": f'{{"path": "pkg/mod_{index}.py"}}',
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "file_read",
                "content": f"# pkg/mod_{index}.py\n{_RESULT_BODY}",
            }
        )
    return messages

def _default_shaper():
    return _build_default_shaper(_Llm(), _Summarizer())

def test_the_default_pipeline_leaves_a_small_history_byte_identical():
    """Thirty exchanges is deep enough to age past any keep-recent window."""
    messages = _small_history(30)
    original = copy.deepcopy(messages)

    shaped = _default_shaper().shape(messages)

    assert shaped == original
    assert messages == original

def test_no_rung_reports_firing_on_a_small_history():
    """The per-rung account has to agree: nothing acted, and it says so."""
    _shaped, reports = _default_shaper().shape_with_report(_small_history(30))

    assert [report["rung"] for report in reports] == ["none"]

def test_growing_the_history_by_one_exchange_only_appends():
    """Rewriting a newly aged tool result invalidates the provider-cache prefix at that depth."""
    shaper = _default_shaper()

    before = shaper.shape(_small_history(30))
    after = shaper.shape(_small_history(31))

    assert after[: len(before)] == before
