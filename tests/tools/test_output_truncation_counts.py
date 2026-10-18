"""The omission count includes characters displaced by the marker itself."""

import re

import pytest

from opencollab.adapters.tools._output import truncate


@pytest.mark.parametrize("label", [None, "stdout", "Unicode \u8f93\u51fa"])
@pytest.mark.parametrize(("size", "cap"), [(1000, 64), (1100, 150), (10100, 1000), (10000, 9980)])
def test_truncation_count_matches_source_characters_actually_removed(label, size, cap):
    text = "α" * size

    result = truncate(text, cap, label)

    match = re.search(r"\n\n\.\.\. \[(\d+) chars(?: of .*?)? truncated\] \.\.\.\n\n", result)
    assert match is not None
    kept = result[:match.start()] + result[match.end():]
    assert int(match.group(1)) == len(text) - len(kept)
    assert len(result) <= cap

def test_truncation_leaves_exactly_fitting_output_unchanged():
    assert truncate("α" * 50, 50) == "α" * 50
