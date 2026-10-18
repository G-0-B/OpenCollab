"""Explicit content matching preserves strict default editing behavior."""

import pytest

from opencollab.adapters.tools.apply_patch import ApplyPatchTool
from tests.tools.test_edit_tool import _runtime, run


def _apply(ws, params):
    return run(ApplyPatchTool().execute_with_runtime(params, _runtime(ws)))

def _file(tmp_path, text, name="f.py"):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    target = ws / name
    target.write_text(text, encoding="utf-8")
    return ws, target

def test_a_hunk_header_with_no_numbers_still_applies(tmp_path):
    ws, target = _file(tmp_path, "a\nb\nc\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "unified_diff",
        "normalize_hunks": True,
        "patch": "@@\n a\n-b\n+x\n",
    })

    assert result.startswith("Applied unified_diff")
    assert target.read_text(encoding="utf-8") == "a\nx\nc\n"

def test_numberless_hunks_with_repeated_context_are_refused(tmp_path):
    # Repeated context does not identify one position without coordinates.
    ws, target = _file(tmp_path, "b\nmiddle\nb\ntail\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "unified_diff",
        "normalize_hunks": True,
        "patch": "@@\n-b\n+first\n@@\n-b\n+second\n",
    })

    assert "matched 2 ranges" in result
    assert target.read_text(encoding="utf-8") == "b\nmiddle\nb\ntail\n"

def test_a_hunk_whose_context_is_not_in_the_file_is_still_refused(tmp_path):
    ws, target = _file(tmp_path, "a\nb\nc\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "unified_diff",
        "normalize_hunks": True,
        "patch": "@@ -1,2 +1,2 @@\n zzz\n-nope\n+x\n",
    })

    assert "did not match the file" in result
    assert target.read_text(encoding="utf-8") == "a\nb\nc\n"

def test_a_header_with_an_empty_body_is_still_refused(tmp_path):
    ws, target = _file(tmp_path, "a\nb\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "unified_diff",
        "normalize_hunks": True,
        "patch": "@@\n@@\n a\n-b\n+x\n",
    })

    assert "has no body" in result
    assert target.read_text(encoding="utf-8") == "a\nb\n"

def test_line_replace_lands_on_a_uniquely_matching_expected_str(tmp_path):
    ws, target = _file(tmp_path, "one\ntwo\nthree\nfour\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "line_replace",
        "relocate_expected": True,
        "start_line": 1,
        "end_line": 1,
        "expected_str": "three",
        "new_str": "THREE",
    })

    assert result.startswith("Applied line_replace")
    assert target.read_text(encoding="utf-8") == "one\ntwo\nTHREE\nfour\n"

def test_a_relocated_line_replace_says_where_it_actually_landed(tmp_path):
    # Silence here would leave the caller planning its next edit against line
    # numbers that were already wrong.
    ws, _ = _file(tmp_path, "one\ntwo\nthree\nfour\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "line_replace",
        "relocate_expected": True,
        "start_line": 1,
        "end_line": 1,
        "expected_str": "three",
        "new_str": "THREE",
    })

    assert "did not match lines 1-1" in result
    assert "matched lines 3-3 uniquely" in result

def test_line_replace_refuses_an_expected_str_that_matches_twice(tmp_path):
    ws, target = _file(tmp_path, "dup\nmiddle\ndup\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "line_replace",
        "relocate_expected": True,
        "start_line": 2,
        "end_line": 2,
        "expected_str": "dup",
        "new_str": "X",
    })

    assert "appears 2 times" in result
    assert target.read_text(encoding="utf-8") == "dup\nmiddle\ndup\n"

def test_line_replace_refuses_an_expected_str_that_is_not_in_the_file(tmp_path):
    ws, target = _file(tmp_path, "a\nb\n")

    result = _apply(ws, {
        "path": "f.py",
        "mode": "line_replace",
        "relocate_expected": True,
        "start_line": 1,
        "end_line": 1,
        "expected_str": "absent",
        "new_str": "X",
    })

    assert "does not appear anywhere in the file" in result
    assert target.read_text(encoding="utf-8") == "a\nb\n"

def test_unified_diff_count_normalization_is_explicit(tmp_path):
    ws, target = _file(tmp_path, "a\nb\nc\n")
    result = _apply(ws, {
        "path": "f.py", "mode": "unified_diff", "normalize_hunks": True,
        "patch": "@@ -1,3 +1,3 @@\n a\n-b\n+x\n",
    })
    assert result.startswith("Applied unified_diff")
    assert target.read_text(encoding="utf-8") == "a\nx\nc\n"

def test_relocation_counts_a_terminated_line_once_and_keeps_following_blank(tmp_path):
    ws, target = _file(tmp_path, "zero\na\n\ntail\n")
    result = _apply(ws, {
        "path": "f.py", "mode": "line_replace", "relocate_expected": True,
        "start_line": 1, "end_line": 1, "expected_str": "a\n", "new_str": "B",
    })
    assert result.startswith("Applied line_replace")
    assert "matched lines 2-2 uniquely" in result
    assert target.read_text() == "zero\nB\n\ntail\n"

def test_relocation_preserves_an_explicit_blank_line_in_the_requested_range(tmp_path):
    ws, target = _file(tmp_path, "zero\na\n\ntail\n")
    result = _apply(ws, {
        "path": "f.py", "mode": "line_replace", "relocate_expected": True,
        "start_line": 1, "end_line": 2, "expected_str": "a\n", "new_str": "B",
    })
    assert result.startswith("Applied line_replace")
    assert "matched lines 2-3 uniquely" in result
    assert target.read_text() == "zero\nB\ntail\n"

def test_relocation_keeps_a_blank_line_quoted_with_its_terminator(tmp_path):
    ws, target = _file(tmp_path, "zero\na\n\ntail\n")
    result = _apply(ws, {
        "path": "f.py", "mode": "line_replace", "relocate_expected": True,
        "start_line": 1, "end_line": 2, "expected_str": "a\n\n", "new_str": "B",
    })
    assert result.startswith("Applied line_replace")
    assert target.read_text() == "zero\nB\ntail\n"

def test_relocation_still_rejects_two_distinct_terminated_matches(tmp_path):
    source = "zero\na\n\na\ntail\n"
    ws, target = _file(tmp_path, source)
    result = _apply(ws, {
        "path": "f.py", "mode": "line_replace", "relocate_expected": True,
        "start_line": 1, "end_line": 1, "expected_str": "a\n", "new_str": "B",
    })
    assert "appears 2 times" in result
    assert target.read_text() == source

def test_relocation_with_a_terminator_keeps_crlf_bytes(tmp_path):
    ws, target = _file(tmp_path, "zero\na\n\ntail\n")
    target.write_bytes(b"zero\r\na\r\n\r\ntail\r\n")
    result = _apply(ws, {
        "path": "f.py", "mode": "line_replace", "relocate_expected": True,
        "start_line": 1, "end_line": 1, "expected_str": "a\r\n", "new_str": "B",
    })
    assert result.startswith("Applied line_replace")
    assert target.read_bytes() == b"zero\r\nB\r\n\r\ntail\r\n"

@pytest.mark.parametrize("target_text", ["value\n", "value", "value\n\n"])
def test_empty_file_unified_diff_matches_git_eof_bytes(tmp_path, target_text):
    import subprocess

    ws, target = _file(tmp_path, "")
    for args in [
        ["init", "-q"], ["config", "user.name", "Test User"],
        ["config", "user.email", "tests@example.invalid"], ["add", "f.py"],
        ["-c", "commit.gpgsign=false", "commit", "-qm", "empty file"],
    ]:
        subprocess.run(["git", *args], cwd=ws, check=True, capture_output=True)
    target.write_text(target_text)
    patch = subprocess.run(
        ["git", "diff", "--", "f.py"], cwd=ws, check=True, capture_output=True, text=True,
    ).stdout
    target.write_text("")
    subprocess.run(["git", "apply", "-"], cwd=ws, input=patch, text=True, check=True, capture_output=True)
    assert target.read_bytes() == target_text.encode()
    target.write_text("")
    result = _apply(ws, {"path": "f.py", "mode": "unified_diff", "patch": patch})
    assert result.startswith("Applied unified_diff")
    assert target.read_bytes() == target_text.encode()
