"""Exercise translation checks against real PR-shaped Git histories."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.packaging.git_support import git
from tests.support.paths import REPO_ROOT

_SCRIPT = REPO_ROOT / "scripts/check_doc_translations.py"
_PAIRS = [
    ("README.md", "README.zh-CN.md"),
    ("docs/duo.md", "docs/duo/README.zh-CN.md"),
    ("examples/mini-edict/README.md", "examples/mini-edict/README.zh-CN.md"),
    ("examples/arcbench-evolution-r6/README.md", "examples/arcbench-evolution-r6/README.zh-CN.md"),
    ("docs/new-guide.md", "docs/new-guide.zh-CN.md"),
]

def _commit(repository: Path, changes: dict[str, str | None]) -> str:
    for name, content in changes.items():
        path = repository / name
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    git(repository, "add", "-A")
    git(repository, "commit", "-m", "\u66f4\u65b0\u6d4b\u8bd5\u6587\u6863")
    return git(repository, "rev-parse", "HEAD")

@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Translation Test")
    git(tmp_path, "config", "user.email", "translation@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    files = {path: "Original\n" for pair in _PAIRS for path in pair}
    files["docs/english-only.md"] = "English only\n"
    return tmp_path, _commit(tmp_path, files)

def _run(repository: Path, base: str, head: str = "HEAD") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), base, head],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

@pytest.mark.parametrize(("english", "chinese"), _PAIRS)
def test_english_change_requires_its_existing_translation(repository, english, chinese):
    root, base = repository
    _commit(root, {english: "Updated\n"})

    result = _run(root, base)

    assert result.returncode == 1
    assert english in result.stderr
    assert chinese in result.stderr

def test_bilingual_updates_can_use_separate_commits_in_one_pr(repository):
    root, base = repository
    _commit(root, {en: "Updated\n" for en, _ in _PAIRS})
    _commit(root, {zh: "\u5df2\u66f4\u65b0\n" for _, zh in _PAIRS})

    assert _run(root, base).returncode == 0

@pytest.mark.parametrize(
    "changes",
    [
        {"README.zh-CN.md": "\u4e2d\u6587\u7ea0\u9519\n"},
        {"docs/english-only.md": "Updated\n"},
        {"docs/fresh.md": "New English-only guide\n"},
        {"source.py": "pass\n"},
    ],
)
def test_unpaired_and_chinese_only_changes_pass(repository, changes):
    root, base = repository
    _commit(root, changes)

    assert _run(root, base).returncode == 0

def test_new_translation_pair_passes(repository):
    root, base = repository
    _commit(root, {"docs/fresh.md": "New guide\n", "docs/fresh.zh-CN.md": "\u65b0\u6307\u5357\n"})

    assert _run(root, base).returncode == 0

def test_changing_a_different_translation_does_not_satisfy_the_pair(repository):
    root, base = repository
    _commit(root, {"README.md": "Updated\n", "docs/duo/README.zh-CN.md": "\u5df2\u66f4\u65b0\n"})

    result = _run(root, base)

    assert result.returncode == 1
    assert "README.zh-CN.md" in result.stderr

def test_reverted_translation_change_does_not_count(repository):
    root, base = repository
    _commit(root, {"README.md": "Updated\n", "README.zh-CN.md": "\u5df2\u66f4\u65b0\n"})
    _commit(root, {"README.zh-CN.md": "Original\n"})

    assert _run(root, base).returncode == 1

def test_target_branch_changes_do_not_count_as_pr_changes(repository):
    root, base = repository
    git(root, "checkout", "-b", "feature")
    head = _commit(root, {"README.md": "Updated\n"})
    git(root, "checkout", "-b", "target", base)
    target = _commit(
        root,
        {
            "README.zh-CN.md": "\u76ee\u6807\u5206\u652f\u7684\u5176\u4ed6\u4fee\u6539\n",
            "docs/duo.md": "Other change\n",
        },
    )

    result = _run(root, target, head)

    assert result.returncode == 1
    assert "README.zh-CN.md" in result.stderr
    assert "docs/duo" not in result.stderr

@pytest.mark.parametrize("remove_chinese", [False, True])
def test_removing_english_also_requires_a_translation_change(repository, remove_chinese):
    root, base = repository
    changes = {"README.md": None}
    if remove_chinese:
        changes["README.zh-CN.md"] = None
    _commit(root, changes)

    assert _run(root, base).returncode == (0 if remove_chinese else 1)

def test_renaming_both_documents_passes(repository):
    root, base = repository
    _commit(
        root,
        {
            "README.md": None,
            "README.zh-CN.md": None,
            "overview.md": "Original\n",
            "overview.zh-CN.md": "Original\n",
        },
    )

    assert _run(root, base).returncode == 0

def test_missing_history_fails_with_an_actionable_error(repository):
    root, _ = repository

    result = _run(root, "missing-branch")

    assert result.returncode == 2
    assert "Fetch both revisions" in result.stderr
