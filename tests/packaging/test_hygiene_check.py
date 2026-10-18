"""Behavior tests for the added-file hygiene command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.packaging.git_support import git as _git
from tests.support.paths import REPO_ROOT

_REPO_ROOT = REPO_ROOT
_SCRIPT = _REPO_ROOT / "scripts" / "check_added_files.py"
_FIXTURE_COMMIT = "test: \u8c03\u6574\u4e34\u65f6\u6d4b\u8bd5\u6587\u4ef6"

# Cover new, grown, renamed, and complete-tree module detection as warnings.

def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Hygiene Test")
    _git(repository, "config", "user.email", "hygiene@example.invalid")
    (repository / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repository, "add", "base.txt")
    _git(repository, "commit", "-m", "base")
    return repository, _git(repository, "rev-parse", "HEAD")

def _run(
    repository: Path,
    base: str,
    head: str,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), base, head, *extra],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

def test_hygiene_rejects_oversized_file_with_newline_name(tmp_path):
    repository, base = _repository(tmp_path)
    name = "oversized\nartifact.bin"
    (repository / name).write_bytes(b"x" * 512_001)
    _git(repository, "add", name)
    _git(repository, "commit", "-m", "add oversized file")

    result = _run(repository, base, "HEAD")

    assert result.returncode == 1
    assert "oversized%0Aartifact.bin" in result.stdout
    assert "512001 bytes" in result.stdout

def test_hygiene_accepts_small_files_with_git_legal_special_names(tmp_path):
    repository, base = _repository(tmp_path)
    names = (
        "with space.txt",
        "with\ttab.txt",
        "with\nnewline.txt",
        'with"quote.txt',
        "café.txt",
        "with:colon,comma.txt",
        "literal[glob]*?.txt",
        "-leading-dash.txt",
    )
    for name in names:
        (repository / name).write_text("small\n", encoding="utf-8")
    _git(repository, "add", "--", *names)
    _git(repository, "commit", "-m", "add special names")

    result = _run(repository, base, "HEAD")

    assert result.returncode == 0
    assert "checks passed" in result.stdout

@pytest.mark.parametrize("committed_size", [10, 512_001])
@pytest.mark.parametrize("checkout", ["base", "modified", "missing", "symlink"])
def test_hygiene_measures_committed_bytes_with_a_different_worktree(
    tmp_path, committed_size, checkout
):
    repository, base = _repository(tmp_path)
    target = repository / "base.txt"
    target.write_bytes(b"x" * committed_size)
    _git(repository, "add", "base.txt")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    if checkout == "base":
        _git(repository, "checkout", "--detach", base)
    elif checkout == "modified":
        target.write_bytes(b"x" * (10 if committed_size > 512_000 else 512_001))
    else:
        target.unlink()
        if checkout == "symlink":
            target.symlink_to("missing-target")

    result = _run(repository, base, head)

    assert result.returncode == (1 if committed_size > 512_000 else 0)
    if committed_size > 512_000:
        assert "512001 bytes" in result.stdout
        assert "grown past the limit by this change" in result.stdout
    else:
        assert "checks passed" in result.stdout

@pytest.mark.parametrize("mode", ["symlink", "submodule"])
def test_hygiene_uses_committed_file_type_with_a_regular_worktree(tmp_path, mode):
    repository, base = _repository(tmp_path)
    target = repository / "artifact.bin"
    if mode == "symlink":
        target.symlink_to("base.txt")
        _git(repository, "add", "artifact.bin")
    else:
        _git(repository, "update-index", "--add", "--cacheinfo", f"160000,{base},artifact.bin")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    if mode == "symlink":
        target.unlink()
    target.write_bytes(b"x" * 512_001)

    result = _run(repository, base, head)

    assert result.returncode == 0
    assert "checks passed" in result.stdout

@pytest.mark.parametrize("mode", ["symlink", "submodule"])
@pytest.mark.parametrize("content", ["small", "oversized", "python"])
def test_hygiene_checks_a_link_replaced_by_a_regular_file(tmp_path, mode, content):
    repository, seed = _repository(tmp_path)
    name = "module.py" if content == "python" else "artifact.bin"
    target = repository / name
    if mode == "symlink":
        target.symlink_to("base.txt")
        _git(repository, "add", name)
    else:
        _git(repository, "update-index", "--add", "--cacheinfo", f"160000,{seed},{name}")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    base = _git(repository, "rev-parse", "HEAD")
    if mode == "symlink":
        target.unlink()
    payload = b"pass\n" * 801 if content == "python" else b"x" * (
        512_001 if content == "oversized" else 10
    )
    target.write_bytes(payload)
    _git(repository, "add", name)
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    assert _git(repository, "diff", "--name-status", base, head) == f"T\t{name}"

    result = _run(repository, base, head)

    assert result.returncode == (1 if content == "oversized" else 0)
    if content == "oversized":
        assert "512001 bytes" in result.stdout
    elif content == "python":
        assert "::warning" in result.stdout
        assert "801 lines" in result.stdout
    else:
        assert "checks passed" in result.stdout

@pytest.mark.parametrize("mode", ["symlink", "submodule"])
def test_hygiene_accepts_a_regular_file_replaced_by_a_link(tmp_path, mode):
    repository, seed = _repository(tmp_path)
    name = "artifact.bin"
    target = repository / name
    target.write_bytes(b"x" * 512_001)
    _git(repository, "add", name)
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    base = _git(repository, "rev-parse", "HEAD")
    target.unlink()
    if mode == "symlink":
        target.symlink_to("base.txt")
        _git(repository, "add", name)
    else:
        _git(repository, "update-index", "--cacheinfo", f"160000,{seed},{name}")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    assert _git(repository, "diff", "--name-status", base, head) == f"T\t{name}"

    result = _run(repository, base, head)

    assert result.returncode == 0
    assert "checks passed" in result.stdout

@pytest.mark.parametrize("size", [10, 512_001])
def test_hygiene_accepts_an_executable_bit_change_without_growth(tmp_path, size):
    repository, _seed = _repository(tmp_path)
    target = repository / "artifact.bin"
    target.write_bytes(b"x" * size)
    _git(repository, "add", target.name)
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    base = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-index", "--chmod=+x", target.name)
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    assert _git(repository, "diff", "--name-status", base, head) == f"M\t{target.name}"

    result = _run(repository, base, head)

    assert result.returncode == 0
    assert "checks passed" in result.stdout

def test_hygiene_measures_committed_executable_bytes(tmp_path):
    repository, base = _repository(tmp_path)
    target = repository / "artifact.bin"
    target.write_bytes(b"x" * 512_001)
    target.chmod(0o755)
    _git(repository, "add", "artifact.bin")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    target.write_bytes(b"x")

    result = _run(repository, base, head)

    assert result.returncode == 1
    assert "512001 bytes" in result.stdout

@pytest.mark.parametrize("diff_relative", ["false", "true"])
def test_hygiene_checks_the_complete_change_from_a_subdirectory(tmp_path, diff_relative):
    repository, base = _repository(tmp_path)
    nested = repository / "nested"
    nested.mkdir()
    (repository / "base.txt").write_bytes(b"x" * 512_001)
    (nested / "module.py").write_text("pass\n" * 801, encoding="utf-8")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    _git(repository, "config", "diff.relative", diff_relative)

    result = _run(nested, base, head)

    assert result.returncode == 1
    assert "file=base.txt" in result.stdout
    assert "512001 bytes" in result.stdout
    assert "file=nested/module.py" in result.stdout
    assert "801 lines" in result.stdout

def test_hygiene_warns_about_a_python_module_over_line_limit(tmp_path):
    repository, base = _repository(tmp_path)
    (repository / "large.py").write_text("pass\n" * 801, encoding="utf-8")
    _git(repository, "add", "large.py")
    _git(repository, "commit", "-m", "add large module")

    result = _run(repository, base, "HEAD")

    assert result.returncode == 0
    assert "::warning" in result.stdout
    assert "801 lines" in result.stdout

def _committed_module(repository: Path, name: str, lines: int) -> str:
    (repository / name).write_text("pass\n" * lines, encoding="utf-8")
    _git(repository, "add", name)
    _git(repository, "commit", "-m", f"write {name}")
    return _git(repository, "rev-parse", "HEAD")

@pytest.mark.parametrize("checkout", ["base", "modified", "missing"])
def test_hygiene_measures_committed_python_lines_with_a_different_worktree(tmp_path, checkout):
    repository, _base = _repository(tmp_path)
    base = _committed_module(repository, "module.py", 700)
    head = _committed_module(repository, "module.py", 801)
    if checkout == "base":
        _git(repository, "checkout", "--detach", base)
    elif checkout == "modified":
        (repository / "module.py").write_text("pass\n", encoding="utf-8")
    else:
        (repository / "module.py").unlink()

    result = _run(repository, base, head)

    assert result.returncode == 0
    assert "::warning" in result.stdout
    assert "801 lines" in result.stdout
    assert "grown past the limit by this change" in result.stdout

def test_hygiene_warns_about_a_change_that_grows_a_module_past_the_line_limit(tmp_path):
    repository, _base = _repository(tmp_path)
    base = _committed_module(repository, "module.py", 700)
    _committed_module(repository, "module.py", 801)

    result = _run(repository, base, "HEAD")

    assert result.returncode == 0
    assert "::warning" in result.stdout
    assert "801 lines" in result.stdout
    assert "grown past the limit by this change" in result.stdout

def test_hygiene_accepts_a_change_that_keeps_a_module_under_the_line_limit(tmp_path):
    repository, _base = _repository(tmp_path)
    base = _committed_module(repository, "module.py", 700)
    _committed_module(repository, "module.py", 799)

    result = _run(repository, base, "HEAD")

    assert result.returncode == 0
    assert "checks passed" in result.stdout

def test_hygiene_leaves_a_module_that_was_already_oversized_to_the_main_tree_check(tmp_path):
    repository, _base = _repository(tmp_path)
    base = _committed_module(repository, "module.py", 900)
    _committed_module(repository, "module.py", 910)

    result = _run(repository, base, "HEAD")

    assert result.returncode == 0
    assert "checks passed" in result.stdout

def test_hygiene_still_reports_an_already_oversized_module_on_the_complete_tree(tmp_path):
    repository, _base = _repository(tmp_path)
    _committed_module(repository, "module.py", 900)
    empty_tree = _git(repository, "hash-object", "-t", "tree", "/dev/null")

    result = _run(repository, empty_tree, "HEAD", "--require-files")

    assert result.returncode == 0
    assert "::warning" in result.stdout
    assert "900 lines" in result.stdout

def test_hygiene_rejects_a_change_that_grows_a_file_past_the_byte_limit(tmp_path):
    repository, _base = _repository(tmp_path)
    (repository / "asset.bin").write_bytes(b"x" * 400_000)
    _git(repository, "add", "asset.bin")
    _git(repository, "commit", "-m", "add asset")
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "asset.bin").write_bytes(b"x" * 512_001)
    _git(repository, "add", "asset.bin")
    _git(repository, "commit", "-m", "grow asset")

    result = _run(repository, base, "HEAD")

    assert result.returncode == 1
    assert "512001 bytes" in result.stdout
    assert "grown past the limit by this change" in result.stdout

def test_hygiene_leaves_already_oversized_bytes_to_the_complete_tree_check(tmp_path):
    repository, _base = _repository(tmp_path)
    target = repository / "asset.bin"
    target.write_bytes(b"x" * 512_001)
    _git(repository, "add", "asset.bin")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    base = _git(repository, "rev-parse", "HEAD")
    target.write_bytes(b"x" * 512_010)
    _git(repository, "add", "asset.bin")
    _git(repository, "commit", "-m", _FIXTURE_COMMIT)
    head = _git(repository, "rev-parse", "HEAD")
    target.write_bytes(b"x")

    change_result = _run(repository, base, head)
    empty_tree = _git(repository, "hash-object", "-t", "tree", "/dev/null")
    tree_result = _run(repository, empty_tree, head, "--require-files")

    assert change_result.returncode == 0
    assert "checks passed" in change_result.stdout
    assert tree_result.returncode == 1
    assert "512010 bytes" in tree_result.stdout

def test_hygiene_measures_a_module_renamed_and_grown_in_one_step(tmp_path):
    repository, _base = _repository(tmp_path)
    base = _committed_module(repository, "module.py", 700)
    (repository / "module.py").unlink()
    (repository / "renamed.py").write_text("pass\n" * 801, encoding="utf-8")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-m", "rename and grow")

    result = _run(repository, base, "HEAD")

    assert result.returncode == 0
    assert "::warning" in result.stdout
    assert "801 lines" in result.stdout

def test_hygiene_fails_when_git_diff_cannot_be_computed(tmp_path):
    repository, _base = _repository(tmp_path)

    result = _run(repository, "missing-base", "HEAD")

    assert result.returncode == 2
    assert "git diff failed" in result.stderr

def test_hygiene_checks_a_complete_root_tree(tmp_path):
    repository, _base = _repository(tmp_path)
    empty_tree = _git(repository, "hash-object", "-t", "tree", "/dev/null")

    result = _run(repository, empty_tree, "HEAD", "--require-files")

    assert result.returncode == 0
    assert "checks passed" in result.stdout

def test_hygiene_rejects_an_empty_complete_tree(tmp_path):
    repository, base = _repository(tmp_path)
    empty_tree = _git(repository, "hash-object", "-t", "tree", "/dev/null")

    result = _run(repository, base, empty_tree, "--require-files")

    assert result.returncode == 1
    assert "no files were available" in result.stdout
