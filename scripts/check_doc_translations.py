#!/usr/bin/env python3
"""Require a Chinese update when a PR changes an English document with a translation."""

from __future__ import annotations

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

# Most pairs share a directory. Duo keeps its translation in a guide directory.
_SEPARATE_PAIRS = {"docs/duo.md": "docs/duo/README.zh-CN.md"}

def _git(repository: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repository)

def _paths(payload: bytes) -> set[str]:
    return {os.fsdecode(path) for path in payload.split(b"\0") if path}

def missing_translations(repository: Path, base: str, head: str) -> list[tuple[str, str]]:
    """Compare the PR's net changes from its merge base, including removed pairs."""
    start = _git(repository, "merge-base", base, head).decode().strip()
    changed = _paths(_git(repository, "diff", "--name-only", "--no-renames", "-z", start, head))
    files = _paths(_git(repository, "ls-tree", "-r", "--name-only", "-z", start))
    files |= _paths(_git(repository, "ls-tree", "-r", "--name-only", "-z", head))
    pairs = {(path.removesuffix(".zh-CN.md") + ".md", path) for path in files if path.endswith(".zh-CN.md")}
    pairs.update(_SEPARATE_PAIRS.items())
    return sorted(
        (english, chinese)
        for english, chinese in pairs
        if english in files and chinese in files and english in changed and chinese not in changed
    )

def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", help="PR target commit or branch")
    parser.add_argument("head", help="PR head commit or branch")
    args = parser.parse_args(argv)
    try:
        missing = missing_translations(Path.cwd(), args.base, args.head)
    except subprocess.CalledProcessError:
        print("Unable to compare documentation changes. Fetch both revisions and their history.", file=sys.stderr)
        return 2
    for english, chinese in missing:
        print(f"Update {chinese} in the same PR as {english}.", file=sys.stderr)
    if missing:
        return 1
    print("Documentation translation check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
