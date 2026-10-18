"""Interpret Git evidence for host and container worktrees.

Both environments supply Git output to the same attribution rules for the
starting base, current HEAD, and own commits.
"""

from __future__ import annotations

import re

# ``git log -g --format=%H%x09%P%x09%gs`` emits HEAD, parents, and action.
# Object ids have 40 characters in SHA-1 repositories and 64 in SHA-256.
REFLOG_ENTRY_RE = re.compile(r"^([0-9a-f]{40}|[0-9a-f]{64})\t([0-9a-f ]*)\t(.*)$")
# Reflog messages for a commit this worktree made itself: ``commit: <subject>``,
# and the parenthesised variants ``commit (initial)``, ``commit (amend)``,
# ``commit (merge)``. Cherry-pick and revert also append work to the current history.
OWN_COMMIT_REFLOG_PREFIX = ("commit", "cherry-pick:", "revert:")
# Pull prefixes the same rebase lifecycle with its command line. Try bare pull
# first, then its arguments lazily, before any markers in the commit subject.
REBASE_REFLOG_RE = re.compile(r"^(?:rebase(?: -i)?|pull|pull [^\r\n]*?) \(([^)]+)\):")
# The capped list has a separately reported uncapped total.
OWN_COMMIT_LIMIT = 64
# An empty old value atomically claims an absent ref with ``git update-ref``.
# Unlike an all-zero object id, it works with every repository hash length.
ABSENT_REF_OLD_VALUE = ""
BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,180}$")

def validate_worktree_branch(name: str) -> str:
    """Return ``name`` if it is one safe Git ref component, else raise."""
    if not isinstance(name, str) or BRANCH_NAME_RE.fullmatch(name) is None:
        raise ValueError("worktree branch name must be one safe Git ref component")
    return name

def _previous_work_entry(
    entries: list[tuple[str, str, str]], index: int, commit: str,
) -> int | None:
    """Find where this worktree last produced a restored commit.

    A later checkout or fast-forward adoption of the same commit begins a new
    stretch. No-op moves and earlier restores keep its original provenance.
    """
    for previous in range(index + 1, len(entries)):
        earlier_commit, parents, message = entries[previous]
        if earlier_commit != commit:
            continue
        if previous + 1 < len(entries) and entries[previous + 1][0] == commit:
            continue
        rebase = REBASE_REFLOG_RE.match(message)
        if message.startswith(OWN_COMMIT_REFLOG_PREFIX) or (
            len(parents.split()) == 2 and ": Merge made by " in message
        ) or (rebase is not None and rebase.group(1) not in {"start", "abort", "finish", "reset"}):
            return previous
        if message.startswith("reset:") or rebase is not None:
            continue
        return None
    return None

def select_diff_base(reflog: str, *, fallback: str) -> str:
    """The external history this worktree's current work grew from, or ``fallback``.

    ``reflog`` is ``git log -g --format=%H%x09%P%x09%gs HEAD``, newest first.
    Checkout, reset, and fast-forward merge adopt the commit they move onto.
    Commit, amend, cherry-pick, and revert extend or revise the current work.
    Rebase adopts its start commit, retaining the replayed work in the diff.
    A two-parent merge adopts its incoming parent, retaining the first parent's
    work and any conflict resolution. Aborted rebases leave the previous base
    intact, including when the rebase had already replayed some commits.
    Reset or rebase onto this worktree's own history resumes that earlier
    stretch, preserving repairs that remain in the restored commit.

    ``fallback`` is the commit the worktree was created on, which is the honest
    answer when there is nothing to read -- a repository with
    ``core.logAllRefUpdates`` off keeps no such record.
    """
    entries = [
        match.groups() for line in reflog.split("\n")
        if (match := REFLOG_ENTRY_RE.match(line)) is not None
    ]
    aborted_rebases = 0
    resume_at = 0
    for index, (commit, parents, message) in enumerate(entries):
        if index < resume_at:
            continue
        rebase = REBASE_REFLOG_RE.match(message)
        if rebase is not None:
            action = rebase.group(1)
            if action == "abort":
                aborted_rebases += 1
            elif action == "start":
                if aborted_rebases:
                    aborted_rebases -= 1
                else:
                    previous_work = _previous_work_entry(entries, index, commit)
                    if previous_work is not None:
                        resume_at = previous_work
                    elif index + 1 == len(entries) or entries[index + 1][0] != commit:
                        return commit
            continue
        if aborted_rebases:
            continue
        parent_commits = parents.split()
        if len(parent_commits) == 2 and (
            message.startswith("commit (merge):") or ": Merge made by " in message
        ):
            return parent_commits[1]
        if message.startswith(OWN_COMMIT_REFLOG_PREFIX):
            continue
        if message.startswith("reset:"):
            previous_work = _previous_work_entry(entries, index, commit)
            if previous_work is not None:
                resume_at = previous_work
                continue
        # Creating a branch at the current HEAD does not adopt a new base.
        if index + 1 < len(entries) and entries[index + 1][0] == commit:
            continue
        return commit
    return fallback

def parse_own_commits(rev_list: str) -> tuple[tuple[str, ...], int]:
    """Split ``git rev-list <base>..HEAD`` output into a capped list and its total.

    The list is exactly the set of shas this agent could hand to a teammate --
    every commit reachable from HEAD but not from the base it started this
    stretch of work on -- so a handoff joins as ``tester.diff_base in
    coder.commits``. The total is returned separately because a capped list must
    not read as a shorter one.
    """
    commits = rev_list.split()
    return tuple(commits[:OWN_COMMIT_LIMIT]), len(commits)
