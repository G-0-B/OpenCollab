"""A Git worktree carved out inside an already-running task container."""

from __future__ import annotations

import logging
import posixpath
import shlex
import uuid
from collections.abc import Callable

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters._env_worktree_submodules import _initialize_source_available_submodules
from opencollab.adapters._git_patch_bytes import _complete_patch
from opencollab.adapters.git_patch import guarded_staged_diff_command
from opencollab.adapters.git_worktree_evidence import (
    ABSENT_REF_OLD_VALUE,
    parse_own_commits,
    select_diff_base,
    validate_worktree_branch,
)

logger = logging.getLogger(__name__)

CONTAINER_GIT_TIMEOUT_SECONDS = 60.0

class ContainerWorktreeEnvironment(DockerEnvironment):
    """An isolated checkout of the task repository inside its container.

    Git runs through ``docker exec`` so a container-only repository needs no host
    mount. Worktrees live outside the repository root to keep them out of the
    harness's patch archive.

    The root may contain a sibling agent's uncommitted edits. Each worktree still
    starts from a clean checkout of the repository root's HEAD, excluding those edits.

    Reports the same ``get_diff``, ``diff_base``, ``head_commit``, ``own_commits``
    and ``own_commit_count`` evidence as ``WorktreeEnvironment``.
    """

    def __init__(
        self,
        *,
        container_id: str,
        repository_root: str,
        worktree_root: str,
        branch_name: str | None = None,
        command_prefix: Callable[[str], str] | str | None = None,
        timeout_returncode: int = -1,
    ) -> None:
        branch = validate_worktree_branch(
            branch_name or f"opencollab-wt-{uuid.uuid4().hex[:12]}"
        )
        repository_root = _absolute_container_path(repository_root, "repository root")
        worktree_root = _absolute_container_path(worktree_root, "worktree root")
        if posixpath.commonpath((repository_root, worktree_root)) == repository_root:
            raise ValueError("container worktree root must be outside the source repository")
        worktree_dir = posixpath.join(worktree_root, branch)
        super().__init__(
            workspace=worktree_dir,
            container_id=container_id,
            exec_workdir=worktree_dir,
            command_prefix=command_prefix,
            timeout_returncode=timeout_returncode,
        )
        self.source_workspace = repository_root
        self._repository_root = repository_root
        self._worktree_root = worktree_root
        self._worktree_dir = worktree_dir
        self._source_subdir = ""
        self._branch = branch
        self._branch_owned = False
        self._worktree_registered = False
        self._worktree_files_pending = False
        self._registered_worktree_path: str | None = None
        self._git_diff_delivery_pending = False
        self._base_commit: str | None = None
        self._diff_base: str | None = None
        self._head_commit: str | None = None
        self._own_commits: tuple[str, ...] = ()
        self._own_commit_count: int | None = None

    @property
    def diff_path_prefix(self) -> str:
        """Repository-relative prefix of this environment's execution workspace."""
        return self._source_subdir

    @property
    def diff_base(self) -> str | None:
        """The revision the last ``get_diff`` measured against, if any."""
        return self._diff_base

    @property
    def head_commit(self) -> str | None:
        """Where HEAD stood at that same reading, if it could be read."""
        return self._head_commit

    @property
    def own_commits(self) -> tuple[str, ...]:
        """The commits this worktree made since ``diff_base``, newest first."""
        return self._own_commits

    @property
    def own_commit_count(self) -> int | None:
        """How many commits ``own_commits`` was cut from, or ``None``."""
        return self._own_commit_count

    async def _git(self, workdir: str, *args: str) -> ExecResult:
        """Run one Git command in the container, outside the agent's shell.

        Not ``exec_cmd``: that runs what the agent's session runs, through the
        image's login shell and whatever command prefix the harness set. The
        worktree's own bookkeeping has to be argv-exact and independent of that.
        """
        await self._bind_attached()
        container_id = self._container_id
        if container_id is None:
            raise RuntimeError("container worktree is not attached to a container")
        result = await self._docker(
            "exec",
            "-w",
            workdir,
            "--",
            container_id,
            "git",
            "-c",
            f"safe.directory={workdir}",
            *args,
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        return result.to_exec_result()

    async def _source_git_marker_exists(self, source_module: str) -> bool:
        await self._bind_attached()
        container_id = self._container_id
        if container_id is None:
            raise RuntimeError("container worktree is not attached to a container")
        marker = shlex.quote(posixpath.join(source_module, ".git"))
        result = await self._docker(
            "exec", "--", container_id, "sh", "-c",
            f"test -e {marker} || test -L {marker}",
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError("cannot inspect source submodule Git metadata")
        return result.returncode == 0

    async def setup(self, mount_dir: str | None = None) -> str:
        await super().setup(mount_dir)
        self._ensure_active()
        await self._create_worktree()
        self.workspace = posixpath.join(self._worktree_dir, self._source_subdir).rstrip("/")
        self._exec_workdir = self.workspace
        return self.workspace

    async def _create_worktree(self) -> None:
        root = await self._git(self.source_workspace, "rev-parse", "--show-toplevel")
        prefix = await self._git(self.source_workspace, "rev-parse", "--show-prefix")
        if any(
            result.returncode != 0 or result.stdout_truncated or result.stderr_truncated
            for result in (root, prefix)
        ):
            raise RuntimeError("cannot resolve container source repository and workspace")
        self._repository_root = _absolute_container_path(
            root.stdout.removesuffix("\n"), "repository root",
        )
        self._source_subdir = prefix.stdout.removesuffix("\n").rstrip("/")
        if posixpath.commonpath((self._repository_root, self._worktree_root)) == self._repository_root:
            raise ValueError("container worktree root must be outside the source repository")
        made = await self._docker(
            "exec",
            "--",
            self._container_id or "",
            "mkdir",
            "-p",
            "--",
            self._worktree_root,
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        if made.returncode != 0:
            raise RuntimeError("cannot create the container worktree root")
        base = await self._git(self._repository_root, "rev-parse", "--verify", "HEAD^{commit}")
        if base.returncode != 0 or base.stdout_truncated:
            raise RuntimeError("cannot resolve container worktree base commit")
        self._base_commit = base.stdout.strip()
        # The claimed ref is an ownership lease, not the worktree's HEAD: a
        # detached worktree keeps its own commits from moving that lease, so a
        # later external advance is observable and cannot be deleted by us.
        claimed = await self._git(
            self._repository_root,
            "update-ref",
            f"refs/heads/{self._branch}",
            self._base_commit,
            ABSENT_REF_OLD_VALUE,
        )
        if claimed.returncode != 0:
            raise RuntimeError(
                f"cannot atomically claim container worktree branch: {claimed.stderr.strip()}"
            )
        self._branch_owned = True
        added = await self._git(
            self._repository_root,
            "worktree",
            "add",
            "--detach",
            self._worktree_dir,
            self._base_commit,
        )
        if added.returncode != 0:
            raise RuntimeError(f"container git worktree add failed: {added.stderr.strip()}")
        self._worktree_registered = True
        self._worktree_files_pending = True
        # Git's registry uses physical paths, even when the configured root
        # contains symlinks. Keep that identity for failed-removal recovery.
        location = await self._git(self._worktree_dir, "rev-parse", "--show-toplevel")
        if location.returncode == 0 and not location.stdout_truncated and not location.stderr_truncated:
            self._registered_worktree_path = location.stdout.removesuffix("\n")
        await _initialize_source_available_submodules(
            self._repository_root, self._worktree_dir, git_in=self._git,
            source_prefix=self._source_subdir,
            require_initialized=False, local_filesystem=False,
            git_marker_exists=self._source_git_marker_exists,
        )

    def retain_changes(self) -> str:
        """Keep the worktree when a bounded delivery omitted part of its patch."""
        self._git_diff_delivery_pending = True
        return f"{self._container_id}:{self._worktree_dir}"

    @property
    def recovery_location(self) -> str | None:
        return f"{self._container_id}:{self._worktree_dir}" if self._git_diff_delivery_pending else None

    async def get_diff(self) -> str:
        """This worktree's changes since the point its current work started from."""
        self._ensure_active()
        self._git_diff_delivery_pending = True
        if self._base_commit is None:
            raise RuntimeError("container worktree base commit is unavailable")
        self._diff_base = await self._resolve_diff_base()
        await self._resolve_own_commits(self._diff_base)
        result = await self.exec_cmd(
            f"cd -- {shlex.quote(self._worktree_dir)} && "
            + guarded_staged_diff_command(base_revision=self._diff_base)
        )
        if result.stdout_truncated or result.stderr_truncated:
            raise RuntimeError(
                f"container worktree diff exceeded capture limit at {self._worktree_dir}"
            )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"git exited with status {result.returncode}"
            raise RuntimeError(
                f"container worktree diff extraction failed: {detail}; "
                f"worktree retained at {self._worktree_dir}"
            )
        patch = await _complete_patch(
            result, "container worktree diff conversion", self,
            self._worktree_dir, self._diff_base, None, captured_blobs=True,
        )
        self._git_diff_delivery_pending = False
        return patch

    async def _resolve_diff_base(self) -> str:
        assert self._base_commit is not None
        reflog = await self._git(
            self._worktree_dir, "log", "-g", "--format=%H%x09%P%x09%gs", "HEAD"
        )
        if reflog.returncode != 0 or reflog.stdout_truncated:
            return self._base_commit
        return select_diff_base(reflog.stdout, fallback=self._base_commit)

    async def _resolve_own_commits(self, base_revision: str) -> None:
        self._head_commit = None
        self._own_commits = ()
        self._own_commit_count = None
        head = await self._git(self._worktree_dir, "rev-parse", "HEAD")
        if head.returncode != 0 or head.stdout_truncated:
            return
        self._head_commit = head.stdout.strip() or None
        listed = await self._git(
            self._worktree_dir, "rev-list", f"{base_revision}..HEAD"
        )
        if listed.returncode != 0 or listed.stdout_truncated:
            return
        self._own_commits, self._own_commit_count = parse_own_commits(listed.stdout)

    async def cleanup(self) -> None:
        """Remove the worktree and release its branch lease, then detach.

        Retained resources are reported to the pool so ownership survives for
        cleanup retry. The caller-owned container is never removed here.
        """
        if self._git_diff_delivery_pending:
            raise RuntimeError(
                "refusing to clean undelivered Git worktree changes; "
                f"worktree retained at {self._worktree_dir}"
            )
        failures: list[str] = []
        if self._worktree_registered:
            removed = await self._git(
                self._repository_root, "worktree", "remove", "--force", self._worktree_dir
            )
            if removed.returncode != 0:
                failures.append(f"worktree retained at {self._worktree_dir}: {removed.stderr.strip()}")
                logger.warning(
                    "container worktree not removed at %s: %s",
                    self._worktree_dir,
                    removed.stderr.strip(),
                )
                listed = await self._git(self._repository_root, "worktree", "list", "--porcelain", "-z")
                if (
                    self._registered_worktree_path is not None
                    and listed.returncode == 0
                    and not listed.stdout_truncated
                    and not listed.stderr_truncated
                ):
                    self._worktree_registered = (
                        f"worktree {self._registered_worktree_path}" in listed.stdout.split("\0")
                    )
            else:
                self._worktree_registered = False
                self._worktree_files_pending = False
        if not self._worktree_registered and self._worktree_files_pending:
            removed_files = await self._docker(
                "exec", "--", self._container_id or "", "rm", "-rf", "--", self._worktree_dir,
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            if removed_files.returncode == 0:
                self._worktree_files_pending = False
            else:
                failures.append(
                    f"worktree files retained at {self._worktree_dir}: "
                    f"{removed_files.stderr.decode('utf-8', errors='replace').strip()}"
                )
        if not self._worktree_registered and self._branch_owned and self._base_commit is not None:
            # Delete only a lease that still stands where it was claimed, so a
            # branch something else advanced is left for its owner to explain.
            released = await self._git(
                self._repository_root,
                "update-ref",
                "-d",
                f"refs/heads/{self._branch}",
                self._base_commit,
            )
            if released.returncode == 0:
                self._branch_owned = False
            else:
                failures.append(f"worktree branch retained: {released.stderr.strip()}")
                logger.warning(
                    "container worktree branch lease retained: %s", released.stderr.strip()
                )
        await super().cleanup()
        if failures:
            raise OSError("container worktree cleanup failed: " + "; ".join(failures))

def _absolute_container_path(path: str, label: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or "\0" in path:
        raise ValueError(f"container {label} must be an absolute path without NUL bytes")
    normalized = posixpath.normpath("/" + path.lstrip("/"))
    if normalized == "/":
        raise ValueError(f"container {label} must not be the container root")
    return normalized

__all__ = ["CONTAINER_GIT_TIMEOUT_SECONDS", "ContainerWorktreeEnvironment"]
