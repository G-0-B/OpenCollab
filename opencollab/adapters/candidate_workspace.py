"""Environment-backed isolated candidate worktrees."""

from __future__ import annotations

import os
import posixpath
import shlex
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from opencollab.adapters._candidate_increment import _apply_increment as _apply_increment
from opencollab.adapters._candidate_increment import _safe_path
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters._env_process import ProcessCleanupError
from opencollab.adapters._env_worktree_submodules import _initialize_source_available_submodules
from opencollab.adapters._git_patch_bytes import _complete, _complete_patch
from opencollab.adapters._git_patch_bytes import _patch_temporary as _candidate_temporary
from opencollab.adapters.env import DockerWorkspaceEnvironment
from opencollab.adapters.git_patch import (
    _git_command_and_config_guard,
    _submodule_patch_guard,
    guarded_staged_diff_command,
)
from opencollab.application.async_timeout import await_owned_operation
from opencollab.application.exception_notes import add_exception_note

CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS = 900.0

def _repository_exclusions(prefix: str, paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(posixpath.join(prefix, _safe_path(path)) for path in paths)

async def _raw_diff(environment: Any, exclude_paths: Sequence[str] = ()) -> str:
    workspace = getattr(environment, "workspace", None)
    if not isinstance(workspace, str) or not workspace:
        raise RuntimeError("candidate diff workspace is unavailable")
    return await _raw_diff_at(environment, workspace, exclude_paths)

async def _raw_diff_at(
    environment: Any,
    workspace: str,
    exclude_paths: Sequence[str] = (),
    *,
    base_revision: str = "HEAD",
    index_file: str | None = None,
    untracked_exclude_paths: Sequence[str] = (),
) -> str:
    excluded = tuple(_safe_path(path) for path in exclude_paths)
    pathspec = ""
    if excluded:
        pathspec = " -- . " + " ".join(
            shlex.quote(f":(exclude,literal){path}") for path in excluded
        )
    git = f"git -c core.quotePath=true -C {shlex.quote(workspace)}"
    cached = ""
    if index_file is not None:
        git = f"GIT_INDEX_FILE={shlex.quote(index_file)} {git}"
        cached = " --cached"
    tracked = await _complete_patch(
        await environment.exec_cmd(
            f"{git} -c core.filemode=true --no-pager diff{cached} {shlex.quote(base_revision)} --binary --full-index "
            "--no-ext-diff --no-textconv --no-color --unified=3 --src-prefix=a/ --dst-prefix=b/"
            + pathspec,
            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
        ),
        "candidate tracked diff", environment, workspace, base_revision, index_file,
    )
    untracked = _complete(
        await environment.exec_cmd(
            f"{git} ls-files --others --exclude-standard -z"
            + pathspec,
            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
        ),
        "candidate untracked listing",
    )
    parts = [tracked] if tracked.strip() else []
    untracked_excluded = tuple(_safe_path(path) + "/" for path in untracked_exclude_paths)
    for path in (item for item in untracked.split("\0") if item):
        if path.startswith(untracked_excluded):
            continue
        patch = await _complete_patch(
            await environment.exec_cmd(
                "git -c core.quotePath=true -C "
                f"{shlex.quote(workspace)} --no-pager diff --no-index --binary --full-index "
                "--no-ext-diff --no-textconv --no-color --unified=3 --src-prefix=a/ --dst-prefix=b/ "
                f"-- /dev/null {shlex.quote(path)}",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            f"candidate untracked diff for {path}", environment, workspace, base_revision, None,
            allowed=(0, 1),
        )
        if patch.strip():
            parts.append(patch)
    return "".join(parts)

async def _assert_patchable_submodules(
    environment: Any,
    workspace: str,
    base_revision: str,
    *,
    index_file: str | None = None,
    reject_ignored_untracked: bool = True,
) -> None:
    git = f"git -C {shlex.quote(workspace)}"
    indexed_git = f"GIT_INDEX_FILE={shlex.quote(index_file)} {git}" if index_file is not None else None
    _complete(
        await environment.exec_cmd(
            _submodule_patch_guard(
                git, base_revision, indexed_git_command=indexed_git,
                reject_ignored_untracked=reject_ignored_untracked,
            ),
            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
        ),
        "submodule patch inspection",
    )

@dataclass(slots=True)
class _CandidateLease:
    base_environment: Any
    environment: Any | None
    source_workspace: str
    candidate_workspace: str
    base_revision: str
    workspace_prefix: str = ""
    cleaned: bool = False

    async def diff(self, exclude_paths: Sequence[str] = ()) -> str:
        async with _candidate_temporary(
            self.base_environment, self.source_workspace,
            "", prefix=".candidate-capture-index-",
        ) as index_file:
            git = (
                f"GIT_INDEX_FILE={shlex.quote(index_file)} "
                f"git -c core.filemode=true -C {shlex.quote(self.candidate_workspace)}"
            )
            # Stage the current contents in an owned temporary index. Candidate
            # commits and index resets leave the same delivered file changes.
            _complete(
                await self.base_environment.exec_cmd(
                    f"{git} read-tree HEAD && {git} add --all -- .",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate contents capture",
            )
            original_git = f"git -C {shlex.quote(self.candidate_workspace)}"
            known = _complete(
                await self.base_environment.exec_cmd(
                    f"{original_git} ls-tree -r --name-only -z {shlex.quote(self.base_revision)} "
                    f"&& {original_git} ls-files --cached -z",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate known paths",
            )
            remaining = _complete(
                await self.base_environment.exec_cmd(
                    f"{git} ls-files --others -z",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate remaining files",
            )
            # Retain existing source and staged files when new ignore rules hide
            # them. The remaining listing supplies current files, so deletions
            # and file/directory replacements follow their actual contents.
            omitted = sorted(set(known.split("\0")).intersection(remaining.split("\0")) - {""})
            if omitted:
                async with _candidate_temporary(
                    self.base_environment, self.source_workspace,
                    "".join(f":(literal){path}\0" for path in omitted),
                    prefix=".candidate-capture-paths-",
                ) as paths_file:
                    _complete(
                        await self.base_environment.exec_cmd(
                            f"{git} add --force --pathspec-from-file={shlex.quote(paths_file)} --pathspec-file-nul",
                            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                        ),
                        "candidate known files capture",
                    )
            # A repository patch carries gitlink identities, not the module's
            # files. Ordinary Git apply also leaves module checkouts unchanged.
            # Let the caller retain this lease instead of delivering a partial
            # result and deleting the only workspace with the module changes.
            await _assert_patchable_submodules(
                self.base_environment, self.candidate_workspace,
                self.base_revision, index_file=index_file,
            )
            return await _raw_diff_at(
                self.base_environment,
                self.candidate_workspace,
                _repository_exclusions(self.workspace_prefix, exclude_paths),
                base_revision=self.base_revision,
                index_file=index_file,
            )

    async def cleanup(self) -> None:
        await await_owned_operation(self._cleanup_resources(), propagate_cancellation=True)

    async def _cleanup_resources(self) -> None:
        if self.cleaned:
            return
        if self.environment is not None:
            await self.environment.cleanup()
        git = f"git -C {shlex.quote(self.source_workspace)}"
        listed = _complete(
            await self.base_environment.exec_cmd(f"{git} worktree list --porcelain -z", timeout=120),
            "candidate worktree cleanup listing",
        )
        normalize = os.path.realpath if isinstance(self.base_environment, LocalEnvironment) else posixpath.normpath
        registered = {
            normalize(field.removeprefix("worktree "))
            for field in listed.split("\0")
            if field.startswith("worktree ")
        }
        if normalize(self.candidate_workspace) in registered:
            command = f"{git} worktree remove --force -- {shlex.quote(self.candidate_workspace)}"
        else:
            command = f"rm -rf -- {shlex.quote(self.candidate_workspace)}"
        _complete(
            await self.base_environment.exec_cmd(command, timeout=120),
            "candidate worktree cleanup",
        )
        self.cleaned = True

class EnvCandidateWorkspace:
    """Candidate workspace port implemented over one task environment."""

    def __init__(
        self, environment: Any, *, workspace: str | None = None,
        owned_save_dir: str | None = None,
    ) -> None:
        self._environment = environment
        self._workspace = workspace or getattr(environment, "workspace", None)
        if not isinstance(self._workspace, str) or not self._workspace:
            raise ValueError("candidate source workspace is unavailable")
        self._owned_save_dir: str | None = None
        if owned_save_dir is not None and isinstance(environment, LocalEnvironment):
            source = os.path.realpath(self._workspace)
            save_dir = os.path.realpath(owned_save_dir)
            try:
                covers_source = os.path.commonpath((source, save_dir)) == save_dir
            except ValueError:
                covers_source = False
            if not covers_source:
                self._owned_save_dir = save_dir

    def _owned_untracked_paths(self, repository_root: str) -> tuple[str, ...]:
        if self._owned_save_dir is None:
            return ()
        repository_root = os.path.realpath(repository_root)
        try:
            in_repository = os.path.commonpath((
                self._owned_save_dir, repository_root,
            )) == repository_root
        except ValueError:
            in_repository = False
        if not in_repository:
            return ()
        return (os.path.relpath(self._owned_save_dir, repository_root).replace(os.sep, "/"),)

    async def _repository_root(self) -> str:
        return _complete(
            await self._environment.exec_cmd(
                f"git -C {shlex.quote(self._workspace)} rev-parse --show-toplevel",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            "candidate repository root",
        ).removesuffix("\n")

    async def _workspace_prefix(self) -> str:
        return _complete(
            await self._environment.exec_cmd(
                f"git -C {shlex.quote(self._workspace)} rev-parse --show-prefix",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            "candidate workspace location",
        ).removesuffix("\n")

    async def _candidate_environment(self, path: str) -> Any:
        if isinstance(self._environment, LocalEnvironment):
            return LocalEnvironment(path)
        if isinstance(self._environment, DockerEnvironment):
            await self._environment.setup()
            container_id = self._environment._container_id
            if not container_id:
                raise RuntimeError("candidate container identity is unavailable")
            return DockerWorkspaceEnvironment(
                container_id=container_id,
                repo_root=path,
                command_prefix=self._environment._command_prefix,
                timeout_returncode=self._environment._timeout_returncode,
            )
        raise RuntimeError("candidate worktrees require a local or Docker environment")

    async def acquire(self, label: str) -> _CandidateLease:
        return await self._acquire(label)

    async def _acquire(
        self, label: str, *, strict_capture: bool = False,
        lease_observer: Callable[[_CandidateLease], None] | None = None,
        strict_untracked_exclusions: Sequence[str] = (),
    ) -> _CandidateLease:
        repository_root = await self._repository_root()
        workspace_prefix = await self._workspace_prefix()
        token = uuid.uuid4().hex
        if isinstance(self._environment, LocalEnvironment):
            path = tempfile.mkdtemp(prefix="opencollab-candidate-")
            os.rmdir(path)
        else:
            path = f"/tmp/opencollab-candidate-{token}"
        base_revision = _complete(
            await self._environment.exec_cmd(
                f"git -C {shlex.quote(repository_root)} rev-parse --verify HEAD",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            "candidate base revision",
        ).strip()
        if strict_capture:
            source_patch = await _complete_patch(
                await self._environment.exec_cmd(
                    f"cd -- {shlex.quote(repository_root)} && "
                    + guarded_staged_diff_command(
                        base_revision=base_revision,
                        reject_ignored_untracked=False,
                        exclude_untracked_paths=(
                            *self._owned_untracked_paths(repository_root),
                            *(_safe_path(path) for path in strict_untracked_exclusions),
                        ),
                    ),
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "team source contents capture", self._environment, repository_root,
                base_revision, None, captured_blobs=True,
            )
        else:
            await _assert_patchable_submodules(
                self._environment, repository_root, base_revision, reject_ignored_untracked=False,
            )
            source_patch = await _raw_diff_at(
                self._environment, repository_root, base_revision=base_revision,
                untracked_exclude_paths=self._owned_untracked_paths(repository_root),
            )
        lease = _CandidateLease(
            base_environment=self._environment,
            environment=None,
            source_workspace=repository_root,
            candidate_workspace=path,
            base_revision=base_revision,
            workspace_prefix=workspace_prefix,
        )
        if lease_observer is not None:
            lease_observer(lease)
        try:
            # Environment adapters terminate their command group before raising
            # cancellation. Cleanup starts only after that command has settled.
            create_command = (
                "git -C "
                f"{shlex.quote(repository_root)} worktree add --detach -- "
                f"{shlex.quote(path)} {shlex.quote(base_revision)}"
            )
            if strict_capture:
                git, guard = _git_command_and_config_guard()
                create_command = (
                    f"cd -- {shlex.quote(repository_root)} && {guard}{git} "
                    f"worktree add --detach -- {shlex.quote(path)} {shlex.quote(base_revision)}"
                )
            result = await self._environment.exec_cmd(
                create_command,
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            )
            _complete(result, f"candidate worktree setup for {label}")
            candidate_workspace = path
            if workspace_prefix:
                candidate_workspace = posixpath.join(path, _safe_path(workspace_prefix))
                _complete(
                    await self._environment.exec_cmd(
                        f"mkdir -p -- {shlex.quote(candidate_workspace)}",
                        timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                    ),
                    "candidate workspace directory",
                )
            lease.environment = await self._candidate_environment(candidate_workspace)
            await lease.environment.setup()
            if isinstance(self._environment, LocalEnvironment):
                source_repository = os.path.realpath(repository_root)
                source_workspace = os.path.realpath(self._workspace)
                try:
                    if os.path.commonpath((source_repository, source_workspace)) != source_repository:
                        raise RuntimeError("candidate source workspace is outside its Git repository")
                except ValueError as exc:
                    raise RuntimeError("candidate source workspace is outside its Git repository") from exc

            async def git_in(workspace: str, *arguments: str) -> Any:
                return await self._environment.exec_cmd(
                    shlex.join(("git", "-C", workspace, *arguments)),
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                )

            async def git_marker_exists(source_module: str) -> bool:
                marker = shlex.quote(posixpath.join(source_module, ".git"))
                result = await self._environment.exec_cmd(
                    f"test -e {marker} || test -L {marker}",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                )
                _complete(result, "source submodule Git metadata inspection", allowed=(0, 1))
                return result.returncode == 0

            await _initialize_source_available_submodules(
                repository_root,
                path,
                git_in=git_in,
                source_prefix=workspace_prefix.rstrip("/"),
                require_initialized=False,
                local_filesystem=isinstance(self._environment, LocalEnvironment),
                git_marker_exists=git_marker_exists,
            )
            if source_patch.strip():
                async with _candidate_temporary(
                    self._environment, repository_root, source_patch,
                    prefix=".candidate-source-", suffix=".patch",
                ) as source_file:
                    apply_command = (
                        f"git -C {shlex.quote(path)} apply --index --binary --whitespace=nowarn "
                        f"-- {shlex.quote(source_file)}"
                    )
                    if strict_capture:
                        git, guard = _git_command_and_config_guard()
                        apply_command = (
                            f"cd -- {shlex.quote(path)} && {guard}{git} "
                            f"apply --index --binary --whitespace=nowarn -- {shlex.quote(source_file)}"
                        )
                    _complete(
                        await self._environment.exec_cmd(
                            apply_command,
                            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                        ),
                        "candidate source contents",
                    )
                # The candidate's index records its starting contents, including
                # source untracked files, while the source index stays untouched.
                lease.base_revision = _complete(
                    await self._environment.exec_cmd(
                        f"git -C {shlex.quote(path)} write-tree",
                        timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                    ),
                    "candidate source tree",
                ).strip()
            # A worktree-local ref retains the original contents for recovery
            # throughout the lease, including after commits or Git collection.
            _complete(
                await self._environment.exec_cmd(
                    f"git -C {shlex.quote(path)} update-ref refs/worktree/opencollab-source "
                    f"{shlex.quote(lease.base_revision)}",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate source recovery reference",
            )
            return lease
        except BaseException as failure:
            if isinstance(failure, ProcessCleanupError) or getattr(self._environment, "revoked", False):
                if lease.environment is not None:
                    try:
                        await await_owned_operation(lease.environment.cleanup())
                    except BaseException as cleanup_error:
                        add_exception_note(failure, f"candidate environment cleanup failed: {cleanup_error}")
                add_exception_note(failure, f"candidate worktree retained until command cleanup completes: {path}")
            else:
                try:
                    await await_owned_operation(lease.cleanup())
                except BaseException as cleanup_error:
                    add_exception_note(
                        failure,
                        f"candidate initialization cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}",
                    )
            raise

    async def source_diff(self, exclude_paths: Sequence[str] = ()) -> str:
        repository_root = await self._repository_root()
        excluded = _repository_exclusions(await self._workspace_prefix(), exclude_paths) if exclude_paths else ()
        return await _raw_diff_at(
            self._environment,
            repository_root,
            excluded,
            untracked_exclude_paths=self._owned_untracked_paths(repository_root),
        )

    async def _source_changed(self, exclude_paths: Sequence[str] = ()) -> bool:
        repository_root = await self._repository_root()
        pathspec = ""
        if exclude_paths:
            excluded = _repository_exclusions(await self._workspace_prefix(), exclude_paths)
            pathspec = " -- . " + " ".join(
                shlex.quote(f":(exclude,literal){path}") for path in excluded
            )
        status = _complete(
            await self._environment.exec_cmd(
                f"git --no-optional-locks -C {shlex.quote(repository_root)} "
                "status --porcelain=v1 -z --untracked-files=all"
                + pathspec,
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            "candidate source status",
        )
        owned = tuple(path + "/" for path in self._owned_untracked_paths(repository_root))
        return any(
            record and not (record.startswith("?? ") and record[3:].startswith(owned))
            for record in status.split("\0")
        )

    async def source_revision(self) -> str:
        return _complete(
            await self._environment.exec_cmd(
                f"git -C {shlex.quote(await self._repository_root())} rev-parse --verify HEAD",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            "candidate source revision",
        ).strip()

    async def restore_source(self, patch: str) -> None:
        await self._replace_source_diff(patch)

    async def _replace_source_diff(self, patch: str) -> None:
        current = await self.source_diff()
        async with _candidate_temporary(
            self._environment, self._workspace, current,
            prefix=".candidate-current-", suffix=".patch", cleanup_best_effort=True,
        ) as current_file:
            async with _candidate_temporary(
                self._environment, self._workspace, patch,
                prefix=".candidate-target-", suffix=".patch", cleanup_best_effort=True,
            ) as target_file:
                reversed_current = False
                try:
                    if current.strip():
                        await self._apply(
                            current_file,
                            "candidate current-source removal",
                            reverse=True,
                        )
                        reversed_current = True
                    if patch.strip():
                        await self._apply(target_file, "candidate source restoration")
                except BaseException as failure:
                    if reversed_current:
                        try:
                            await self._apply(
                                current_file,
                                "candidate source restoration rollback",
                            )
                        except BaseException as rollback:
                            raise RuntimeError(
                                "candidate source restoration and rollback both failed"
                            ) from rollback
                    raise failure

    async def _apply(self, path: str, operation: str, *, reverse: bool = False) -> None:
        reverse_flag = " --reverse" if reverse else ""
        repository_root = await self._repository_root()
        result = await self._environment.exec_cmd(
            "git -c core.filemode=true -C "
            f"{shlex.quote(repository_root)} apply --binary --whitespace=nowarn"
            f"{reverse_flag} -- {shlex.quote(path)}",
            timeout=120,
        )
        _complete(result, operation)

    async def adopt(
        self,
        patch: str,
        preserve_paths: Sequence[str] = (),
    ) -> None:
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("candidate patch must be non-empty")
        preserved = (
            _repository_exclusions(await self._workspace_prefix(), preserve_paths)
            if preserve_paths else ()
        )
        await _apply_increment(
            self._environment, await self._repository_root(), patch, preserved,
        )

__all__ = ["EnvCandidateWorkspace"]
