"""Create, track, and clean up per-agent worktree environments."""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import shlex
import uuid
from collections.abc import Sequence
from contextlib import nullcontext

from opencollab.adapters._git_patch_bytes import _complete
from opencollab.adapters._team_candidate_environment import _TeamCandidateEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace, _apply_increment
from opencollab.adapters.env import (
    ContainerWorktreeEnvironment,
    DockerEnvironment,
    Environment,
    LocalEnvironment,
    WorktreeEnvironment,
)
from opencollab.application.async_timeout import await_owned_operation
from opencollab.application.exception_notes import add_exception_note
from opencollab.domain.identity import role_storage_slug, validate_role_identity

logger = logging.getLogger(__name__)

# Where per-agent checkouts go inside a container. Deliberately not under the
# repository root: an archive of that root is what a harness reads the run's
# result out of, and a worktree nested in it would arrive there as a directory
# full of files no agent wrote.
CONTAINER_WORKTREE_ROOT = "/tmp/opencollab-worktrees"

async def _finish_cleanup(operation):
    return await await_owned_operation(operation, propagate_cancellation=True)

class WorktreePool:
    """Lend isolated environments and track them for cleanup.

    With ``use_worktrees=False``, return ``LocalEnvironment(workspace)``.
    ``base_environment`` selects host worktrees by default or worktrees inside
    an attached container. Container worktrees avoid host mounts so bounded
    container archives remain usable for patch extraction.
    """

    def __init__(
        self,
        workspace: str,
        *,
        use_worktrees: bool,
        base_environment: Environment | None = None,
        owned_save_dir: str | None = None,
        enable_adoption: bool = False,
        owned_output_paths: Sequence[str] = (),
    ):
        self._workspace = workspace
        self._use_worktrees = use_worktrees
        self._base_environment = base_environment
        self._owned_save_dir = owned_save_dir
        self._enable_adoption = enable_adoption
        self._owned_output_paths = tuple(os.path.realpath(path) for path in owned_output_paths)
        self._delivery_lock = asyncio.Lock()
        self._owned_delivery_environment: LocalEnvironment | None = None
        self._envs: list[Environment] = []
        self._lifecycle = asyncio.Condition()
        self._release_lock = asyncio.Lock()
        self._release_active = False
        self._generation = 0
        self._inflight_acquires = 0

    async def acquire(self, role: str) -> Environment:
        """Create (and remember) an isolated env for a spawned agent of this role."""
        role = validate_role_identity(role)
        base = self._base_environment
        if not self._use_worktrees:
            # Without isolation every agent works where the run works, which is
            # the supplied environment when there is one and the host workspace
            # otherwise. The shared environment is not tracked for cleanup here:
            # it belongs to whoever handed it in.
            if base is not None:
                return base
            env = LocalEnvironment(self._workspace)
            generation = await self._begin_acquire()
            return await self._finish_acquire(env, generation)

        branch = f"opencollab-{role_storage_slug(role)}-{uuid.uuid4().hex[:8]}"
        generation = await self._begin_acquire()
        env = None
        try:
            async with (self._delivery_lock if self._enable_adoption else nullcontext()):
                if self._enable_adoption:
                    source = self._delivery_environment()
                    probe = await source.exec_cmd(
                        f"git -C {shlex.quote(source.workspace)} rev-parse --show-toplevel",
                    )
                    if probe.returncode == 0:
                        root = _complete(probe, "team source repository").removesuffix("\n")
                        excluded = ()
                        if isinstance(source, LocalEnvironment):
                            excluded = tuple(
                                os.path.relpath(path, root).replace(os.sep, "/")
                                for path in self._owned_output_paths
                                if os.path.commonpath((path, root)) == root and path != root
                            )
                        def remember_lease(lease) -> None:
                            nonlocal env
                            env = _TeamCandidateEnvironment(lease, source.workspace)

                        await EnvCandidateWorkspace(
                            source, owned_save_dir=self._owned_save_dir,
                        )._acquire(role, strict_capture=True, lease_observer=remember_lease,
                                   strict_untracked_exclusions=excluded)
                if env is None:
                    env = self._build_worktree(branch, base)
                    await env.setup()
        except BaseException as original:
            try:
                if env is not None:
                    await _finish_cleanup(env.cleanup())
            except BaseException as cleanup_exc:
                async with self._lifecycle:
                    self._envs.append(env)
                logger.warning("partial worktree cleanup failed", exc_info=True)
                add_exception_note(
                    original,
                    "partial worktree retained for cleanup retry: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                )
            await _finish_cleanup(self._end_acquire())
            raise original
        return await self._finish_acquire(env, generation)

    def _delivery_environment(self) -> Environment:
        base = self._base_environment
        if base is not None and not getattr(base, "local_filesystem", False):
            return base
        if self._owned_delivery_environment is None:
            # Preserve the pool's existing local_filesystem delegation for
            # third-party local ports as well as native LocalEnvironment.
            self._owned_delivery_environment = LocalEnvironment(
                base.workspace if base is not None else self._workspace,
            )
        return self._owned_delivery_environment

    async def adopt(self, patch: str) -> None:
        if not self._enable_adoption:
            raise PermissionError("this team has not declared adopt_agent")
        await self._begin_acquire()
        try:
            async with self._delivery_lock:
                source = self._delivery_environment()
                probe = await source.exec_cmd(
                    f"git -C {shlex.quote(source.workspace)} rev-parse --show-toplevel",
                )
                repository = probe.returncode == 0
                root = (
                    _complete(probe, "team delivery repository").removesuffix("\n")
                    if repository else source.workspace
                )
                await _apply_increment(source, root, patch, repository=repository, strict=True)
        finally:
            await _finish_cleanup(self._end_acquire())

    async def _begin_acquire(self) -> int:
        async with self._lifecycle:
            while self._release_active:
                await self._lifecycle.wait()
            self._inflight_acquires += 1
            return self._generation

    async def _end_acquire(self) -> None:
        async with self._lifecycle:
            self._inflight_acquires -= 1
            self._lifecycle.notify_all()

    async def _finish_acquire(self, env: Environment, generation: int) -> Environment:
        async with self._lifecycle:
            invalidated = generation != self._generation
            if not invalidated:
                self._envs.append(env)
                self._inflight_acquires -= 1
                self._lifecycle.notify_all()
        if invalidated:
            try:
                await _finish_cleanup(env.cleanup())
            except BaseException as cleanup_exc:
                async with self._lifecycle:
                    self._envs.append(env)
                raise RuntimeError(
                    "pool release invalidated an in-flight acquire; cleanup retained for retry"
                ) from cleanup_exc
            finally:
                await _finish_cleanup(self._end_acquire())
            raise RuntimeError("pool release invalidated an in-flight acquire")
        return env

    def _build_worktree(self, branch: str, base: Environment | None) -> Environment:
        """An isolated view of wherever the run's repository actually is."""
        output_options = (
            {"owned_output_paths": self._owned_output_paths}
            if self._enable_adoption and self._owned_output_paths else {}
        )
        if base is None:
            return WorktreeEnvironment(
                self._workspace, branch_name=branch,
                owned_save_dir=self._owned_save_dir,
                **output_options,
            )
        if isinstance(base, DockerEnvironment) and base.container_reference is not None:
            worktree_root = CONTAINER_WORKTREE_ROOT
            if posixpath.commonpath((base.workspace, worktree_root)) == base.workspace:
                worktree_root = "/var/tmp/opencollab-worktrees"
            return ContainerWorktreeEnvironment(
                container_id=base.container_reference,
                repository_root=base.workspace,
                worktree_root=worktree_root,
                branch_name=branch,
                command_prefix=base.command_prefix,
            )
        if getattr(base, "local_filesystem", False):
            return WorktreeEnvironment(
                base.workspace, branch_name=branch,
                owned_save_dir=self._owned_save_dir,
                **output_options,
            )
        raise TypeError(
            "worktree isolation is not available for this environment: "
            f"{type(base).__name__}"
        )

    async def release(self) -> None:
        """Tear down every environment this pool has handed out.

        One failing teardown must not abort the others, so each is isolated;
        the failure is logged rather than swallowed so it is diagnosable.
        """
        await _finish_cleanup(self._release_owned())

    async def _release_owned(self) -> None:
        async with self._release_lock:
            async with self._lifecycle:
                self._release_active = True
                self._generation += 1
                while self._inflight_acquires:
                    await self._lifecycle.wait()
            try:
                await self._cleanup_owned()
            finally:
                async with self._lifecycle:
                    self._release_active = False
                    self._lifecycle.notify_all()

    async def _cleanup_owned(self) -> None:
        failures: list[str] = []
        for env in tuple(self._envs):
            try:
                await _finish_cleanup(env.cleanup())
            except BaseException as exc:
                failures.append(f"{env.workspace}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "environment cleanup failed for %s",
                    env.workspace,
                    exc_info=True,
                )
            else:
                self._envs.remove(env)
        if failures:
            raise OSError(
                "environment pool cleanup failed; retry state retained: "
                + "; ".join(failures)
            )
        if self._owned_delivery_environment is not None:
            await self._owned_delivery_environment.cleanup()
            self._owned_delivery_environment = None

    async def release_env(self, env: Environment) -> None:
        """Release one failed spawn's environment without touching siblings."""
        async with self._release_lock:
            if env not in self._envs:
                return
            try:
                await _finish_cleanup(env.cleanup())
            except BaseException:
                logger.warning(
                    "environment cleanup failed for %s", env.workspace, exc_info=True
                )
                raise
            self._envs.remove(env)

    async def cleanup(self) -> None:
        """Compatibility alias for older callers."""
        await self.release()
