"""Initialize worktree submodules from already available local sources."""

from __future__ import annotations

import os
import posixpath
from collections.abc import Awaitable, Callable

from opencollab.adapters._env_base import ExecResult


async def _initialize_source_available_submodules(
    source_repository: str,
    target_repository: str,
    *,
    git_in: Callable[..., Awaitable[ExecResult]],
    source_prefix: str = "",
    require_initialized: bool = True,
    local_filesystem: bool = True,
    git_marker_exists: Callable[[str], Awaitable[bool]] | None = None,
) -> None:
    tree = _SourceSubmoduleTree(
        git_in, require_initialized=require_initialized, local_filesystem=local_filesystem,
        git_marker_exists=git_marker_exists,
    )
    await tree._initialize_source_submodule_tree(
        source_repository,
        target_repository,
        source_prefix=source_prefix,
        active_sources=set(),
    )

class _SourceSubmoduleTree:
    def __init__(
        self, git_in: Callable[..., Awaitable[ExecResult]], *,
        require_initialized: bool, local_filesystem: bool,
        git_marker_exists: Callable[[str], Awaitable[bool]] | None,
    ) -> None:
        self._git_in = git_in
        self._require_initialized = require_initialized
        self._local_filesystem = local_filesystem
        self._git_marker_exists = git_marker_exists
        self._path = os.path if local_filesystem else posixpath

    def _normalize_path(self, path: str) -> str:
        return os.path.realpath(path) if self._local_filesystem else posixpath.normpath(path)

    async def _configured_source_submodules(
        self,
        source_repository: str,
        target_repository: str,
        *,
        source_prefix: str,
    ) -> list[tuple[str, str, str]]:
        configuration_repository = source_repository if self._local_filesystem else target_repository
        modules_file = self._path.join(configuration_repository, ".gitmodules")
        if self._local_filesystem and not os.path.isfile(modules_file):
            return []
        configured = await self._git_in(
            configuration_repository,
            "config",
            "--null",
            "--file",
            modules_file,
            "--get-regexp",
            r"^submodule\..*\.path$",
        )
        if configured.returncode == 1:
            return []
        if (
            configured.returncode != 0
            or configured.stdout_truncated
            or configured.stderr_truncated
        ):
            raise RuntimeError("cannot inspect Git submodule configuration")

        submodules: list[tuple[str, str, str]] = []
        records = configured.stdout.split("\0")
        if records.pop() != "":
            raise RuntimeError("invalid Git submodule path configuration")
        for record in records:
            try:
                key, configured_path = record.split("\n", 1)
            except ValueError as exc:
                raise RuntimeError("invalid Git submodule path configuration") from exc
            prefix = "submodule."
            suffix = ".path"
            if not key.startswith(prefix) or not key.endswith(suffix):
                raise RuntimeError("invalid Git submodule path configuration")
            name = key[len(prefix) : -len(suffix)]
            normalized_path = configured_path.replace("\\", "/").strip("/")
            if (
                not normalized_path
                or normalized_path == ".."
                or normalized_path.startswith("../")
            ):
                raise RuntimeError("invalid Git submodule path configuration")
            if source_prefix and not (
                normalized_path == source_prefix
                or normalized_path.startswith(f"{source_prefix}/")
            ):
                continue
            source_module = self._normalize_path(
                self._path.join(source_repository, *normalized_path.split("/"))
            )
            try:
                contained = self._path.commonpath((source_repository, source_module))
            except ValueError:
                contained = ""
            if contained != source_repository:
                raise RuntimeError(
                    f"Git submodule is not initialized in source workspace: {configured_path}"
                )
            available = None
            if not self._local_filesystem:
                available = await self._git_in(source_module, "rev-parse", "--show-toplevel")
            initialized = (
                os.path.lexists(os.path.join(source_module, ".git"))
                if self._local_filesystem else (
                    available is not None
                    and available.returncode == 0
                    and not available.stdout_truncated
                    and not available.stderr_truncated
                    and self._normalize_path(available.stdout.removesuffix("\n")) == source_module
                )
            )
            if not self._require_initialized and not initialized:
                if not self._local_filesystem:
                    if self._git_marker_exists is None:
                        raise RuntimeError("cannot inspect source submodule Git metadata")
                    if await self._git_marker_exists(source_module):
                        raise RuntimeError(
                            f"Git submodule is not initialized in source workspace: {configured_path}"
                        )
                state = await self._git_in(
                    configuration_repository, "submodule", "status", "--", f":(literal){normalized_path}",
                )
                records = state.stdout.splitlines()
                if (
                    state.returncode != 0
                    or state.stdout_truncated
                    or state.stderr_truncated
                    or len(records) != 1
                    or not records[0]
                    or records[0][0] not in "- +"
                ):
                    raise RuntimeError(f"cannot inspect Git submodule initialization: {configured_path}")
                if records[0].startswith("-"):
                    continue
            if self._local_filesystem and not os.path.isdir(source_module):
                raise RuntimeError(
                    f"Git submodule is not initialized in source workspace: {configured_path}"
                )
            if available is None:
                available = await self._git_in(source_module, "rev-parse", "--show-toplevel")
            if (
                available.returncode != 0
                or available.stdout_truncated
                or available.stderr_truncated
                or self._normalize_path(available.stdout.removesuffix("\n")) != source_module
            ):
                raise RuntimeError(
                    f"Git submodule is not initialized in source workspace: {configured_path}"
                )
            submodules.append((name, normalized_path, source_module))
        return submodules

    async def _initialize_source_submodule_tree(
        self,
        source_repository: str,
        target_repository: str,
        *,
        source_prefix: str,
        active_sources: set[str],
    ) -> None:
        source_repository = self._normalize_path(source_repository)
        if not self._local_filesystem:
            resolved = await self._git_in(source_repository, "rev-parse", "--show-toplevel")
            if resolved.returncode != 0 or resolved.stdout_truncated or resolved.stderr_truncated:
                raise RuntimeError("cannot resolve Git submodule source repository")
            source_repository = self._normalize_path(resolved.stdout.removesuffix("\n"))
        if source_repository in active_sources:
            raise RuntimeError("cyclic initialized Git submodule source")
        active_sources.add(source_repository)
        try:
            submodules = await self._configured_source_submodules(
                source_repository,
                target_repository,
                source_prefix=source_prefix,
            )
            for name, normalized_path, source_module in submodules:
                updated = await self._git_in(
                    target_repository,
                    "-c",
                    "protocol.allow=never",
                    "-c",
                    "protocol.file.allow=always",
                    "-c",
                    f"submodule.{name}.url={source_module}",
                    "submodule",
                    "update",
                    "--init",
                    "--no-fetch",
                    "--",
                    normalized_path,
                )
                if (
                    updated.returncode != 0
                    or updated.stdout_truncated
                    or updated.stderr_truncated
                ):
                    detail = updated.stderr.strip() or "submodule checkout failed"
                    raise RuntimeError(
                        "cannot initialize source-available submodule "
                        f"{normalized_path}: {detail}"
                    )
                target_module = self._normalize_path(
                    self._path.join(
                        target_repository,
                        *normalized_path.split("/"),
                    )
                )
                await self._initialize_source_submodule_tree(
                    source_module,
                    target_module,
                    source_prefix="",
                    active_sources=active_sources,
                )
        finally:
            active_sources.remove(source_repository)
