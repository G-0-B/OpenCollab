"""Preserve legacy text bytes in environment-backed Git patches."""

from __future__ import annotations

import posixpath
import shlex
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.application.async_timeout import await_owned_operation
from opencollab.patches import git_diff_endpoint, git_header_tokens, patch_entries, split_patch_blocks

_PATCH_GIT_TIMEOUT_SECONDS = 900.0

def _directory_patch_path(token: str, side: str, directories: tuple[str, str]) -> str:
    start = int(token.startswith('"'))
    for directory in directories:
        prefix = f"{side}/{directory}/"
        if token.startswith(prefix, start):
            return token[:start] + f"{side}/" + token[start + len(prefix):]
    return token

def _directory_patch_bytes(payload: bytes, before: str, after: str) -> bytes:
    directories = (posixpath.basename(before), posixpath.basename(after))
    blocks = split_patch_blocks(payload.decode("utf-8", errors="surrogateescape"))
    for block in blocks:
        if not block[0].startswith("diff --git "):
            continue
        tokens = git_header_tokens(block[0])
        header = block[0][len("diff --git "):].removesuffix("\n")
        if not header.startswith('"'):
            # Directory comparisons disable renames, so both endpoints have
            # the same relative path. Use that equality to disambiguate an
            # unquoted filename containing Git's own " b/" separator.
            separator = header.find(" b/")
            while separator >= 0:
                candidate = [header[:separator], header[separator + 1:]]
                old, new = (
                    _directory_patch_path(token, side, directories)
                    for token, side in zip(candidate, ("a", "b"))
                )
                if git_diff_endpoint(old, "a") == git_diff_endpoint(new, "b"):
                    tokens = candidate
                    break
                separator = header.find(" b/", separator + 1)
        if len(tokens) == 2:
            old, new = (
                _directory_patch_path(token, side, directories)
                for token, side in zip(tokens, ("a", "b"))
            )
            ending = "\n" if block[0].endswith("\n") else ""
            block[0] = f"diff --git {old} {new}{ending}"
        for index, line in enumerate(block[1:], 1):
            if line.startswith(("@@", "GIT binary patch", "Binary files ")):
                break
            if line.startswith(("--- ", "+++ ")):
                side = "a" if line.startswith("--- ") else "b"
                block[index] = line[:4] + _directory_patch_path(line[4:], side, directories)
    return "".join("".join(block) for block in blocks).encode("utf-8", errors="surrogateescape")

def _complete(result: Any, operation: str, allowed: tuple[int, ...] = (0,)) -> str:
    if (
        getattr(result, "returncode", -1) not in allowed
        or getattr(result, "stdout_truncated", False)
        or getattr(result, "stderr_truncated", False)
    ):
        detail = str(getattr(result, "stderr", "") or "").strip()
        raise RuntimeError(f"{operation} failed: {detail or 'incomplete command result'}")
    return str(getattr(result, "stdout", "") or "")

@asynccontextmanager
async def _patch_temporary(
    environment: Any,
    workspace: str,
    content: str,
    *,
    prefix: str,
    suffix: str = ".tmp",
    cleanup_best_effort: bool = False,
    repository: bool = True,
) -> AsyncIterator[str]:
    if isinstance(environment, LocalEnvironment):
        git_directory = None
        if repository:
            git_directory = _complete(
                await environment.exec_cmd(
                    f"git -C {shlex.quote(workspace)} rev-parse --absolute-git-dir",
                    timeout=_PATCH_GIT_TIMEOUT_SECONDS,
                ),
                "candidate temporary storage",
            ).strip()
        # Git metadata is outside ordinary file snapshots, including linked
        # worktrees. A private directory also owns any transient Git lock files.
        with tempfile.TemporaryDirectory(prefix="opencollab-candidate-", dir=git_directory) as directory:
            owner = LocalEnvironment(directory)
            try:
                yield await owner.write_temp_file(content, prefix=prefix, suffix=suffix)
            finally:
                try:
                    await await_owned_operation(owner.cleanup(), propagate_cancellation=True)
                except Exception:
                    if not cleanup_best_effort:
                        raise
    else:
        path = await environment.write_temp_file(content, prefix=prefix, suffix=suffix)
        try:
            yield path
        finally:
            try:
                await await_owned_operation(environment.remove_file(path), propagate_cancellation=True)
            except Exception:
                if not cleanup_best_effort:
                    raise

async def _binary_patch_block(
    environment: Any,
    workspace: str,
    block: list[str],
    base_revision: str,
    index_file: str | None,
    before_directory: str | None = None,
    captured_blobs: bool = False,
) -> str:
    old_path, new_path = patch_entries("".join(block))[0]
    git = f"git -C {shlex.quote(workspace)}"
    if captured_blobs:
        git = f"GIT_NO_REPLACE_OBJECTS=1 {git}"
    async with _patch_temporary(
        environment, workspace, "", prefix=".candidate-binary-index-",
        repository=before_directory is None,
    ) as temporary:
        directory = temporary + ".parts"
        quoted_directory = shlex.quote(directory)
        old_file, new_file = (posixpath.join(directory, name) for name in ("old", "new"))
        attributes = posixpath.join(directory, "attributes")
        try:
            _complete(await environment.exec_cmd(
                f"mkdir -m 700 -- {quoted_directory} && printf '%s\\n' '* -diff' > {shlex.quote(attributes)}",
                timeout=_PATCH_GIT_TIMEOUT_SECONDS,
            ), "candidate binary temporary storage")
            if before_directory is not None:
                old_file = posixpath.join(before_directory, old_path) if old_path else "/dev/null"
                new_file = posixpath.join(workspace, new_path) if new_path else "/dev/null"
            elif captured_blobs:
                identities = next(line for line in block if line.startswith("index ")).split()[1].split("..")
                for path, identity, destination in zip(
                    (old_path, new_path), identities, (old_file, new_file),
                ):
                    if path:
                        _complete(await environment.exec_cmd(
                            f"{git} cat-file blob {shlex.quote(identity)} > {shlex.quote(destination)}",
                            timeout=_PATCH_GIT_TIMEOUT_SECONDS,
                        ), "worktree binary captured contents")
            elif old_path:
                _complete(await environment.exec_cmd(
                    f"{git} show {shlex.quote(base_revision + ':' + old_path)} > {shlex.quote(old_file)}",
                    timeout=_PATCH_GIT_TIMEOUT_SECONDS,
                ), "candidate binary original contents")
            if new_path and before_directory is None and not captured_blobs:
                captured_index = index_file or temporary
                indexed_git = f"GIT_INDEX_FILE={shlex.quote(captured_index)} {git}"
                if index_file is None:
                    # Git add retains the repository's clean/filter semantics
                    # while leaving the source index untouched.
                    _complete(await environment.exec_cmd(
                        f"{indexed_git} read-tree {shlex.quote(base_revision)} && "
                        f"{indexed_git} add --force -- {shlex.quote(':(literal)' + new_path)}",
                        timeout=_PATCH_GIT_TIMEOUT_SECONDS,
                    ), "candidate binary contents capture")
                _complete(await environment.exec_cmd(
                    f"{indexed_git} show {shlex.quote(':' + new_path)} > {shlex.quote(new_file)}",
                    timeout=_PATCH_GIT_TIMEOUT_SECONDS,
                ), "candidate binary updated contents")
            binary = _complete(await environment.exec_cmd(
                f"GIT_CEILING_DIRECTORIES={shlex.quote(posixpath.dirname(directory))} "
                f"git -C {quoted_directory} -c core.attributesFile={shlex.quote(attributes)} "
                "--no-pager diff --no-index --binary --no-ext-diff --no-textconv --no-color -- "
                f"{shlex.quote(old_file if old_path else '/dev/null')} "
                f"{shlex.quote(new_file if new_path else '/dev/null')}",
                timeout=_PATCH_GIT_TIMEOUT_SECONDS,
            ), "candidate binary patch", allowed=(0, 1))
            marker = "GIT binary patch\n"
            if marker not in binary:
                raise RuntimeError("Git did not produce a binary patch for legacy text")
            # Keep the source paths, full blob identities and mode metadata.
            # Git generates only the opaque payload in the private directory.
            header = block[:next(index for index, line in enumerate(block) if line.startswith("--- "))]
            return "".join(header) + marker + binary.split(marker, 1)[1]
        finally:
            _complete(await await_owned_operation(environment.exec_cmd(
                f"rm -rf -- {quoted_directory}", timeout=120,
            ), propagate_cancellation=True), "candidate binary temporary cleanup")

async def _complete_patch(
    result: Any,
    operation: str,
    environment: Any,
    workspace: str,
    base_revision: str,
    index_file: str | None,
    *,
    allowed: tuple[int, ...] = (0,),
    before_directory: str | None = None,
    captured_blobs: bool = False,
) -> str:
    display = _complete(result, operation, allowed)
    payload = getattr(result, "stdout_bytes", None)
    if payload is None:
        payload = display.encode("utf-8")
    if before_directory is not None:
        payload = _directory_patch_bytes(payload, before_directory, workspace)
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        blocks = split_patch_blocks(payload.decode("utf-8", errors="surrogateescape"))
    parts = []
    for block in blocks:
        text = "".join(block)
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            text = await _binary_patch_block(
                environment, workspace, block, base_revision, index_file, before_directory,
                captured_blobs,
            )
        parts.append(text)
    return "".join(parts)
