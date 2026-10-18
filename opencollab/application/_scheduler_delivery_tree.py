"""Record agent 0's delivered tree at turn and message boundaries.

Teammate worktree changes reach this tree only after adoption. Serialized
turn-start snapshots bracket one seat's work, with the final delivered patch
closing the last interval. Concurrent turns provide approximate attribution.
Accepted messages also trigger a snapshot, while refused messages do not.

Snapshots retain bounded diff text and mark truncation. Repeated states use
``unchanged_since`` to reference an earlier row. Failed probes record
``diff=None`` and ``probe_error`` so failures remain distinct from unchanged
trees and do not interrupt execution.

Recording is enabled only when the scheduler receives a tree probe.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cap boundary diff text while retaining line attribution and marking truncation.
DELIVERY_DIFF_SNAPSHOT_CHARS = 20_000

def _delivery_diff_files(text: str) -> list[str]:
    """Every path a working-tree diff touches, in diff order, without repeats."""
    paths: list[str] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" b/", 1)
            if len(parts) == 2 and parts[1] not in paths:
                paths.append(parts[1])
    return paths

class SchedulerDeliveryTreeMixin:
    """Records the graded tree at seat boundaries, when a probe is wired."""

    @property
    def delivery_tree_snapshots(self) -> tuple[dict[str, Any], ...]:
        """Every boundary recorded so far, oldest first. Empty when off."""
        return tuple(self._delivery_tree_snapshots)

    async def snapshot_delivery_tree(
        self,
        at: str,
        *,
        aid: int,
        to_aid: int | None = None,
    ) -> None:
        """Record the graded tree at one boundary. No-op without a probe."""
        probe = self._delivery_tree_probe
        if probe is None:
            return
        record: dict[str, Any] = {
            "at": at,
            "aid": aid,
            "role": self._traced_role(aid),
        }
        if to_aid is not None:
            record["to_aid"] = to_aid
            record["to_role"] = self._traced_role(to_aid)
        try:
            text = await probe.diff()
        except Exception as exc:  # noqa: BLE001 — observation must never fail a run
            record["diff"] = None
            record["probe_error"] = f"{type(exc).__name__}: {exc}"
            self._delivery_tree_snapshots.append(record)
            logger.error("delivery-tree snapshot at %s failed: %s", at, exc)
            return
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record["chars"] = len(text)
        record["sha256"] = digest
        already = self._delivery_tree_shas.get(digest)
        if already is not None:
            # Reference the stored tree row instead of repeating unchanged diff text.
            record["unchanged_since"] = already
            self._delivery_tree_snapshots.append(record)
            return
        self._delivery_tree_shas[digest] = len(self._delivery_tree_snapshots)
        record["files"] = _delivery_diff_files(text)
        record["diff"] = text[:DELIVERY_DIFF_SNAPSHOT_CHARS]
        record["truncated"] = len(text) > DELIVERY_DIFF_SNAPSHOT_CHARS
        self._delivery_tree_snapshots.append(record)

__all__ = ["DELIVERY_DIFF_SNAPSHOT_CHARS", "SchedulerDeliveryTreeMixin"]
