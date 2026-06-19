"""Opt-in append-only spend record (m6).

v0.1's ``agentfuse status`` had nothing to read: the per-task ledger lives and
dies with the process, so once a run finished its real spend was gone. This
module adds a *minimal*, opt-in record — one JSONL line per finished task — so
that ``agentfuse status --log <path>`` can summarise the last task's REAL spend
across process boundaries.

Deliberately scoped to **execution-adjacent record-keeping**, NOT a standalone
cost dashboard: there is no visualization, no live monitoring service, no
cross-run budget rollover. It is an append-only log of facts (name, ceiling,
spent, whether the fuse tripped, timestamp) that the existing CLI can tail.

Usage::

    from agentfuse import Fuse, record_task

    with Fuse(max_spend_usd=5.0, name="nightly-scan") as budget:
        try:
            run_my_agent()
        finally:
            record_task(budget, tripped=..., log_path="~/.agentfuse/spend.jsonl")

The store never raises on a bad path or a serialization hiccup — recording spend
must never crash a task that already did its work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentfuse.budget import Budget

logger = logging.getLogger("agentfuse.store")

# Bump if the on-disk line shape ever changes incompatibly.
RECORD_VERSION = 1


@dataclass(frozen=True)
class TaskRecord:
    """One finished task's spend, as written to / read from the JSONL store."""

    name: str
    ceiling_usd: float
    spent_usd: float
    tripped: bool
    ts: str
    ceiling_tokens: int | None = None
    spent_tokens: int = 0
    v: int = RECORD_VERSION

    def to_json(self) -> str:
        """Serialize to a single compact JSON line (no trailing newline)."""
        return json.dumps(
            {
                "v": self.v,
                "name": self.name,
                "ceiling_usd": self.ceiling_usd,
                "spent_usd": self.spent_usd,
                "ceiling_tokens": self.ceiling_tokens,
                "spent_tokens": self.spent_tokens,
                "tripped": self.tripped,
                "ts": self.ts,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskRecord":
        return cls(
            name=str(d.get("name", "task")),
            ceiling_usd=float(d.get("ceiling_usd", 0.0)),
            spent_usd=float(d.get("spent_usd", 0.0)),
            tripped=bool(d.get("tripped", False)),
            ts=str(d.get("ts", "")),
            ceiling_tokens=(
                int(d["ceiling_tokens"]) if d.get("ceiling_tokens") is not None else None
            ),
            spent_tokens=int(d.get("spent_tokens", 0) or 0),
            v=int(d.get("v", RECORD_VERSION)),
        )


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_task(
    budget: Budget,
    *,
    tripped: bool,
    log_path: str | Path,
    ts: str | None = None,
) -> TaskRecord | None:
    """Append one line summarising ``budget``'s final spend to ``log_path``.

    Reads the task's final state straight from the :class:`Budget` snapshot, so
    the recorded ``spent`` is the REAL accumulated spend. Creates parent
    directories as needed. Returns the :class:`TaskRecord` written, or ``None``
    if writing failed (a record write must never crash a finished task).

    Args:
        budget: The task's :class:`Budget` (after the ``with Fuse(...)`` block).
        tripped: Whether the fuse tripped on this task (caller knows; typically
            ``True`` if a :class:`~agentfuse.exceptions.BudgetExceeded` was
            raised inside the scope).
        log_path: JSONL file to append to (``~`` is expanded).
        ts: Optional ISO timestamp override (defaults to UTC now).
    """
    snap = budget.snapshot()
    record = TaskRecord(
        name=snap.name,
        ceiling_usd=snap.ceiling_usd,
        spent_usd=snap.spent_usd,
        tripped=bool(tripped),
        ts=ts or _now_iso(),
        ceiling_tokens=snap.ceiling_tokens,
        spent_tokens=snap.spent_tokens,
    )
    path = Path(log_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")
    except OSError as exc:
        logger.warning("could not append spend record to %s: %s", path, exc)
        return None
    return record


def read_records(log_path: str | Path) -> list[TaskRecord]:
    """Read all task records from ``log_path`` (oldest first).

    Skips blank lines and any line that fails to parse, so a partially-written
    or hand-edited log never raises. Returns ``[]`` if the file does not exist.
    """
    path = Path(log_path).expanduser()
    if not path.exists():
        return []
    records: list[TaskRecord] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read spend record %s: %s", path, exc)
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(TaskRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug("skipping malformed spend-record line: %s (%s)", line, exc)
    return records


def last_record(log_path: str | Path) -> TaskRecord | None:
    """Return the most recent :class:`TaskRecord`, or ``None`` if the log is empty."""
    records = read_records(log_path)
    return records[-1] if records else None


__all__ = [
    "TaskRecord",
    "RECORD_VERSION",
    "record_task",
    "read_records",
    "last_record",
]
