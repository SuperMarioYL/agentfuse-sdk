"""The per-task spend ledger — AgentFuse's core primitive.

A :class:`Budget` is a small, thread-safe ledger for one task: it knows the USD
ceiling, tracks the confirmed spend so far, and answers the one question the
fuse needs before every call — *would this next call push me over?*

The flow the wrapper drives (in later stages) is:

1. estimate the upper-bound cost of the next call (see :mod:`agentfuse.pricing`),
2. ask :meth:`Budget.would_exceed` — if ``True`` the call is blocked and
   :class:`~agentfuse.exceptions.BudgetExceeded` is raised *before* anything is
   sent,
3. otherwise send the call, then :meth:`Budget.commit` the real spend read back
   from the response's ``Usage``.

This module is the m1 "running per-task ledger". The pre-call estimate and the
post-call real-cost both flow through it; the actual ``litellm`` wrapping lives
in a later stage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from agentfuse.exceptions import BudgetExceeded


@dataclass(frozen=True)
class BudgetSnapshot:
    """An immutable point-in-time view of a :class:`Budget`, for CLI/demo output."""

    name: str
    ceiling_usd: float
    spent_usd: float
    remaining_usd: float


class Budget:
    """A per-task USD spend ledger with a hard ceiling.

    Args:
        ceiling_usd: The maximum USD this task is allowed to spend. Must be
            non-negative.
        name: A human label for this task, shown in snapshots and the CLI.

    The ledger starts at ``0`` confirmed spend. :meth:`commit` accumulates real
    spend after each call; :meth:`would_exceed` is the pre-call admission check.
    All mutation is guarded by a lock so a multi-threaded agent can share one
    budget safely.
    """

    def __init__(self, ceiling_usd: float, name: str = "task") -> None:
        if ceiling_usd < 0:
            raise ValueError(f"ceiling_usd must be >= 0, got {ceiling_usd!r}")
        self.ceiling_usd: float = float(ceiling_usd)
        self.name: str = name
        self._spent_usd: float = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        """USD confirmed-spent on this task so far."""
        with self._lock:
            return self._spent_usd

    def would_exceed(self, estimated_usd: float) -> bool:
        """Return ``True`` if committing ``estimated_usd`` would cross the ceiling.

        The fuse is strict-greater-than: spending *up to and including* the
        ceiling is allowed; only spend that lands *past* it trips the fuse.
        """
        with self._lock:
            return self._spent_usd + estimated_usd > self.ceiling_usd

    def check(self, estimated_usd: float) -> None:
        """Raise :class:`BudgetExceeded` if ``estimated_usd`` would cross the ceiling.

        This is the pre-call gate in raising form: call it before delegating to
        ``litellm`` so the over-budget call is never sent.
        """
        with self._lock:
            spent = self._spent_usd
            if spent + estimated_usd > self.ceiling_usd:
                raise BudgetExceeded(
                    spent=spent,
                    ceiling=self.ceiling_usd,
                    would_spend=estimated_usd,
                )

    def commit(self, actual_usd: float) -> float:
        """Add ``actual_usd`` of confirmed spend to the ledger; return new total.

        Called after a call returns, with the real cost from its ``Usage``.
        Negative amounts are rejected — the ledger only moves forward.
        """
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd!r}")
        with self._lock:
            self._spent_usd += float(actual_usd)
            return self._spent_usd

    def remaining(self) -> float:
        """USD left before the ceiling. Clamped at ``0`` once spend reaches it."""
        with self._lock:
            return max(0.0, self.ceiling_usd - self._spent_usd)

    def snapshot(self) -> BudgetSnapshot:
        """Return an immutable :class:`BudgetSnapshot` of the current state."""
        with self._lock:
            return BudgetSnapshot(
                name=self.name,
                ceiling_usd=self.ceiling_usd,
                spent_usd=self._spent_usd,
                remaining_usd=max(0.0, self.ceiling_usd - self._spent_usd),
            )

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"Budget(name={self.name!r}, "
                f"spent=${self._spent_usd:.4f}, "
                f"ceiling=${self.ceiling_usd:.2f}, "
                f"remaining=${max(0.0, self.ceiling_usd - self._spent_usd):.4f})"
            )


def gate(budget: Budget, estimated_usd: float) -> None:
    """Pre-call admission check: block the call if it would cross the ceiling.

    A thin functional wrapper over :meth:`Budget.check`. Raises
    :class:`~agentfuse.exceptions.BudgetExceeded` (before any call is delegated to
    ``litellm``) when ``budget.spent + estimated_usd`` would exceed
    ``budget.ceiling_usd``; otherwise returns ``None`` and the caller may proceed.
    """
    budget.check(estimated_usd)
