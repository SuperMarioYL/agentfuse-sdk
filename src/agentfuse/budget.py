"""The per-task spend ledger — AgentFuse's core primitive.

A :class:`Budget` is a small, thread-safe ledger for one task: it knows the USD
ceiling (and, optionally, a token ceiling and a per-call hard cap), tracks the
confirmed spend so far, and answers the one question the fuse needs before every
call — *would this next call push me over?*

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

Three ceilings can trip the fuse, **first-to-trip wins**:

* ``ceiling_usd`` — cumulative USD spend (the original, always required),
* ``ceiling_tokens`` — cumulative token spend (optional, closes the m2 spec gap),
* ``single_call_ceiling`` — a per-call USD hard cap so one oversized prompt
  cannot blow the whole budget in a single shot (optional).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from agentfuse.exceptions import BudgetExceeded


def _require_finite_positive(value: float, label: str) -> float:
    """Validate that ``value`` is a finite, strictly-positive number.

    A ceiling of ``0``, ``NaN``, or ``±inf`` produces a fuse state
    indistinguishable from "off" (or worse, perverts the arithmetic), so we
    reject them up front rather than silently disabling the circuit-breaker.
    """
    f = float(value)
    if not math.isfinite(f) or f <= 0:
        raise ValueError(f"{label} must be a finite number > 0, got {value!r}")
    return f


@dataclass(frozen=True)
class BudgetSnapshot:
    """An immutable point-in-time view of a :class:`Budget`, for CLI/demo output."""

    name: str
    ceiling_usd: float
    spent_usd: float
    remaining_usd: float
    ceiling_tokens: int | None = None
    spent_tokens: int = 0
    single_call_ceiling: float | None = None


class Budget:
    """A per-task spend ledger with hard ceilings.

    Args:
        ceiling_usd: The maximum USD this task is allowed to spend. Must be a
            finite number greater than 0.
        name: A human label for this task, shown in snapshots and the CLI.
        ceiling_tokens: Optional maximum cumulative tokens (prompt + completion
            estimate) for this task. ``None`` disables the token ceiling. When
            set it must be a positive integer.
        single_call_ceiling: Optional per-call USD hard cap. A single call whose
            estimate alone exceeds this trips the fuse independently of the
            cumulative ledger. ``None`` disables it. When set it must be a finite
            number greater than 0.

    The ledger starts at ``0`` confirmed spend. :meth:`commit` accumulates real
    spend after each call; :meth:`would_exceed` is the pre-call admission check.
    All mutation is guarded by a lock so a multi-threaded agent can share one
    budget safely.
    """

    def __init__(
        self,
        ceiling_usd: float,
        name: str = "task",
        *,
        ceiling_tokens: int | None = None,
        single_call_ceiling: float | None = None,
        on_unpriced: str = "block",
    ) -> None:
        self.ceiling_usd: float = _require_finite_positive(ceiling_usd, "ceiling_usd")
        self.name: str = name
        # Policy carried alongside the ledger so the gate (which only has the
        # active Budget in hand) can honour it when a model is unpriced.
        self.on_unpriced: str = on_unpriced

        if ceiling_tokens is not None:
            ct = int(ceiling_tokens)
            if ct <= 0:
                raise ValueError(f"ceiling_tokens must be a positive int, got {ceiling_tokens!r}")
            self.ceiling_tokens: int | None = ct
        else:
            self.ceiling_tokens = None

        if single_call_ceiling is not None:
            self.single_call_ceiling: float | None = _require_finite_positive(
                single_call_ceiling, "single_call_ceiling"
            )
        else:
            self.single_call_ceiling = None

        self._spent_usd: float = 0.0
        self._spent_tokens: int = 0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        """USD confirmed-spent on this task so far."""
        with self._lock:
            return self._spent_usd

    @property
    def spent_tokens(self) -> int:
        """Tokens confirmed-spent on this task so far."""
        with self._lock:
            return self._spent_tokens

    def would_exceed(self, estimated_usd: float, estimated_tokens: int = 0) -> bool:
        """Return ``True`` if committing this estimate would trip any ceiling.

        Checks the cumulative USD ceiling, the optional cumulative token ceiling,
        and the optional per-call USD cap — **first-to-trip wins**. The fuse is
        strict-greater-than: spending *up to and including* a ceiling is allowed;
        only spend that lands *past* it trips the fuse.
        """
        with self._lock:
            return self._tripped_reason(estimated_usd, estimated_tokens) is not None

    def _tripped_reason(self, estimated_usd: float, estimated_tokens: int) -> str | None:
        """Return the first tripped ceiling's label, or ``None``. Caller holds the lock."""
        if (
            self.single_call_ceiling is not None
            and estimated_usd > self.single_call_ceiling
        ):
            return "single_call"
        if self._spent_usd + estimated_usd > self.ceiling_usd:
            return "usd"
        if (
            self.ceiling_tokens is not None
            and self._spent_tokens + estimated_tokens > self.ceiling_tokens
        ):
            return "tokens"
        return None

    def check(self, estimated_usd: float, estimated_tokens: int = 0) -> None:
        """Raise :class:`BudgetExceeded` if this estimate would trip any ceiling.

        This is the pre-call gate in raising form: call it before delegating to
        ``litellm`` so the over-budget call is never sent. Whichever ceiling
        trips first (per-call USD cap, cumulative USD, or cumulative tokens)
        determines the structured fields on the raised exception.
        """
        with self._lock:
            reason = self._tripped_reason(estimated_usd, estimated_tokens)
            if reason is None:
                return
            if reason == "single_call":
                raise BudgetExceeded(
                    spent=self._spent_usd,
                    ceiling=self.single_call_ceiling,  # type: ignore[arg-type]
                    would_spend=estimated_usd,
                    limit_kind="single_call",
                )
            if reason == "tokens":
                raise BudgetExceeded(
                    spent=self._spent_usd,
                    ceiling=self.ceiling_usd,
                    would_spend=estimated_usd,
                    limit_kind="tokens",
                    spent_tokens=self._spent_tokens,
                    ceiling_tokens=self.ceiling_tokens,
                    would_spend_tokens=estimated_tokens,
                )
            raise BudgetExceeded(
                spent=self._spent_usd,
                ceiling=self.ceiling_usd,
                would_spend=estimated_usd,
                limit_kind="usd",
            )

    def commit(self, actual_usd: float, actual_tokens: int = 0) -> float:
        """Add confirmed spend to the ledger; return the new USD total.

        Called after a call returns, with the real cost (and, optionally, real
        token count) from its ``Usage``. Negative amounts are rejected — the
        ledger only moves forward.
        """
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd!r}")
        if actual_tokens < 0:
            raise ValueError(f"actual_tokens must be >= 0, got {actual_tokens!r}")
        with self._lock:
            self._spent_usd += float(actual_usd)
            self._spent_tokens += int(actual_tokens)
            return self._spent_usd

    def remaining(self) -> float:
        """USD left before the ceiling. Clamped at ``0`` once spend reaches it."""
        with self._lock:
            return max(0.0, self.ceiling_usd - self._spent_usd)

    def remaining_tokens(self) -> int | None:
        """Tokens left before the token ceiling, or ``None`` if no token ceiling."""
        with self._lock:
            if self.ceiling_tokens is None:
                return None
            return max(0, self.ceiling_tokens - self._spent_tokens)

    def snapshot(self) -> BudgetSnapshot:
        """Return an immutable :class:`BudgetSnapshot` of the current state."""
        with self._lock:
            return BudgetSnapshot(
                name=self.name,
                ceiling_usd=self.ceiling_usd,
                spent_usd=self._spent_usd,
                remaining_usd=max(0.0, self.ceiling_usd - self._spent_usd),
                ceiling_tokens=self.ceiling_tokens,
                spent_tokens=self._spent_tokens,
                single_call_ceiling=self.single_call_ceiling,
            )

    def __repr__(self) -> str:
        with self._lock:
            extra = ""
            if self.ceiling_tokens is not None:
                extra += f", tokens={self._spent_tokens}/{self.ceiling_tokens}"
            if self.single_call_ceiling is not None:
                extra += f", per_call<=${self.single_call_ceiling:.2f}"
            return (
                f"Budget(name={self.name!r}, "
                f"spent=${self._spent_usd:.4f}, "
                f"ceiling=${self.ceiling_usd:.2f}, "
                f"remaining=${max(0.0, self.ceiling_usd - self._spent_usd):.4f}{extra})"
            )


def gate(budget: Budget, estimated_usd: float, estimated_tokens: int = 0) -> None:
    """Pre-call admission check: block the call if it would trip any ceiling.

    A thin functional wrapper over :meth:`Budget.check`. Raises
    :class:`~agentfuse.exceptions.BudgetExceeded` (before any call is delegated to
    ``litellm``) when the estimate would cross the USD ceiling, the token ceiling,
    or the per-call cap — whichever trips first; otherwise returns ``None`` and
    the caller may proceed.
    """
    budget.check(estimated_usd, estimated_tokens)
