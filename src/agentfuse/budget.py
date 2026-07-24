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

import logging
import math
import threading
from dataclasses import dataclass
from typing import Callable

from agentfuse.exceptions import BudgetExceeded

logger = logging.getLogger("agentfuse.budget")

# A user-supplied trip hook. It receives the fully-structured BudgetExceeded that
# the fuse is *about* to raise (so it can read spent / ceiling / would_spend /
# limit_kind / token-ledger fields) and is invoked BEFORE the over-budget call
# is delegated to litellm — the same moment the stderr trip banner fires. Any
# exception raised by the callback is logged and swallowed: a user hook must
# NEVER change whether the over-budget call is blocked. ``None`` (default)
# means no hook — the fuse simply raises, exactly as it did before v0.4.0.
TripCallback = Callable[["BudgetExceeded"], None]


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
    pending_usd: float = 0.0
    pending_tokens: int = 0


class Reservation:
    """Handle to a pending spend reservation created by :meth:`Budget.reserve`.

    Threading the reservation through ``check -> commit/release`` makes the
    pre-call gate atomic across the awaited LLM call: a concurrent caller that
    gates while this call is still in-flight sees the reserved amount in the
    budget's pending balance, so two concurrent estimates cannot both pass
    against the still-uncommitted ledger and overshoot the ceiling. Pass the
    handle back to :meth:`Budget.commit` (on success — releases the reservation
    and commits the real cost) or :meth:`Budget.release` (on the error path —
    releases the reservation without committing). Failing to do either leaks the
    reservation into ``pending`` until the handle is eventually released.

    The handle carries the reserved ``(estimated_usd, estimated_tokens)`` so
    releases are attributable to the specific concurrent call that reserved them
    (the budget cannot infer which reservation a late commit corresponds to once
    completion order diverges from reservation order).
    """

    __slots__ = ("estimated_usd", "estimated_tokens", "_active")

    def __init__(self, estimated_usd: float, estimated_tokens: int) -> None:
        self.estimated_usd = float(estimated_usd)
        self.estimated_tokens = int(estimated_tokens)
        self._active = True

    @property
    def active(self) -> bool:
        """``True`` until :meth:`Budget.commit` / :meth:`Budget.release` consumes it."""
        return self._active

    def __repr__(self) -> str:
        return (
            f"Reservation(estimated_usd=${self.estimated_usd:.6f}, "
            f"estimated_tokens={self.estimated_tokens}, active={self._active})"
        )


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
        on_trip: Optional callback invoked with the :class:`BudgetExceeded` the
            fuse is *about* to raise, right before the over-budget call is
            blocked. It sees the same structured trip context (``spent`` /
            ``ceiling`` / ``would_spend`` / ``limit_kind`` / token-ledger fields)
            the trip banner uses, so an operator can wire the trip event into
            their own Slack/Feishu webhook, audit DB, or metric counter in-process.
            The hook is fail-soft: any exception it raises is logged and
            swallowed, so a user callback can never change whether the
            over-budget call is blocked. ``None`` (default) means no hook.

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
        on_trip: TripCallback | None = None,
    ) -> None:
        self.ceiling_usd: float = _require_finite_positive(ceiling_usd, "ceiling_usd")
        self.name: str = name
        # Policy carried alongside the ledger so the gate (which only has the
        # active Budget in hand) can honour it when a model is unpriced.
        self.on_unpriced: str = on_unpriced
        # In-process trip hook (v0.4.0). Fail-soft: see _invoke_on_trip.
        self.on_trip: TripCallback | None = on_trip

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
        # v0.5.0: a pending-reserve balance so the pre-call gate is atomic
        # across the awaited LLM call. `reserve()` adds an estimate here (so a
        # concurrent caller sees spent+pending+its_estimate against the
        # ceiling); `commit()`/`release()` subtract the matching reservation.
        self._pending_usd: float = 0.0
        self._pending_tokens: int = 0
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

    @property
    def pending_usd(self) -> float:
        """USD reserved by in-flight calls (not yet committed or released)."""
        with self._lock:
            return self._pending_usd

    @property
    def pending_tokens(self) -> int:
        """Tokens reserved by in-flight calls (not yet committed or released)."""
        with self._lock:
            return self._pending_tokens

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
        """Return the first tripped ceiling's label, or ``None``. Caller holds the lock.

        Considers the **pending** reserve from in-flight calls alongside the
        confirmed spend, so a concurrent caller that gates while another call is
        still awaited cannot also pass against the still-uncommitted ledger
        (the v0.5.0 fix-budget-check-commit-race fix).
        """
        if (
            self.single_call_ceiling is not None
            and estimated_usd > self.single_call_ceiling
        ):
            return "single_call"
        if self._spent_usd + self._pending_usd + estimated_usd > self.ceiling_usd:
            return "usd"
        if (
            self.ceiling_tokens is not None
            and self._spent_tokens + self._pending_tokens + estimated_tokens > self.ceiling_tokens
        ):
            return "tokens"
        return None

    def _build_exceeded(
        self, reason: str, estimated_usd: float, estimated_tokens: int
    ) -> BudgetExceeded:
        """Construct the :class:`BudgetExceeded` for a tripped ceiling. Caller holds the lock."""
        if reason == "single_call":
            return BudgetExceeded(
                spent=self._spent_usd,
                ceiling=self.single_call_ceiling,  # type: ignore[arg-type]
                would_spend=estimated_usd,
                limit_kind="single_call",
            )
        if reason == "tokens":
            return BudgetExceeded(
                spent=self._spent_usd,
                ceiling=self.ceiling_usd,
                would_spend=estimated_usd,
                limit_kind="tokens",
                spent_tokens=self._spent_tokens,
                ceiling_tokens=self.ceiling_tokens,
                would_spend_tokens=estimated_tokens,
            )
        return BudgetExceeded(
            spent=self._spent_usd,
            ceiling=self.ceiling_usd,
            would_spend=estimated_usd,
            limit_kind="usd",
        )

    def _invoke_on_trip(self, err: BudgetExceeded) -> None:
        """Run the user-supplied ``on_trip`` hook fail-soft, OUTSIDE the ledger lock.

        The hook sees the fully-structured trip exception (the same fields the
        stderr banner composes its body from). Any exception it raises is logged
        at debug and swallowed — a user callback must never change whether the
        over-budget call is blocked, so the fuse still raises ``err`` afterwards.
        """
        if self.on_trip is None:
            return
        try:
            self.on_trip(err)
        except Exception:  # noqa: BLE001 - never let a user hook break the gate
            logger.debug(
                "on_trip callback raised for task %r; swallowing (fuse still trips)",
                self.name,
                exc_info=True,
            )

    def check(self, estimated_usd: float, estimated_tokens: int = 0) -> None:
        """Raise :class:`BudgetExceeded` if this estimate would trip any ceiling.

        This is the pre-call gate in raising form: call it before delegating to
        ``litellm`` so the over-budget call is never sent. Whichever ceiling
        trips first (per-call USD cap, cumulative USD, or cumulative tokens)
        determines the structured fields on the raised exception.

        .. note::
            ``check`` is a **non-mutating peek**: it considers the current
            ``spent`` + ``pending`` reserve but reserves nothing itself. The
            race-free gate (the v0.5.0 fix-budget-check-commit-race fix) is
            :meth:`reserve`, which the litellm wrapper thread through
            :meth:`commit` / :meth:`release` so concurrent fan-out cannot both
            pass against the still-uncommitted ledger. Use ``check`` /
            ``would_exceed`` only for a side-effect-free peek at the gate state.

        If an ``on_trip`` callback is set on this budget (v0.4.0), it is invoked
        with the about-to-be-raised :class:`BudgetExceeded` — fail-soft, outside
        the ledger lock — *before* the exception is raised, so an operator can
        wire the trip event into their own alerting/audit/metrics in-process.
        """
        with self._lock:
            reason = self._tripped_reason(estimated_usd, estimated_tokens)
            if reason is None:
                return
            err = self._build_exceeded(reason, estimated_usd, estimated_tokens)
        # Lock released before the user hook runs, so a callback that touches
        # this budget (e.g. reads .snapshot()) cannot deadlock against the gate.
        self._invoke_on_trip(err)
        raise err

    def reserve(
        self, estimated_usd: float, estimated_tokens: int = 0
    ) -> Reservation:
        """Reserve ``estimated_usd`` / ``estimated_tokens`` and return a handle, or raise.

        Like :meth:`check` but **mutating**: on the pass path it adds the estimate
        to the budget's ``pending`` reserve (so a concurrent caller that gates
        while this call is still in-flight sees ``spent + pending +
        its_estimate`` against the ceiling and cannot also pass) and returns a
        :class:`Reservation` handle. The handle MUST be handed to
        :meth:`commit` (on success — releases the reservation and commits the real
        cost) or :meth:`release` (on the error path — releases without
        committing); otherwise the reservation leaks into ``pending``. On the
        trip path it raises :class:`BudgetExceeded` (after the ``on_trip`` hook),
        reserving nothing, exactly like :meth:`check`.

        This closes the Budget.check→await→commit race: before v0.5.0,
        ``check`` released the lock before ``acompletion`` awaited the delegate,
        so two concurrent calls both gated against the still-uncommitted ledger
        and both committed, overshooting by up to ``(concurrency * per-call
        estimate)``. ``reserve`` holds the estimate in ``pending`` across the
        await without holding the lock across the HTTP call, and ``commit`` /
        ``release`` settle it.
        """
        with self._lock:
            reason = self._tripped_reason(estimated_usd, estimated_tokens)
            if reason is None:
                self._pending_usd += float(estimated_usd)
                self._pending_tokens += int(estimated_tokens)
                return Reservation(estimated_usd, estimated_tokens)
            err = self._build_exceeded(reason, estimated_usd, estimated_tokens)
        # Lock released before the user hook runs (mirrors check()).
        self._invoke_on_trip(err)
        raise err

    def _release_locked(self, reservation: "Reservation | None") -> None:
        """Release a pending reservation. Caller MUST hold ``self._lock``.

        Idempotent: ``None`` or an already-released handle is a no-op. Subtracts
        the reservation's reserved estimate from ``pending`` (clamped at 0 so a
        misattributed handle cannot drive it negative) and marks the handle
        inactive.
        """
        if reservation is None or not reservation._active:
            return
        self._pending_usd = max(0.0, self._pending_usd - reservation.estimated_usd)
        self._pending_tokens = max(0, self._pending_tokens - reservation.estimated_tokens)
        reservation._active = False

    def commit(
        self,
        actual_usd: float,
        actual_tokens: int = 0,
        *,
        reservation: "Reservation | None" = None,
    ) -> float:
        """Add confirmed spend to the ledger; return the new USD total.

        Called after a call returns, with the real cost (and, optionally, real
        token count) from its ``Usage``. Negative amounts are rejected — the
        ledger only moves forward. When a ``reservation`` handle from
        :meth:`reserve` is supplied (the v0.5.0 race-free gate path used by the
        litellm wrapper), it is released first so the pending estimate is
        settled atomically with the real-cost commit (concurrent callers stop
        seeing it as in-flight). Without a handle the commit is the legacy
        plain accumulate (backward-compatible for callers not on the reserve
        path, e.g. tests and the streaming fallback when no usage was emitted).
        """
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd!r}")
        if actual_tokens < 0:
            raise ValueError(f"actual_tokens must be >= 0, got {actual_tokens!r}")
        with self._lock:
            if reservation is not None:
                self._release_locked(reservation)
            self._spent_usd += float(actual_usd)
            self._spent_tokens += int(actual_tokens)
            return self._spent_usd

    def release(self, reservation: "Reservation | None") -> None:
        """Release a pending reservation WITHOUT committing spend (error path).

        Called when a gated call raised before it could commit (e.g. the delegate
        raised) so the reserved estimate does not stay pinned in ``pending``
        forever and block every subsequent call. Idempotent: ``None`` or an
        already-released handle is a no-op.
        """
        with self._lock:
            self._release_locked(reservation)

    def remaining(self) -> float:
        """USD left before the ceiling, accounting for in-flight reservations.

        Clamped at ``0`` once confirmed spend + pending reserve reaches the
        ceiling. (Pending-aware since v0.5.0 so the value reflects concurrent
        in-flight calls, not just committed spend.)
        """
        with self._lock:
            return max(0.0, self.ceiling_usd - self._spent_usd - self._pending_usd)

    def remaining_tokens(self) -> int | None:
        """Tokens left before the token ceiling (pending-aware), or ``None``."""
        with self._lock:
            if self.ceiling_tokens is None:
                return None
            return max(0, self.ceiling_tokens - self._spent_tokens - self._pending_tokens)

    def snapshot(self) -> BudgetSnapshot:
        """Return an immutable :class:`BudgetSnapshot` of the current state."""
        with self._lock:
            return BudgetSnapshot(
                name=self.name,
                ceiling_usd=self.ceiling_usd,
                spent_usd=self._spent_usd,
                remaining_usd=max(0.0, self.ceiling_usd - self._spent_usd - self._pending_usd),
                ceiling_tokens=self.ceiling_tokens,
                spent_tokens=self._spent_tokens,
                single_call_ceiling=self.single_call_ceiling,
                pending_usd=self._pending_usd,
                pending_tokens=self._pending_tokens,
            )

    def __repr__(self) -> str:
        with self._lock:
            extra = ""
            if self._pending_usd > 0 or self._pending_tokens > 0:
                extra += (
                    f", pending=${self._pending_usd:.4f}/{self._pending_tokens}t"
                )
            if self.ceiling_tokens is not None:
                extra += f", tokens={self._spent_tokens}/{self.ceiling_tokens}"
            if self.single_call_ceiling is not None:
                extra += f", per_call<=${self.single_call_ceiling:.2f}"
            return (
                f"Budget(name={self.name!r}, "
                f"spent=${self._spent_usd:.4f}, "
                f"ceiling=${self.ceiling_usd:.2f}, "
                f"remaining=${max(0.0, self.ceiling_usd - self._spent_usd - self._pending_usd):.4f}{extra})"
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
