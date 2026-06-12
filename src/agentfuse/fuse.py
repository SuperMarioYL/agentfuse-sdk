"""The enforcement layer + ergonomic public API.

This module binds a per-task :class:`~agentfuse.budget.Budget` to the current
execution context (a :mod:`contextvars` slot, so it is async- and thread-safe)
and exposes the surfaces an agent author actually touches:

* :func:`task` — a context manager (``with task(ceiling_usd=5.0): ...``) that
  scopes an active budget for every gated LLM call made inside it.
* :class:`Fuse` — the public name used in the plan's two-line quickstart
  (``with Fuse(max_spend_usd=5.0): ...``). It is a thin alias over :func:`task`.
* :func:`fuse` / :func:`fused` — a decorator (``@fuse(ceiling_usd=5.0)``) that
  wraps a function so every gated call made while it runs is bound to a fresh
  per-call budget.
* :func:`current_budget` — the active :class:`Budget`, or ``None`` outside a task.
* :func:`gate` — the core pre-call check. It estimates the call's upper-bound
  cost, asks the active budget whether that would cross the ceiling, prints the
  user-facing ``🔌 FUSE TRIPPED`` banner if so, then re-raises
  :class:`~agentfuse.exceptions.BudgetExceeded` — *before* the call is ever
  delegated to ``litellm``.

The actual ``litellm.completion`` wrapping that calls :func:`gate` lives in
:mod:`agentfuse.wrap`; this module owns the budget binding and the trip path.
"""

from __future__ import annotations

import contextvars
import functools
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from agentfuse.budget import Budget
from agentfuse.exceptions import BudgetExceeded
from agentfuse.pricing import actual_cost, estimate_prompt_cost

# The active per-task budget for the current context (thread/async-local).
_ACTIVE_BUDGET: contextvars.ContextVar[Budget | None] = contextvars.ContextVar(
    "agentfuse_active_budget", default=None
)

F = TypeVar("F", bound=Callable[..., Any])

# The banner printed on the user-facing trip path. The emoji belongs here (the
# moment a human sees the fuse blow), per the plan's §1/§5 demo script.
TRIP_BANNER = "🔌 FUSE TRIPPED"


def current_budget() -> Budget | None:
    """Return the :class:`Budget` bound to the current context, or ``None``.

    Outside any :func:`task` / :class:`Fuse` / :func:`fuse` scope this is
    ``None``, which the wrapper treats as "no fuse installed — pass the call
    straight through, ungated".
    """
    return _ACTIVE_BUDGET.get()


def _print_trip_banner(err: BudgetExceeded) -> None:
    """Print the human-facing ``🔌 FUSE TRIPPED`` banner for a tripped fuse.

    Composed from the exception's structured fields (not its message, which would
    duplicate the "FUSE TRIPPED" text) so the emoji-led banner reads cleanly.
    """
    print(
        f"\n{TRIP_BANNER} — task halted at ${err.spent:.2f} / ${err.ceiling:.2f} "
        f"ceiling (next call est. +${err.would_spend:.2f} would cross it; "
        f"call not sent)",
        file=sys.stderr,
        flush=True,
    )


def gate(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int | None = None,
    budget: Budget | None = None,
) -> float:
    """Pre-call admission check for one LLM call. Raise before any spend.

    Estimates an upper bound on the call's USD cost
    (:func:`~agentfuse.pricing.estimate_prompt_cost`), then asks ``budget``
    (defaulting to the active per-task budget) whether committing that estimate
    would cross the ceiling. If it would, prints the ``🔌 FUSE TRIPPED`` banner
    and raises :class:`~agentfuse.exceptions.BudgetExceeded` — *before* the call
    is delegated to ``litellm``, so the over-budget call is never sent.

    Returns the estimated upper-bound cost (USD) when the call is allowed, so the
    caller can log it. Returns ``0.0`` and gates nothing when there is no active
    budget (the fuse is a no-op outside a task scope).
    """
    active = budget if budget is not None else current_budget()
    if active is None:
        # No fuse installed for this context: do not gate, do not estimate-block.
        return 0.0

    estimate = estimate_prompt_cost(model, messages, max_tokens=max_tokens)
    try:
        active.check(estimate)  # raises BudgetExceeded if it would cross ceiling
    except BudgetExceeded as err:
        _print_trip_banner(err)
        raise
    return estimate


def commit_actual(response: Any, *, budget: Budget | None = None) -> float:
    """Post-call metering: add the real cost of ``response`` to the ledger.

    Reads the confirmed USD cost from the response's ``Usage``
    (:func:`~agentfuse.pricing.actual_cost`) and commits it to ``budget``
    (defaulting to the active per-task budget). Returns the committed amount, or
    ``0.0`` when there is no active budget.
    """
    active = budget if budget is not None else current_budget()
    if active is None:
        return 0.0
    cost = actual_cost(response)
    active.commit(cost)
    return cost


@contextmanager
def task(ceiling_usd: float, name: str = "task") -> Iterator[Budget]:
    """Scope a per-task spend ceiling for every gated call made inside the block.

    Binds a fresh :class:`Budget` to the current context for the duration of the
    ``with`` block, then restores the previous binding on exit (so tasks can
    nest). Yields the :class:`Budget` so the caller can inspect ``.spent`` /
    ``.snapshot()`` afterwards::

        with task(ceiling_usd=5.0, name="nightly-scan") as budget:
            run_my_agent()        # every wrapped LLM call is gated
        print(budget.snapshot())  # what the task actually spent

    Args:
        ceiling_usd: Hard USD ceiling for this task.
        name: Human label shown in the ledger / CLI / trip banner.
    """
    budget = Budget(ceiling_usd=ceiling_usd, name=name)
    token = _ACTIVE_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _ACTIVE_BUDGET.reset(token)


class Fuse:
    """Context manager form of :func:`task`, named per the plan's quickstart.

    ``with Fuse(max_spend_usd=5.0): run_my_agent()`` is the two-line integration
    from §1 of the MVP plan. ``max_spend_usd`` is the public keyword; it maps to
    a per-task :class:`Budget` ceiling. The bound budget is exposed as
    :attr:`budget` once the block is entered.
    """

    def __init__(self, max_spend_usd: float, name: str = "task") -> None:
        self.max_spend_usd = max_spend_usd
        self.name = name
        self.budget: Budget | None = None
        self._cm: Iterator[Budget] | None = None

    def __enter__(self) -> Budget:
        self._cm = task(self.max_spend_usd, name=self.name)
        self.budget = self._cm.__enter__()  # type: ignore[attr-defined]
        return self.budget

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        assert self._cm is not None
        return bool(self._cm.__exit__(exc_type, exc, tb))  # type: ignore[attr-defined]


def fuse(
    ceiling_usd: float | None = None,
    *,
    max_spend_usd: float | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorator binding a per-call spend ceiling to a function.

    Every gated LLM call made while the decorated function runs is scoped to a
    fresh :class:`Budget`. Accepts either ``ceiling_usd`` or the quickstart alias
    ``max_spend_usd``::

        @fuse(ceiling_usd=5.0)
        def run_agent():
            ...

    The decorated function's name is used as the budget label unless ``name`` is
    given.
    """
    resolved = ceiling_usd if ceiling_usd is not None else max_spend_usd
    if resolved is None:
        raise TypeError("fuse() requires ceiling_usd (or max_spend_usd)")
    limit = float(resolved)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with task(limit, name=name or getattr(func, "__name__", "task")):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# Alias matching the plan's `@fused(...)` spelling.
fused = fuse


__all__ = [
    "Fuse",
    "task",
    "fuse",
    "fused",
    "gate",
    "commit_actual",
    "current_budget",
    "TRIP_BANNER",
]
