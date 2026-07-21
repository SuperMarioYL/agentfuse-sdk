"""Tests for the v0.4.0 in-process ``on_trip`` callback hook.

The hook is invoked with the fully-structured :class:`BudgetExceeded` right
before the over-budget call is blocked (i.e. before it is delegated to litellm),
fail-soft. These tests pin the three load-bearing properties:

1. the callback receives the real trip exception with the right ``limit_kind``,
2. a callback that raises is swallowed — the fuse still trips with the same
   error (a user hook can never change whether the over-budget call is blocked),
3. the kwarg threads from ``Fuse(on_trip=...)`` / ``@fuse(on_trip=...)`` /
   ``task(on_trip=...)`` down to ``Budget.on_trip``.
"""

from __future__ import annotations

import importlib

import pytest

from agentfuse import Budget, BudgetExceeded, Fuse
from agentfuse.budget import TripCallback  # noqa: F401  - public type alias sanity
from agentfuse.fuse import fuse, task

wrap_mod = importlib.import_module("agentfuse.wrap")

MODEL = "gpt-4o"
MESSAGES = [
    {"role": "system", "content": "You are an autonomous agent."},
    {"role": "user", "content": "Scan the network and report back in detail."},
]


# --------------------------------------------------------------------------- #
# Budget.check: on_trip invoked with the right exception, before the raise
# --------------------------------------------------------------------------- #


def test_on_trip_callback_invoked_with_exception_before_raise():
    seen: list[BudgetExceeded] = []

    def hook(err: BudgetExceeded) -> None:
        seen.append(err)

    b = Budget(ceiling_usd=0.0001, on_trip=hook)
    with pytest.raises(BudgetExceeded) as excinfo:
        b.check(estimated_usd=1.0, estimated_tokens=10)

    # The hook saw the same exception that was ultimately raised.
    assert len(seen) == 1
    assert seen[0] is excinfo.value
    # And it carried the right structured fields.
    assert seen[0].limit_kind == "usd"
    assert seen[0].ceiling == pytest.approx(0.0001)
    assert seen[0].would_spend == pytest.approx(1.0)


def test_on_trip_callback_not_invoked_when_within_budget():
    seen: list[BudgetExceeded] = []

    def hook(err: BudgetExceeded) -> None:
        seen.append(err)

    b = Budget(ceiling_usd=100.0, on_trip=hook)
    # Within budget -> no trip, no hook call.
    b.check(estimated_usd=0.001, estimated_tokens=5)
    assert seen == []


def test_on_trip_reports_token_limit_kind():
    seen: list[BudgetExceeded] = []

    def hook(err: BudgetExceeded) -> None:
        seen.append(err)

    b = Budget(ceiling_usd=1_000.0, ceiling_tokens=5, on_trip=hook)
    with pytest.raises(BudgetExceeded):
        b.check(estimated_usd=0.0001, estimated_tokens=100)
    assert seen[0].limit_kind == "tokens"
    assert seen[0].ceiling_tokens == 5


# --------------------------------------------------------------------------- #
# Fail-soft: a callback that raises is swallowed; the fuse still trips.
# --------------------------------------------------------------------------- #


def test_on_trip_callback_exception_is_swallowed():
    def bad_hook(err: BudgetExceeded) -> None:
        raise RuntimeError("user webhook is down")

    b = Budget(ceiling_usd=0.0001, on_trip=bad_hook)
    # The bad callback must NOT mask the BudgetExceeded — the fuse still trips.
    with pytest.raises(BudgetExceeded) as excinfo:
        b.check(estimated_usd=1.0, estimated_tokens=10)
    assert excinfo.value.limit_kind == "usd"


def test_on_trip_callback_default_none_is_noop():
    # No hook set -> check() behaves exactly as before v0.4.0.
    b = Budget(ceiling_usd=0.0001)  # on_trip defaults to None
    with pytest.raises(BudgetExceeded):
        b.check(estimated_usd=1.0, estimated_tokens=10)


# --------------------------------------------------------------------------- #
# The hook fires on the real gate path (wrap.completion), before the delegate.
# --------------------------------------------------------------------------- #


def test_on_trip_fires_before_delegate_on_over_budget_call():
    calls: list[dict] = []
    trips: list[BudgetExceeded] = []

    def hook(err: BudgetExceeded) -> None:
        trips.append(err)

    def stub_delegate(*args, **kwargs):  # would be the real litellm.completion
        calls.append(kwargs)
        raise AssertionError("delegate must NOT be called when over budget")

    with task(ceiling_usd=0.0001, name="t", on_trip=hook):
        with pytest.raises(BudgetExceeded):
            wrap_mod.completion(
                model=MODEL,
                messages=MESSAGES,
                max_tokens=4000,
                real=stub_delegate,
            )

    # The hook fired once with the usd-trip context...
    assert len(trips) == 1
    assert trips[0].limit_kind == "usd"
    # ...and the over-budget call never reached the delegate.
    assert calls == []


# --------------------------------------------------------------------------- #
# The kwarg threads from Fuse / @fuse down to Budget.on_trip
# --------------------------------------------------------------------------- #


def test_on_trip_wired_through_fuse_context_manager():
    seen: list[BudgetExceeded] = []

    def hook(err: BudgetExceeded) -> None:
        seen.append(err)

    with Fuse(max_spend_usd=0.0001, on_trip=hook) as budget:
        assert budget.on_trip is hook
        with pytest.raises(BudgetExceeded):
            budget.check(estimated_usd=1.0, estimated_tokens=10)
    assert len(seen) == 1
    assert seen[0].limit_kind == "usd"


def test_on_trip_wired_through_fuse_decorator():
    seen: list[BudgetExceeded] = []

    def hook(err: BudgetExceeded) -> None:
        seen.append(err)

    @fuse(ceiling_usd=0.0001, on_trip=hook)
    def run_agent():
        # One in-scope over-budget gate call trips the fuse.
        from agentfuse.fuse import current_budget

        current_budget().check(estimated_usd=1.0, estimated_tokens=10)

    with pytest.raises(BudgetExceeded):
        run_agent()
    assert len(seen) == 1
    assert seen[0].limit_kind == "usd"
