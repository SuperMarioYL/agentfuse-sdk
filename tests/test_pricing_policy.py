"""Tests for the v0.2 hardening: unpriced-model policy, non-positive ceiling
guard, token ceiling, and the per-call hard cap.

All pure-Python, no network. The pricing tests exercise the local model-cost
table and tiktoken; they never issue an LLM call.
"""

from __future__ import annotations

import math

import pytest

from agentfuse import Budget, BudgetExceeded, Fuse, UnpricedModelError, fuse
from agentfuse.fuse import task
from agentfuse.pricing import (
    FALLBACK_USD_PER_TOKEN,
    estimate_call,
    estimate_prompt_cost,
)

MODEL = "gpt-4o"
UNPRICED = "totally-made-up-model-xyz"
MESSAGES = [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Scan the DN42 network and report back."},
]


# --------------------------------------------------------------------------- #
# fix-unpriced-model-noop: fail closed by default
# --------------------------------------------------------------------------- #


def test_unpriced_blocks_by_default():
    with pytest.raises(UnpricedModelError) as excinfo:
        estimate_prompt_cost(UNPRICED, MESSAGES, max_tokens=256)
    assert UNPRICED in str(excinfo.value)
    assert excinfo.value.model == UNPRICED


def test_unpriced_warn_pass_returns_zero():
    cost = estimate_prompt_cost(UNPRICED, MESSAGES, max_tokens=256, on_unpriced="warn-pass")
    assert cost == 0.0


def test_unpriced_fallback_is_positive_and_conservative():
    # fallback prices every estimated token at the conservative fallback rate.
    usd, tokens = estimate_call(UNPRICED, MESSAGES, max_tokens=256, on_unpriced="fallback")
    assert tokens > 0
    assert usd == pytest.approx(tokens * FALLBACK_USD_PER_TOKEN)
    assert usd > 0.0


def test_priced_model_unaffected_by_policy():
    # A priced model ignores on_unpriced entirely.
    a = estimate_prompt_cost(MODEL, MESSAGES, max_tokens=256, on_unpriced="block")
    b = estimate_prompt_cost(MODEL, MESSAGES, max_tokens=256, on_unpriced="warn-pass")
    assert a == pytest.approx(b)
    assert a > 0.0


def test_unpriced_block_trips_the_fuse_end_to_end():
    # The whole point: an unpriced model no longer silently disables the fuse.
    with pytest.raises(UnpricedModelError):
        with task(ceiling_usd=100.0, name="t") as budget:
            from agentfuse.fuse import gate as fuse_gate

            fuse_gate(UNPRICED, MESSAGES, budget=budget)


def test_unpriced_warn_pass_via_task_passes_through():
    with task(ceiling_usd=100.0, name="t", on_unpriced="warn-pass") as budget:
        from agentfuse.fuse import gate as fuse_gate

        est = fuse_gate(UNPRICED, MESSAGES, budget=budget)
    assert est == 0.0


# --------------------------------------------------------------------------- #
# fix-nonpositive-ceiling-footgun: reject 0 / NaN / inf ceilings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_budget_rejects_non_positive_or_non_finite_ceiling(bad):
    with pytest.raises(ValueError):
        Budget(ceiling_usd=bad)


def test_fuse_surfaces_ceiling_validation():
    with pytest.raises(ValueError):
        with Fuse(max_spend_usd=0.0):
            pass


def test_fuse_decorator_surfaces_ceiling_validation():
    @fuse(ceiling_usd=float("nan"))
    def run():  # pragma: no cover - body never reached
        return "x"

    with pytest.raises(ValueError):
        run()


def test_valid_small_positive_ceiling_ok():
    b = Budget(ceiling_usd=0.0001)
    assert b.ceiling_usd == pytest.approx(0.0001)


# --------------------------------------------------------------------------- #
# m4_token_ceiling: token bound trips alongside USD, first-to-trip wins
# --------------------------------------------------------------------------- #


def test_token_ceiling_trips_independently_of_usd():
    # Huge USD ceiling, tiny token ceiling -> tokens trip first.
    b = Budget(ceiling_usd=1_000.0, ceiling_tokens=5)
    with pytest.raises(BudgetExceeded) as excinfo:
        b.check(estimated_usd=0.0001, estimated_tokens=100)
    err = excinfo.value
    assert err.limit_kind == "tokens"
    assert err.ceiling_tokens == 5
    assert err.would_spend_tokens == 100
    assert "tokens" in str(err)


def test_token_ceiling_allows_up_to_limit():
    b = Budget(ceiling_usd=1_000.0, ceiling_tokens=100)
    b.commit(0.01, actual_tokens=90)
    # 90 + 10 == 100, exactly the ceiling -> allowed (strict greater-than)
    b.check(estimated_usd=0.01, estimated_tokens=10)
    assert b.spent_tokens == 90
    assert b.remaining_tokens() == 10


def test_usd_trips_before_tokens_when_usd_is_tighter():
    # USD ceiling tight, token ceiling loose -> USD wins (first-to-trip).
    b = Budget(ceiling_usd=0.0001, ceiling_tokens=1_000_000)
    with pytest.raises(BudgetExceeded) as excinfo:
        b.check(estimated_usd=1.0, estimated_tokens=10)
    assert excinfo.value.limit_kind == "usd"


def test_no_token_ceiling_means_no_token_limit():
    b = Budget(ceiling_usd=1_000.0)
    assert b.remaining_tokens() is None
    # Even a giant token estimate passes when no token ceiling is set.
    b.check(estimated_usd=0.01, estimated_tokens=10_000_000)


def test_token_ceiling_must_be_positive():
    with pytest.raises(ValueError):
        Budget(ceiling_usd=5.0, ceiling_tokens=0)
    with pytest.raises(ValueError):
        Budget(ceiling_usd=5.0, ceiling_tokens=-10)


def test_commit_tracks_tokens_and_snapshot():
    b = Budget(ceiling_usd=5.0, ceiling_tokens=100, name="scan")
    b.commit(0.5, actual_tokens=30)
    b.commit(0.25, actual_tokens=20)
    snap = b.snapshot()
    assert snap.spent_tokens == 50
    assert snap.ceiling_tokens == 100
    assert b.spent_tokens == 50


def test_commit_rejects_negative_tokens():
    b = Budget(ceiling_usd=5.0)
    with pytest.raises(ValueError):
        b.commit(0.0, actual_tokens=-1)


# --------------------------------------------------------------------------- #
# m5_per_call_ceiling: one oversized call trips even with budget to spare
# --------------------------------------------------------------------------- #


def test_single_call_ceiling_trips_on_oversized_call():
    # Plenty of cumulative budget, but one call's estimate alone is too big.
    b = Budget(ceiling_usd=1_000.0, single_call_ceiling=0.50)
    with pytest.raises(BudgetExceeded) as excinfo:
        b.check(estimated_usd=0.75)
    err = excinfo.value
    assert err.limit_kind == "single_call"
    assert err.ceiling == pytest.approx(0.50)
    assert err.would_spend == pytest.approx(0.75)
    assert "per-call" in str(err)


def test_single_call_ceiling_allows_at_limit():
    b = Budget(ceiling_usd=1_000.0, single_call_ceiling=0.50)
    # exactly at the per-call cap -> allowed (strict greater-than)
    b.check(estimated_usd=0.50)


def test_single_call_ceiling_independent_of_cumulative():
    # Many small calls are fine; the cap only fires on a single oversized one.
    b = Budget(ceiling_usd=1_000.0, single_call_ceiling=0.50)
    for _ in range(10):
        b.check(estimated_usd=0.40)
        b.commit(0.40)
    assert b.spent == pytest.approx(4.0)  # cumulative grew, cap never tripped
    with pytest.raises(BudgetExceeded) as excinfo:
        b.check(estimated_usd=0.60)
    assert excinfo.value.limit_kind == "single_call"


def test_single_call_ceiling_must_be_positive_finite():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Budget(ceiling_usd=5.0, single_call_ceiling=bad)


def test_per_call_cap_via_fuse_context_manager():
    with Fuse(max_spend_usd=1_000.0, single_call_ceiling=0.25) as budget:
        assert budget.single_call_ceiling == pytest.approx(0.25)
        with pytest.raises(BudgetExceeded):
            budget.check(estimated_usd=0.30)


def test_token_ceiling_via_fuse_max_total_tokens_kwarg():
    # v0.3.0 — the cumulative token ceiling keyword is `max_total_tokens` (the old
    # `max_tokens` is a deprecated alias, covered in test_stream.py).
    with Fuse(max_spend_usd=1_000.0, max_total_tokens=10) as budget:
        assert budget.ceiling_tokens == 10
        with pytest.raises(BudgetExceeded) as excinfo:
            budget.check(estimated_usd=0.0001, estimated_tokens=50)
        assert excinfo.value.limit_kind == "tokens"


def test_finite_guard_helper_directly():
    # sanity: the validator accepts a normal value and rejects edge cases.
    from agentfuse.budget import _require_finite_positive

    assert _require_finite_positive(5.0, "x") == 5.0
    for bad in (0.0, -0.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            _require_finite_positive(bad, "x")
