"""Tests for the m1 spend meter: the per-task ledger and the cost estimator.

These are pure-Python, no-network tests. The pricing test exercises
``estimate_prompt_cost`` against LiteLLM's local model-cost table and tiktoken;
it never issues an LLM call.
"""

from __future__ import annotations

import pytest

from agentfuse import Budget, gate
from agentfuse.budget import BudgetSnapshot
from agentfuse.exceptions import BudgetExceeded
from agentfuse.pricing import estimate_prompt_cost


# --------------------------------------------------------------------------- #
# Ledger basics
# --------------------------------------------------------------------------- #


def test_ledger_starts_at_zero():
    b = Budget(ceiling_usd=5.0)
    assert b.spent == 0.0
    assert b.remaining() == 5.0
    assert b.name == "task"


def test_commit_accumulates():
    b = Budget(ceiling_usd=5.0)
    assert b.commit(1.25) == pytest.approx(1.25)
    assert b.commit(0.75) == pytest.approx(2.0)
    assert b.spent == pytest.approx(2.0)
    assert b.remaining() == pytest.approx(3.0)


def test_commit_rejects_negative():
    b = Budget(ceiling_usd=5.0)
    with pytest.raises(ValueError):
        b.commit(-0.01)


def test_negative_ceiling_rejected():
    with pytest.raises(ValueError):
        Budget(ceiling_usd=-1.0)


# --------------------------------------------------------------------------- #
# would_exceed boundary: just under / exactly at / over the ceiling
# --------------------------------------------------------------------------- #


def test_would_exceed_just_under():
    b = Budget(ceiling_usd=5.0)
    b.commit(4.0)
    # 4.00 + 0.99 = 4.99 <= 5.00 -> allowed
    assert b.would_exceed(0.99) is False


def test_would_exceed_exactly_at_ceiling():
    b = Budget(ceiling_usd=5.0)
    b.commit(4.0)
    # 4.00 + 1.00 = 5.00, exactly the ceiling -> allowed (strict greater-than)
    assert b.would_exceed(1.0) is False


def test_would_exceed_over_ceiling():
    b = Budget(ceiling_usd=5.0)
    b.commit(4.0)
    # 4.00 + 1.01 = 5.01 > 5.00 -> blocked
    assert b.would_exceed(1.01) is True


def test_check_raises_budget_exceeded_with_fields():
    b = Budget(ceiling_usd=5.0)
    b.commit(4.98)
    with pytest.raises(BudgetExceeded) as excinfo:
        b.check(0.10)
    err = excinfo.value
    assert err.spent == pytest.approx(4.98)
    assert err.ceiling == pytest.approx(5.0)
    assert err.would_spend == pytest.approx(0.10)
    assert "FUSE TRIPPED" in str(err)


def test_check_allows_at_ceiling():
    b = Budget(ceiling_usd=5.0)
    b.commit(4.5)
    # exactly hits ceiling -> no raise
    b.check(0.5)


def test_gate_function_matches_check():
    b = Budget(ceiling_usd=1.0)
    b.commit(0.9)
    gate(b, 0.1)  # 1.00 == ceiling, allowed
    with pytest.raises(BudgetExceeded):
        gate(b, 0.2)  # 1.10 > ceiling, blocked


# --------------------------------------------------------------------------- #
# remaining() math + snapshot/repr
# --------------------------------------------------------------------------- #


def test_remaining_clamps_at_zero():
    b = Budget(ceiling_usd=2.0)
    b.commit(2.5)  # overshoot (e.g. a real cost landed above estimate)
    assert b.remaining() == 0.0
    assert b.spent == pytest.approx(2.5)


def test_snapshot_and_repr():
    b = Budget(ceiling_usd=10.0, name="nightly-scan")
    b.commit(3.0)
    snap = b.snapshot()
    assert isinstance(snap, BudgetSnapshot)
    assert snap.name == "nightly-scan"
    assert snap.ceiling_usd == pytest.approx(10.0)
    assert snap.spent_usd == pytest.approx(3.0)
    assert snap.remaining_usd == pytest.approx(7.0)
    r = repr(b)
    assert "nightly-scan" in r
    assert "remaining" in r


# --------------------------------------------------------------------------- #
# Pricing estimator (no network)
# --------------------------------------------------------------------------- #

_MESSAGES = [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Scan the DN42 network and report back."},
]


def test_estimate_known_model_positive():
    cost = estimate_prompt_cost("gpt-4o", _MESSAGES, max_tokens=256)
    assert cost > 0.0


def test_estimate_unknown_model_returns_zero_no_crash():
    cost = estimate_prompt_cost("totally-made-up-model-xyz", _MESSAGES, max_tokens=256)
    assert cost == 0.0


def test_estimate_grows_with_max_tokens():
    low = estimate_prompt_cost("gpt-4o", _MESSAGES, max_tokens=10)
    high = estimate_prompt_cost("gpt-4o", _MESSAGES, max_tokens=1000)
    assert high > low


def test_estimate_default_max_tokens_when_unset():
    # No max_tokens -> still finite, positive, and does not raise.
    cost = estimate_prompt_cost("gpt-4o", _MESSAGES)
    assert cost > 0.0
