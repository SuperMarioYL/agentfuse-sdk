"""Tests for the m2 enforcement fuse and the m3 litellm wrapper.

The load-bearing assertion of the whole product: the over-budget call is gated
in AgentFuse's OWN wrapper code and the underlying delegate is NEVER invoked when
the fuse trips. These tests use a stub delegate / litellm ``mock_response`` — no
real network, no API key.
"""

from __future__ import annotations

import pytest

import agentfuse
from agentfuse import Budget, BudgetExceeded, Fuse
from agentfuse.fuse import current_budget, fuse, fused, gate, task

# NOTE: `agentfuse.wrap` the *attribute* is the install() callable (the public
# verb re-exported by __init__), which shadows the submodule of the same name.
# Reach the actual module object via importlib for the wrapper-internals tests.
import importlib

wrap_mod = importlib.import_module("agentfuse.wrap")

# A model that exists in litellm.model_cost with a non-zero price, so estimates
# are positive.
MODEL = "gpt-4o"
SYSTEM = {"role": "system", "content": "You are an autonomous agent."}
USER = {"role": "user", "content": "Scan the DN42 network and report back in detail."}
MESSAGES = [SYSTEM, USER]


# --------------------------------------------------------------------------- #
# gate(): pre-call trip vs pass-through
# --------------------------------------------------------------------------- #


def test_gate_passes_through_with_no_active_budget():
    # Outside any task scope, gate is a no-op and returns 0.0 (no fuse installed).
    assert current_budget() is None
    assert gate(MODEL, MESSAGES, max_tokens=100) == 0.0


def test_gate_returns_estimate_when_within_budget():
    b = Budget(ceiling_usd=100.0)
    est = gate(MODEL, MESSAGES, max_tokens=100, budget=b)
    assert est > 0.0


def test_gate_raises_before_spend_when_over_budget():
    b = Budget(ceiling_usd=0.0001)  # tiny ceiling -> any real call trips
    with pytest.raises(BudgetExceeded) as excinfo:
        gate(MODEL, MESSAGES, max_tokens=4000, budget=b)
    assert excinfo.value.ceiling == pytest.approx(0.0001)


# --------------------------------------------------------------------------- #
# The core guarantee: delegate is NOT called when the fuse trips.
# --------------------------------------------------------------------------- #


def test_over_budget_call_blocks_before_delegate_is_invoked():
    calls: list[dict] = []

    def stub_delegate(*args, **kwargs):  # would be the real litellm.completion
        calls.append(kwargs)
        raise AssertionError("delegate must NOT be called when over budget")

    with task(ceiling_usd=0.0001, name="t"):
        with pytest.raises(BudgetExceeded):
            wrap_mod.completion(
                model=MODEL,
                messages=MESSAGES,
                max_tokens=4000,
                real=stub_delegate,
            )

    # The whole point: the over-budget call was never delegated.
    assert calls == [], "over-budget call must not reach the delegate"


def test_within_budget_call_delegates_and_commits():
    sent: list[dict] = []

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class FakeResponse:
        model = MODEL
        usage = FakeUsage()

    def stub_delegate(*args, **kwargs):
        sent.append(kwargs)
        return FakeResponse()

    with task(ceiling_usd=100.0, name="t") as budget:
        resp = wrap_mod.completion(
            model=MODEL,
            messages=MESSAGES,
            max_tokens=50,
            real=stub_delegate,
        )
        assert isinstance(resp, FakeResponse)
        # The real call went out exactly once...
        assert len(sent) == 1
        # ...and its real cost was committed to the ledger (gpt-4o priced > 0).
        assert budget.spent > 0.0


# --------------------------------------------------------------------------- #
# task() / Fuse / decorator scoping
# --------------------------------------------------------------------------- #


def test_task_scopes_budget_and_restores_on_exit():
    assert current_budget() is None
    with task(ceiling_usd=5.0, name="scan") as b:
        assert current_budget() is b
        assert b.name == "scan"
    assert current_budget() is None


def test_task_scopes_nest_and_restore():
    with task(ceiling_usd=5.0, name="outer") as outer:
        assert current_budget() is outer
        with task(ceiling_usd=1.0, name="inner") as inner:
            assert current_budget() is inner
        assert current_budget() is outer


def test_fuse_context_manager_alias():
    with Fuse(max_spend_usd=3.0, name="cm") as b:
        assert current_budget() is b
        assert b.ceiling_usd == pytest.approx(3.0)
    assert current_budget() is None


def test_fuse_decorator_binds_budget():
    seen = {}

    @fuse(ceiling_usd=2.5)
    def run():
        seen["budget"] = current_budget()
        return "done"

    assert current_budget() is None
    assert run() == "done"
    assert current_budget() is None
    assert seen["budget"] is not None
    assert seen["budget"].ceiling_usd == pytest.approx(2.5)


def test_fused_alias_and_max_spend_kwarg():
    @fused(max_spend_usd=1.0)
    def run():
        return current_budget().ceiling_usd

    assert run() == pytest.approx(1.0)


def test_fuse_decorator_requires_ceiling():
    with pytest.raises(TypeError):
        fuse()  # neither ceiling_usd nor max_spend_usd


# --------------------------------------------------------------------------- #
# install() / uninstall() monkeypatch round-trip
# --------------------------------------------------------------------------- #


def test_install_uninstall_restores_litellm_completion():
    import litellm

    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    wrap_mod.install()
    try:
        assert wrap_mod.is_installed() is True
        assert litellm.completion is not original_completion
        assert litellm.acompletion is not original_acompletion
    finally:
        wrap_mod.uninstall()

    assert wrap_mod.is_installed() is False
    assert litellm.completion is original_completion
    assert litellm.acompletion is original_acompletion


def test_installed_wrapper_gates_via_active_budget(monkeypatch):
    import litellm

    calls: list[dict] = []

    def fake_real(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("must not be reached when over budget")

    # Point the wrapper's captured 'real' completion at our stub, then install.
    monkeypatch.setattr(wrap_mod, "_REAL_COMPLETION", fake_real)
    wrap_mod.install()
    try:
        with task(ceiling_usd=0.0001, name="t"):
            with pytest.raises(BudgetExceeded):
                litellm.completion(model=MODEL, messages=MESSAGES, max_tokens=4000)
    finally:
        wrap_mod.uninstall()

    assert calls == []


# --------------------------------------------------------------------------- #
# End-to-end offline: the runaway loop trips the fuse (mock_response).
# --------------------------------------------------------------------------- #


def test_runaway_loop_trips_with_mock_response():
    """Full path through agentfuse.completion with litellm mock_response (offline)."""
    tripped = False
    with task(ceiling_usd=0.05, name="runaway") as budget:
        try:
            for i in range(1000):
                agentfuse.completion(
                    model=MODEL,
                    messages=MESSAGES,
                    max_tokens=4000,
                    mock_response=f"mock step {i}",
                )
        except BudgetExceeded:
            tripped = True
    assert tripped, "the fuse should trip before 1000 mock calls"
    # And it tripped at or below the ceiling (never overshot via a blocked call).
    assert budget.spent <= budget.ceiling_usd
