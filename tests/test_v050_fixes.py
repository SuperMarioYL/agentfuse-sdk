"""Adversarial red→green tests for the v0.5.0 amendment's three `type: fix` milestones.

Each milestone has at least one test that FAILS on the unfixed v0.4.0 source and
PASSES on the v0.5.0 fixed source (the red→green contract the plan demands):

* ``fix-async-decorator-bypasses-fuse`` — `@fuse`/`@fused` on an `async def`
  used to return the coroutine *before* the `with task(...)` block exited, so
  `current_budget()` was ``None`` inside the async body and `acompletion` took
  the no-budget pass-through path (the fuse silently did not break). The async
  wrapper now `await`s the body INSIDE the budget scope.
* ``fix-budget-check-commit-race`` — `Budget.check` released the lock before
  `acompletion` awaited the delegate, so two concurrent calls both gated
  against the still-uncommitted ledger and both committed, overshooting by up to
  ``(concurrency * per-call estimate)``. The gate now RESERVES the estimate
  (`Budget.reserve` → `Reservation` handle) and `commit`/`release` settle it,
  making the gate atomic across the awaited call WITHOUT holding the lock across
  the HTTP call.
* ``fix-streaming-default-completion-underestimate`` —
  `DEFAULT_MAX_COMPLETION_TOKENS=1024` was not an upper bound for streamed
  completions (4k-16k on frontier models), so the pre-call gate passed and the
  post-call commit jumped the ledger past the ceiling with no retroactive trip.
  The default is now 8192 and a model-aware `max_output_tokens` cap is preferred
  when larger, so the estimate is a true upper bound and the pre-call gate trips.

All tests are offline (stub delegates / litellm `mock_response`); no network,
no API key.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from agentfuse import BudgetExceeded, fuse, fused
from agentfuse.budget import Budget, Reservation
from agentfuse.fuse import current_budget, task
from agentfuse.pricing import DEFAULT_MAX_COMPLETION_TOKENS, estimate_call

# `agentfuse.wrap` the *attribute* (re-exported by __init__) is the install()
# callable and shadows the submodule. Reach the actual module via importlib for
# the wrapper-internals tests (same pattern as tests/test_fuse.py).
wrap_mod = importlib.import_module("agentfuse.wrap")

# A model that exists in litellm.model_cost with a non-zero price AND a
# max_output_tokens cap (gpt-4o: max_output_tokens=16384), so the v0.5.0
# model-aware estimate path is exercised.
MODEL = "gpt-4o"
MESSAGES = [
    {"role": "system", "content": "You are an autonomous agent."},
    {"role": "user", "content": "Scan the DN42 network and report back in detail."},
]


# --------------------------------------------------------------------------- #
# Fakes for stubbed litellm responses / streams (mirrors tests/test_stream.py).
# --------------------------------------------------------------------------- #


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _FakeResponse:
    """A plain object exposing ``.model`` + ``.usage`` for the pricing helpers."""

    def __init__(self, usage: _FakeUsage, model: str = MODEL) -> None:
        self.model = model
        self.usage = usage


class _FakeChunk:
    def __init__(self, content: str, usage: _FakeUsage | None = None, model: str = MODEL) -> None:
        self.content = content
        self.model = model
        self.usage = usage


def _make_stream(n_chunks: int, final_usage: _FakeUsage | None):
    for i in range(n_chunks):
        yield _FakeChunk(f"chunk {i}", usage=final_usage if i == n_chunks - 1 else None)


# =========================================================================== #
# fix-async-decorator-bypasses-fuse
# =========================================================================== #


def test_v050_async_decorator_binds_budget_inside_body():
    """RED on v0.4.0 (current_budget() is None inside the async body), GREEN on v0.5.0.

    On the unfixed sync wrapper, `func(*args)` returned the coroutine without
    starting the body, the `with task(...)` block exited (resetting the
    contextvar) BEFORE the caller awaited, so `current_budget()` was None.
    """
    @fuse(ceiling_usd=5.0)
    async def run():
        return current_budget()

    assert current_budget() is None
    result = asyncio.run(run())
    # Outside the decorator scope again.
    assert current_budget() is None
    # The load-bearing assertion: the budget was bound WHILE the async body ran.
    assert result is not None, "async @fuse must bind the budget inside the async body"
    assert isinstance(result, Budget)
    assert result.ceiling_usd == pytest.approx(5.0)


def test_v050_async_decorator_gates_over_budget_acompletion():
    """RED on v0.4.0 (fuse inert -> delegate IS called, no BudgetExceeded), GREEN on v0.5.0.

    An over-budget `acompletion` made inside a decorated async function must
    raise `BudgetExceeded` BEFORE the delegate is invoked — which is only
    possible if the budget is actually bound inside the async body.
    """
    delegated: list[dict] = []

    async def stub_delegate(*args, **kwargs):
        delegated.append(kwargs)
        return _FakeResponse(_FakeUsage(prompt=10, completion=5))

    @fuse(ceiling_usd=0.0001)  # tiny ceiling -> any real gpt-4o call trips
    async def run_over():
        return await wrap_mod.acompletion(
            model=MODEL, messages=MESSAGES, max_tokens=4000, real=stub_delegate
        )

    with pytest.raises(BudgetExceeded):
        asyncio.run(run_over())
    # The whole point: the over-budget call never reached the delegate.
    assert delegated == [], "over-budget async call must not be delegated"


def test_v050_fused_alias_works_on_async():
    """The `@fused` spelling of the decorator must also await the async body in-scope."""
    @fused(max_spend_usd=3.0)
    async def run():
        b = current_budget()
        assert b is not None
        return b.ceiling_usd

    assert asyncio.run(run()) == pytest.approx(3.0)


def test_v050_async_decorator_preserves_metadata():
    """`functools.wraps` must keep the wrapped function's introspection metadata."""
    @fuse(ceiling_usd=5.0)
    async def my_async_agent(x: int) -> int:
        return x  # pragma: no cover - body not the point

    assert my_async_agent.__name__ == "my_async_agent"
    # asyncio.iscoroutinefunction must see the decorated function as async so it
    # can be awaited (and double-decoration / frameworks that branch on it work).
    import inspect

    assert inspect.iscoroutinefunction(my_async_agent) is True


def test_v050_sync_decorator_still_binds_budget():
    """Regression: the sync decorator path is unchanged by the async branch."""
    seen: dict = {}

    @fuse(ceiling_usd=2.5)
    def run():
        seen["budget"] = current_budget()
        return "done"

    assert run() == "done"
    assert seen["budget"] is not None
    assert seen["budget"].ceiling_usd == pytest.approx(2.5)


# =========================================================================== #
# fix-budget-check-commit-race — Budget reserve/release mechanics
# =========================================================================== #


def test_v050_reserve_returns_reservation_handle_and_pends():
    b = Budget(ceiling_usd=5.0)
    r = b.reserve(0.50, 100)
    assert isinstance(r, Reservation)
    assert r.active is True
    assert r.estimated_usd == pytest.approx(0.50)
    assert r.estimated_tokens == 100
    # Reserve adds to pending, NOT to spent.
    assert b.pending_usd == pytest.approx(0.50)
    assert b.pending_tokens == 100
    assert b.spent == 0.0
    assert b.spent_tokens == 0


def test_v050_reserve_blocks_concurrent_gate_via_pending():
    """RED on v0.4.0 (no pending -> second gate passes), GREEN on v0.5.0.

    The core race fix: a second concurrent gate must see the first call's
    pending reservation and trip, so two concurrent estimates cannot both pass
    against the still-uncommitted ledger.
    """
    b = Budget(ceiling_usd=0.60)  # < 2 * 0.40, > 1 * 0.40
    r1 = b.reserve(0.40, 10)
    assert r1.active
    assert b.pending_usd == pytest.approx(0.40)
    # A second concurrent gate for the same estimate now trips:
    #   spent(0) + pending(0.40) + 0.40 = 0.80 > 0.60
    with pytest.raises(BudgetExceeded):
        b.reserve(0.40, 10)
    # The failed reserve created no reservation (no leak into pending).
    assert b.pending_usd == pytest.approx(0.40)
    assert b.pending_tokens == 10


def test_v050_commit_releases_reservation_and_commits_real():
    b = Budget(ceiling_usd=5.0)
    r = b.reserve(0.40, 100)
    assert b.pending_usd == pytest.approx(0.40)
    # Real cost landed below the estimate; commit settles the reservation.
    b.commit(0.35, 90, reservation=r)
    assert b.spent == pytest.approx(0.35)
    assert b.spent_tokens == 90
    assert b.pending_usd == 0.0
    assert b.pending_tokens == 0
    assert r.active is False
    # Idempotent: re-passing the same (now-inactive) handle commits without
    # touching pending again (no double-release / negative pending).
    b.commit(0.10, reservation=r)
    assert b.pending_usd == 0.0
    assert b.spent == pytest.approx(0.45)


def test_v050_release_releases_reservation_without_committing():
    b = Budget(ceiling_usd=5.0)
    r = b.reserve(0.40, 100)
    b.release(r)
    assert b.pending_usd == 0.0
    assert b.pending_tokens == 0
    # Release does NOT commit spend (it is the error-path rollback).
    assert b.spent == 0.0
    assert b.spent_tokens == 0
    assert r.active is False
    # Budget is reusable after release (no stuck reservation).
    b.check(0.40, 100)  # peek passes — pending is clear


def test_v050_release_is_idempotent_and_handles_none():
    b = Budget(ceiling_usd=5.0)
    r = b.reserve(0.40, 100)
    b.release(r)
    b.release(r)  # idempotent
    b.release(None)  # no-op
    assert b.pending_usd == 0.0


def test_v050_check_remains_a_non_mutating_peek():
    """Regression: `check`/`would_exceed` must NOT reserve (legacy peek semantics).

    A standalone `check` that ignored the return value must not leak a
    reservation into pending — that is why the race-free path is a separate
    `reserve()` and `check` stays a side-effect-free peek.
    """
    b = Budget(ceiling_usd=5.0)
    b.check(0.50, 10)
    assert b.pending_usd == 0.0  # check reserved nothing
    assert b.pending_tokens == 0
    assert b.would_exceed(0.50, 10) is False


def test_v050_concurrent_acompletion_cannot_overshoot_ceiling():
    """RED on v0.4.0 (both calls pass the gate and commit -> overshoot, no trip),
    GREEN on v0.5.0 (the second concurrent gate sees the first's pending
    reservation and trips; only one call commits, within the ceiling).

    Mirrors the amendment's verification: `asyncio.gather` of two `acompletion`
    calls against a ceiling < 2x the per-call estimate, with a stub delegate that
    sleeps (so both gate before either commits).
    """
    N = 4000  # explicit max_tokens -> estimate is deterministic
    est_usd, _ = estimate_call(MODEL, MESSAGES, max_tokens=N)
    # Per-call actual cost (prompt=28, completion=N) == estimate (verified).
    actual_per_call = _actual_cost_for(N)
    assert actual_per_call == pytest.approx(est_usd)

    ceiling = est_usd * 1.5  # > 1x estimate (one call fits), < 2x (two don't)

    async def stub_delegate(*args, **kwargs):
        # Yield so the sibling task gates (reserves) before this one commits.
        await asyncio.sleep(0.01)
        return _FakeResponse(_FakeUsage(prompt=28, completion=N))

    async def one_call():
        return await wrap_mod.acompletion(
            model=MODEL, messages=MESSAGES, max_tokens=N, real=stub_delegate
        )

    async def driver():
        with task(ceiling_usd=ceiling, name="race") as budget:
            results = await asyncio.gather(one_call(), one_call(), return_exceptions=True)
            return budget, results

    budget, results = asyncio.run(driver())

    # The race fix: at least one of the two concurrent calls was rejected.
    excs = [r for r in results if isinstance(r, BudgetExceeded)]
    assert excs, (
        "expected >=1 BudgetExceeded from the concurrent fan-out (race not fixed); "
        f"results={results!r}"
    )
    # Exactly one call was sent + committed; the ledger never overshot the ceiling.
    assert budget.spent <= ceiling + 1e-9, (
        f"overshot the ceiling: spent={budget.spent} ceiling={ceiling}"
    )
    # And it did NOT commit both calls (which would be ~2x the estimate).
    assert budget.spent < 2 * est_usd, (
        f"both concurrent calls committed (race not fixed): spent={budget.spent}"
    )
    # The settled reservation left no pending leak.
    assert budget.pending_usd == 0.0
    assert budget.pending_tokens == 0


def test_v050_concurrent_acompletion_releases_reservation_on_delegate_error():
    """If the delegate raises, the reservation is released (not leaked into pending).

    (A correctness pin for the error path — passes on both versions, but guards
    the v0.5.0 release-on-exception branch in `acompletion`.)
    """
    N = 4000
    est_usd, _ = estimate_call(MODEL, MESSAGES, max_tokens=N)

    async def stub_raises(*args, **kwargs):
        await asyncio.sleep(0)  # yield once
        raise RuntimeError("provider 500")

    async def one_call():
        return await wrap_mod.acompletion(
            model=MODEL, messages=MESSAGES, max_tokens=N, real=stub_raises
        )

    async def driver():
        with task(ceiling_usd=est_usd * 1.5, name="err") as budget:
            results = await asyncio.gather(one_call(), return_exceptions=True)
            return budget, results

    budget, results = asyncio.run(driver())
    assert any(isinstance(r, RuntimeError) for r in results)
    # Nothing committed (delegate raised before any usage), and no reservation
    # leaked into pending — the error path released it.
    assert budget.spent == 0.0
    assert budget.pending_usd == 0.0
    assert budget.pending_tokens == 0


def test_v050_concurrent_acompletion_token_ceiling_also_racesafes():
    """The token ceiling is race-safe too: pending tokens count against it."""
    b = Budget(ceiling_usd=1_000_000.0, ceiling_tokens=60)
    r1 = b.reserve(0.01, 40)
    assert r1.active
    assert b.pending_tokens == 40
    # spent(0) + pending(40) + 40 = 80 > 60 -> trip on the token ceiling.
    with pytest.raises(BudgetExceeded) as excinfo:
        b.reserve(0.01, 40)
    assert excinfo.value.limit_kind == "tokens"
    # No leak from the failed reserve.
    assert b.pending_tokens == 40


# =========================================================================== #
# fix-streaming-default-completion-underestimate
# =========================================================================== #


def test_v050_default_completion_default_is_8192_floor():
    assert DEFAULT_MAX_COMPLETION_TOKENS == 8192


def test_v050_default_completion_estimate_is_a_true_upper_bound():
    """RED on v0.4.0 (default was 1024 -> estimate(max_tokens=None) == estimate(max_tokens=1024)),
    GREEN on v0.5.0 (default raised + model-aware cap -> estimate(max_tokens=None) is larger)."""
    est_default, _ = estimate_call(MODEL, MESSAGES, max_tokens=None)
    est_1024, _ = estimate_call(MODEL, MESSAGES, max_tokens=1024)
    assert est_default > est_1024, (
        "the no-max_tokens estimate must be larger than the 1024-based estimate "
        "(pre-v0.5.0 the default WAS 1024, so they were equal)"
    )


def test_v050_default_estimate_uses_model_aware_cap_when_larger():
    """RED on v0.4.0 (1024 default -> token_bound ~1k), GREEN on v0.5.0 (model-aware
    max_output_tokens=16384 for gpt-4o -> token_bound well above the 8192 floor)."""
    _est, token_bound = estimate_call(MODEL, MESSAGES, max_tokens=None)
    assert token_bound > 8192, (
        "the model-aware max_output_tokens cap (gpt-4o: 16384) must be preferred "
        "over the 8192 floor when it is larger"
    )


def test_v050_explicit_max_tokens_still_honoured():
    """Regression: a caller-supplied max_tokens overrides the model-aware/default cap."""
    est_50, _ = estimate_call(MODEL, MESSAGES, max_tokens=50)
    est_5000, _ = estimate_call(MODEL, MESSAGES, max_tokens=5000)
    assert est_5000 > est_50


def test_v050_streamed_call_with_unbounded_max_tokens_trips_pre_call_gate():
    """RED on v0.4.0 (1024-default estimate < ceiling -> gate passes -> stream commits
    a 4000-token completion -> overshoots, no trip), GREEN on v0.5.0 (8192+model-aware
    estimate > ceiling -> pre-call gate raises BudgetExceeded BEFORE the delegate runs).

    The ceiling is tuned to the OLD 1024-default estimate (1.5x it), so the
    unfixed code would pass the gate and overshoot post-commit, while the fixed
    code trips the gate pre-call.
    """
    # The OLD (v0.4.0) per-call estimate used a 1024 completion default. Compute
    # that explicitly (max_tokens=1024 is version-stable) and tune the ceiling
    # just above it but well below the v0.5.0 estimate.
    old_est, _ = estimate_call(MODEL, MESSAGES, max_tokens=1024)
    ceiling = old_est * 1.5  # > old_est (unfixed gate passes), < new_est (fixed gate trips)

    # A stub that WOULD produce a 4000-token streamed completion — but on the
    # fixed code the pre-call gate trips before the delegate is ever called.
    delegate_invoked: list[dict] = []

    def stub_stream(*args, **kwargs):
        delegate_invoked.append(kwargs)
        # prompt=28, completion=4000 -> real cost ~3.9x the old estimate
        return _make_stream(2, _FakeUsage(prompt=28, completion=4000))

    with task(ceiling_usd=ceiling, name="fix3-stream") as budget:
        with pytest.raises(BudgetExceeded):
            stream = wrap_mod.completion(
                model=MODEL, messages=MESSAGES, stream=True, real=stub_stream
            )
            # On the unfixed code the gate passes, the stream is returned, and
            # consuming it commits the 4000-token cost (overshoot) with NO raise
            # -> this line running means the test fails (DID NOT RAISE).
            list(stream)

    # The over-budget call never reached the delegate on the fixed code.
    assert delegate_invoked == [], "streamed over-budget call must not be delegated"
    # And nothing was committed (the gate tripped pre-call).
    assert budget.spent == 0.0
    assert budget.pending_usd == 0.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _actual_cost_for(completion_tokens: int) -> float:
    """Real USD cost of a gpt-4o response with prompt=28, completion=N (offline)."""
    from agentfuse.pricing import actual_cost

    resp = _FakeResponse(_FakeUsage(prompt=28, completion=completion_tokens))
    return actual_cost(resp)
