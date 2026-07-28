"""Adversarial red→green tests for the v0.6.0 amendment's two `type: fix` milestones.

Each milestone has at least one test that FAILS on the unfixed v0.5.0 source and
PASSES on the v0.6.0 fixed source (the red→green contract the plan demands):

* ``fix-stream-detector-misclassifies-usageless-modelresponse`` —
  ``is_stream_response()`` only treated a response as finished when it carried a
  truthy ``.usage``; otherwise any iterable was classified as a stream. litellm's
  ``ModelResponse`` is iterable (pydantic ``__iter__`` yields field tuples) and its
  ``.usage`` is attached dynamically, so a non-stream completion that returns no
  token usage arrives WITHOUT ``.usage`` and was misclassified as a stream. The
  wrapper then returned the metering GENERATOR instead of the ``ModelResponse``
  (``resp.choices[0].message.content`` raised ``AttributeError``) and, because the
  generator was never iterated, its ``finally`` never ran, so the pre-call
  ``Reservation`` leaked into the budget's ``pending`` balance forever.
  ``is_stream_response()`` now rejects litellm's finished ``ModelResponse`` (it
  exposes ``.choices``) BEFORE the iterable check.

* ``fix-stream-meter-drops-token-estimate`` —
  ``meter_sync_stream()`` / ``meter_async_stream()`` took ``estimated_usd`` but NOT
  ``estimated_tokens``; the no-usage fallback committed ``0`` tokens, freezing the
  cumulative token ledger so the ``ceiling_tokens`` fuse never tripped on the
  dominant streamed no-usage call mode (ollama / watsonx self-hosted models that
  omit a usage block). ``estimated_tokens`` is now threaded through both meter
  functions (and the wrap.py call sites) and committed in the no-usage fallback,
  mirroring the existing USD-estimate fallback on the same path.

Tests use real litellm types (``mock_response``) + stub delegates; no network, no
API key.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from agentfuse import Budget, BudgetExceeded
from agentfuse.fuse import task
from agentfuse.pricing import estimate_call
from agentfuse.stream import is_stream_response, meter_async_stream, meter_sync_stream

# `agentfuse.wrap` the *attribute* (re-exported by __init__) is the install()
# callable and shadows the submodule. Reach the actual module via importlib for
# the wrapper-internals tests (same pattern as tests/test_fuse.py).
wrap_mod = importlib.import_module("agentfuse.wrap")

MODEL = "gpt-4o"
MESSAGES = [
    {"role": "system", "content": "You are an autonomous agent."},
    {"role": "user", "content": "Scan the network and report back in detail."},
]
SHORT_MESSAGES = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# Fakes / real-litellm helpers
# --------------------------------------------------------------------------- #


class _NoUsageChunk:
    """A stream chunk carrying ``.content`` / ``.model`` but NO ``.usage`` block.

    Mirrors a provider (ollama / watsonx) that omits the usage block on every
    chunk, so the streamed no-usage fallback path is the only commit route.
    """

    def __init__(self, i: int) -> None:
        self.content = f"chunk {i}"
        self.model = MODEL
        self.usage = None


def _no_usage_stream(n: int):
    """A sync generator of ``n`` chunks, none carrying a usage block."""
    for i in range(n):
        yield _NoUsageChunk(i)


async def _async_no_usage_stream(n: int):
    """An async generator of ``n`` chunks, none carrying a usage block."""
    for i in range(n):
        yield _NoUsageChunk(i)


def _real_model_response_without_usage():
    """A real litellm ``ModelResponse`` with its ``.usage`` stripped.

    ``litellm.completion(mock_response=...)`` returns a ``ModelResponse`` that DOES
    carry a ``.usage`` block, but real no-usage provider paths (ollama / watsonx
    self-hosted models) deliver the finished ``ModelResponse`` WITHOUT ``.usage``.
    The ``ModelResponse`` is iterable (pydantic ``__iter__`` yields field tuples)
    and exposes ``.choices`` — exactly the shape v0.5.0's ``is_stream_response``
    misclassified as a stream. Verified end-to-end through real litellm (per the
    amendment, not a stub).

    Uses the wrapper's import-time-captured ``_REAL_COMPLETION`` rather than the
    ``litellm.completion`` attribute: a prior install/uninstall test in
    ``test_fuse.py`` can leave ``litellm.completion`` transiently pointing at a
    stub, whereas ``_REAL_COMPLETION`` is the genuine litellm completion captured
    once at module import (and only ever swapped by tests' ``monkeypatch``, which
    always reverts). This keeps the test robust to that global-state leak.
    """
    resp = wrap_mod._REAL_COMPLETION(
        model=MODEL, messages=SHORT_MESSAGES, mock_response="hello world"
    )
    # Strip the dynamically-attached .usage so the response mimics a no-usage
    # provider. `del` makes the attribute truly absent (matches real providers);
    # the assignment fallback covers a pydantic version where del is rejected —
    # present-but-None still trips the v0.5.0 iterable check the same way.
    try:
        del resp.usage
    except Exception:
        resp.usage = None
    return resp


# =========================================================================== #
# fix-stream-detector-misclassifies-usageless-modelresponse
# =========================================================================== #


def test_v060_is_stream_response_false_for_usageless_real_modelresponse():
    """RED on v0.5.0 (iterable ModelResponse w/o .usage -> True), GREEN on v0.6.0.

    A finished litellm ``ModelResponse`` is iterable (pydantic ``__iter__``) and,
    when the provider emits no usage, arrives WITHOUT ``.usage``. v0.5.0 classified
    it as a stream; v0.6.0 rejects the finished ``ModelResponse`` (exposes
    ``.choices``) before the iterable check.
    """
    resp = _real_model_response_without_usage()
    # Sanity: this is exactly the misclassified shape.
    assert hasattr(resp, "choices"), "real ModelResponse exposes .choices"
    assert not hasattr(resp, "usage"), ".usage was stripped (no-usage provider path)"
    assert hasattr(resp, "__iter__"), "ModelResponse is iterable (pydantic __iter__)"
    # The load-bearing assertion: a finished ModelResponse is NOT a stream.
    assert is_stream_response(resp) is False


def test_v060_completion_returns_modelresponse_not_generator_when_no_usage():
    """RED on v0.5.0 (wrap returns the meter_sync_stream GENERATOR ->
    AttributeError on ``resp.choices[0].message.content`` + leaked Reservation
    pinned in pending forever), GREEN on v0.6.0 (wrap returns the ModelResponse,
    no leak).

    A non-stream completion whose provider returns no usage must still come back as
    the finished ``ModelResponse`` (so the agent's ``resp.choices[0].message.content``
    works) and must NOT leak the pre-call ``Reservation`` into the budget's pending
    balance.
    """
    def stub_delegate(*args, **kwargs):
        return _real_model_response_without_usage()

    with task(ceiling_usd=100.0, name="fix1-e2e") as budget:
        resp = wrap_mod.completion(
            model=MODEL, messages=SHORT_MESSAGES, max_tokens=200, real=stub_delegate
        )
        # v0.6.0: the finished ModelResponse is returned, not the metering generator.
        assert hasattr(resp, "choices"), (
            "non-stream completion must return the ModelResponse, not a generator"
        )
        assert resp.choices[0].message.content == "hello world"
        # The reservation was settled by commit_actual (released), NOT leaked into
        # pending because the generator's finally never ran.
        assert budget.pending_usd == 0.0
        assert budget.pending_tokens == 0


# =========================================================================== #
# fix-stream-meter-drops-token-estimate
# =========================================================================== #


def test_v060_sync_stream_no_usage_fallback_commits_estimated_tokens():
    """RED on v0.5.0 (``meter_sync_stream`` did not accept ``estimated_tokens``;
    the no-usage fallback committed 0 tokens), GREEN on v0.6.0 (threads
    ``estimated_tokens`` and commits it in the no-usage fallback).

    The pre-call ``reserve()`` had added ``est_tokens`` to ``pending_tokens``; the
    no-usage fallback released the reservation (subtracting ``est_tokens`` from
    pending) but committed ``0`` to ``spent_tokens``, freezing the cumulative
    token ledger. v0.6.0 threads ``estimated_tokens`` through and commits it.
    """
    b = Budget(ceiling_usd=100.0, ceiling_tokens=10_000)
    reservation = b.reserve(0.25, 999)  # pre-call reservation carrying est_tokens
    assert b.pending_tokens == 999
    stream = meter_sync_stream(
        _no_usage_stream(3), b,
        estimated_usd=0.25, estimated_tokens=999, reservation=reservation,
    )
    list(stream)  # exhaust -> no-usage fallback commit
    # v0.6.0: the token estimate was committed (not 0), so the ledger advanced.
    assert b.spent_tokens == 999, (
        f"no-usage fallback must commit the estimated tokens (got {b.spent_tokens})"
    )
    # Reservation settled: pending cleared, not leaked.
    assert b.pending_tokens == 0
    assert reservation.active is False


def test_v060_async_stream_no_usage_fallback_commits_estimated_tokens():
    """Async variant: ``meter_async_stream`` now threads + commits
    ``estimated_tokens`` in the no-usage fallback (RED on v0.5.0, GREEN on v0.6.0)."""
    async def run():
        b = Budget(ceiling_usd=100.0, ceiling_tokens=10_000)
        reservation = b.reserve(0.25, 777)
        out = []
        async for chunk in meter_async_stream(
            _async_no_usage_stream(2), b,
            estimated_usd=0.25, estimated_tokens=777, reservation=reservation,
        ):
            out.append(chunk.content)
        return b, reservation, out

    b, reservation, out = asyncio.run(run())
    assert out == ["chunk 0", "chunk 1"]
    assert b.spent_tokens == 777, (
        f"async no-usage fallback must commit the estimated tokens (got {b.spent_tokens})"
    )
    assert b.pending_tokens == 0
    assert reservation.active is False


def test_v060_streamed_no_usage_call_advances_token_ledger():
    """RED on v0.5.0 (no-usage fallback committed 0 tokens -> spent_tokens==0),
    GREEN on v0.6.0 (commits the pre-call token estimate -> spent_tokens advances).

    A single streamed no-usage call must advance the cumulative token ledger by the
    pre-call estimate (not leave it frozen at 0). The stub emits NO usage block, so
    the only way spent_tokens can equal the pre-call estimate is the no-usage
    fallback committing it.
    """
    max_tokens = 200
    _est_usd, est_tokens = estimate_call(MODEL, MESSAGES, max_tokens=max_tokens)

    def stub_delegate(*args, **kwargs):
        return _no_usage_stream(2)  # provider emits NO usage block

    with task(ceiling_usd=100.0, ceiling_tokens=10_000, name="fix2-advance") as budget:
        stream = wrap_mod.completion(
            model=MODEL, messages=MESSAGES, max_tokens=max_tokens,
            stream=True, real=stub_delegate,
        )
        list(stream)  # exhaust -> no-usage fallback commits the estimate
        # v0.6.0: the token estimate was committed, not 0.
        assert budget.spent_tokens == est_tokens, (
            f"streamed no-usage call must commit the pre-call token estimate "
            f"({est_tokens}); got {budget.spent_tokens}"
        )
        # Reservation settled by the commit, not leaked into pending.
        assert budget.pending_tokens == 0
        assert budget.pending_usd == 0.0


def test_v060_async_streamed_no_usage_call_advances_token_ledger():
    """Async variant: a streamed no-usage ``acompletion`` must advance the token
    ledger (RED on v0.5.0, GREEN on v0.6.0)."""
    max_tokens = 200
    _est_usd, est_tokens = estimate_call(MODEL, MESSAGES, max_tokens=max_tokens)

    async def stub_delegate(*args, **kwargs):
        return _async_no_usage_stream(2)

    async def run():
        with task(ceiling_usd=100.0, ceiling_tokens=10_000, name="fix2-async") as budget:
            stream = await wrap_mod.acompletion(
                model=MODEL, messages=MESSAGES, max_tokens=max_tokens,
                stream=True, real=stub_delegate,
            )
            out = [c.content async for c in stream]
            return budget, out

    budget, out = asyncio.run(run())
    assert out == ["chunk 0", "chunk 1"]
    assert budget.spent_tokens == est_tokens, (
        f"async streamed no-usage call must commit the pre-call token estimate "
        f"({est_tokens}); got {budget.spent_tokens}"
    )
    assert budget.pending_tokens == 0
    assert budget.pending_usd == 0.0


def test_v060_streamed_no_usage_calls_trip_cumulative_token_fuse():
    """RED on v0.5.0 (spent_tokens frozen at 0 -> token ceiling NEVER trips on the
    dominant streamed no-usage call mode), GREEN on v0.6.0 (each no-usage stream
    commits ``est_tokens`` -> ``spent_tokens`` advances -> cumulative token ceiling
    trips, first-to-trip wins against the USD ceiling).

    Mirrors the amendment's verification: repeated streamed no-usage calls under a
    token ceiling. Before the fix the token ledger stayed at 0 and the token fuse
    was inert while the USD ceiling advanced.
    """
    max_tokens = 200
    _est_usd, est_tokens = estimate_call(MODEL, MESSAGES, max_tokens=max_tokens)
    # Tune the token ceiling so >=1 streamed no-usage call fits before the
    # cumulative ceiling trips: > 1x est_tokens (one call fits), < 3x (third trips).
    ceiling_tokens = int(est_tokens * 2.5)

    sent = 0

    def stub_delegate(*args, **kwargs):
        nonlocal sent
        sent += 1
        return _no_usage_stream(2)  # provider emits NO usage block

    with task(ceiling_usd=100.0, ceiling_tokens=ceiling_tokens, name="fix2-ceiling") as budget:
        tripped = False
        for _ in range(50):
            try:
                stream = wrap_mod.completion(
                    model=MODEL, messages=MESSAGES, max_tokens=max_tokens,
                    stream=True, real=stub_delegate,
                )
            except BudgetExceeded:
                tripped = True
                break
            list(stream)  # exhaust -> no-usage fallback commits est_tokens
        assert tripped, (
            "cumulative token fuse must eventually trip on streamed no-usage calls"
        )
        # The token ledger actually advanced (was frozen at 0 before the fix).
        assert budget.spent_tokens > 0
        # Runaway stopped: far fewer than 50 calls went out.
        assert sent < 50
        # No reservation leaked into pending.
        assert budget.pending_tokens == 0
        assert budget.pending_usd == 0.0
