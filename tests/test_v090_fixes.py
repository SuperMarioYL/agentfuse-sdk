"""Adversarial red->green tests for the v0.9.0 amendment's three milestones.

Milestone 1 ``fix-nousage-finished-response-commits-zero-usd``:

A finished (non-stream) response that carries NO usage block at all committed
$0.00 and discarded the pre-call estimate: ``resolve_commit_cost``'s fallback
guard required ``tokens > 0``, but with no ``.usage`` attribute
``actual_tokens`` returns 0, so the guard never fired and
``budget.commit(0.0, 0, reservation=...)`` released the reservation while
adding nothing to the ledger. The stream.py docstring itself documents that
usage-less non-stream completions are "common for open / self-hosted models
via ollama / watsonx" — and the bypass hit priced models too (any provider
that omits usage). The fix treats "no usage information at all" the same way
the streaming no-usage branch already does: commit the conservative pre-call
estimate (USD and, via the newly threaded ``estimated_tokens``, the token
bound). A usage block that REPORTS zero tokens still commits an honest 0.0.

Milestone 2 ``fix-unconsumed-metered-stream-never-settles`` (appended below).

Milestone 3 ``fix-precall-estimate-ignores-n-choices`` (appended below).

Tests use stub delegates + real litellm pricing tables; no network, no API key.
"""

from __future__ import annotations

import importlib

import pytest

from agentfuse import BudgetExceeded, UnpricedModelError
from agentfuse.fuse import task
from agentfuse.pricing import FALLBACK_USD_PER_TOKEN, estimate_call

wrap_mod = importlib.import_module("agentfuse.wrap")

UNPRICED_MODEL = "my-self-hosted-model"
PRICED_MODEL = "gpt-4o"
MESSAGES = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# Fakes: a FINISHED response (carries .choices) that carries NO usage block.
# stream.py:57-63 documents this shape as common for ollama / watsonx.
# --------------------------------------------------------------------------- #


class _Choice:
    message = {"role": "assistant", "content": "ok"}


class _FinishedResponseNoUsage:
    """Mimics a litellm ModelResponse WITHOUT a .usage attribute."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.choices = [_Choice()]
        self.id = "cmpl-x"
        self.object = "chat.completion"


class _FinishedResponseZeroUsage(_FinishedResponseNoUsage):
    """A finished response whose usage block REPORTS zero tokens."""

    def __init__(self, model: str) -> None:
        super().__init__(model)

        class _Usage:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        self.usage = _Usage()


# --------------------------------------------------------------------------- #
# Milestone 1: fix-nousage-finished-response-commits-zero-usd
# --------------------------------------------------------------------------- #


class TestNoUsageFinishedResponseCommitsEstimate:
    def test_unpriced_model_nousage_response_trips_the_usd_fuse(self):
        """50 no-usage non-stream calls under on_unpriced='fallback' must trip.

        RED on v0.8.0: every call committed $0.00 (the tokens > 0 guard never
        fired), spent_usd stayed pinned at $0.000000 and the ceiling never
        tripped (50 sent, 0 trips).
        """
        sent = {"n": 0}

        def delegate(*args, **kwargs):
            sent["n"] += 1
            return _FinishedResponseNoUsage(UNPRICED_MODEL)

        with task(ceiling_usd=0.50, on_unpriced="fallback") as budget:
            for _ in range(50):
                try:
                    wrap_mod.completion(
                        model=UNPRICED_MODEL, messages=MESSAGES, real=delegate
                    )
                except UnpricedModelError:  # pragma: no cover - guard the policy
                    pytest.fail("on_unpriced='fallback' must not block unpriced models")
                except BudgetExceeded:
                    break
            else:  # pragma: no cover
                pytest.fail("USD ceiling never tripped after 50 no-usage calls")
        snap = budget.snapshot()
        assert sent["n"] < 50, "the fuse must trip before the loop is exhausted"
        assert snap.spent_usd > 0.0, "the pre-call estimates must reach the ledger"

    def test_priced_model_nousage_response_trips_the_usd_fuse(self):
        """Same bypass on a PRICED model: usage-less gpt-4o responses must trip.

        RED on v0.8.0: litellm.completion_cost and cost_per_token both fail on
        a response without usage, actual_cost returned 0.0, and the ceiling
        never tripped (50 sent, 0 trips).
        """
        sent = {"n": 0}

        def delegate(*args, **kwargs):
            sent["n"] += 1
            return _FinishedResponseNoUsage(PRICED_MODEL)

        with task(ceiling_usd=0.50) as budget:  # default on_unpriced='block'
            for _ in range(50):
                try:
                    wrap_mod.completion(
                        model=PRICED_MODEL, messages=MESSAGES, real=delegate
                    )
                except BudgetExceeded:
                    break
            else:  # pragma: no cover
                pytest.fail("USD ceiling never tripped after 50 no-usage calls")
        snap = budget.snapshot()
        assert sent["n"] < 50
        assert snap.spent_usd > 0.0

    def test_nousage_response_commits_the_token_estimate_too(self):
        """The token ledger must advance on the no-usage fallback (stream parity).

        RED on v0.8.0: the commit was (0.0, 0) — the token ledger froze too.
        GREEN: the no-usage fallback commits the pre-call estimated_tokens,
        mirroring the streaming no-usage fallback (v0.6.0 fix).
        """

        def delegate(*args, **kwargs):
            return _FinishedResponseNoUsage(UNPRICED_MODEL)

        with task(ceiling_usd=10.0, on_unpriced="fallback") as budget:
            wrap_mod.completion(model=UNPRICED_MODEL, messages=MESSAGES, real=delegate)
        snap = budget.snapshot()
        _usd, est_tokens = estimate_call(
            UNPRICED_MODEL, MESSAGES, on_unpriced="fallback"
        )
        assert snap.spent_usd == pytest.approx(_usd)
        assert snap.spent_tokens == est_tokens
        assert snap.pending_tokens == 0, "the reservation must be released"

    def test_usage_block_reporting_zero_tokens_still_commits_honest_zero(self):
        """Boundary pin: usage PRESENT but zero -> (0.0, 0), no estimate fallback.

        GREEN on v0.8.0 too — the fallback must fire only when the usage block
        is absent (or unpriceable with real tokens), not for a genuine zero.
        """

        def delegate(*args, **kwargs):
            return _FinishedResponseZeroUsage(PRICED_MODEL)

        with task(ceiling_usd=10.0) as budget:
            wrap_mod.completion(model=PRICED_MODEL, messages=MESSAGES, real=delegate)
        snap = budget.snapshot()
        assert snap.spent_usd == 0.0
        assert snap.spent_tokens == 0
        assert snap.pending_usd == 0.0, "the reservation must still be released"


# --------------------------------------------------------------------------- #
# Milestone 2: fix-unconsumed-metered-stream-never-settles
# --------------------------------------------------------------------------- #


class _StreamChunk:
    def __init__(self, usage=None) -> None:
        self.content = "x"
        self.model = UNPRICED_MODEL
        self.usage = usage


class _StreamUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


def _sync_stream_delegate(*args, **kwargs):
    return iter([_StreamChunk() for _ in range(4)])


class _AsyncStream:
    def __init__(self) -> None:
        self._chunks = [_StreamChunk() for _ in range(4)]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


async def _async_stream_delegate(*args, **kwargs):
    return _AsyncStream()


class TestUnconsumedMeteredStreamSettles:
    def test_zero_consumption_stream_settles_on_drop(self):
        """A stream that is NEVER iterated must still settle when dropped.

        RED on v0.8.0: meter_sync_stream is a generator; a generator that was
        never started runs NO body code on close/deallocation, so the finally
        (commit + reservation release) is unreachable — pending_usd stays
        pinned at the estimate forever and the abandoned call (whose delegate
        RAN and was billed) commits nothing.
        """
        with task(ceiling_usd=10.0, on_unpriced="fallback") as budget:
            # Fire the call and drop the metered stream without iterating it.
            wrap_mod.completion(
                model=UNPRICED_MODEL,
                messages=MESSAGES,
                stream=True,
                real=_sync_stream_delegate,
            )
        snap = budget.snapshot()
        est_usd, est_tokens = estimate_call(
            UNPRICED_MODEL, MESSAGES, on_unpriced="fallback"
        )
        assert snap.pending_usd == pytest.approx(0.0), (
            "the reservation must not stay pinned after the stream is dropped"
        )
        assert snap.pending_tokens == 0
        assert snap.spent_usd == pytest.approx(est_usd)
        assert snap.spent_tokens == est_tokens

    def test_zero_consumption_async_stream_settles_on_drop(self):
        """Async variant: a never-iterated metered async stream must settle.

        RED on v0.8.0 for the same unstarted-generator reason.
        """
        import asyncio

        async def _fire_and_drop():
            await wrap_mod.acompletion(
                model=UNPRICED_MODEL,
                messages=MESSAGES,
                stream=True,
                real=_async_stream_delegate,
            )

        async def _scenario():
            with task(ceiling_usd=10.0, on_unpriced="fallback") as budget:
                await _fire_and_drop()
                return budget

        budget = asyncio.run(_scenario())
        snap = budget.snapshot()
        est_usd, est_tokens = estimate_call(
            UNPRICED_MODEL, MESSAGES, on_unpriced="fallback"
        )
        assert snap.pending_usd == pytest.approx(0.0)
        assert snap.pending_tokens == 0
        assert snap.spent_usd == pytest.approx(est_usd)
        assert snap.spent_tokens == est_tokens

    def test_close_settles_without_consuming(self):
        """An explicit close() on a never-iterated metered stream must settle.

        RED on v0.8.0: close() on an unstarted generator runs no body code.
        """
        with task(ceiling_usd=10.0, on_unpriced="fallback") as budget:
            metered = wrap_mod.completion(
                model=UNPRICED_MODEL,
                messages=MESSAGES,
                stream=True,
                real=_sync_stream_delegate,
            )
            metered.close()
            snap = budget.snapshot()
        est_usd, _est_tokens = estimate_call(
            UNPRICED_MODEL, MESSAGES, on_unpriced="fallback"
        )
        assert snap.pending_usd == pytest.approx(0.0)
        assert snap.spent_usd == pytest.approx(est_usd)

    def test_one_chunk_then_drop_still_settles(self):
        """Regression pin (green on v0.8.0 too): partial consumption settles."""
        with task(ceiling_usd=10.0, on_unpriced="fallback") as budget:
            metered = wrap_mod.completion(
                model=UNPRICED_MODEL,
                messages=MESSAGES,
                stream=True,
                real=_sync_stream_delegate,
            )
            next(iter(metered))
            del metered
        snap = budget.snapshot()
        est_usd, est_tokens = estimate_call(
            UNPRICED_MODEL, MESSAGES, on_unpriced="fallback"
        )
        assert snap.pending_usd == pytest.approx(0.0)
        assert snap.spent_usd == pytest.approx(est_usd)
        assert snap.spent_tokens == est_tokens

    def test_full_consume_with_usage_commits_real_usage(self):
        """Regression pin (green on v0.8.0 too): real usage wins when emitted."""
        usage = _StreamUsage(100, 200)

        def delegate(*args, **kwargs):
            return iter([_StreamChunk(), _StreamChunk(usage)])

        with task(ceiling_usd=10.0, on_unpriced="fallback") as budget:
            for _ in wrap_mod.completion(
                model=UNPRICED_MODEL, messages=MESSAGES, stream=True, real=delegate
            ):
                pass
        snap = budget.snapshot()
        assert snap.spent_tokens == 300
        assert snap.pending_usd == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Milestone 3: fix-precall-estimate-ignores-n-choices
# --------------------------------------------------------------------------- #


class TestPrecallEstimateIncludesNChoices:
    def test_estimate_call_n_multiplies_the_completion_bound(self):
        """n=4 must multiply the COMPLETION half of both bounds (max_tokens is
        per choice in OpenAI/litellm semantics).

        RED on v0.8.0: estimate_call had no n parameter and priced exactly one
        completion, so the pre-call 'upper bound' was 1x of a 4x bill.
        """
        import litellm

        from agentfuse.pricing import count_prompt_tokens

        prices = litellm.model_cost[PRICED_MODEL]
        in_cost = float(prices["input_cost_per_token"])
        out_cost = float(prices["output_cost_per_token"])
        prompt = count_prompt_tokens(PRICED_MODEL, MESSAGES)

        usd1, tok1 = estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000)
        usd4, tok4 = estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000, n=4)
        assert tok1 == prompt + 1000
        assert tok4 == prompt + 4 * 1000
        assert usd1 == pytest.approx(prompt * in_cost + 1000 * out_cost)
        assert usd4 == pytest.approx(prompt * in_cost + 4 * 1000 * out_cost)

    def test_estimate_call_n_one_and_omitted_are_identical(self):
        """Regression pin: n=1 (and omitted n) stay bit-identical to v0.8.0."""
        assert estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000, n=1) == (
            estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000)
        )

    def test_estimate_call_fallback_policy_includes_n(self):
        """The on_unpriced='fallback' bound inherits the n factor too."""
        usd1, tok1 = estimate_call(
            UNPRICED_MODEL, MESSAGES, on_unpriced="fallback"
        )
        usd4, tok4 = estimate_call(
            UNPRICED_MODEL, MESSAGES, n=4, on_unpriced="fallback"
        )
        assert tok4 == tok1 + 3 * 8192
        assert usd4 == pytest.approx(tok4 * FALLBACK_USD_PER_TOKEN)

    def test_estimate_call_floors_n_at_one(self):
        """n=0 / negative floor at 1 instead of producing a nonsense bound."""
        assert estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000, n=0) == (
            estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000, n=1)
        )
        assert estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000, n=-3) == (
            estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000, n=1)
        )

    def test_n_call_is_blocked_precall_when_the_n_bound_crosses_the_ceiling(self):
        """End-to-end: an n=4 call whose 4x estimate crosses the ceiling is
        blocked BEFORE the delegate runs, so the over-bill never happens.

        RED on v0.8.0: the gate passed on the 1x estimate, the call went out,
        and the post-call commit of the real 4x usage landed past the ceiling
        with no retroactive trip (one-shot task overshoots).
        """
        est1, _tok1 = estimate_call(PRICED_MODEL, MESSAGES, max_tokens=1000)
        ceiling = est1 * 2  # 1x estimate passes; 4x estimate crosses

        called = {"n": 0}

        def delegate(*args, **kwargs):
            called["n"] += 1
            return _FinishedResponseNoUsage(PRICED_MODEL)

        with task(ceiling_usd=ceiling) as budget:
            with pytest.raises(BudgetExceeded):
                wrap_mod.completion(
                    model=PRICED_MODEL,
                    messages=MESSAGES,
                    max_tokens=1000,
                    n=4,
                    real=delegate,
                )
        assert called["n"] == 0, "the blocked call must never reach the delegate"
        assert budget.snapshot().spent_usd == pytest.approx(0.0)
