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
