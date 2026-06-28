"""Tests for streaming-aware metering (v0.3.0 fix-stream-bypasses-fuse).

Before v0.3.0, a ``stream=True`` call returned a wrapper with no ``.usage``, so
``commit_actual`` committed $0 and the cumulative USD / token ledger never
advanced — the fuse could never trip on a streamed run, which is the dominant
agent call mode. These tests pin the fix: a streamed call advances the ledger
(via real usage when the provider emits it, else the pre-call estimate), and a
streaming loop eventually trips the cumulative fuse.

Also covers the v0.3.0 fix-fuse-max-tokens-name-collision rename: Fuse's
cumulative-token keyword is now ``max_total_tokens``; the old ``max_tokens`` is a
deprecated alias that still maps to the cumulative ceiling.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from agentfuse import Budget, BudgetExceeded, Fuse
from agentfuse.fuse import task
from agentfuse.stream import (
    is_stream_response,
    meter_async_stream,
    meter_sync_stream,
)

wrap_mod = importlib.import_module("agentfuse.wrap")

MODEL = "gpt-4o"
MESSAGES = [
    {"role": "system", "content": "You are an autonomous agent."},
    {"role": "user", "content": "Scan the network and report back in detail."},
]


# --------------------------------------------------------------------------- #
# Fakes: a streamed response is an iterator of chunks; the final chunk may
# carry a .usage block (litellm's stream_options include_usage behaviour).
# --------------------------------------------------------------------------- #


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _FakeChunk:
    def __init__(self, content: str, usage: _FakeUsage | None = None, model: str = MODEL) -> None:
        self.content = content
        self.model = model
        # Only the final chunk carries usage; the rest expose None.
        self.usage = usage


def _make_stream(n_chunks: int, final_usage: _FakeUsage | None):
    """A sync generator of chunks; only the last carries usage (if provided)."""
    for i in range(n_chunks):
        is_last = i == n_chunks - 1
        yield _FakeChunk(f"chunk {i}", usage=final_usage if is_last else None)


async def _make_astream(n_chunks: int, final_usage: _FakeUsage | None):
    for i in range(n_chunks):
        is_last = i == n_chunks - 1
        yield _FakeChunk(f"chunk {i}", usage=final_usage if is_last else None)


# --------------------------------------------------------------------------- #
# is_stream_response detection
# --------------------------------------------------------------------------- #


def test_is_stream_response_true_for_iterator_without_usage():
    assert is_stream_response(_make_stream(2, None)) is True


def test_is_stream_response_false_for_finished_response_with_usage():
    class FinalResp:
        model = MODEL
        usage = _FakeUsage(10, 5)

    assert is_stream_response(FinalResp()) is False


def test_is_stream_response_false_for_plain_containers():
    assert is_stream_response("a string") is False
    assert is_stream_response({"a": 1}) is False
    assert is_stream_response([1, 2, 3]) is False


# --------------------------------------------------------------------------- #
# meter_sync_stream: ledger advances from real usage on exhaustion
# --------------------------------------------------------------------------- #


def test_sync_stream_commits_real_usage_on_exhaustion():
    b = Budget(ceiling_usd=100.0)
    usage = _FakeUsage(prompt=1000, completion=500)
    stream = meter_sync_stream(_make_stream(3, usage), b, estimated_usd=0.01)

    # Nothing committed until the stream is consumed.
    assert b.spent == 0.0
    chunks = list(stream)
    assert len(chunks) == 3
    # Real usage was priced and committed (gpt-4o priced > 0).
    assert b.spent > 0.0
    assert b.spent_tokens == 1500


def test_sync_stream_falls_back_to_estimate_when_no_usage():
    b = Budget(ceiling_usd=100.0)
    stream = meter_sync_stream(_make_stream(3, None), b, estimated_usd=0.25)
    list(stream)
    # No usage block emitted -> commit the pre-call estimate so the ledger moves.
    assert b.spent == pytest.approx(0.25)


def test_sync_stream_yields_chunks_transparently():
    b = Budget(ceiling_usd=100.0)
    out = [c.content for c in meter_sync_stream(_make_stream(4, None), b, estimated_usd=0.0)]
    assert out == ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]


# --------------------------------------------------------------------------- #
# The core guarantee: a streamed loop TRIPS the cumulative fuse.
# (Pre-v0.3.0 this never happened — the ledger stayed at 0 forever.)
# --------------------------------------------------------------------------- #


def test_streamed_calls_trip_cumulative_usd_fuse():
    sent = 0

    def stub_delegate(*args, **kwargs):
        nonlocal sent
        sent += 1
        # A streamed delegate returns an iterator with usage on the final chunk.
        return _make_stream(2, _FakeUsage(prompt=2000, completion=1000))

    with task(ceiling_usd=0.05, name="stream-task") as budget:
        tripped = False
        for _ in range(50):
            try:
                stream = wrap_mod.completion(
                    model=MODEL,
                    messages=MESSAGES,
                    max_tokens=1000,
                    stream=True,
                    real=stub_delegate,
                )
            except BudgetExceeded:
                tripped = True
                break
            # Consume the stream so its usage is committed to the ledger.
            list(stream)
        assert tripped, "cumulative USD fuse must eventually trip on streamed calls"
        assert budget.spent > 0.0
        # The fuse stopped sending before runaway: far fewer than 50 calls went out.
        assert sent < 50


def test_streamed_call_advances_ledger_even_without_usage():
    def stub_delegate(*args, **kwargs):
        return _make_stream(3, None)  # provider emits no usage block

    with task(ceiling_usd=100.0, name="t") as budget:
        stream = wrap_mod.completion(
            model=MODEL,
            messages=MESSAGES,
            max_tokens=200,
            stream=True,
            real=stub_delegate,
        )
        list(stream)  # exhaust -> fallback-commit the estimate
        # Estimate-fallback advanced the ledger instead of leaving it at 0.
        assert budget.spent > 0.0


# --------------------------------------------------------------------------- #
# async streaming path
# --------------------------------------------------------------------------- #


def test_async_stream_commits_real_usage_on_exhaustion():
    async def run():
        b = Budget(ceiling_usd=100.0)
        usage = _FakeUsage(prompt=800, completion=400)
        out = []
        async for chunk in meter_async_stream(_make_astream(2, usage), b, estimated_usd=0.01):
            out.append(chunk.content)
        return b, out

    b, out = asyncio.run(run())
    assert out == ["chunk 0", "chunk 1"]
    assert b.spent > 0.0
    assert b.spent_tokens == 1200


# --------------------------------------------------------------------------- #
# fix-fuse-max-tokens-name-collision: rename + deprecated alias
# --------------------------------------------------------------------------- #


def test_fuse_max_total_tokens_sets_cumulative_ceiling():
    with Fuse(max_spend_usd=100.0, max_total_tokens=4096) as budget:
        assert budget.ceiling_tokens == 4096


def test_fuse_max_tokens_alias_is_deprecated_but_works():
    with pytest.warns(DeprecationWarning):
        f = Fuse(max_spend_usd=100.0, max_tokens=4096)
    # Still maps to the cumulative ceiling for back-compat.
    with f as budget:
        assert budget.ceiling_tokens == 4096
    # The deprecated property still reads back the cumulative value.
    assert f.max_tokens == 4096
    assert f.max_total_tokens == 4096


def test_fuse_rejects_conflicting_token_keywords():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            Fuse(max_spend_usd=100.0, max_total_tokens=1000, max_tokens=2000)


def test_fuse_no_token_ceiling_by_default():
    with Fuse(max_spend_usd=100.0) as budget:
        assert budget.ceiling_tokens is None
