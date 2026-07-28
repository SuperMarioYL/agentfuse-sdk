"""Streaming-aware metering so the cumulative fuse trips on streamed agents.

The wrapper in :mod:`agentfuse.wrap` delegates to ``litellm.completion`` /
``litellm.acompletion`` and then commits the *real* cost read back from the
response's ``Usage``. That works for a normal (non-streamed) response, which
carries a ``.usage`` block. But when a caller passes ``stream=True``, litellm
returns a ``CustomStreamWrapper`` (sync) or an async iterator of chunks — an
object with **no** ``.usage`` until it is consumed. If we naively call
:func:`agentfuse.pricing.actual_cost` on that wrapper it returns ``0.0`` and the
cumulative ledger never advances, so the USD / token ceilings can **never trip on
a streamed run** — and streaming is the dominant call mode for agent runtimes.

This module closes that hole. It wraps the streamed object so AgentFuse:

1. yields every chunk straight through to the caller (transparent — the caller's
   loop is unchanged),
2. watches the chunks for a usage block (litellm emits one on the final chunk
   when ``stream_options={"include_usage": True}``),
3. when the stream is exhausted, commits the **real** cost if a usage block was
   seen, otherwise commits the **pre-call upper-bound estimate** so the ledger
   still advances and the fuse still trips on the *next* call.

This is pure execution-time metering — no dashboard, no visualization, no
monitoring service. It only keeps the existing per-task ledger honest for the one
response shape (streaming) that previously slipped past it.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator

from agentfuse.budget import Budget, Reservation
from agentfuse.pricing import actual_cost, actual_tokens

logger = logging.getLogger("agentfuse.stream")


def is_stream_response(response: Any) -> bool:
    """Return ``True`` if ``response`` is a streamed LLM response, not a final one.

    A non-streamed litellm response is a ``ModelResponse`` carrying a ``.usage``
    block. A streamed response is a ``CustomStreamWrapper`` / async iterator with
    no usable ``.usage`` until consumed. We detect the stream by the absence of a
    populated ``.usage`` combined with the object being iterable (sync or async).

    The check is deliberately conservative: anything that already exposes a
    truthy ``.usage`` is treated as a finished response (the wrapper commits it
    directly), so we never double-wrap a normal response.

    .. note::

        litellm's ``ModelResponse`` is ALSO iterable (pydantic's ``__iter__``
        yields ``(field, value)`` tuples), and ``.usage`` is attached dynamically
        rather than being a guaranteed core field — so a non-stream completion
        that returns no token usage (common for open / self-hosted models via
        ollama / watsonx) arrives WITHOUT ``.usage``. Without the guard below
        such a finished ``ModelResponse`` would be misclassified as a stream:
        the wrapper would return the metering GENERATOR instead of the
        ``ModelResponse`` (so ``resp.choices[0].message.content`` raises
        ``AttributeError``) and, because the generator is never iterated, its
        ``finally`` never runs and the pre-call ``Reservation`` leaks into the
        budget's ``pending`` balance forever. So litellm's finished response type
        is rejected BEFORE the iterable check — an object exposing ``.choices``
        (the ``ModelResponse`` marker the stream wrapper never carries) is a
        finished response regardless of ``.usage``.
    """
    # Reject litellm's finished (non-streamed) ModelResponse BEFORE the iterable
    # check: it carries .choices and is iterable, and may arrive without .usage.
    if hasattr(response, "choices"):
        return False
    usage = getattr(response, "usage", None)
    if usage is not None:
        # A finished response with a real usage block — not a stream.
        return False
    is_sync_iter = hasattr(response, "__iter__") and not isinstance(
        response, (str, bytes, dict, list, tuple)
    )
    is_async_iter = hasattr(response, "__aiter__")
    return bool(is_sync_iter or is_async_iter)


def _commit_from_chunks(
    budget: Budget,
    last_usage_holder: dict[str, Any],
    estimated_usd: float,
    estimated_tokens: int = 0,
    reservation: "Reservation | None" = None,
) -> None:
    """Commit a streamed call's spend once its chunks are exhausted.

    Prefers the real ``Usage`` captured off the final chunk; falls back to the
    pre-call upper-bound estimate so the cumulative ledger still advances (and
    the fuse can still trip on the next call) when the provider gave us no usage.
    The ``reservation`` (from the v0.5.0 race-free gate) is released atomically
    with the commit so the pending estimate is settled alongside the confirmed
    spend and concurrent callers stop seeing it as in-flight.

    The no-usage fallback commits BOTH the pre-call USD estimate AND the pre-call
    token estimate (mirroring the USD-estimate fallback on the same path). The
    pre-call ``reserve()`` had correctly added ``est_tokens`` to the budget's
    pending-token ledger; committing ``0`` tokens here (as v0.5.0 did) would
    release the reservation (subtracting ``est_tokens`` from pending) while
    adding ``0`` to spent-tokens, freezing the cumulative token ledger at ``0``
    for every streamed no-usage call and leaving the ``ceiling_tokens`` fuse
    inert for the dominant streamed call mode (ollama / watsonx self-hosted
    models that omit a usage block).
    """
    usage_obj = last_usage_holder.get("usage")
    if usage_obj is not None:
        # Build a tiny shim that exposes .usage so the existing pricing helpers work.
        shim = _UsageShim(usage_obj, last_usage_holder.get("model"))
        cost = actual_cost(shim)
        tokens = actual_tokens(shim)
        budget.commit(cost, tokens, reservation=reservation)
        return
    # No usage emitted by the provider — commit the conservative pre-call estimate
    # so the ledger moves forward instead of silently staying at 0. Commit the
    # token estimate too so the cumulative token fuse still advances on the
    # dominant streamed no-usage call mode (the USD estimate alone is not enough).
    logger.debug(
        "streamed response carried no usage; committing the pre-call estimate "
        "($%.6f, %d tokens) so the cumulative fuse still advances",
        estimated_usd,
        estimated_tokens,
    )
    budget.commit(float(estimated_usd), int(estimated_tokens), reservation=reservation)


class _UsageShim:
    """Minimal object exposing ``.model`` + ``.usage`` for the pricing helpers."""

    def __init__(self, usage: Any, model: Any) -> None:
        self.usage = usage
        self.model = model or ""


def _extract_usage(chunk: Any, holder: dict[str, Any]) -> None:
    """Capture a usage block / model off a stream chunk if it carries one."""
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        holder["usage"] = usage
    model = getattr(chunk, "model", None)
    if model:
        holder["model"] = model


def meter_sync_stream(
    stream: Iterator[Any],
    budget: Budget,
    estimated_usd: float,
    estimated_tokens: int = 0,
    reservation: "Reservation | None" = None,
) -> Iterator[Any]:
    """Wrap a sync streamed response so AgentFuse meters it on exhaustion.

    Yields each chunk through unchanged; when the generator is fully consumed (or
    closed), commits the real cost if a usage block was seen, else the pre-call
    estimate. The ``reservation`` (from the v0.5.0 race-free gate) is threaded to
    that commit so the pending estimate is released atomically with the real
    spend; ``None`` preserves the legacy no-reservation path (e.g. direct callers
    with no pre-call reserve).

    ``estimated_tokens`` is the pre-call token upper bound (threaded from the
    reservation at the wrap.py call site) so the no-usage fallback commits a
    non-zero token estimate and the cumulative token fuse still advances on the
    dominant streamed no-usage call mode — not just the USD fuse.
    """
    holder: dict[str, Any] = {}
    committed = False
    try:
        for chunk in stream:
            _extract_usage(chunk, holder)
            yield chunk
    finally:
        if not committed:
            _commit_from_chunks(
                budget, holder, estimated_usd, estimated_tokens, reservation
            )
            committed = True


async def meter_async_stream(
    stream: AsyncIterator[Any],
    budget: Budget,
    estimated_usd: float,
    estimated_tokens: int = 0,
    reservation: "Reservation | None" = None,
) -> AsyncIterator[Any]:
    """Async variant of :func:`meter_sync_stream`."""
    holder: dict[str, Any] = {}
    committed = False
    try:
        async for chunk in stream:
            _extract_usage(chunk, holder)
            yield chunk
    finally:
        if not committed:
            _commit_from_chunks(
                budget, holder, estimated_usd, estimated_tokens, reservation
            )
            committed = True


__all__ = [
    "is_stream_response",
    "meter_sync_stream",
    "meter_async_stream",
]
