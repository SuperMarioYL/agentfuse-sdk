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
   still advances and the fuse still trips on the *next* call,
4. settles the ledger even when the caller abandons the stream: consuming ZERO
   chunks must still commit the pre-call estimate and release the pre-call
   :class:`~agentfuse.budget.Reservation` (v0.9.0
   ``fix-unconsumed-metered-stream-never-settles``). The previous lazy-generator
   implementation could never do that — a generator that is never started runs
   no body code on close/deallocation, so its ``finally`` was unreachable and
   the reservation stayed pinned in ``pending`` for the life of the Budget while
   the abandoned (already-billed) call committed nothing. Settlement now lives
   in a small wrapper class and runs on exhaustion, on a provider error,
   on explicit ``close()`` / ``aclose()``, and in ``__del__`` — i.e. as soon as
   the caller drops the stream, GC or not.

This is pure execution-time metering — no dashboard, no visualization, no
monitoring service. It only keeps the existing per-task ledger honest for the
one response shape (streaming) that previously slipped past it.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterable, Iterator

from agentfuse.budget import Budget, Reservation
from agentfuse.pricing import resolve_commit_cost

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
        # Build a tiny shim that exposes .usage so the pricing helpers work.
        shim = _UsageShim(usage_obj, last_usage_holder.get("model"))
        # When the provider emits a usage block but litellm cannot price the
        # model (an unpriced self-hosted model), resolve_commit_cost falls back
        # to the pre-call estimated_usd so the USD ledger still advances and the
        # USD ceiling holds (mirrors the no-usage fallback below) — closing the
        # v0.8.0 on_unpriced='fallback' USD-bypass gap on the streaming path.
        cost, tokens = resolve_commit_cost(shim, estimated_usd, estimated_tokens)
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


class _MeteredStreamBase:
    """Shared settlement logic for the sync / async metered stream wrappers.

    Settlement (commit + reservation release) happens exactly once, via
    :meth:`_settle`, and is triggered by EVERY exit path — exhaustion, a
    provider error, explicit close, and ``__del__`` (the caller simply dropping
    the object). Class-based settlement is what makes the zero-consumption case
    safe: the previous lazy-generator implementation only settled if the
    generator body had STARTED, and an unstarted generator runs no body code on
    close/deallocation, so a never-iterated stream leaked its reservation into
    ``pending`` forever and committed nothing for a call the provider billed.
    """

    def __init__(
        self,
        budget: Budget,
        estimated_usd: float,
        estimated_tokens: int,
        reservation: "Reservation | None",
    ) -> None:
        self._budget = budget
        self._estimated_usd = estimated_usd
        self._estimated_tokens = estimated_tokens
        self._reservation = reservation
        self._holder: dict[str, Any] = {}
        self._settled = False

    def _settle(self) -> None:
        """Commit the call's spend and release the reservation, exactly once."""
        if self._settled:
            return
        self._settled = True
        _commit_from_chunks(
            self._budget,
            self._holder,
            self._estimated_usd,
            self._estimated_tokens,
            self._reservation,
        )

    def __del__(self) -> None:
        # Last-resort settlement for an abandoned stream (never iterated, or
        # dropped mid-iteration). Never raise from __del__.
        try:
            self._settle()
        except Exception:  # noqa: BLE001 - __del__ must not raise
            pass


class _MeteredSyncStream(_MeteredStreamBase):
    """Sync metered wrapper: iterable, settle-on-any-exit (see module docstring)."""

    def __init__(self, stream: Iterable[Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._iterator = iter(stream)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
        except BaseException:
            # StopIteration (exhaustion) or a provider error: settle, then let
            # the exception propagate unchanged.
            self._settle()
            raise
        _extract_usage(chunk, self._holder)
        return chunk

    def close(self) -> None:
        """Settle early (e.g. the caller is abandoning the stream)."""
        self._settle()


class _MeteredAsyncStream(_MeteredStreamBase):
    """Async variant of :class:`_MeteredSyncStream`."""

    def __init__(self, stream: AsyncIterator[Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._iterator = stream.__aiter__()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._iterator.__anext__()
        except BaseException:
            # StopAsyncIteration (exhaustion) or a provider error.
            self._settle()
            raise
        _extract_usage(chunk, self._holder)
        return chunk

    async def aclose(self) -> None:
        """Settle early (e.g. the caller is abandoning the stream)."""
        self._settle()


def meter_sync_stream(
    stream: Iterator[Any],
    budget: Budget,
    estimated_usd: float,
    estimated_tokens: int = 0,
    reservation: "Reservation | None" = None,
) -> _MeteredSyncStream:
    """Wrap a sync streamed response so AgentFuse meters it on ANY exit.

    Yields each chunk through unchanged; when the stream is exhausted, closed
    early, hits a provider error, OR is simply dropped by the caller (even with
    zero chunks consumed), commits the real cost if a usage block was seen, else
    the pre-call estimate, and releases the threaded
    :class:`~agentfuse.budget.Reservation`. ``None`` preserves the legacy
    no-reservation path (e.g. direct callers with no pre-call reserve).

    ``estimated_tokens`` is the pre-call token upper bound (threaded from the
    reservation at the wrap.py call site) so the no-usage fallback commits a
    non-zero token estimate and the cumulative token fuse still advances on the
    dominant streamed no-usage call mode — not just the USD fuse.

    .. versionchanged:: 0.9.0
        Returns a settle-explicit wrapper object instead of a generator: a
        generator that is never started runs no ``finally``, so a zero-consumption
        abandoned stream could never settle its reservation (v0.9.0
        ``fix-unconsumed-metered-stream-never-settles``). Iteration semantics
        are unchanged.
    """
    return _MeteredSyncStream(
        stream, budget, estimated_usd, estimated_tokens, reservation
    )


def meter_async_stream(
    stream: AsyncIterator[Any],
    budget: Budget,
    estimated_usd: float,
    estimated_tokens: int = 0,
    reservation: "Reservation | None" = None,
) -> _MeteredAsyncStream:
    """Async variant of :func:`meter_sync_stream` (see it for the contract)."""
    return _MeteredAsyncStream(
        stream, budget, estimated_usd, estimated_tokens, reservation
    )


__all__ = [
    "is_stream_response",
    "meter_sync_stream",
    "meter_async_stream",
]
