"""Zero-touch interception of ``litellm.completion`` / ``litellm.acompletion``.

This is the abort point. The whole product depends on getting the interception
*location* right (verified in ``schema_smoke``):

    The budget gate MUST run in AgentFuse's OWN wrapper code, BEFORE delegating
    to ``litellm`` — NOT in a LiteLLM pre-call callback. LiteLLM wraps its
    in-process pre-call callbacks in a ``[Non-Blocking]`` try/except that LOGS
    AND SWALLOWS any exception, so a ``raise`` there does NOT abort and the HTTP
    call still goes out. Raising inside our own wrapper, by contrast, is ordinary
    Python control flow: the over-budget call is never reached, so the money is
    never spent.

Per-call flow inside :func:`completion` / :func:`acompletion`:

1. **estimate** the call's upper-bound cost,
2. **gate** it against the active per-task :class:`~agentfuse.budget.Budget`
   (raises :class:`~agentfuse.exceptions.BudgetExceeded` *before* step 3),
3. **delegate** to the real ``litellm.completion`` only if within budget,
4. **commit** the real cost read back from the response (post-call metering).

Two integration styles are offered:

* a direct callable — ``agentfuse.completion(...)`` / ``agentfuse.acompletion(...)``;
* a monkeypatch — :func:`install` swaps ``litellm.completion`` /
  ``litellm.acompletion`` for the gated wrappers so existing agent code that
  already calls ``litellm.completion`` is gated with zero edits;
  :func:`uninstall` restores the originals.

``wrap`` (re-exported by :mod:`agentfuse`) is :func:`install`.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Mapping, Sequence

import litellm

from agentfuse.fuse import commit_actual, current_budget, gate_with_reservation
from agentfuse.stream import is_stream_response, meter_async_stream, meter_sync_stream

# The genuine litellm callables, captured at import time so install/uninstall is
# idempotent and we always delegate to the real thing (never to our own wrapper).
_REAL_COMPLETION: Callable[..., Any] = litellm.completion
_REAL_ACOMPLETION: Callable[..., Any] = litellm.acompletion

_installed = False


def _extract_call_args(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[str, Sequence[Mapping[str, Any]], int | None]:
    """Pull ``(model, messages, max_tokens)`` from a litellm-style call signature.

    ``litellm.completion(model=..., messages=...)`` is almost always called with
    keywords, but the first two positionals are ``model`` then ``messages`` too,
    so we accept both.
    """
    model = kwargs.get("model")
    if model is None and len(args) >= 1:
        model = args[0]
    messages = kwargs.get("messages")
    if messages is None and len(args) >= 2:
        messages = args[1]
    max_tokens = kwargs.get("max_tokens")
    return str(model or ""), list(messages or []), max_tokens


def completion(*args: Any, real: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Gated synchronous ``litellm.completion``.

    Estimates and gates against the active per-task budget *before* delegating;
    raises :class:`agentfuse.exceptions.BudgetExceeded` (so the call is never
    sent) when it would cross the ceiling. On success, commits the real cost.
    With no active budget this is a transparent pass-through.

    The gate RESERVES the estimate (v0.5.0 fix-budget-check-commit-race) so a
    concurrent caller cannot also pass against the still-uncommitted ledger; the
    reservation is released by the commit (success) or by ``release`` (error).
    """
    delegate = real if real is not None else _REAL_COMPLETION
    model, messages, max_tokens = _extract_call_args(args, kwargs)

    active = current_budget()
    # (1) estimate + (2) gate — raises BudgetExceeded (reserving nothing) BEFORE
    # the delegate runs; on the pass path it RESERVES the estimate so a
    # concurrent caller cannot also pass against the still-uncommitted ledger.
    estimate, reservation = gate_with_reservation(
        model, messages, max_tokens=max_tokens, budget=active
    )

    try:
        # (3) delegate to the real litellm only when within budget.
        response = delegate(*args, **kwargs)
    except BaseException:
        # Error path: release the reservation so the pending estimate does not
        # pin the budget forever (reservation is None when no budget is active).
        if reservation is not None:
            active.release(reservation)
        raise

    # (4) post-call commit of the real cost. A streamed response (stream=True)
    # carries no .usage until consumed, so meter it via the stream wrapper
    # (which threads the reservation to its commit) instead — otherwise
    # commit_actual would commit $0 and the cumulative fuse would never trip on
    # streamed runs.
    if active is not None and is_stream_response(response):
        # Thread the pre-call token estimate (carried on the reservation) so the
        # streaming no-usage fallback commits non-zero tokens and the cumulative
        # token fuse still advances on the dominant streamed no-usage call mode.
        est_tokens = reservation.estimated_tokens if reservation is not None else 0
        return meter_sync_stream(response, active, estimate, est_tokens, reservation)
    commit_actual(response, budget=active, reservation=reservation)
    return response


async def acompletion(*args: Any, real: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Gated asynchronous ``litellm.acompletion`` (async variant of :func:`completion`).

    The race-free gate is most consequential here: ``await delegate`` yields
    control to other tasks in the same loop, so without the reservation a second
    concurrent ``acompletion`` would gate against the still-uncommitted ledger
    and both would commit, overshooting the ceiling. The reservation pins the
    estimate in ``pending`` across the await and is settled by the commit
    (success) or ``release`` (error).
    """
    delegate = real if real is not None else _REAL_ACOMPLETION
    model, messages, max_tokens = _extract_call_args(args, kwargs)

    active = current_budget()
    estimate, reservation = gate_with_reservation(
        model, messages, max_tokens=max_tokens, budget=active
    )

    try:
        response = await delegate(*args, **kwargs)
    except BaseException:
        if reservation is not None:
            active.release(reservation)
        raise

    if active is not None and is_stream_response(response):
        # Thread the pre-call token estimate (carried on the reservation) so the
        # streaming no-usage fallback commits non-zero tokens and the cumulative
        # token fuse still advances on the dominant streamed no-usage call mode.
        est_tokens = reservation.estimated_tokens if reservation is not None else 0
        return meter_async_stream(response, active, estimate, est_tokens, reservation)
    commit_actual(response, budget=active, reservation=reservation)
    return response


def install() -> None:
    """Monkeypatch ``litellm.completion`` / ``litellm.acompletion`` to be gated.

    After this, any code (yours or a third-party agent runtime) that calls
    ``litellm.completion`` is automatically gated against the active per-task
    budget. Idempotent: a second call is a no-op. Pair with :func:`uninstall`.
    """
    global _installed
    if _installed:
        return

    @functools.wraps(_REAL_COMPLETION)
    def _patched_completion(*args: Any, **kwargs: Any) -> Any:
        return completion(*args, real=_REAL_COMPLETION, **kwargs)

    @functools.wraps(_REAL_ACOMPLETION)
    async def _patched_acompletion(*args: Any, **kwargs: Any) -> Any:
        return await acompletion(*args, real=_REAL_ACOMPLETION, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
    _installed = True


def uninstall() -> None:
    """Restore the original ``litellm.completion`` / ``litellm.acompletion``."""
    global _installed
    litellm.completion = _REAL_COMPLETION
    litellm.acompletion = _REAL_ACOMPLETION
    _installed = False


def is_installed() -> bool:
    """Return whether the litellm monkeypatch is currently active."""
    return _installed


# `wrap` is the name `agentfuse.__init__` imports. Installing the monkeypatch IS
# the act of "wrapping" litellm, so the public verb maps to install().
wrap = install


__all__ = [
    "completion",
    "acompletion",
    "install",
    "uninstall",
    "is_installed",
    "wrap",
]
