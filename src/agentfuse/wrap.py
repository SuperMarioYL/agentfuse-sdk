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

from agentfuse.fuse import commit_actual, current_budget, gate
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
    raises :class:`~agentfuse.exceptions.BudgetExceeded` (so the call is never
    sent) when it would cross the ceiling. On success, commits the real cost.
    With no active budget this is a transparent pass-through.
    """
    delegate = real if real is not None else _REAL_COMPLETION
    model, messages, max_tokens = _extract_call_args(args, kwargs)

    # (1) estimate + (2) gate — raises BudgetExceeded BEFORE the delegate runs.
    # gate() returns the pre-call upper-bound estimate, kept as the streaming
    # fallback so a streamed call with no usage block still advances the ledger.
    estimate = gate(model, messages, max_tokens=max_tokens, budget=current_budget())

    # (3) delegate to the real litellm only when within budget.
    response = delegate(*args, **kwargs)

    # (4) post-call commit of the real cost. A streamed response (stream=True)
    # carries no .usage until consumed, so meter it via the stream wrapper
    # instead — otherwise commit_actual would commit $0 and the cumulative fuse
    # would never trip on streamed runs.
    budget = current_budget()
    if budget is not None and is_stream_response(response):
        return meter_sync_stream(response, budget, estimate)
    commit_actual(response, budget=budget)
    return response


async def acompletion(*args: Any, real: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Gated asynchronous ``litellm.acompletion`` (async variant of :func:`completion`)."""
    delegate = real if real is not None else _REAL_ACOMPLETION
    model, messages, max_tokens = _extract_call_args(args, kwargs)

    estimate = gate(model, messages, max_tokens=max_tokens, budget=current_budget())

    response = await delegate(*args, **kwargs)

    budget = current_budget()
    if budget is not None and is_stream_response(response):
        return meter_async_stream(response, budget, estimate)
    commit_actual(response, budget=budget)
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
