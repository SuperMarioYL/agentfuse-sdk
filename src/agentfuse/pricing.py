"""Cost estimation for AgentFuse's pre-call gate and post-call commit.

Two jobs live here:

* :func:`estimate_prompt_cost` — a *conservative upper-bound* estimate of what a
  call will cost, computed **before** the call is sent. It counts the prompt
  tokens and assumes the completion runs all the way to ``max_tokens`` (or a sane
  default), then prices both halves from LiteLLM's per-model cost table. The gate
  in :mod:`agentfuse.budget` compares this estimate against the remaining budget.
* :func:`actual_cost` — the *real* USD spend read back from a completed response's
  ``Usage`` (prompt/completion tokens), used to commit confirmed spend after the
  call returns.

Pricing data and token counting come from LiteLLM (``litellm.model_cost``,
``litellm.token_counter``, ``litellm.cost_per_token``, ``litellm.completion_cost``)
with tiktoken as a fallback token counter.

**Unpriced models.** When a model is missing from ``litellm.model_cost`` the fuse
cannot price the call. Returning ``0.0`` (the v0.1 behaviour) silently disables
the circuit-breaker exactly when runaway risk is highest — on self-hosted / custom
models that are *not* in the table. So v0.2 fails closed by default
(``on_unpriced='block'`` → raise :class:`~agentfuse.exceptions.UnpricedModelError`).
Callers can opt into ``'fallback'`` (price at a conservative per-token rate) or
``'warn-pass'`` (the old pass-through: estimate ``0.0`` and log a warning).
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Mapping, Sequence

import litellm

from agentfuse.exceptions import UnpricedModelError

logger = logging.getLogger("agentfuse.pricing")

# When a caller does not cap completion length we still need a finite upper bound
# for the pre-call estimate. Real streamed completions routinely run 4k-16k
# tokens on modern frontier models, so the v0.1-v0.4 default of 1024 was NOT an
# upper bound: the pre-call gate passed on the small estimate and the post-call
# commit of the real (larger) usage jumped the ledger past the ceiling with no
# retroactive trip — the fuse only gates the NEXT call, which for a one-shot
# streamed task never comes. The floor is now 8192, and when
# ``litellm.model_cost`` carries a model-specific ``max_output_tokens`` that is
# larger, that genuine cap is preferred (see :func:`_model_max_output_tokens`),
# so the estimate is an actual upper bound for typical streamed completions.
DEFAULT_MAX_COMPLETION_TOKENS = 8192

# Policy for a model that is not in litellm.model_cost.
OnUnpriced = Literal["block", "fallback", "warn-pass"]
DEFAULT_ON_UNPRICED: OnUnpriced = "block"

# Conservative per-token USD rate used by the ``'fallback'`` policy when a model
# is unpriced. Deliberately high (~frontier-model output pricing) so an unpriced
# call is over-, never under-, estimated — better to trip early than to overshoot.
FALLBACK_USD_PER_TOKEN = 1.5e-5


def _model_prices(model: str) -> tuple[float, float] | None:
    """Return ``(input_cost_per_token, output_cost_per_token)`` for ``model``.

    Returns ``None`` when the model is not present in ``litellm.model_cost`` so
    callers can fall back to a zero estimate instead of raising.
    """
    entry: Mapping[str, Any] | None = litellm.model_cost.get(model)
    if not entry:
        return None
    in_cost = entry.get("input_cost_per_token")
    out_cost = entry.get("output_cost_per_token")
    if in_cost is None and out_cost is None:
        return None
    return float(in_cost or 0.0), float(out_cost or 0.0)


def _model_max_output_tokens(model: str) -> int | None:
    """Return the model's own ``max_output_tokens`` from ``litellm.model_cost``.

    Used as a model-aware upper bound on completion length when the caller omits
    ``max_tokens`` (the common case for streamed agent calls). Returns ``None``
    when the model is unpriced or the field is absent / non-positive, so the
    caller falls back to :data:`DEFAULT_MAX_COMPLETION_TOKENS` as the floor.
    """
    entry: Mapping[str, Any] | None = litellm.model_cost.get(model)
    if not entry:
        return None
    cap = entry.get("max_output_tokens")
    if cap is None:
        return None
    try:
        n = int(cap)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def count_prompt_tokens(model: str, messages: Sequence[Mapping[str, Any]]) -> int:
    """Count prompt tokens for ``messages`` under ``model``.

    Prefers ``litellm.token_counter`` (provider-aware via LiteLLM's normalization);
    falls back to a tiktoken ``cl100k_base`` encode of the concatenated message
    text, and finally to a crude character heuristic so this never raises.
    """
    msg_list = [dict(m) for m in messages]
    try:
        return int(litellm.token_counter(model=model, messages=msg_list))
    except Exception as exc:  # noqa: BLE001 - never let counting crash the gate
        logger.debug("litellm.token_counter failed for %s: %s; using tiktoken", model, exc)

    text = " ".join(str(m.get("content", "")) for m in msg_list)
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as exc:  # noqa: BLE001
        logger.debug("tiktoken counting failed for %s: %s; using char heuristic", model, exc)
        # ~4 chars per token is the usual rough rule of thumb.
        return max(1, len(text) // 4)


def estimate_call(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int | None = None,
    *,
    on_unpriced: OnUnpriced = DEFAULT_ON_UNPRICED,
) -> tuple[float, int]:
    """Estimate ``(usd_upper_bound, token_upper_bound)`` for one call, pre-send.

    The token bound is ``prompt_tokens`` plus a worst-case completion of
    ``max_tokens`` (or :data:`DEFAULT_MAX_COMPLETION_TOKENS`). The USD bound
    prices both halves from ``litellm.model_cost``.

    When ``model`` is not in ``litellm.model_cost`` the behaviour is governed by
    ``on_unpriced``:

    * ``'block'`` (default) — raise :class:`~agentfuse.exceptions.UnpricedModelError`
      so the fuse fails closed instead of silently passing the call.
    * ``'fallback'`` — price every estimated token at
      :data:`FALLBACK_USD_PER_TOKEN` (a deliberately conservative rate).
    * ``'warn-pass'`` — log a warning and return ``(0.0, token_bound)`` (the v0.1
      pass-through; the USD gate cannot block this call).
    """
    prompt_tokens = count_prompt_tokens(model, messages)
    if max_tokens is not None and max_tokens > 0:
        # Caller explicitly capped completion length — honour it.
        completion_tokens = int(max_tokens)
    else:
        # No caller cap: use a genuinely conservative upper bound. Prefer the
        # model's own max_output_tokens when litellm knows it and it is larger;
        # DEFAULT_MAX_COMPLETION_TOKENS (8192) is the floor fallback so the
        # estimate is an actual upper bound for typical streamed completions
        # (real ones run 4k-16k on frontier models — the old 1024 default was
        # not an upper bound and let streamed calls overshoot post-commit).
        model_cap = _model_max_output_tokens(model)
        completion_tokens = max(DEFAULT_MAX_COMPLETION_TOKENS, model_cap or 0)
    token_bound = prompt_tokens + completion_tokens

    prices = _model_prices(model)
    if prices is None:
        if on_unpriced == "block":
            raise UnpricedModelError(model)
        if on_unpriced == "fallback":
            logger.warning(
                "model %r not found in litellm.model_cost; pricing %d tokens at the "
                "conservative fallback rate ($%.2e/token) so the fuse still bounds it",
                model,
                token_bound,
                FALLBACK_USD_PER_TOKEN,
            )
            return float(token_bound * FALLBACK_USD_PER_TOKEN), token_bound
        # warn-pass
        logger.warning(
            "model %r not found in litellm.model_cost; estimating $0.00 "
            "(on_unpriced='warn-pass' — the fuse cannot price this call and will "
            "not block it on USD)",
            model,
        )
        return 0.0, token_bound

    input_price, output_price = prices
    cost = prompt_tokens * input_price + completion_tokens * output_price
    return float(cost), token_bound


def estimate_prompt_cost(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int | None = None,
    *,
    on_unpriced: OnUnpriced = DEFAULT_ON_UNPRICED,
) -> float:
    """Estimate an **upper bound** on the USD cost of one call, before sending it.

    Thin wrapper over :func:`estimate_call` returning only the USD bound. See
    :func:`estimate_call` for the ``on_unpriced`` policy on models missing from
    ``litellm.model_cost``.
    """
    usd, _tokens = estimate_call(model, messages, max_tokens, on_unpriced=on_unpriced)
    return usd


def actual_cost(response: Any) -> float:
    """Return the real USD cost of a completed LiteLLM response.

    Prefers ``litellm.completion_cost(completion_response=...)``; falls back to
    pricing the response's ``Usage`` (prompt/completion tokens) via
    ``litellm.cost_per_token``. Returns ``0.0`` and warns if neither path can
    price the response, so post-call commit never crashes a finished task.
    """
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception as exc:  # noqa: BLE001
        logger.debug("litellm.completion_cost failed: %s; pricing from Usage", exc)

    model = getattr(response, "model", None) or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None

    if model and prompt_tokens is not None and completion_tokens is not None:
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
            )
            return float(prompt_cost) + float(completion_cost)
        except Exception as exc:  # noqa: BLE001
            logger.debug("litellm.cost_per_token failed for %s: %s", model, exc)

    logger.warning(
        "could not determine actual cost for response (model=%r); committing $0.00",
        model,
    )
    return 0.0


def actual_tokens(response: Any) -> int:
    """Return the real total token count of a completed LiteLLM response.

    Reads ``response.usage.total_tokens`` (falling back to
    ``prompt_tokens + completion_tokens``). Returns ``0`` when the response
    carries no usable usage, so post-call commit never crashes a finished task.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    total = getattr(usage, "total_tokens", None)
    if total is not None:
        try:
            return int(total)
        except (TypeError, ValueError):
            pass
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    try:
        return int(prompt or 0) + int(completion or 0)
    except (TypeError, ValueError):
        return 0
