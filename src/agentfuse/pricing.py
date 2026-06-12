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
with tiktoken as a fallback token counter. Everything degrades gracefully: if a
model is missing from the price table we estimate ``0.0`` and log a warning rather
than crash, because a fuse that crashes is worse than a fuse that occasionally
under-estimates.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import litellm

logger = logging.getLogger("agentfuse.pricing")

# When a caller does not cap completion length we still need a finite upper bound
# for the pre-call estimate. This is deliberately generous so the gate stays
# conservative (better to trip a little early than to overshoot the ceiling).
DEFAULT_MAX_COMPLETION_TOKENS = 1024


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


def estimate_prompt_cost(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int | None = None,
) -> float:
    """Estimate an **upper bound** on the USD cost of one call, before sending it.

    The estimate is ``prompt_tokens * input_price`` plus a worst-case completion
    of ``max_tokens`` (or :data:`DEFAULT_MAX_COMPLETION_TOKENS`) priced at the
    model's output rate. Returns ``0.0`` (and logs a warning) when ``model`` is
    not in ``litellm.model_cost``, so an unknown model can never crash the gate.
    """
    prices = _model_prices(model)
    if prices is None:
        logger.warning(
            "model %r not found in litellm.model_cost; estimating $0.00 "
            "(the fuse cannot price this call and will not block it)",
            model,
        )
        return 0.0

    input_price, output_price = prices
    prompt_tokens = count_prompt_tokens(model, messages)
    completion_tokens = (
        max_tokens if max_tokens is not None and max_tokens > 0 else DEFAULT_MAX_COMPLETION_TOKENS
    )

    cost = prompt_tokens * input_price + completion_tokens * output_price
    return float(cost)


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
