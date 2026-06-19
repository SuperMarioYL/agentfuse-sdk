"""Exception types raised by AgentFuse.

The flagship is :class:`BudgetExceeded`. It is raised *inside AgentFuse's own
wrapper code, before the call is delegated to ``litellm.completion``* — so the
outbound HTTP request is never issued and the money is never spent. This is
ordinary Python control flow, not a swallowed LiteLLM non-blocking callback.
"""

from __future__ import annotations


class AgentFuseError(Exception):
    """Base class for every error AgentFuse raises."""


class UnpricedModelError(AgentFuseError):
    """Raised pre-call when a model is absent from the price table and the fuse
    is configured to fail closed (``on_unpriced='block'``, the default).

    A budget circuit-breaker that cannot price a call cannot enforce a ceiling on
    it; rather than silently passing the call through (which disables the fuse
    exactly when runaway risk is highest, e.g. on self-hosted / custom models),
    AgentFuse fails closed by default. Use ``on_unpriced='warn-pass'`` to opt back
    into the old pass-through behaviour, or ``on_unpriced='fallback'`` to price
    unpriced models at a conservative per-token rate.
    """

    def __init__(self, model: str, message: str | None = None) -> None:
        self.model: str = model
        super().__init__(message or self._default_message())

    def _default_message(self) -> str:
        return (
            f"model {self.model!r} is not in litellm.model_cost, so AgentFuse "
            f"cannot price this call and cannot enforce a ceiling on it. The fuse "
            f"is configured to fail closed (on_unpriced='block'). Pass "
            f"on_unpriced='fallback' for a conservative estimate, or "
            f"on_unpriced='warn-pass' to send the call ungated."
        )


class BudgetExceeded(AgentFuseError):
    """Raised pre-call when the next LLM call would push spend past a ceiling.

    Because this is raised in AgentFuse's wrapper *before* delegating to the
    real ``litellm.completion``, the blocked call is never sent and the
    estimated spend is never incurred.

    Attributes:
        spent: USD already confirmed-spent on the current task.
        ceiling: The ceiling that would be crossed (the per-task USD ceiling, or
            the per-call cap when ``limit_kind == 'single_call'``).
        would_spend: Estimated upper-bound USD cost of the blocked call. Adding
            this to ``spent`` is what crosses ``ceiling`` (for the USD ceiling).
        limit_kind: Which ceiling tripped — ``'usd'`` (cumulative USD, default),
            ``'tokens'`` (cumulative token ceiling), or ``'single_call'`` (the
            per-call hard cap).
        spent_tokens / ceiling_tokens / would_spend_tokens: Token-ledger context,
            populated when ``limit_kind == 'tokens'``.
    """

    def __init__(
        self,
        spent: float,
        ceiling: float,
        would_spend: float,
        message: str | None = None,
        *,
        limit_kind: str = "usd",
        spent_tokens: int | None = None,
        ceiling_tokens: int | None = None,
        would_spend_tokens: int | None = None,
    ) -> None:
        self.spent: float = spent
        self.ceiling: float = ceiling
        self.would_spend: float = would_spend
        self.limit_kind: str = limit_kind
        self.spent_tokens = spent_tokens
        self.ceiling_tokens = ceiling_tokens
        self.would_spend_tokens = would_spend_tokens
        super().__init__(message or self._default_message())

    def _default_message(self) -> str:
        if self.limit_kind == "single_call":
            return (
                f"FUSE TRIPPED - call blocked: estimated +${self.would_spend:.2f} "
                f"exceeds the ${self.ceiling:.2f} per-call ceiling "
                f"(call not sent)"
            )
        if self.limit_kind == "tokens":
            spent_t = self.spent_tokens or 0
            ceiling_t = self.ceiling_tokens or 0
            would_t = self.would_spend_tokens or 0
            return (
                f"FUSE TRIPPED - task halted at {spent_t:,} / {ceiling_t:,} tokens "
                f"(next call est. +{would_t:,} tokens would cross it; call not sent)"
            )
        return (
            f"FUSE TRIPPED - task halted at ${self.spent:.2f} / "
            f"${self.ceiling:.2f} ceiling "
            f"(next call est. +${self.would_spend:.2f} would cross it; "
            f"call not sent)"
        )

    def __str__(self) -> str:
        return self.args[0] if self.args else self._default_message()
