"""Exception types raised by AgentFuse.

The flagship is :class:`BudgetExceeded`. It is raised *inside AgentFuse's own
wrapper code, before the call is delegated to ``litellm.completion``* — so the
outbound HTTP request is never issued and the money is never spent. This is
ordinary Python control flow, not a swallowed LiteLLM non-blocking callback.
"""

from __future__ import annotations


class AgentFuseError(Exception):
    """Base class for every error AgentFuse raises."""


class BudgetExceeded(AgentFuseError):
    """Raised pre-call when the next LLM call would push spend past the ceiling.

    Because this is raised in AgentFuse's wrapper *before* delegating to the
    real ``litellm.completion``, the blocked call is never sent and the
    estimated spend is never incurred.

    Attributes:
        spent: USD already confirmed-spent on the current task.
        ceiling: The per-task USD ceiling that must not be crossed.
        would_spend: Estimated upper-bound USD cost of the blocked call. Adding
            this to ``spent`` is what crosses ``ceiling``.
    """

    def __init__(
        self,
        spent: float,
        ceiling: float,
        would_spend: float,
        message: str | None = None,
    ) -> None:
        self.spent: float = spent
        self.ceiling: float = ceiling
        self.would_spend: float = would_spend
        super().__init__(message or self._default_message())

    def _default_message(self) -> str:
        return (
            f"FUSE TRIPPED - task halted at ${self.spent:.2f} / "
            f"${self.ceiling:.2f} ceiling "
            f"(next call est. +${self.would_spend:.2f} would cross it; "
            f"call not sent)"
        )

    def __str__(self) -> str:
        return self.args[0] if self.args else self._default_message()
