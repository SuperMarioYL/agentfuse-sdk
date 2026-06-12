"""A self-contained, offline demo of AgentFuse tripping on a runaway agent.

This reproduces, in miniature, the failure mode from the viral "AI agent
bankrupted its operator while scanning DN42" thread: an autonomous agent stuck
in a loop, firing paid LLM call after paid LLM call with nobody watching the
meter.

Here the loop is wrapped by ``agentfuse`` with a per-task ceiling. The running
ledger climbs with every call, and the moment the *next* call's estimated cost
would push spend past the ceiling, AgentFuse prints ``🔌 FUSE TRIPPED`` and
raises :class:`~agentfuse.exceptions.BudgetExceeded` **before that call is ever
sent** — so the over-budget call never goes out.

It runs fully offline: every ``litellm.completion`` is answered with a litellm
``mock_response`` carrying *realistic* token usage, so NO API key and NO network
are needed yet the per-call cost (priced from litellm's real gpt-4o rate) is
genuine. The pre-call gate prices a worst-case completion of ``max_tokens`` and
the running ledger climbs by each call's real cost — so the fuse trips on honest
arithmetic, not a rigged number.

Run it with::

    agentfuse demo
    # or
    python examples/runaway_agent.py
"""

from __future__ import annotations

from litellm.types.utils import Choices, Message, ModelResponse, Usage

import agentfuse
from agentfuse import BudgetExceeded
from agentfuse.fuse import current_budget, task

# Each looped "thought" asks for up to this many completion tokens, and the
# (mocked) response actually returns a chunky completion — a realistic runaway
# step. The pre-call gate prices a worst-case completion of MAX_TOKENS_PER_CALL.
MAX_TOKENS_PER_CALL = 4000
MOCK_PROMPT_TOKENS = 1200
MOCK_COMPLETION_TOKENS = 2000
MODEL = "gpt-4o"


def _mock_response(step: int) -> ModelResponse:
    """A canned litellm response with realistic usage (keeps the demo offline).

    litellm's plain string ``mock_response`` reports fixed tiny usage; passing a
    fully-formed :class:`ModelResponse` lets the demo carry real-sized token
    counts, so ``actual_cost`` (priced from litellm's real gpt-4o rate) is a
    genuine per-call cost the ledger can accumulate.
    """
    return ModelResponse(
        model=MODEL,
        choices=[
            Choices(
                index=0,
                message=Message(
                    role="assistant",
                    content=f"(mock) scanning... step {step} complete, continuing the loop.",
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=MOCK_PROMPT_TOKENS,
            completion_tokens=MOCK_COMPLETION_TOKENS,
            total_tokens=MOCK_PROMPT_TOKENS + MOCK_COMPLETION_TOKENS,
        ),
    )


def _agent_step(i: int) -> str:
    """One step of a 'runaway' agent: a single paid (here, mocked) LLM call.

    Routed through ``agentfuse.completion`` so the per-task budget gates it
    BEFORE the (mocked) call is delegated to litellm. ``mock_response`` keeps the
    whole thing offline.
    """
    messages = [
        {"role": "system", "content": "You are an autonomous network-scanning agent."},
        {
            "role": "user",
            "content": (
                f"Step {i}: keep scanning the DN42 network and decide the next "
                "set of probes to run. Be thorough and exhaustive."
            ),
        },
    ]
    response = agentfuse.completion(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS_PER_CALL,
        mock_response=_mock_response(i),
    )
    return response.choices[0].message.content


def run_demo(ceiling_usd: float = 0.50, max_iterations: int = 100) -> None:
    """Run the runaway loop under a ``$ceiling_usd`` per-task fuse and trip it.

    Prints the climbing ledger each iteration; when the fuse trips it reports how
    much spend was saved by halting before the over-budget call. Returns normally
    after the fuse trips (the :class:`BudgetExceeded` is caught here so the demo
    can show the summary).
    """
    print(f"Runaway-agent demo — per-task ceiling ${ceiling_usd:.2f}")
    print("Running offline (litellm mock_response — no API key needed).\n")

    with task(ceiling_usd=ceiling_usd, name="runaway-scan") as budget:
        try:
            for i in range(1, max_iterations + 1):
                _agent_step(i)
                snap = budget.snapshot()
                print(
                    f"  call #{i:>2} ok  | spent ${snap.spent_usd:0.4f} "
                    f"/ ${snap.ceiling_usd:0.2f}  | remaining ${snap.remaining_usd:0.4f}"
                )
        except BudgetExceeded as err:
            # The banner ("🔌 FUSE TRIPPED ...") was already printed to stderr by
            # the gate; here we add the operator-facing summary.
            saved_call_estimate = err.would_spend
            print()
            print(
                f"The fuse halted the run at ${err.spent:.4f} of the "
                f"${err.ceiling:.2f} ceiling."
            )
            print(
                f"The next call (est. +${saved_call_estimate:.4f}) was BLOCKED "
                "before it was ever sent — that spend was never incurred."
            )
            print()
            print("Without AgentFuse, this loop would have kept burning money.")
            return

    # If we somehow finish the loop without tripping (e.g. a tiny ceiling raise
    # mid-flight), report the final ledger honestly.
    snap = budget.snapshot()
    print(f"\nTask finished — spent ${snap.spent_usd:.4f} / ${snap.ceiling_usd:.2f}.")


if __name__ == "__main__":
    run_demo()
