"""Thin repo-root entrypoint for the bundled runaway-agent demo.

The demo itself is vendored inside the installed package at
:mod:`agentfuse._runaway_demo` so ``agentfuse demo`` works from a ``pip install``
of the wheel — ``examples/`` is NOT shipped in the wheel (only ``src/agentfuse``
is packaged by ``[tool.hatch.build.targets.wheel] packages = ["src/agentfuse"]``),
so the demo could not previously be found from a wheel install. This file
re-exports the vendored module so ``python examples/runaway_agent.py`` from a
source checkout still runs unchanged.

Run it with::

    agentfuse demo
    # or
    python examples/runaway_agent.py
"""

from __future__ import annotations

# Re-export the vendored demo so this entrypoint keeps working from a checkout.
from agentfuse._runaway_demo import (  # noqa: F401
    MAX_TOKENS_PER_CALL,
    MOCK_COMPLETION_TOKENS,
    MOCK_PROMPT_TOKENS,
    MODEL,
    _agent_step,
    _mock_response,
    run_demo,
)

if __name__ == "__main__":  # pragma: no cover
    run_demo()
