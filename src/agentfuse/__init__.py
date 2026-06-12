"""AgentFuse — an enforcing per-task spend circuit-breaker for autonomous agents."""

from __future__ import annotations

__version__ = "0.1.0"

# Public API surface. `exceptions` ships in stage A; `Budget`, `gate`, and `wrap`
# are created in stages B/C. The cross-module imports are guarded so that
# `import agentfuse` succeeds even before those modules land.
from agentfuse.exceptions import AgentFuseError, BudgetExceeded

try:  # stage B: per-task ledger + pre-call gate
    from agentfuse.budget import Budget, gate
except ImportError:  # pragma: no cover - wired in stage B
    Budget = None  # type: ignore[assignment]
    gate = None  # type: ignore[assignment]

try:  # stage C: enforcement layer + ergonomic API (Fuse / task / @fuse)
    from agentfuse.fuse import Fuse, current_budget, fuse, fused, task
except ImportError:  # pragma: no cover - wired in stage C
    Fuse = None  # type: ignore[assignment]
    task = None  # type: ignore[assignment]
    fuse = None  # type: ignore[assignment]
    fused = None  # type: ignore[assignment]
    current_budget = None  # type: ignore[assignment]

try:  # stage C: zero-touch litellm wrapper (install/uninstall + completion)
    from agentfuse.wrap import acompletion, completion, install, uninstall, wrap
except ImportError:  # pragma: no cover - wired in stage C
    wrap = None  # type: ignore[assignment]
    install = None  # type: ignore[assignment]
    uninstall = None  # type: ignore[assignment]
    completion = None  # type: ignore[assignment]
    acompletion = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "AgentFuseError",
    "BudgetExceeded",
    "Budget",
    "gate",
    # ergonomic API
    "Fuse",
    "task",
    "fuse",
    "fused",
    "current_budget",
    # litellm interception
    "wrap",
    "install",
    "uninstall",
    "completion",
    "acompletion",
]
