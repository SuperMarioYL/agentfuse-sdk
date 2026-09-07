"""Adversarial red->green tests for the v0.8.0 amendment's two milestones.

Milestone 1 ``fix-fallback-unpriced-commits-zero-usd-bypasses-fuse``:

``on_unpriced='fallback'`` is marketed to bound unpriced/self-hosted models
(pricing.py docstring: "price unpriced models at a conservative per-token rate
so the fuse still bounds it"). The pre-call gate conservatively reserves a
non-zero USD estimate, BUT the post-call commit threw it away: for a model
missing from ``litellm.model_cost`` whose ``Usage`` still reports real tokens,
``pricing.actual_cost`` returns ``0.0`` (``litellm.completion_cost`` raises and
the ``cost_per_token`` fallback also fails), so ``commit_actual`` committed
``0.0`` USD — freezing the cumulative USD ledger and silently bypassing the USD
ceiling for the exact self-hosted/unpriced audience ``fallback`` targets. This
is the USD-cost half of the same bug class the v0.6.0
``fix-stream-meter-drops-token-estimate`` milestone corrected on the
streaming-no-usage path (committing the pre-call estimate instead of 0); the USD
half was never fixed on the non-stream ``commit_actual`` path or the
stream-with-usage ``_commit_from_chunks`` path. The fix adds
``pricing.resolve_commit_cost(response, estimated_usd)`` which falls back to
the pre-call ``estimated_usd`` when the real cost resolves to ``0.0`` with real
usage, mirroring the no-usage fallback; ``warn-pass`` (estimate ``0.0``) is left
committing ``0.0`` by design (the ``estimated_usd > 0`` guard).

Milestone 2 ``fix-agentfuse-demo-broken-on-wheel-install``:

The README quickstart (``pip install agentfuse`` -> ``agentfuse demo``) was
broken on the wheel install path: ``pyproject.toml``'s
``[tool.hatch.build.targets.wheel] packages = ["src/agentfuse"]`` ships only
``src/agentfuse``, so ``examples/runaway_agent.py`` (repo root) is absent from
the wheel and ``cli._load_runaway_agent`` raised ``ClickException``. The demo is
now vendored as ``agentfuse._runaway_demo`` (always importable from the
package), with ``examples/runaway_agent.py`` a thin re-export.

Tests use real litellm types (``mock_response``/``ModelResponse``) + stub
delegates; no network, no API key.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest
from click.testing import CliRunner

from agentfuse import Budget, BudgetExceeded
from agentfuse.cli import main
from agentfuse.fuse import task
from agentfuse.pricing import FALLBACK_USD_PER_TOKEN, estimate_call

# `agentfuse.wrap` the *attribute* (re-exported by __init__) shadows the
# submodule; reach the actual module via importlib (same pattern as
# tests/test_v060_fixes.py).
wrap_mod = importlib.import_module("agentfuse.wrap")

MODEL_PRICED = "gpt-4o"
MODEL_UNPRICED = "some-unpriced-selfhosted-model"  # absent from litellm.model_cost
MESSAGES = [
    {"role": "system", "content": "You are an autonomous agent."},
    {"role": "user", "content": "Scan the network and report back in detail."},
]
SHORT_MESSAGES = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# Fakes / real-litellm helpers
# --------------------------------------------------------------------------- #

from litellm.types.utils import Choices, Message, ModelResponse, Usage


def _unpriced_model_response_with_usage(
    prompt: int = 1000, completion: int = 2000
) -> ModelResponse:
    """A real litellm ``ModelResponse`` for an UNPRICED model carrying real usage.

    The model is absent from ``litellm.model_cost`` so ``actual_cost`` cannot
    price it (returns 0.0), but the ``Usage`` reports real tokens — the exact
    shape of a self-hosted / ollama / watsonx call whose provider emits usage but
    litellm has no price row for.
    """
    return ModelResponse(
        model=MODEL_UNPRICED,
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="(mock) continuing the loop."),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


class _UsageChunk:
    """A stream chunk carrying ``.content`` / ``.model`` and a ``.usage`` block."""

    def __init__(self, content: str, usage: Usage | None, model: str = MODEL_UNPRICED) -> None:
        self.content = content
        self.model = model
        self.usage = usage


def _stream_with_final_usage(n_chunks: int, final_usage: Usage):
    """A sync generator whose final chunk carries a usage block (litellm include_usage)."""
    for i in range(n_chunks - 1):
        yield _UsageChunk(f"chunk {i}", None)
    yield _UsageChunk(f"chunk {n_chunks - 1} (final)", final_usage)


async def _async_stream_with_final_usage(n_chunks: int, final_usage: Usage):
    for i in range(n_chunks - 1):
        yield _UsageChunk(f"chunk {i}", None)
    yield _UsageChunk(f"chunk {n_chunks - 1} (final)", final_usage)


# =========================================================================== #
# fix-fallback-unpriced-commits-zero-usd-bypasses-fuse  (non-stream path)
# =========================================================================== #


def test_v080_fallback_unpriced_nonstream_advances_usd_ledger():
    """RED on v0.7.0 (actual_cost returns 0.0 for unpriced model -> commit 0.0 ->
    spent frozen at ~0), GREEN on v0.8.0 (resolve_commit_cost falls back to the
    pre-call estimate -> spent advances).

    A single non-stream call to an unpriced model under ``on_unpriced='fallback'``
    whose response carries real usage must advance the cumulative USD ledger by
    the conservative pre-call estimate (not commit $0).
    """
    max_tokens = 2000
    estimate, est_tokens = estimate_call(
        MODEL_UNPRICED, SHORT_MESSAGES, max_tokens=max_tokens, on_unpriced="fallback"
    )
    assert estimate > 0.0, "fallback pre-call estimate must be non-zero for an unpriced model"

    def stub_delegate(*args, **kwargs):
        return _unpriced_model_response_with_usage()

    with task(ceiling_usd=100.0, name="fb-advance", on_unpriced="fallback") as budget:
        wrap_mod.completion(
            model=MODEL_UNPRICED, messages=SHORT_MESSAGES, max_tokens=max_tokens,
            real=stub_delegate,
        )
        # v0.8.0: the pre-call fallback estimate was committed (not 0.0).
        assert budget.spent > 0.0, (
            f"unpriced-model call under on_unpriced='fallback' must advance the USD "
            f"ledger (spent={budget.spent!r}); v0.7.0 committed $0 (actual_cost cannot "
            f"price an unpriced model) and the USD fuse was silently bypassed."
        )
        # No reservation leaked into pending.
        assert budget.pending_usd == 0.0


def test_v080_fallback_unpriced_nonstream_trips_usd_ceiling():
    """RED on v0.7.0 (50 calls -> spent frozen at ~0 -> USD ceiling NEVER trips),
    GREEN on v0.8.0 (each call commits the estimate -> spent climbs -> USD ceiling
    trips).

    Repeated non-stream calls to an unpriced model under ``fallback`` must trip
    the cumulative USD ceiling — the self-hosted-model runaway scenario the
    ``fallback`` policy exists to bound.
    """
    max_tokens = 2000

    def stub_delegate(*args, **kwargs):
        return _unpriced_model_response_with_usage()

    sent = 0
    with task(ceiling_usd=0.50, name="fb-ceiling", on_unpriced="fallback") as budget:
        tripped = False
        for _ in range(50):
            try:
                wrap_mod.completion(
                    model=MODEL_UNPRICED, messages=SHORT_MESSAGES, max_tokens=max_tokens,
                    real=stub_delegate,
                )
                sent += 1
            except BudgetExceeded:
                tripped = True
                break
        assert tripped, (
            "cumulative USD fuse must trip on repeated unpriced-model calls under "
            "on_unpriced='fallback' (v0.7.0 committed $0 per call so spent stayed ~0 "
            "and the ceiling never tripped — the fuse was silently bypassed for the "
            "exact self-hosted/unpriced audience fallback markets)."
        )
        assert budget.spent > 0.0
        assert sent < 50, "runaway must stop well before 50 calls"
        assert budget.pending_usd == 0.0


def test_v080_warn_pass_unpriced_still_commits_zero_usd():
    """The fix must NOT over-reach into ``on_unpriced='warn-pass'``'s documented
    opt-out. Under warn-pass the pre-call estimate is ``0.0`` (the USD fuse
    cannot bound an unpriced call by design), so ``resolve_commit_cost``'s
    ``estimated_usd > 0`` guard must keep committing ``0.0`` USD.
    """
    max_tokens = 2000
    estimate, _ = estimate_call(
        MODEL_UNPRICED, SHORT_MESSAGES, max_tokens=max_tokens, on_unpriced="warn-pass"
    )
    assert estimate == 0.0, "warn-pass pre-call estimate is 0.0 by design"

    def stub_delegate(*args, **kwargs):
        return _unpriced_model_response_with_usage()

    with task(ceiling_usd=100.0, name="warnpass", on_unpriced="warn-pass") as budget:
        wrap_mod.completion(
            model=MODEL_UNPRICED, messages=SHORT_MESSAGES, max_tokens=max_tokens,
            real=stub_delegate,
        )
        # warn-pass opts out of USD bounding: spent stays 0 (the documented
        # behaviour, preserved by the estimated_usd > 0 guard).
        assert budget.spent == 0.0
        assert budget.pending_usd == 0.0


# =========================================================================== #
# fix-fallback-unpriced-commits-zero-usd-bypasses-fuse  (stream-with-usage path)
# =========================================================================== #


def test_v080_fallback_unpriced_stream_with_usage_advances_usd_ledger():
    """RED on v0.7.0 (the stream-with-usage path: _commit_from_chunks builds a
    _UsageShim, actual_cost(shim) returns 0.0 for the unpriced model -> commit
    0.0 USD -> USD ledger frozen), GREEN on v0.8.0 (resolve_commit_cost falls
    back to estimated_usd).

    A streamed call to an unpriced model whose final chunk carries a usage block
    must still advance the USD ledger (the v0.6.0 fix only covered the NO-usage
    fallback; the WITH-usage-but-unpriced path committed 0.0 USD).
    """
    max_tokens = 2000
    estimate, est_tokens = estimate_call(
        MODEL_UNPRICED, MESSAGES, max_tokens=max_tokens, on_unpriced="fallback"
    )
    assert estimate > 0.0

    final_usage = Usage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)

    def stub_delegate(*args, **kwargs):
        return _stream_with_final_usage(3, final_usage)

    with task(ceiling_usd=100.0, ceiling_tokens=10_000, name="fb-stream", on_unpriced="fallback") as budget:
        stream = wrap_mod.completion(
            model=MODEL_UNPRICED, messages=MESSAGES, max_tokens=max_tokens,
            stream=True, real=stub_delegate,
        )
        list(stream)  # exhaust -> commit on the with-usage path
        assert budget.spent > 0.0, (
            f"streamed unpriced-model call with a usage block must advance the USD "
            f"ledger (spent={budget.spent!r}); v0.7.0 committed $0 on the with-usage "
            f"path because actual_cost(shim) returned 0.0 for the unpriced model."
        )
        assert budget.pending_usd == 0.0


def test_v080_async_fallback_unpriced_stream_with_usage_advances_usd_ledger():
    """Async variant of the stream-with-usage fallback fix (RED on v0.7.0, GREEN)."""
    max_tokens = 2000
    final_usage = Usage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)

    async def stub_delegate(*args, **kwargs):
        return _async_stream_with_final_usage(3, final_usage)

    async def run():
        with task(ceiling_usd=100.0, ceiling_tokens=10_000, name="fb-stream-async", on_unpriced="fallback") as budget:
            stream = await wrap_mod.acompletion(
                model=MODEL_UNPRICED, messages=MESSAGES, max_tokens=max_tokens,
                stream=True, real=stub_delegate,
            )
            out = [c.content async for c in stream]
            return budget, out

    budget, out = asyncio.run(run())
    assert len(out) == 3
    assert budget.spent > 0.0, (
        f"async streamed unpriced-model call with usage must advance the USD ledger "
        f"(spent={budget.spent!r})."
    )
    assert budget.pending_usd == 0.0


# =========================================================================== #
# fix-agentfuse-demo-broken-on-wheel-install
# =========================================================================== #


def test_v080_demo_module_is_importable_from_package():
    """RED on v0.7.0 (no agentfuse._runaway_demo module -> ImportError), GREEN on
    v0.8.0.

    The demo must live inside the installed package (``agentfuse._runaway_demo``)
    so it ships in the wheel; ``examples/`` is not packaged.
    """
    from agentfuse import _runaway_demo

    assert hasattr(_runaway_demo, "run_demo"), "vendored demo must expose run_demo"
    assert hasattr(_runaway_demo, "_agent_step"), "vendored demo must expose _agent_step"


def test_v080_demo_cli_runs_and_trips_from_package_module(monkeypatch):
    """RED on v0.7.0 (``agentfuse demo`` raised ClickException on a wheel install
    because examples/ was not shipped), GREEN on v0.8.0 (the vendored package
    module is used, so the demo runs and trips the fuse with no examples/ on the
    path).

    Simulate a wheel install by hiding the ``examples`` package from import so
    only the package-internal ``agentfuse._runaway_demo`` can satisfy the load.
    """
    import builtins

    real_import = builtins.__import__

    def _block_examples(name, *args, **kwargs):
        # Block the `from examples import runaway_agent` path (absent in a wheel).
        if name == "examples" or name.startswith("examples."):
            raise ImportError("examples/ is not shipped in the wheel (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_examples)

    # _load_runaway_agent must still succeed via the package module.
    from agentfuse.cli import _load_runaway_agent

    module = _load_runaway_agent()
    assert hasattr(module, "run_demo"), (
        "demo must load from the package module when examples/ is unavailable"
    )

    # And `agentfuse demo` must run and trip the fuse (exit 0, trip summary).
    result = CliRunner().invoke(main, ["demo", "--ceiling", "0.50"])
    assert result.exit_code == 0, result.output
    assert "FUSE TRIPPED" in result.output or "halting" in result.output.lower() or (
        "halted" in result.output.lower()
    ), f"demo must trip the fuse; got:\n{result.output}"


def test_v080_examples_entrypoint_re_exports_vendored_demo():
    """The repo-root ``examples/runaway_agent.py`` is now a thin re-export of the
    vendored module, so ``python examples/runaway_agent.py`` from a checkout still
    works (RED on v0.7.0 where examples/ held the full copy; the re-export is the
    v0.8.0 change)."""
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    entry = repo_root / "examples" / "runaway_agent.py"
    assert entry.exists(), "examples/runaway_agent.py must still exist as the entrypoint"
    spec = importlib.util.spec_from_file_location("agentfuse_test_ra_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The thin entrypoint re-exports the vendored demo's API.
    assert hasattr(module, "run_demo")
    assert hasattr(module, "_agent_step")
    # And it delegates to the same vendored implementation.
    from agentfuse import _runaway_demo

    assert module.run_demo is _runaway_demo.run_demo
