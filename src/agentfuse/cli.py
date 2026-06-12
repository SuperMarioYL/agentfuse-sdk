"""The ``agentfuse`` command-line interface.

Minimal and honest about what persists. Per the MVP plan, a persistent spend
store is explicitly out of scope for v0.1 — AgentFuse is an in-process library,
the ledger lives and dies with the agent process. So:

* ``agentfuse status`` reports the configured default ceiling and is upfront that
  there is no cross-process active task to read (the live ledger only exists
  inside a running ``with Fuse(...)`` scope).
* ``agentfuse demo`` runs the bundled offline ``runaway_agent`` example so you
  can watch the fuse trip without any API key.
* ``agentfuse --version`` prints the package version.
"""

from __future__ import annotations

import click

from agentfuse import __version__

# The default per-task ceiling the demo and docs use. Not persisted anywhere —
# v0.1 has no spend store (out of scope); this is just the documented default.
DEFAULT_CEILING_USD = 5.00


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="agentfuse")
@click.pass_context
def main(ctx: click.Context) -> None:
    """AgentFuse — a per-task spend circuit-breaker that halts an Agent
    before it burns your budget.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
def status() -> None:
    """Show the current task ledger summary.

    v0.1 keeps no persistent spend store (out of scope), so there is no
    cross-process "active task" to read from outside a running process. This
    prints the configured default ceiling and explains where the live ledger
    actually lives.
    """
    click.echo("AgentFuse status")
    click.echo("=" * 40)
    click.echo("active task : none")
    click.echo(
        f"default ceiling : ${DEFAULT_CEILING_USD:.2f} per task "
        "(used by `agentfuse demo` and the docs)"
    )
    click.echo("")
    click.echo(
        "AgentFuse is an in-process library: the live ledger only exists inside a\n"
        "running `with Fuse(max_spend_usd=...)` scope and is not persisted between\n"
        "runs (a persistent spend store is out of scope for v0.1). Inspect a task's\n"
        "spend in-process via the Budget returned by `task(...)` / `Fuse(...)`,\n"
        "e.g. `print(budget.snapshot())`."
    )
    click.echo("")
    click.echo("Try `agentfuse demo` to watch the fuse trip on a runaway agent.")


@main.command()
@click.option(
    "--ceiling",
    "ceiling_usd",
    type=float,
    default=0.50,
    show_default=True,
    help="Per-task USD ceiling for the demo run.",
)
def demo(ceiling_usd: float) -> None:
    """Run the bundled offline runaway-agent demo and watch the fuse trip.

    Runs entirely offline (litellm mock responses) — no API key required.
    """
    runaway_agent = _load_runaway_agent()
    runaway_agent.run_demo(ceiling_usd=ceiling_usd)


def _load_runaway_agent():
    """Import the bundled ``examples/runaway_agent`` demo module.

    Tries the importable ``examples`` package first (works when the repo root is
    on ``sys.path``, e.g. running from a clone). Falls back to loading the file
    directly relative to this package, so ``agentfuse demo`` works even when only
    the wheel is installed and ``examples/`` was shipped alongside it.
    """
    try:
        from examples import runaway_agent  # type: ignore[import-not-found]

        return runaway_agent
    except ImportError:
        pass

    import importlib.util
    from pathlib import Path

    here = Path(__file__).resolve()
    # repo layout: <root>/src/agentfuse/cli.py and <root>/examples/runaway_agent.py
    candidate = here.parents[2] / "examples" / "runaway_agent.py"
    if not candidate.exists():
        raise click.ClickException(
            "Bundled demo (examples/runaway_agent.py) not found. Run it directly "
            "from a source checkout: `python examples/runaway_agent.py`."
        )
    spec = importlib.util.spec_from_file_location("agentfuse_demo_runaway", candidate)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":  # pragma: no cover
    main()
