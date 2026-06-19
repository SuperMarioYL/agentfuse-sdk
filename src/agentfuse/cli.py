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
@click.option(
    "--log",
    "log_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to an opt-in JSONL spend record (see agentfuse.record_task). "
    "When given, summarise the last recorded task's REAL spend.",
)
def status(log_path: str | None) -> None:
    """Show the current task ledger summary.

    Without ``--log`` this is a static reminder that the live ledger lives only
    inside a running ``with Fuse(...)`` scope. With ``--log <path>`` it reads the
    opt-in append-only JSONL spend record (written by ``agentfuse.record_task``)
    and summarises the last finished task's REAL spend across processes.
    """
    if log_path is not None:
        _status_from_log(log_path)
        return

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
        "running `with Fuse(max_spend_usd=...)` scope. For cross-process history,\n"
        "opt into the append-only spend record via `agentfuse.record_task(budget,\n"
        "tripped=..., log_path=...)` and read it back with `agentfuse status --log\n"
        "<path>`. Inspect a live task in-process via `print(budget.snapshot())`."
    )
    click.echo("")
    click.echo("Try `agentfuse demo` to watch the fuse trip on a runaway agent.")


def _status_from_log(log_path: str) -> None:
    """Render the last recorded task from an opt-in JSONL spend record."""
    from agentfuse.store import read_records

    records = read_records(log_path)
    click.echo("AgentFuse status")
    click.echo("=" * 40)
    click.echo(f"spend record : {log_path}")
    if not records:
        click.echo("last task    : none recorded yet")
        click.echo("")
        click.echo(
            "No records found. Write one with `agentfuse.record_task(budget, "
            "tripped=..., log_path=...)`\nafter a `with Fuse(...)` block finishes."
        )
        return

    last = records[-1]
    state = "TRIPPED 🔌" if last.tripped else "ok"
    click.echo(f"tasks logged : {len(records)}")
    click.echo(f"last task    : {last.name}  [{state}]")
    click.echo(
        f"  spent      : ${last.spent_usd:.4f} / ${last.ceiling_usd:.2f} ceiling"
    )
    if last.ceiling_tokens is not None:
        click.echo(
            f"  tokens     : {last.spent_tokens:,} / {last.ceiling_tokens:,}"
        )
    elif last.spent_tokens:
        click.echo(f"  tokens     : {last.spent_tokens:,}")
    click.echo(f"  finished   : {last.ts}")


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
