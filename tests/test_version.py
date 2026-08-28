"""Regression tests for the v0.7.0 version-drift fix.

The shipped v0.6.0 release forgot the version bump: ``src/agentfuse/__init__.py``
and ``pyproject.toml`` both carried ``0.5.0`` while the latest tag was ``v0.6.0``,
so ``agentfuse --version`` reported ``0.5.0`` on a v0.6.0 install and ``pip
install`` of v0.6.0 installed a wheel named ``agentfuse-0.5.0`` (git history proves
every prior release bumped ``__version__``; the v0.6.0 tag did not). v0.7.0 bumps
both to ``0.7.0`` and adds this guard so the version stays honest and the two
sources can never drift apart again.

* ``test_version_is_0_7_0`` is the red->green pin: it FAILS on the unfixed ``0.5.0``
  source (the exact failure v0.6.0 shipped) and is updated per release.
* ``test_version_matches_pyproject`` is the durable anti-drift guard: the package
  version must always equal the version declared in ``pyproject.toml``.
* ``test_cli_version_reports_0_7_0`` pins the user-facing ``agentfuse --version``.
"""

from __future__ import annotations

import re
from pathlib import Path

import agentfuse
from agentfuse.cli import main

from click.testing import CliRunner

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# The shipped release this source is tagged as. Bump together with
# src/agentfuse/__init__.py and pyproject.toml on every release (the v0.6.0
# release forgot this bump — this pin makes a forgotten bump fail CI).
SHIPPED_VERSION = "0.7.0"


def test_version_is_0_7_0():
    """RED on the unfixed 0.5.0 source (v0.6.0 forgot the bump); GREEN after."""
    assert agentfuse.__version__ == SHIPPED_VERSION, (
        f"agentfuse.__version__ is {agentfuse.__version__!r}, expected "
        f"{SHIPPED_VERSION!r} — the package version must match the shipped "
        f"release tag (v0.6.0 shipped with a stale 0.5.0 string)."
    )


def test_version_matches_pyproject():
    """The package version must equal the version declared in pyproject.toml.

    The v0.6.0 drift happened because there was no guard tying
    ``__version__`` to ``pyproject.toml``'s ``[project] version``; both were
    stale at 0.5.0. This keeps the two version sources in lock-step so a future
    release that bumps one but not the other fails CI.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, re.MULTILINE)
    assert match, "pyproject.toml has no [project] version field"
    pyproject_version = match.group(1)
    assert agentfuse.__version__ == pyproject_version, (
        f"agentfuse.__version__ ({agentfuse.__version__!r}) does not match "
        f"pyproject.toml version ({pyproject_version!r}) — the two version "
        f"sources drifted apart (the v0.6.0 root cause)."
    )


def test_cli_version_reports_0_7_0():
    """``agentfuse --version`` must report the shipped release (cli.py wires it
    to ``__version__`` via ``click.version_option``)."""
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert SHIPPED_VERSION in result.output, (
        f"`agentfuse --version` did not report {SHIPPED_VERSION!r}; got: "
        f"{result.output!r}"
    )
