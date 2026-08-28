"""Regression tests for the v0.7.0 release-notes-drift fix.

The shipped v0.6.0 source carried STALE release notes: `CHANGELOG.md` had an
empty ``## [Unreleased]`` immediately followed by ``## [0.5.0]`` (no ``## [0.6.0]``
section, so the two v0.6.0 correctness fixes shipped undocumented), the
``[Unreleased]`` link reference was still based at ``compare/v0.5.0...HEAD`` (so
the v0.6.0 commits rendered under "Unreleased"), and there was no ``[0.6.0]:``
link reference. This is the exact drift the v0.4.0
``fix-changelog-missing-v030-link-and-stale-unreleased-base`` milestone
corrected, recurring for v0.6.0.

These tests pin the fix (red on the unfixed v0.6.0 source, green after):
* a ``## [0.6.0]`` section exists (the v0.6.0 fixes are finally documented),
* a ``## [0.7.0]`` section exists (this release is documented),
* the ``[Unreleased]`` compare is based at the latest release (``v0.7.0``),
* ``[0.6.0]`` and ``[0.7.0]`` link references exist.
"""

from __future__ import annotations

import re
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def _text() -> str:
    assert CHANGELOG.exists(), f"{CHANGELOG} not found"
    return CHANGELOG.read_text(encoding="utf-8")


def test_changelog_documents_v060_release():
    """RED on v0.6.0 (no ## [0.6.0] section), GREEN after backfill.

    The v0.6.0 fixes (stream-detector misclassification + streaming-meter token
    estimate) shipped with no CHANGELOG entry at all.
    """
    text = _text()
    assert re.search(r"^##\s+\[0\.6\.0\]\s", text, re.MULTILINE), (
        "CHANGELOG.md must have a ## [0.6.0] section documenting the v0.6.0 "
        "fixes (v0.6.0 shipped without one)."
    )


def test_changelog_documents_v070_release():
    """This release (v0.7.0) must be documented in the CHANGELOG."""
    text = _text()
    assert re.search(r"^##\s+\[0\.7\.0\]\s", text, re.MULTILINE), (
        "CHANGELOG.md must have a ## [0.7.0] section documenting this release."
    )


def test_unreleased_compare_based_at_latest_release():
    """RED on v0.6.0 ([Unreleased] based at v0.5.0...HEAD), GREEN after re-base.

    The [Unreleased] compare link must be based at the latest released tag
    (v0.7.0), not at a stale earlier tag — otherwise the latest release's own
    commits render under "Unreleased" on GitHub.
    """
    text = _text()
    match = re.search(r"^\[Unreleased\]:\s*(\S+)", text, re.MULTILINE)
    assert match, "CHANGELOG.md must define an [Unreleased] link reference"
    url = match.group(1)
    assert "v0.7.0...HEAD" in url, (
        f"[Unreleased] compare must be based at v0.7.0...HEAD (was stale at "
        f"v0.5.0...HEAD on the v0.6.0 source); got: {url!r}"
    )


def test_changelog_has_link_references_for_v060_and_v070():
    """RED on v0.6.0 (no [0.6.0] link ref), GREEN after.

    Without a ``[0.6.0]:`` reference a ``## [0.6.0]`` header renders as dead
    plain text on GitHub (the v0.4.0 fix established this discipline).
    """
    text = _text()
    assert re.search(r"^\[0\.6\.0\]:\s*\S+", text, re.MULTILINE), (
        "CHANGELOG.md must define a [0.6.0] link reference"
    )
    assert re.search(r"^\[0\.7\.0\]:\s*\S+", text, re.MULTILINE), (
        "CHANGELOG.md must define a [0.7.0] link reference"
    )
