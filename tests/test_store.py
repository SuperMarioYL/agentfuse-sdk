"""Tests for the v0.2 opt-in append-only spend record (store.py) and the
``agentfuse status --log <path>`` summary it feeds.

No network; the store is plain JSONL on a tmp_path.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from agentfuse import Budget, last_record, read_records, record_task
from agentfuse.cli import main
from agentfuse.store import TaskRecord


def _budget(spent_usd=0.0, spent_tokens=0, **kw):
    b = Budget(ceiling_usd=kw.pop("ceiling_usd", 5.0), **kw)
    if spent_usd or spent_tokens:
        b.commit(spent_usd, spent_tokens)
    return b


# --------------------------------------------------------------------------- #
# record_task: append-only, returns the written record
# --------------------------------------------------------------------------- #


def test_record_task_appends_one_line(tmp_path):
    log = tmp_path / "spend.jsonl"
    b = _budget(spent_usd=1.5, name="scan-1")
    rec = record_task(b, tripped=False, log_path=log)
    assert isinstance(rec, TaskRecord)
    assert rec.name == "scan-1"
    assert rec.spent_usd == pytest.approx(1.5)
    assert rec.tripped is False
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["name"] == "scan-1"
    assert parsed["spent_usd"] == pytest.approx(1.5)
    assert parsed["tripped"] is False
    assert parsed["v"] == 1
    assert parsed["ts"].endswith("Z")


def test_record_task_is_append_only(tmp_path):
    log = tmp_path / "spend.jsonl"
    record_task(_budget(spent_usd=1.0, name="a"), tripped=False, log_path=log)
    record_task(_budget(spent_usd=2.0, name="b"), tripped=True, log_path=log)
    record_task(_budget(spent_usd=3.0, name="c"), tripped=False, log_path=log)
    records = read_records(log)
    assert [r.name for r in records] == ["a", "b", "c"]
    assert [r.tripped for r in records] == [False, True, False]


def test_record_task_creates_parent_dirs(tmp_path):
    log = tmp_path / "nested" / "deep" / "spend.jsonl"
    rec = record_task(_budget(spent_usd=0.1), tripped=False, log_path=log)
    assert rec is not None
    assert log.exists()


def test_record_task_captures_tokens(tmp_path):
    log = tmp_path / "spend.jsonl"
    b = _budget(spent_usd=0.5, spent_tokens=320, ceiling_tokens=1000, name="tok")
    rec = record_task(b, tripped=True, log_path=log)
    assert rec.spent_tokens == 320
    assert rec.ceiling_tokens == 1000
    back = read_records(log)[0]
    assert back.spent_tokens == 320
    assert back.ceiling_tokens == 1000


def test_record_task_bad_path_returns_none_not_raises(tmp_path):
    # A directory where a file is expected -> OSError swallowed, returns None.
    bad = tmp_path / "iamadir"
    bad.mkdir()
    rec = record_task(_budget(spent_usd=1.0), tripped=False, log_path=bad)
    assert rec is None


# --------------------------------------------------------------------------- #
# read_records / last_record: tolerant readers
# --------------------------------------------------------------------------- #


def test_read_records_missing_file_is_empty(tmp_path):
    assert read_records(tmp_path / "nope.jsonl") == []
    assert last_record(tmp_path / "nope.jsonl") is None


def test_read_records_skips_blank_and_malformed_lines(tmp_path):
    log = tmp_path / "spend.jsonl"
    good = TaskRecord(
        name="ok", ceiling_usd=5.0, spent_usd=1.0, tripped=False, ts="2026-06-19T00:00:00Z"
    ).to_json()
    log.write_text(f"\n{good}\nnot json at all\n   \n")
    records = read_records(log)
    assert len(records) == 1
    assert records[0].name == "ok"


def test_last_record_returns_most_recent(tmp_path):
    log = tmp_path / "spend.jsonl"
    record_task(_budget(spent_usd=1.0, name="first"), tripped=False, log_path=log)
    record_task(_budget(spent_usd=2.0, name="last"), tripped=True, log_path=log)
    last = last_record(log)
    assert last is not None
    assert last.name == "last"
    assert last.tripped is True


# --------------------------------------------------------------------------- #
# CLI: agentfuse status --log
# --------------------------------------------------------------------------- #


def test_cli_status_log_summarises_last_task(tmp_path):
    log = tmp_path / "spend.jsonl"
    record_task(_budget(spent_usd=1.0, name="early"), tripped=False, log_path=log)
    record_task(
        _budget(spent_usd=4.98, ceiling_usd=5.0, name="runaway"),
        tripped=True,
        log_path=log,
    )
    result = CliRunner().invoke(main, ["status", "--log", str(log)])
    assert result.exit_code == 0
    assert "runaway" in result.output
    assert "TRIPPED" in result.output
    assert "4.98" in result.output
    assert "5.00" in result.output
    assert "tasks logged : 2" in result.output


def test_cli_status_log_empty(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    result = CliRunner().invoke(main, ["status", "--log", str(log)])
    assert result.exit_code == 0
    assert "none recorded yet" in result.output


def test_cli_status_log_missing_file(tmp_path):
    result = CliRunner().invoke(main, ["status", "--log", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 0
    assert "none recorded yet" in result.output


def test_cli_status_without_log_is_static(tmp_path):
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    assert "active task : none" in result.output
