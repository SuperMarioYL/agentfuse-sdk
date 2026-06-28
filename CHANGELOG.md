# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-29

Two correctness fixes that keep the fuse honest where it was quietly failing.
Both stay *executive* guardrails (they halt / meter), never charts.

### Fixed

- **Streamed responses now advance the cumulative ledger** (`agentfuse.stream`,
  wired through `wrap.completion` / `wrap.acompletion`). A `stream=True` call
  returns a wrapper with no `.usage` until consumed, so post-call metering used
  to commit `$0` and the cumulative USD / token ceilings could **never trip on a
  streamed run** — the dominant agent call mode. The wrapper now meters the
  stream on exhaustion: it commits the real cost when the provider emits a usage
  block (litellm's `stream_options={"include_usage": True}`), and otherwise
  commits the pre-call upper-bound estimate so the ledger still advances and the
  fuse still trips on the next call. Chunks pass through transparently; sync and
  async streams are both covered.

### Changed

- **`Fuse(max_total_tokens=...)` replaces the confusingly-named
  `Fuse(max_tokens=...)`** for the cumulative whole-task token ceiling. In the
  LLM ecosystem `max_tokens` means a *single call's* completion length, so
  `Fuse(max_tokens=4096)` silently capped the *whole task* at 4096 tokens and
  tripped after roughly one call. The cumulative ceiling keyword is now
  `max_total_tokens` (matching `task()` / `@fuse()`'s `ceiling_tokens`).
  `max_tokens` is kept as a **deprecated alias** for one release — it emits a
  `DeprecationWarning` and still maps to the cumulative ceiling, so existing code
  keeps working.

## [0.2.0] - 2026-06-19

Hardening the fuse so it can't silently disable itself, plus two new ceilings
and an opt-in spend record. Every change is still an *executive* guardrail (it
halts), not a chart.

### Changed

- **Fail-closed on unpriced models** (`pricing.estimate_prompt_cost` /
  `estimate_call`) — a model missing from `litellm.model_cost` used to estimate
  `$0.00`, which silently passed *every* call on an unknown / self-hosted model:
  the fuse no-op'd exactly when runaway risk was highest. There is now an
  `on_unpriced` policy, defaulting to `'block'` (raise the new
  `UnpricedModelError`). Opt into `'fallback'` (price at a conservative
  per-token rate) or `'warn-pass'` (the old pass-through) per task via
  `Fuse(..., on_unpriced=...)` / `@fuse(on_unpriced=...)` / `task(...)`.

### Added

- **Token ceiling** (`Budget(ceiling_tokens=...)`, `Fuse(max_tokens=...)`) —
  closes the m2 spec gap: a task can be capped by USD, by cumulative tokens, or
  by whichever trips first. `spent_tokens` / `remaining_tokens()` are tracked
  alongside USD and surfaced in `snapshot()`.
- **Per-call hard cap** (`single_call_ceiling=...`) — an optional per-call USD
  ceiling that trips independently of the cumulative ledger, so one oversized
  prompt cannot blow the whole budget in a single shot.
- **Opt-in spend record** (`agentfuse.store`, `record_task` / `read_records` /
  `last_record`) — an append-only JSONL log of finished tasks (name, ceiling,
  spent, tripped?, timestamp, tokens). `agentfuse status --log <path>` now
  summarises the last task's REAL spend across processes. Execution-adjacent
  record-keeping only — no visualization, no monitoring service, no cross-run
  budget rollover.
- `UnpricedModelError` exception; `BudgetExceeded` now carries a `limit_kind`
  (`'usd'` / `'tokens'` / `'single_call'`) plus token-ledger context so the trip
  banner and message say *which* ceiling blew.

### Hardened

- **Reject zero / non-finite ceilings** — `Budget.__init__` now requires a
  finite `ceiling_usd > 0` (was only `>= 0`), so `0.0` / `NaN` / `±inf` ceilings
  no longer produce a fuse state indistinguishable from "off". The same
  validation is surfaced through `Fuse.__init__` and `fuse()` / `fused()`.

## [0.1.0] - 2026-06-13

First public release — an *enforcing* per-task spend circuit-breaker for
autonomous agents. The fuse trips **before** the over-budget call is sent, so
the money is never spent.

### Added

- **Per-task spend ledger** (`Budget`) — a thread-safe USD ledger with a hard
  ceiling, a pre-call admission check (`would_exceed` / `check`), post-call
  `commit` of confirmed spend, and immutable `snapshot()`s for the CLI/demo.
- **Pre-call cost meter** (`pricing.estimate_prompt_cost` / `actual_cost`) — a
  conservative upper-bound estimate (prompt tokens + worst-case `max_tokens`
  completion, priced from `litellm.model_cost`, tiktoken fallback) for the gate,
  plus real-cost readback from the response `Usage` for post-call metering.
  Degrades gracefully when a model is missing from the price table.
- **Pre-call gate** (`gate`) — estimates the next call's upper-bound cost and
  raises `BudgetExceeded` *before* delegating to litellm when it would cross the
  ceiling. Prints the `🔌 FUSE TRIPPED` banner on the trip path.
- **litellm wrapper** (`wrap.completion` / `acompletion`, `install` /
  `uninstall`) — gates `litellm.completion` / `litellm.acompletion` in
  AgentFuse's own code *before* the call goes out (sync + async); `install()`
  monkeypatches litellm so existing agent code is gated with zero edits.
  Post-call metering reads the real `Usage` back from the response.
- **Ergonomic API** — `Fuse(max_spend_usd=...)` context manager, `task(...)`
  context manager, and the `@fuse` / `@fused` decorator, all binding a per-task
  budget via `contextvars` (async- and thread-safe). `current_budget()` exposes
  the active ledger.
- **CLI** (`agentfuse`) — `agentfuse status`, `agentfuse demo` (runs the bundled
  offline runaway-agent example), and `agentfuse --version`.
- **Offline demo** (`examples/runaway_agent.py`) — reproduces the "AI agent
  bankrupted its operator scanning DN42" loop with realistic token usage via
  litellm `mock_response` (no API key, no network) and trips the fuse on honest
  arithmetic.
- `BudgetExceeded` / `AgentFuseError` exceptions carrying structured
  `spent` / `ceiling` / `would_spend` fields.
- 30 tests (`test_budget` ×16, `test_fuse` ×14); CI on Python 3.11 / 3.12.

[Unreleased]: https://github.com/supermario_leo/agentfuse/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/supermario_leo/agentfuse/releases/tag/v0.2.0
[0.1.0]: https://github.com/supermario_leo/agentfuse/releases/tag/v0.1.0
