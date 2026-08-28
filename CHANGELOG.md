# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-08-29

Two maintenance fixes that close the version/release-notes drift the v0.6.0 ship
introduced. No behavior change to the enforcement core; both stay *executive*
metadata/doc fixes. Pinned by `tests/test_version.py` and `tests/test_changelog.py`.

### Fixed

- **Package version no longer lies about its own release.** The v0.6.0 tag
  shipped with `__version__ = "0.5.0"` in `src/agentfuse/__init__.py` and
  `version = "0.5.0"` in `pyproject.toml` (every prior release bumped it; the
  v0.6.0 ship forgot), so `agentfuse --version` reported `0.5.0` on a v0.6.0
  install and `pip install` of v0.6.0 installed `agentfuse-0.5.0`; `web/site.json`
  `content_version` was `v0.5.0` for the same reason. Bumped all to `0.7.0` and
  added `tests/test_version.py` pinning `agentfuse.__version__ == "0.7.0"` (red on
  the unfixed source) and asserting it matches the `[project] version` in
  `pyproject.toml` so the two sources can't drift apart again.
- **CHANGELOG + README release notes back in step with the shipped releases.**
  The v0.6.0 fixes shipped with NO `## [0.6.0]` CHANGELOG section and the
  `[Unreleased]` compare was still based at `v0.5.0...HEAD` (the exact drift the
  v0.4.0 `fix-changelog-missing-v030-link-and-stale-unreleased-base` milestone
  corrected, recurring). Backfilled the `[0.6.0]` section (below), added this
  `[0.7.0]` section, added the `[0.6.0]`/`[0.7.0]` link references, re-based
  `[Unreleased]` at `v0.7.0...HEAD`, and appended `v0.5.0`/`v0.6.0`/`v0.7.0`
  entries to the README roadmap. `tests/test_changelog.py` pins the structure.

## [0.6.0] - 2026-07-29

Two correctness fixes that keep the streaming fuse honest where it was quietly
leaking. Both stay *executive* guardrails; pinned by `tests/test_v060_fixes.py`.

### Fixed

- **The stream detector no longer misclassifies a usage-less litellm
  `ModelResponse` as a stream (`agentfuse.stream.is_stream_response`).** litellm's
  finished `ModelResponse` is iterable (pydantic `__iter__`) and, when a provider
  (ollama / watsonx self-hosted models) emits no token usage, arrives WITHOUT
  `.usage`. The v0.5.0 detector only treated a response as finished when it
  carried a truthy `.usage`, so such a finished `ModelResponse` was classified as
  a stream: the wrapper returned the metering GENERATOR instead of the
  `ModelResponse` (`resp.choices[0].message.content` raised `AttributeError`)
  and, because the generator was never iterated, its `finally` never ran, so the
  pre-call `Reservation` leaked into the budget's `pending` balance forever,
  pinning the budget. `is_stream_response` now rejects litellm's finished
  `ModelResponse` (it exposes `.choices`) BEFORE the iterable check.
- **The streaming meter no longer drops the pre-call token estimate
  (`agentfuse.stream.meter_sync_stream` / `meter_async_stream`).** The v0.5.0
  streaming meter took `estimated_usd` but NOT `estimated_tokens`; the no-usage
  fallback committed `0` tokens, so the cumulative token ledger stayed frozen at
  `0` for every streamed no-usage call and the `ceiling_tokens` fuse never tripped
  on the dominant streamed no-usage call mode. `estimated_tokens` is now threaded
  through both meter functions (and the `wrap.py` call sites) and committed in the
  no-usage fallback, mirroring the existing USD-estimate fallback on the same
  path.

## [0.5.0] - 2026-07-25

Three correctness fixes that close the last ways the fuse could silently fail to
break. All three stay *executive* guardrails (they halt / reserve / gate) — no
dashboard, no monitoring service, no new surface. Each was empirically verified
end-to-end and is pinned by adversarial red→green tests (`tests/test_v050_fixes.py`).

### Fixed

- **`@fuse` / `@fused` no longer silently bypass the fuse for `async def`
  functions.** The decorator's `wrapper` was a sync `def`: for a coroutine `func`,
  `func(*args)` returned the coroutine object without starting the body, then the
  `with task(...)` block exited (resetting the `contextvars` budget binding) BEFORE
  the caller ever `await`ed the returned coroutine. When the async body finally
  ran, `current_budget()` was `None`, so `acompletion` took the no-budget
  pass-through path — `gate()` returned `0.0` and `commit_actual` was a no-op.
  The fuse was completely inert for every decorated async function, with no error
  and no warning (a circuit-breaker that silently did not break) — exactly the
  async-agent audience the GTM targets (hermes-agent / OpenViking runtimes). The
  decorator now branches on `inspect.iscoroutinefunction(func)`: for async, an
  `async def wrapper` does `return await func(...)` INSIDE the `with task(...)`
  block so the budget stays bound while the coroutine runs; the sync path is
  unchanged. `functools.wraps` preserves metadata in both branches.
- **The pre-call budget gate is now atomic across the awaited LLM call so
  concurrent fan-out cannot overshoot the ceiling.** `Budget.check` released the
  lock before `acompletion` awaited the delegate, so two concurrent calls both
  gated against the still-uncommitted ledger (`spent` unchanged because neither
  had committed yet), both passed, and both committed — overshooting the ceiling
  by up to `(concurrency * per-call estimate)`. Any async agent that fans out
  (`asyncio.gather` of tool calls / parallel completions) could silently overspend.
  `Budget` now keeps a `pending_usd` / `pending_tokens` reserve held under the
  existing lock: a new `Budget.reserve(estimate)` is the mutating gate that adds
  the estimate to `pending` (so the gate considers `spent + pending + estimate`
  against the ceiling) and returns a `Reservation` handle; `Budget.commit(...,
  reservation=)` releases the reservation and adds the real cost (success path);
  `Budget.release(reservation)` releases it without committing (error path).
  Threading the reservation handle through `check → commit/release` makes the
  gate atomic across the awaited call WITHOUT holding the lock across the HTTP
  call, and concurrent releases are attributable to the specific call that
  reserved them (the handle carries the reserved estimate). The litellm wrapper
  (`wrap.completion` / `wrap.acompletion`, and the streamed metering path
  `meter_sync_stream` / `meter_async_stream`) thread the reservation
  end-to-end and `release` on the delegate-exception path so no reservation leaks
  into `pending`. `Budget.check` / `would_exceed` remain non-mutating peeks
  (backward-compatible) so legacy callers and tests that ignore the return value
  cannot leak a reservation; the race-free path is `reserve`.
- **The pre-call completion-token upper bound is now a true upper bound for
  streamed calls without `max_tokens`.** `DEFAULT_MAX_COMPLETION_TOKENS = 1024`
  (used by `estimate_call` whenever a caller omits `max_tokens` — the common case
  for streamed agent calls) was not an upper bound: real streamed completions
  routinely produce 4k-16k tokens on modern frontier models, so the "upper-bound"
  estimate passed the pre-call gate, and the post-call commit of the real (larger)
  usage jumped the ledger past the ceiling with no retroactive trip — the fuse
  only gates the NEXT call, which for a one-shot streamed task never comes. The
  default is now `8192`, and when `litellm.model_cost` carries a model-specific
  `max_output_tokens` that is larger, that genuine cap is preferred (the static
  8192 is the floor fallback), so the estimate is an actual upper bound for
  typical streamed completions and the pre-call gate trips instead of the
  post-call overshoot. A caller-supplied `max_tokens` still overrides both
  (unchanged).

### Added

- `agentfuse.budget.Reservation` — the pending-reservation handle returned by
  `Budget.reserve`, threaded through the litellm wrapper to `commit` / `release`
  so concurrent fan-out is race-free. Exposes `estimated_usd` /
  `estimated_tokens` / `.active`.
- `agentfuse.fuse.gate_with_reservation` — the reserving variant of `gate` used
  by `wrap.completion` / `wrap.acompletion`; returns `(estimate, Reservation)`
  (or `(0.0, None)` with no active budget). `gate` is kept as the float-returning
  non-mutating peek for backward compatibility.

## [0.4.0] - 2026-07-21

Two doc/metadata correctness fixes plus one small in-process trip hook. Every
change is still an *executive* guardrail (it halts) or a doc fix — no dashboard,
no monitoring service.

### Fixed

- **Package metadata + README repo links now point at the real repo.**
  `pyproject.toml`'s PyPI `Homepage`/`Repository` URLs and both READMEs' CI-badge
  link, "Share this" paste text, and copyright-footer profile link all pointed
  at `https://github.com/supermario_leo/agentfuse` — a repo that does not exist
  (`supermario_leo` is the local git author identity, not the GitHub login that
  owns the repo). Every PyPI Homepage/Repository link 404'd. Corrected to the
  real `https://github.com/SuperMarioYL/agentfuse-sdk`.
- **CHANGELOG link references repaired.** The `[0.3.0]` header shipped without a
  matching `[0.3.0]: ...` link reference (so it rendered as dead plain text on
  GitHub), the `[Unreleased]` compare was still based at `v0.2.0...HEAD` instead
  of `v0.3.0...HEAD`, and all compare/release links used the wrong repo. Fixed:
  added the missing `[0.3.0]` reference, re-based `[Unreleased]` on `v0.3.0`, and
  rewrote every link to the real `SuperMarioYL/agentfuse-sdk` repo.

### Added

- **In-process `on_trip` callback hook.** `Budget(on_trip=...)` (threaded through
  `task(...)` / `Fuse(...)` / `@fuse(...)` alongside the existing `on_unpriced`
  kwarg) is invoked fail-soft with the fully-structured `BudgetExceeded` right
  before the over-budget call is blocked — *before* it is delegated to litellm,
  so the over-budget call is still never sent. Any exception the callback raises
  is logged and swallowed, so a user hook can never change whether the call is
  blocked. This advances the plan §1 `--report-endpoint` / `report_to=` /
  AgentFuse Cloud GTM in-process: an operator can wire
  `Fuse(max_spend_usd=5.0, on_trip=lambda e: requests.post(..., json=e.__dict__))`
  today, and the eventual `report_to="cloud"` upload hook is a thin `on_trip`
  wired to the hosted endpoint. It is still an executive guardrail, not a
  dashboard.

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

[Unreleased]: https://github.com/SuperMarioYL/agentfuse-sdk/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/SuperMarioYL/agentfuse-sdk/releases/tag/v0.7.0
[0.6.0]: https://github.com/SuperMarioYL/agentfuse-sdk/releases/tag/v0.6.0
[0.5.0]: https://github.com/SuperMarioYL/agentfuse-sdk/releases/tag/v0.5.0
[0.4.0]: https://github.com/SuperMarioYL/agentfuse-sdk/releases/tag/v0.4.0
[0.3.0]: https://github.com/SuperMarioYL/agentfuse-sdk/releases/tag/v0.3.0
[0.2.0]: https://github.com/SuperMarioYL/agentfuse-sdk/releases/tag/v0.2.0
[0.1.0]: https://github.com/SuperMarioYL/agentfuse-sdk/releases/tag/v0.1.0
