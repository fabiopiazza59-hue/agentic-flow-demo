# Implementation Plan — AMZN Daily Close Predictor

## Architecture
```
                       ┌─────────────────── run_daily.py (orchestrator) ───────────────────┐
                       │  guard → score → reflect → predict → report → commit              │
                       └───────────────────────────────────────────────────────────────────┘
   data/                evals/                 agents/                    report.py
   ├ providers.py       ├ metrics.py (pure)    ├ analysts.py (5×Sonnet)   ├ results.csv
   ├ market_calendar.py └ scorecard.py         ├ meta_judge.py (Opus)     ├ metrics.json
   features.py                                  └ reflector.py (Opus)      ├ RESULTS.md
                                                                           └ site/ (Pages)
   state: data/predictions.jsonl · learnings/{STRATEGY.md,scorecards.json,*.md} · results/*
```

## Module responsibilities
- **config.py** — Pydantic `Settings`: `SYMBOL="AMZN"`, `PASS_THRESHOLD=0.01`, `ROLLING_WINDOW=20`,
  model IDs, lookback, paths (resolved relative to project root), provider selection from env.
- **utils.py** — `extract_json(text)` (markdown-fence / preamble tolerant), `iso_today()`, atomic
  jsonl append/read, safe float parsing.
- **data/providers.py** — `get_quote`, `get_actual_close`, `get_history` with fallback chains.
  History: Alpha Vantage `TIME_SERIES_DAILY` (keyed) → yfinance (keyless) → Stooq+PoW (last resort).
  Quote: Finnhub → Alpha Vantage → yfinance → derived from history. (Stooq now PoW/API-key gated, so
  yfinance is the practical keyless source and Alpha Vantage the CI-reliable keyed one.)
- **data/market_calendar.py** — `is_trading_day(date)`, `previous_trading_day(date)`,
  `last_n_trading_days(n)` via NYSE calendar.
- **features.py** — `build_features(history, quote)` → dict: returns (1/5/20d), realized vol, RSI(14),
  SMA(5/20/50), distance-to-SMA, 52w hi/lo position, gap vs prev close, recent OHLC table (text).
- **evals/metrics.py** — pure funcs: `ape`, `is_pass`, `direction`, `directional_hit`,
  `baseline_ape`, `brier_component`, `aggregate(rows, window)`.
- **evals/scorecard.py** — `update_scorecards(scorecards, analyst_preds, actual)`,
  `weight_hints(scorecards)`.
- **agents/analysts.py** — `ANALYSTS` registry (technical, momentum, contrarian, news, macro). Each:
  system prompt + feature payload → strict JSON. News analyst attaches `web_search` tool. Runs via
  `ThreadPoolExecutor`. Failures dropped gracefully.
- **agents/meta_judge.py** — Opus call: inputs analyst preds + scorecards + STRATEGY.md +
  last-N learnings → final `{predicted_close, direction, confidence, weights, rationale}`.
- **agents/reflector.py** — Opus call: scored result + history → post-mortem md + bounded STRATEGY.md
  note (returns text to append; orchestrator does the file write).
- **loop/run_daily.py** — CLI orchestrator with modes + `--dry-run`; wires everything; git commit.
- **report.py** — ledger → results.csv, metrics.json, RESULTS.md, results/site/data.json (+ copy
  static index.html if missing).

## Claude integration details
- `anthropic.Anthropic()` auto-reads `ANTHROPIC_API_KEY`.
- Analysts: `model=claude-sonnet-4-6`, `max_tokens≈1024`, temperature modest; respond JSON-only.
- News analyst: add `tools=[{"type":"web_search_20250305","name":"web_search","max_uses":3}]`.
- Meta-judge / reflector: `model=claude-opus-4-8`, `max_tokens≈1500`.
- All model JSON parsed through `extract_json`; on parse failure, analyst is skipped.
- `--dry-run` without `ANTHROPIC_API_KEY` uses deterministic stub predictions (prev_close ± small
  feature-driven nudge) so the full pipeline is testable offline.

## Mock/test strategy
- HTTP providers mocked with `responses`/monkeypatch fixtures (no network in unit tests).
- `metrics.py` tested at the ±1% boundary (0.0099, 0.0100, 0.0101), direction edge (flat), baseline
  beat/miss, Brier.
- `extract_json` tested on fenced, prefixed, and trailing-comma inputs.
- End-to-end `--dry-run` uses a fixture ledger + Stooq stub + stub Claude → asserts ledger/score/
  report artifacts are produced and idempotent.

## Sequencing (maps to tasks.md)
config/utils → data → evals → features/agents → orchestrator/report → deps/env/workflow/site →
tests/seed/e2e.
