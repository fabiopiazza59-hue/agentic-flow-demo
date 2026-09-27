# Tasks — AMZN Daily Close Predictor

Ordered, verifiable build tasks. Each closes when its acceptance check passes.

## T1 — Spec foundation
- [x] constitution.md, spec.md, plan.md, tasks.md
- **Accept:** specs cover mission, eval contract, data contracts, FRs/NFRs.

## T2 — Config & utils
- [ ] `src/config.py` (Pydantic Settings, paths, thresholds, model IDs, provider selection)
- [ ] `src/utils.py` (`extract_json`, jsonl IO, date helpers)
- **Accept:** `python -c "from src.config import settings; print(settings.SYMBOL)"` works;
  `extract_json` unit tests pass.

## T3 — Data layer
- [ ] `src/data/providers.py` (Finnhub/AlphaVantage/Stooq + fallback; `get_quote`,
  `get_actual_close`, `get_history`)
- [ ] `src/data/market_calendar.py` (trading-day guard + prev/last-n helpers)
- **Accept:** Stooq history fetch returns a non-empty DataFrame; provider-fallback unit test passes;
  `is_trading_day` correct for a known weekend/holiday.

## T4 — Evals
- [ ] `src/evals/metrics.py` (pure metric funcs + `aggregate`)
- [ ] `src/evals/scorecard.py` (per-strategy update + weight hints)
- **Accept:** boundary tests (±1%), baseline beat/miss, Brier, aggregate over synthetic ledger pass.

## T5 — Features & agents
- [ ] `src/features.py`
- [ ] `src/agents/analysts.py` (5 analysts, parallel, web_search on news, graceful failure)
- [ ] `src/agents/meta_judge.py` (Opus synthesis from scorecards + STRATEGY.md + learnings)
- [ ] `src/agents/reflector.py` (post-mortem + bounded STRATEGY.md note)
- **Accept:** dry-run stub path produces a valid prediction dict without network/keys.

## T6 — Orchestrator & reporting
- [ ] `src/loop/run_daily.py` (modes: daily/score/predict/report/backfill; `--dry-run`; git commit)
- [ ] `src/report.py` (results.csv, metrics.json, RESULTS.md, site/data.json)
- **Accept:** `python -m src.loop.run_daily --mode daily --dry-run` runs end-to-end, writes ledger +
  results/*, is idempotent on re-run.

## T7 — Packaging, env, automation, dashboard
- [ ] `requirements.txt`, `.env.example`, `README.md`
- [ ] `results/site/index.html` (Chart.js dashboard)
- [ ] `.github/workflows/amzn-predict.yml` at **repo root** (daily job + deploy-pages job)
- **Accept:** workflow YAML lints; README documents secrets + Pages enablement.

## T8 — Tests, seed, e2e
- [ ] `tests/` (metrics, providers fallback, extract_json, e2e dry-run)
- [ ] Seed backfill + one mocked end-to-end
- **Accept:** `pytest` green; seeded ledger + RESULTS.md + Pages data render.

## Activation (manual, post-build)
- [ ] Add repo secrets: `ANTHROPIC_API_KEY` + `FINNHUB_API_KEY` (or `ALPHAVANTAGE_API_KEY`).
- [ ] Enable GitHub Pages (Settings → Pages → Source: GitHub Actions).
- [ ] Merge workflow to the default branch so the schedule fires.
- [ ] Trigger `workflow_dispatch` once to smoke-test.
