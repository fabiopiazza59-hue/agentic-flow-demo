# Constitution — AMZN Daily Close Predictor

## Mission
Build a **self-improving agentic loop** that predicts Amazon (AMZN) stock's daily closing price
before each US market open, validates itself against reality the next trading day, and refines its
own strategy over time from accumulated learnings — held to a hard, honest skill bar.

This is a learning experiment, **not financial advice and not a trading system**. It places no
orders and moves no money. Its only output is a prediction and an honest scorecard.

## Principles
1. **Honesty over vanity.** A wrong prediction is signal, not shame. Every miss is recorded with a
   post-mortem. The dashboard never hides FAILs.
2. **Beat the baseline or admit no edge.** The system is only "working" if it beats a naive
   random-walk forecast (`predicted = previous close`) on rolling MAPE. Until then, the README and
   dashboard say so plainly.
3. **State lives in the repo.** Predictions, learnings, and scorecards are committed every run. Git
   history *is* the experiment log. Anyone can reconstruct the full story from the repo.
4. **Self-improvement is measured, not assumed.** Each strategy carries a tracked accuracy
   scorecard; the meta-judge reweights toward what has actually worked. The reflector edits the
   living STRATEGY.md within guardrails — append insight, never wipe history.
5. **Cheap, reproducible, robust.** One run/day, pennies of API spend. Validation never breaks: a
   keyless Stooq fallback always provides the truth close even if the keyed provider is down.
6. **Spec before code.** This repo is built spec-first. Behavior is defined in `spec/` before it is
   implemented.

## Success criteria
- **Operational:** the GitHub Action runs every trading day, scores yesterday, predicts today, and
  commits results — unattended — for 20+ consecutive trading days.
- **Eval primary:** rolling-20-day PASS rate (PASS = `|pred − actual| / actual ≤ 1.0%`) is tracked
  and trending up.
- **Skill bar:** rolling-20-day MAPE beats the random-walk baseline. This is the headline verdict.
- **Self-improvement evidence:** per-strategy weights shift measurably toward higher-accuracy
  strategies over time, and STRATEGY.md accrues concrete, tested insights.
- **Transparency:** `RESULTS.md` + a live GitHub Pages dashboard reflect the latest run.

## Tech stack
- **Language:** Python 3.11.
- **Intelligence:** Anthropic Claude API — analysts on `claude-sonnet-4-6`, meta-judge + reflector
  on `claude-opus-4-8`. News analyst uses the server-side `web_search` tool for live context.
- **Market data:** Alpha Vantage (recommended keyed source — also serves daily history) or Finnhub
  (live quote) via free API key; **yfinance** as the keyless fallback (Stooq is a best-effort last
  resort now that it gates CSV behind proof-of-work + an API key).
- **Libs:** `anthropic`, `pandas`, `pandas_market_calendars`, `requests`, `pydantic`,
  `python-dotenv`, `pytest`.
- **Automation:** GitHub Actions (cron + `workflow_dispatch`), commits state back, deploys a Pages
  dashboard.

## Non-goals
- No live trading, brokerage integration, or money movement.
- No intraday/high-frequency prediction — one daily close prediction per trading day.
- No guarantee of profit or accuracy; the experiment may conclude "no edge," and that is a valid,
  honestly-reported outcome.
