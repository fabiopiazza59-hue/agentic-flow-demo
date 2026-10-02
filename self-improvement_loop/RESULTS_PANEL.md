# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (4 sessions, 120 name-forecasts; mean daily APE gain 0.004%, CS n=4 sessions (too few)).

**Pending (2026-10-02):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.004% | n=4 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.002% | n=4 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0018 | n=4 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.072% | n=4 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.075% | n=4 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.26%, free baseline 1.26%, prior close 1.19%; LLM 80% interval coverage 77.50%; names moved per session 1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 4 | 1.43% | 1.43% | 0 |
| ADBE | 4 | 1.49% | 1.49% | 0 |
| AMD | 4 | 1.25% | 1.25% | 0 |
| AMZN | 4 | 0.81% | 0.81% | 0 |
| AVGO | 4 | 1.80% | 1.80% | 0 |
| BAC | 4 | 1.51% | 1.51% | 0 |
| CAT | 4 | 1.13% | 1.13% | 0 |
| COST | 4 | 0.56% | 0.53% | 1 |
| CRM | 4 | 2.20% | 2.20% | 0 |
| CVX | 4 | 0.86% | 0.86% | 0 |
| DIS | 4 | 1.05% | 1.08% | 1 |
| GOOGL | 4 | 1.21% | 1.21% | 0 |
| HD | 4 | 1.03% | 1.09% | 1 |
| JNJ | 4 | 1.40% | 1.40% | 0 |
| JPM | 4 | 1.15% | 1.15% | 0 |
| KO | 4 | 0.54% | 0.54% | 0 |
| LLY | 4 | 0.99% | 0.99% | 0 |
| MA | 4 | 0.92% | 0.92% | 0 |
| META | 4 | 2.50% | 2.50% | 0 |
| MSFT | 4 | 0.69% | 0.69% | 0 |
| NFLX | 4 | 2.24% | 2.24% | 0 |
| NVDA | 4 | 0.81% | 0.81% | 0 |
| ORCL | 4 | 2.30% | 2.30% | 0 |
| PEP | 4 | 0.75% | 0.75% | 0 |
| PG | 4 | 1.51% | 1.51% | 0 |
| TSLA | 4 | 1.55% | 1.55% | 0 |
| UNH | 4 | 0.93% | 0.93% | 0 |
| V | 4 | 0.65% | 0.65% | 0 |
| WMT | 4 | 1.50% | 1.54% | 1 |
| XOM | 4 | 0.98% | 0.98% | 0 |

_Research experiment, not financial advice._
