# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (5 sessions, 150 name-forecasts; mean daily APE gain 0.003%, CS n=5 sessions (too few)).

**Pending (2026-10-05):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.003% | n=5 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.002% | n=5 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0015 | n=5 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.045% | n=5 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.048% | n=5 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.22%, free baseline 1.22%, prior close 1.17%; LLM 80% interval coverage 80.00%; names moved per session 0.8. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 0 | — | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 5 | 1.31% | 1.31% | 0 |
| ADBE | 5 | 1.51% | 1.51% | 0 |
| AMD | 5 | 1.52% | 1.52% | 0 |
| AMZN | 5 | 0.86% | 0.86% | 0 |
| AVGO | 5 | 1.99% | 1.99% | 0 |
| BAC | 5 | 1.22% | 1.22% | 0 |
| CAT | 5 | 1.36% | 1.36% | 0 |
| COST | 5 | 0.55% | 0.53% | 1 |
| CRM | 5 | 1.95% | 1.95% | 0 |
| CVX | 5 | 0.71% | 0.71% | 0 |
| DIS | 5 | 1.01% | 1.04% | 1 |
| GOOGL | 5 | 1.22% | 1.22% | 0 |
| HD | 5 | 0.84% | 0.89% | 1 |
| JNJ | 5 | 1.37% | 1.37% | 0 |
| JPM | 5 | 0.97% | 0.97% | 0 |
| KO | 5 | 0.57% | 0.57% | 0 |
| LLY | 5 | 0.93% | 0.93% | 0 |
| MA | 5 | 0.93% | 0.93% | 0 |
| META | 5 | 2.03% | 2.03% | 0 |
| MSFT | 5 | 0.64% | 0.64% | 0 |
| NFLX | 5 | 2.05% | 2.05% | 0 |
| NVDA | 5 | 0.86% | 0.86% | 0 |
| ORCL | 5 | 2.38% | 2.38% | 0 |
| PEP | 5 | 0.62% | 0.62% | 0 |
| PG | 5 | 1.34% | 1.34% | 0 |
| TSLA | 5 | 2.05% | 2.05% | 0 |
| UNH | 5 | 1.10% | 1.10% | 0 |
| V | 5 | 0.56% | 0.56% | 0 |
| WMT | 5 | 1.24% | 1.28% | 1 |
| XOM | 5 | 0.79% | 0.79% | 0 |

_Research experiment, not financial advice._
