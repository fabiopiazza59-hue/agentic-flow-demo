# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (2 sessions, 60 name-forecasts; mean daily APE gain 0.007%, CS n=2 sessions (too few)).

**Pending (2026-09-30):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.007% | n=2 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.003% | n=2 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0042 | n=2 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.034% | n=2 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.041% | n=2 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.23%, free baseline 1.24%, prior close 1.20%; LLM 80% interval coverage 85.00%; names moved per session 2. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 2 | 1.83% | 1.83% | 0 |
| ADBE | 2 | 1.39% | 1.39% | 0 |
| AMD | 2 | 2.13% | 2.13% | 0 |
| AMZN | 2 | 0.85% | 0.85% | 0 |
| AVGO | 2 | 1.39% | 1.39% | 0 |
| BAC | 2 | 1.67% | 1.67% | 0 |
| CAT | 2 | 0.48% | 0.48% | 0 |
| COST | 2 | 0.25% | 0.19% | 1 |
| CRM | 2 | 2.04% | 2.04% | 0 |
| CVX | 2 | 1.06% | 1.06% | 0 |
| DIS | 2 | 0.04% | 0.12% | 1 |
| GOOGL | 2 | 0.39% | 0.39% | 0 |
| HD | 2 | 0.95% | 1.06% | 1 |
| JNJ | 2 | 1.01% | 1.01% | 0 |
| JPM | 2 | 1.30% | 1.30% | 0 |
| KO | 2 | 0.51% | 0.51% | 0 |
| LLY | 2 | 0.18% | 0.18% | 0 |
| MA | 2 | 0.47% | 0.47% | 0 |
| META | 2 | 3.65% | 3.65% | 0 |
| MSFT | 2 | 0.88% | 0.88% | 0 |
| NFLX | 2 | 2.17% | 2.17% | 0 |
| NVDA | 2 | 1.34% | 1.34% | 0 |
| ORCL | 2 | 3.64% | 3.64% | 0 |
| PEP | 2 | 0.12% | 0.12% | 0 |
| PG | 2 | 1.39% | 1.39% | 0 |
| TSLA | 2 | 2.88% | 2.88% | 0 |
| UNH | 2 | 0.44% | 0.44% | 0 |
| V | 2 | 0.28% | 0.28% | 0 |
| WMT | 2 | 1.19% | 1.28% | 1 |
| XOM | 2 | 1.00% | 1.00% | 0 |

_Research experiment, not financial advice._
