# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (11 sessions, 329 name-forecasts; mean daily APE gain -0.003%, CS [-0.016%, +0.011%]).

**Pending (2026-10-13):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 3 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.016%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.002% | [-0.012%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.096 | n=3 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0003 | [-0.003, +0.004] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.023% | [-0.096%, +0.049%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.021% | [-0.097%, +0.055%] | ≈ not distinguishable |

MAPE — LLM 1.23%, free baseline 1.23%, prior close 1.21%; LLM 80% interval coverage 79.05%; names moved per session 0.8. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 6 | +0.001% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 5 | +0.000% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 11 | 1.05% | 1.05% | 0 |
| ADBE | 11 | 1.50% | 1.50% | 0 |
| AMD | 11 | 1.78% | 1.78% | 0 |
| AMZN | 11 | 1.22% | 1.22% | 0 |
| AVGO | 11 | 2.10% | 2.05% | 2 |
| BAC | 11 | 0.87% | 0.87% | 0 |
| CAT | 11 | 1.63% | 1.63% | 0 |
| COST | 11 | 0.70% | 0.69% | 1 |
| CRM | 11 | 1.63% | 1.63% | 0 |
| CVX | 11 | 0.91% | 0.91% | 0 |
| DIS | 11 | 0.99% | 1.00% | 1 |
| GOOGL | 11 | 0.91% | 0.91% | 0 |
| HD | 11 | 1.11% | 1.13% | 1 |
| JNJ | 11 | 1.24% | 1.24% | 0 |
| JPM | 11 | 0.67% | 0.67% | 0 |
| KO | 11 | 0.83% | 0.83% | 0 |
| LLY | 11 | 1.04% | 1.04% | 0 |
| MA | 11 | 1.14% | 1.14% | 0 |
| META | 10 | 1.64% | 1.64% | 0 |
| MSFT | 11 | 0.95% | 0.93% | 1 |
| NFLX | 11 | 1.60% | 1.61% | 0 |
| NVDA | 11 | 1.13% | 1.16% | 1 |
| ORCL | 11 | 2.56% | 2.48% | 1 |
| PEP | 11 | 0.98% | 0.98% | 0 |
| PG | 11 | 1.17% | 1.17% | 0 |
| TSLA | 11 | 1.55% | 1.55% | 0 |
| UNH | 11 | 1.11% | 1.11% | 0 |
| V | 11 | 0.92% | 0.92% | 0 |
| WMT | 11 | 1.31% | 1.32% | 1 |
| XOM | 11 | 0.86% | 0.86% | 0 |

_Research experiment, not financial advice._
