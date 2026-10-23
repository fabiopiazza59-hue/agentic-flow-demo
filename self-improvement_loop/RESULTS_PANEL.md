# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (19 sessions, 569 name-forecasts; mean daily APE gain -0.007%, CS [-0.023%, +0.009%]).

**Pending (2026-10-23):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.007% | [-0.023%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.003% | [-0.014%, +0.009%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.108 | n=9 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0016 | [-0.005, +0.002] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.010% | [-0.110%, +0.131%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.017% | [-0.100%, +0.134%] | ≈ not distinguishable |

MAPE — LLM 1.40%, free baseline 1.39%, prior close 1.41%; LLM 80% interval coverage 76.64%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 14 | +0.000% | [-0.046%, +0.046%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 13 | +0.002% | [-0.042%, +0.046%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 19 | 0.95% | 0.95% | 0 |
| ADBE | 19 | 2.01% | 2.01% | 0 |
| AMD | 19 | 2.33% | 2.33% | 0 |
| AMZN | 19 | 1.50% | 1.50% | 0 |
| AVGO | 19 | 1.99% | 1.96% | 4 |
| BAC | 19 | 0.93% | 0.95% | 1 |
| CAT | 19 | 1.66% | 1.66% | 0 |
| COST | 19 | 0.86% | 0.85% | 1 |
| CRM | 19 | 1.70% | 1.70% | 0 |
| CVX | 19 | 1.03% | 1.03% | 0 |
| DIS | 19 | 0.95% | 0.96% | 1 |
| GOOGL | 19 | 1.20% | 1.20% | 0 |
| HD | 19 | 1.29% | 1.30% | 1 |
| JNJ | 19 | 1.61% | 1.59% | 1 |
| JPM | 19 | 0.72% | 0.70% | 1 |
| KO | 19 | 0.88% | 0.88% | 0 |
| LLY | 19 | 1.50% | 1.47% | 1 |
| MA | 19 | 1.02% | 1.02% | 0 |
| META | 18 | 2.15% | 2.15% | 0 |
| MSFT | 19 | 1.01% | 1.00% | 1 |
| NFLX | 19 | 1.52% | 1.52% | 0 |
| NVDA | 19 | 1.38% | 1.32% | 3 |
| ORCL | 19 | 3.15% | 3.06% | 2 |
| PEP | 19 | 1.14% | 1.14% | 0 |
| PG | 19 | 1.27% | 1.28% | 1 |
| TSLA | 19 | 1.55% | 1.55% | 0 |
| UNH | 19 | 1.51% | 1.49% | 1 |
| V | 19 | 1.00% | 1.00% | 0 |
| WMT | 19 | 1.29% | 1.30% | 1 |
| XOM | 19 | 0.96% | 0.96% | 0 |

_Research experiment, not financial advice._
