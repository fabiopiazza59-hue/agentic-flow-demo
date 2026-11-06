# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (29 sessions, 869 name-forecasts; mean daily APE gain -0.003%, CS [-0.017%, +0.012%]).

**Pending (2026-11-06):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.017%, +0.012%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.000% | [-0.011%, +0.011%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.041 | [-0.288, +0.207] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0013 | [-0.004, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.068% | [-0.093%, +0.229%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.071% | [-0.084%, +0.226%] | ≈ not distinguishable |

MAPE — LLM 1.60%, free baseline 1.60%, prior close 1.67%; LLM 80% interval coverage 77.57%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 24 | -0.008% | [-0.046%, +0.030%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 23 | -0.004% | [-0.043%, +0.036%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 29 | 1.21% | 1.21% | 1 |
| ADBE | 29 | 1.95% | 1.95% | 0 |
| AMD | 29 | 3.30% | 3.31% | 2 |
| AMZN | 29 | 1.85% | 1.85% | 0 |
| AVGO | 29 | 2.23% | 2.21% | 4 |
| BAC | 29 | 1.08% | 1.09% | 1 |
| CAT | 29 | 2.33% | 2.30% | 1 |
| COST | 29 | 1.02% | 1.01% | 1 |
| CRM | 29 | 1.56% | 1.56% | 0 |
| CVX | 29 | 1.25% | 1.26% | 1 |
| DIS | 29 | 1.09% | 1.09% | 1 |
| GOOGL | 29 | 1.28% | 1.29% | 1 |
| HD | 29 | 1.53% | 1.54% | 1 |
| JNJ | 29 | 1.58% | 1.57% | 1 |
| JPM | 29 | 1.06% | 1.05% | 1 |
| KO | 29 | 0.93% | 0.92% | 1 |
| LLY | 29 | 1.45% | 1.47% | 3 |
| MA | 29 | 1.33% | 1.34% | 1 |
| META | 28 | 1.98% | 1.98% | 0 |
| MSFT | 29 | 1.46% | 1.45% | 2 |
| NFLX | 29 | 1.58% | 1.58% | 0 |
| NVDA | 29 | 1.72% | 1.68% | 3 |
| ORCL | 29 | 3.47% | 3.42% | 2 |
| PEP | 29 | 1.22% | 1.22% | 0 |
| PG | 29 | 1.23% | 1.24% | 1 |
| TSLA | 29 | 1.89% | 1.89% | 0 |
| UNH | 29 | 1.50% | 1.49% | 1 |
| V | 29 | 1.25% | 1.25% | 0 |
| WMT | 29 | 1.34% | 1.35% | 1 |
| XOM | 29 | 1.35% | 1.37% | 1 |

_Research experiment, not financial advice._
