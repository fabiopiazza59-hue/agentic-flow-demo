# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (32 sessions, 959 name-forecasts; mean daily APE gain -0.003%, CS [-0.016%, +0.011%]).

**Pending (2026-11-11):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.016%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.000% | [-0.010%, +0.009%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.041 | [-0.288, +0.207] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0012 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.073% | [-0.077%, +0.223%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.076% | [-0.069%, +0.220%] | ≈ not distinguishable |

MAPE — LLM 1.60%, free baseline 1.60%, prior close 1.67%; LLM 80% interval coverage 78.84%; names moved per session 1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 27 | -0.009% | [-0.043%, +0.025%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 26 | -0.003% | [-0.038%, +0.032%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 32 | 1.19% | 1.19% | 1 |
| ADBE | 32 | 1.86% | 1.86% | 0 |
| AMD | 32 | 3.61% | 3.61% | 2 |
| AMZN | 32 | 1.83% | 1.83% | 0 |
| AVGO | 32 | 2.42% | 2.40% | 4 |
| BAC | 32 | 1.08% | 1.10% | 1 |
| CAT | 32 | 2.40% | 2.38% | 1 |
| COST | 32 | 0.99% | 0.99% | 1 |
| CRM | 32 | 1.53% | 1.53% | 0 |
| CVX | 32 | 1.20% | 1.22% | 1 |
| DIS | 32 | 1.07% | 1.07% | 1 |
| GOOGL | 32 | 1.24% | 1.26% | 1 |
| HD | 32 | 1.53% | 1.53% | 1 |
| JNJ | 32 | 1.58% | 1.57% | 1 |
| JPM | 32 | 1.06% | 1.05% | 1 |
| KO | 32 | 0.89% | 0.88% | 1 |
| LLY | 32 | 1.51% | 1.53% | 3 |
| MA | 32 | 1.25% | 1.26% | 1 |
| META | 31 | 1.96% | 1.96% | 0 |
| MSFT | 32 | 1.48% | 1.47% | 2 |
| NFLX | 32 | 1.55% | 1.55% | 0 |
| NVDA | 32 | 1.99% | 1.96% | 3 |
| ORCL | 32 | 3.30% | 3.25% | 2 |
| PEP | 32 | 1.16% | 1.16% | 0 |
| PG | 32 | 1.20% | 1.21% | 1 |
| TSLA | 32 | 2.00% | 2.00% | 0 |
| UNH | 32 | 1.47% | 1.46% | 1 |
| V | 32 | 1.15% | 1.15% | 0 |
| WMT | 32 | 1.26% | 1.27% | 1 |
| XOM | 32 | 1.26% | 1.27% | 1 |

_Research experiment, not financial advice._
