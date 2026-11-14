# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (35 sessions, 1048 name-forecasts; mean daily APE gain -0.003%, CS [-0.014%, +0.009%]).

**Pending (2026-11-13):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.014%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.011%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.035 | [-0.264, +0.195] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0014 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.076% | [-0.062%, +0.214%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.079% | [-0.054%, +0.212%] | ≈ not distinguishable |

MAPE — LLM 1.58%, free baseline 1.58%, prior close 1.66%; LLM 80% interval coverage 79.39%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 30 | -0.008% | [-0.039%, +0.023%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 29 | -0.002% | [-0.034%, +0.030%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 35 | 1.17% | 1.16% | 1 |
| ADBE | 35 | 1.77% | 1.77% | 0 |
| AMD | 35 | 3.72% | 3.75% | 3 |
| AMZN | 35 | 1.75% | 1.75% | 0 |
| AVGO | 35 | 2.48% | 2.48% | 5 |
| BAC | 35 | 1.04% | 1.05% | 1 |
| CAT | 35 | 2.34% | 2.32% | 1 |
| COST | 35 | 0.95% | 0.95% | 1 |
| CRM | 35 | 1.47% | 1.47% | 0 |
| CVX | 35 | 1.21% | 1.22% | 1 |
| DIS | 35 | 1.05% | 1.04% | 2 |
| GOOGL | 35 | 1.26% | 1.28% | 2 |
| HD | 35 | 1.42% | 1.43% | 1 |
| JNJ | 35 | 1.49% | 1.48% | 1 |
| JPM | 35 | 0.99% | 0.98% | 1 |
| KO | 35 | 0.90% | 0.89% | 1 |
| LLY | 35 | 1.46% | 1.48% | 3 |
| MA | 35 | 1.22% | 1.22% | 1 |
| META | 33 | 1.93% | 1.93% | 0 |
| MSFT | 35 | 1.46% | 1.45% | 2 |
| NFLX | 35 | 1.58% | 1.58% | 0 |
| NVDA | 35 | 2.14% | 2.06% | 4 |
| ORCL | 35 | 3.30% | 3.26% | 2 |
| PEP | 35 | 1.16% | 1.16% | 0 |
| PG | 35 | 1.14% | 1.15% | 1 |
| TSLA | 35 | 2.04% | 2.04% | 0 |
| UNH | 35 | 1.40% | 1.39% | 1 |
| V | 35 | 1.14% | 1.14% | 0 |
| WMT | 35 | 1.22% | 1.22% | 1 |
| XOM | 35 | 1.24% | 1.25% | 1 |

_Research experiment, not financial advice._
