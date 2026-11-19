# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (39 sessions, 1168 name-forecasts; mean daily APE gain -0.002%, CS [-0.012%, +0.009%]).

**Pending (2026-11-20):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.012%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.009%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.019 | [-0.239, +0.200] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0012 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.074% | [-0.061%, +0.210%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.076% | [-0.056%, +0.208%] | ≈ not distinguishable |

MAPE — LLM 1.63%, free baseline 1.63%, prior close 1.70%; LLM 80% interval coverage 78.25%; names moved per session 1.0. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 34 | -0.011% | [-0.040%, +0.018%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 33 | -0.003% | [-0.031%, +0.025%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 39 | 1.13% | 1.12% | 1 |
| ADBE | 39 | 1.92% | 1.92% | 0 |
| AMD | 39 | 3.72% | 3.74% | 3 |
| AMZN | 39 | 1.87% | 1.87% | 0 |
| AVGO | 39 | 2.46% | 2.47% | 5 |
| BAC | 39 | 1.05% | 1.06% | 1 |
| CAT | 39 | 2.46% | 2.44% | 1 |
| COST | 39 | 0.92% | 0.92% | 1 |
| CRM | 39 | 1.77% | 1.77% | 0 |
| CVX | 39 | 1.26% | 1.27% | 1 |
| DIS | 39 | 1.10% | 1.10% | 2 |
| GOOGL | 39 | 1.38% | 1.40% | 2 |
| HD | 39 | 1.32% | 1.33% | 1 |
| JNJ | 39 | 1.50% | 1.49% | 1 |
| JPM | 39 | 1.10% | 1.09% | 1 |
| KO | 39 | 0.87% | 0.86% | 1 |
| LLY | 39 | 1.44% | 1.46% | 3 |
| MA | 39 | 1.22% | 1.23% | 1 |
| META | 37 | 1.92% | 1.92% | 0 |
| MSFT | 39 | 1.47% | 1.46% | 2 |
| NFLX | 39 | 1.60% | 1.60% | 0 |
| NVDA | 39 | 2.30% | 2.24% | 5 |
| ORCL | 39 | 3.35% | 3.31% | 2 |
| PEP | 39 | 1.18% | 1.18% | 0 |
| PG | 39 | 1.13% | 1.14% | 1 |
| TSLA | 39 | 1.99% | 1.99% | 0 |
| UNH | 39 | 1.83% | 1.82% | 1 |
| V | 39 | 1.17% | 1.17% | 0 |
| WMT | 39 | 1.16% | 1.16% | 1 |
| XOM | 39 | 1.32% | 1.33% | 1 |

_Research experiment, not financial advice._
