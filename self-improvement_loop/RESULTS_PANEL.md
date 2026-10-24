# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (20 sessions, 599 name-forecasts; mean daily APE gain -0.006%, CS [-0.021%, +0.009%]).

**Pending (2026-10-26):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.006% | [-0.021%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.003% | [-0.014%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.108 | n=9 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0015 | [-0.005, +0.002] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.011% | [-0.103%, +0.124%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.017% | [-0.093%, +0.128%] | ≈ not distinguishable |

MAPE — LLM 1.40%, free baseline 1.40%, prior close 1.42%; LLM 80% interval coverage 77.31%; names moved per session 1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 15 | -0.000% | [-0.043%, +0.042%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 14 | +0.002% | [-0.038%, +0.042%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 20 | 0.92% | 0.92% | 0 |
| ADBE | 20 | 2.02% | 2.02% | 0 |
| AMD | 20 | 2.32% | 2.32% | 0 |
| AMZN | 20 | 1.48% | 1.48% | 0 |
| AVGO | 20 | 1.90% | 1.87% | 4 |
| BAC | 20 | 0.94% | 0.96% | 1 |
| CAT | 20 | 1.70% | 1.70% | 0 |
| COST | 20 | 0.87% | 0.87% | 1 |
| CRM | 20 | 1.66% | 1.66% | 0 |
| CVX | 20 | 1.11% | 1.11% | 0 |
| DIS | 20 | 0.92% | 0.92% | 1 |
| GOOGL | 20 | 1.21% | 1.21% | 0 |
| HD | 20 | 1.34% | 1.35% | 1 |
| JNJ | 20 | 1.61% | 1.59% | 1 |
| JPM | 20 | 0.76% | 0.74% | 1 |
| KO | 20 | 0.99% | 0.99% | 0 |
| LLY | 20 | 1.56% | 1.53% | 1 |
| MA | 20 | 1.06% | 1.06% | 0 |
| META | 19 | 2.06% | 2.06% | 0 |
| MSFT | 20 | 0.98% | 0.97% | 1 |
| NFLX | 20 | 1.53% | 1.53% | 0 |
| NVDA | 20 | 1.34% | 1.29% | 3 |
| ORCL | 20 | 3.03% | 2.95% | 2 |
| PEP | 20 | 1.19% | 1.19% | 0 |
| PG | 20 | 1.35% | 1.37% | 1 |
| TSLA | 20 | 1.48% | 1.48% | 0 |
| UNH | 20 | 1.51% | 1.49% | 1 |
| V | 20 | 1.03% | 1.03% | 0 |
| WMT | 20 | 1.30% | 1.31% | 1 |
| XOM | 20 | 1.03% | 1.03% | 0 |

_Research experiment, not financial advice._
