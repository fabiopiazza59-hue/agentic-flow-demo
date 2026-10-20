# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (16 sessions, 479 name-forecasts; mean daily APE gain -0.008%, CS [-0.026%, +0.011%]).

**Pending (2026-10-20):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.008% | [-0.026%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.004% | [-0.016%, +0.009%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.139 | n=7 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0015 | [-0.006, +0.003] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.027% | [-0.128%, +0.073%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.020% | [-0.115%, +0.075%] | ≈ not distinguishable |

MAPE — LLM 1.38%, free baseline 1.37%, prior close 1.35%; LLM 80% interval coverage 78.72%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 11 | +0.007% | [-0.043%, +0.058%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 10 | +0.011% | [-0.035%, +0.057%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 16 | 0.88% | 0.88% | 0 |
| ADBE | 16 | 1.74% | 1.74% | 0 |
| AMD | 16 | 2.29% | 2.29% | 0 |
| AMZN | 16 | 1.47% | 1.47% | 0 |
| AVGO | 16 | 2.09% | 2.06% | 4 |
| BAC | 16 | 1.01% | 1.04% | 1 |
| CAT | 16 | 1.64% | 1.64% | 0 |
| COST | 16 | 0.84% | 0.83% | 1 |
| CRM | 16 | 1.80% | 1.80% | 0 |
| CVX | 16 | 1.13% | 1.13% | 0 |
| DIS | 16 | 0.90% | 0.91% | 1 |
| GOOGL | 16 | 1.07% | 1.07% | 0 |
| HD | 16 | 1.16% | 1.17% | 1 |
| JNJ | 16 | 1.53% | 1.50% | 1 |
| JPM | 16 | 0.79% | 0.77% | 1 |
| KO | 16 | 0.87% | 0.87% | 0 |
| LLY | 16 | 1.30% | 1.30% | 0 |
| MA | 16 | 1.04% | 1.04% | 0 |
| META | 15 | 1.79% | 1.79% | 0 |
| MSFT | 16 | 1.03% | 1.02% | 1 |
| NFLX | 16 | 1.50% | 1.50% | 0 |
| NVDA | 16 | 1.38% | 1.32% | 3 |
| ORCL | 16 | 3.34% | 3.24% | 2 |
| PEP | 16 | 1.09% | 1.09% | 0 |
| PG | 16 | 1.14% | 1.14% | 0 |
| TSLA | 16 | 1.64% | 1.64% | 0 |
| UNH | 16 | 1.48% | 1.46% | 1 |
| V | 16 | 1.00% | 1.00% | 0 |
| WMT | 16 | 1.27% | 1.28% | 1 |
| XOM | 16 | 1.08% | 1.08% | 0 |

_Research experiment, not financial advice._
