# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (10 sessions, 299 name-forecasts; mean daily APE gain -0.002%, CS [-0.017%, +0.014%]).

**Pending (2026-10-12):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.017%, +0.014%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.002% | [-0.013%, +0.010%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.009 | n=2 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0003 | [-0.004, +0.004] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.022% | [-0.103%, +0.060%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.020% | [-0.106%, +0.066%] | ≈ not distinguishable |

MAPE — LLM 1.24%, free baseline 1.23%, prior close 1.21%; LLM 80% interval coverage 77.29%; names moved per session 0.8. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 5 | -0.000% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 4 | -0.002% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 10 | 1.01% | 1.01% | 0 |
| ADBE | 10 | 1.47% | 1.47% | 0 |
| AMD | 10 | 1.72% | 1.72% | 0 |
| AMZN | 10 | 1.23% | 1.23% | 0 |
| AVGO | 10 | 2.07% | 2.04% | 1 |
| BAC | 10 | 0.93% | 0.93% | 0 |
| CAT | 10 | 1.77% | 1.77% | 0 |
| COST | 10 | 0.64% | 0.62% | 1 |
| CRM | 10 | 1.58% | 1.58% | 0 |
| CVX | 10 | 0.86% | 0.86% | 0 |
| DIS | 10 | 1.06% | 1.07% | 1 |
| GOOGL | 10 | 0.97% | 0.97% | 0 |
| HD | 10 | 1.21% | 1.23% | 1 |
| JNJ | 10 | 1.26% | 1.26% | 0 |
| JPM | 10 | 0.71% | 0.71% | 0 |
| KO | 10 | 0.73% | 0.73% | 0 |
| LLY | 10 | 1.10% | 1.10% | 0 |
| MA | 10 | 1.15% | 1.15% | 0 |
| META | 9 | 1.44% | 1.44% | 0 |
| MSFT | 10 | 0.93% | 0.91% | 1 |
| NFLX | 10 | 1.72% | 1.72% | 0 |
| NVDA | 10 | 1.12% | 1.14% | 1 |
| ORCL | 10 | 2.51% | 2.43% | 1 |
| PEP | 10 | 1.03% | 1.03% | 0 |
| PG | 10 | 1.16% | 1.16% | 0 |
| TSLA | 10 | 1.58% | 1.58% | 0 |
| UNH | 10 | 1.18% | 1.18% | 0 |
| V | 10 | 0.91% | 0.91% | 0 |
| WMT | 10 | 1.28% | 1.30% | 1 |
| XOM | 10 | 0.75% | 0.75% | 0 |

_Research experiment, not financial advice._
