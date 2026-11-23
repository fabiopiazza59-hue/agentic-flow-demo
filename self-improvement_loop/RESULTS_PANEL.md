# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (41 sessions, 1228 name-forecasts; mean daily APE gain -0.002%, CS [-0.012%, +0.008%]).

**Pending (2026-11-24):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.012%, +0.008%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.009%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.019 | [-0.239, +0.200] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0011 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.101% | [-0.042%, +0.243%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.102% | [-0.037%, +0.242%] | ≈ not distinguishable |

MAPE — LLM 1.70%, free baseline 1.70%, prior close 1.80%; LLM 80% interval coverage 78.33%; names moved per session 0.9. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 36 | -0.010% | [-0.037%, +0.017%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 35 | -0.003% | [-0.029%, +0.023%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 41 | 1.16% | 1.16% | 1 |
| ADBE | 41 | 1.90% | 1.90% | 0 |
| AMD | 41 | 3.89% | 3.91% | 3 |
| AMZN | 41 | 1.91% | 1.91% | 0 |
| AVGO | 41 | 2.68% | 2.68% | 5 |
| BAC | 41 | 1.12% | 1.13% | 1 |
| CAT | 41 | 2.57% | 2.55% | 1 |
| COST | 41 | 0.91% | 0.91% | 1 |
| CRM | 41 | 1.72% | 1.72% | 0 |
| CVX | 41 | 1.26% | 1.27% | 1 |
| DIS | 41 | 1.14% | 1.13% | 2 |
| GOOGL | 41 | 1.49% | 1.51% | 2 |
| HD | 41 | 1.32% | 1.33% | 1 |
| JNJ | 41 | 1.46% | 1.45% | 1 |
| JPM | 41 | 1.16% | 1.15% | 1 |
| KO | 41 | 0.87% | 0.86% | 1 |
| LLY | 41 | 1.40% | 1.42% | 3 |
| MA | 41 | 1.21% | 1.21% | 1 |
| META | 39 | 2.08% | 2.08% | 0 |
| MSFT | 41 | 1.46% | 1.45% | 2 |
| NFLX | 41 | 1.64% | 1.64% | 0 |
| NVDA | 41 | 3.07% | 3.02% | 5 |
| ORCL | 41 | 3.61% | 3.57% | 2 |
| PEP | 41 | 1.15% | 1.15% | 0 |
| PG | 41 | 1.10% | 1.11% | 1 |
| TSLA | 41 | 2.29% | 2.29% | 0 |
| UNH | 41 | 1.83% | 1.82% | 1 |
| V | 41 | 1.14% | 1.14% | 0 |
| WMT | 41 | 1.16% | 1.16% | 1 |
| XOM | 41 | 1.32% | 1.34% | 1 |

_Research experiment, not financial advice._
