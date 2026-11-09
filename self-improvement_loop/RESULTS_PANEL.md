# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (31 sessions, 929 name-forecasts; mean daily APE gain -0.003%, CS [-0.016%, +0.011%]).

**Pending (2026-11-10):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.016%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.000% | [-0.010%, +0.010%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.041 | [-0.288, +0.207] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0013 | [-0.004, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.078% | [-0.077%, +0.232%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.080% | [-0.068%, +0.229%] | ≈ not distinguishable |

MAPE — LLM 1.62%, free baseline 1.61%, prior close 1.69%; LLM 80% interval coverage 78.37%; names moved per session 1.0. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 26 | -0.010% | [-0.045%, +0.026%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 25 | -0.003% | [-0.040%, +0.033%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 31 | 1.22% | 1.21% | 1 |
| ADBE | 31 | 1.91% | 1.91% | 0 |
| AMD | 31 | 3.70% | 3.70% | 2 |
| AMZN | 31 | 1.84% | 1.84% | 0 |
| AVGO | 31 | 2.37% | 2.35% | 4 |
| BAC | 31 | 1.11% | 1.12% | 1 |
| CAT | 31 | 2.40% | 2.37% | 1 |
| COST | 31 | 1.00% | 1.00% | 1 |
| CRM | 31 | 1.56% | 1.56% | 0 |
| CVX | 31 | 1.23% | 1.25% | 1 |
| DIS | 31 | 1.09% | 1.09% | 1 |
| GOOGL | 31 | 1.25% | 1.27% | 1 |
| HD | 31 | 1.56% | 1.57% | 1 |
| JNJ | 31 | 1.61% | 1.60% | 1 |
| JPM | 31 | 1.09% | 1.08% | 1 |
| KO | 31 | 0.89% | 0.87% | 1 |
| LLY | 31 | 1.51% | 1.53% | 3 |
| MA | 31 | 1.27% | 1.28% | 1 |
| META | 30 | 2.02% | 2.02% | 0 |
| MSFT | 31 | 1.46% | 1.45% | 2 |
| NFLX | 31 | 1.57% | 1.57% | 0 |
| NVDA | 31 | 1.84% | 1.80% | 3 |
| ORCL | 31 | 3.35% | 3.30% | 2 |
| PEP | 31 | 1.15% | 1.15% | 0 |
| PG | 31 | 1.23% | 1.24% | 1 |
| TSLA | 31 | 2.04% | 2.04% | 0 |
| UNH | 31 | 1.45% | 1.44% | 1 |
| V | 31 | 1.18% | 1.18% | 0 |
| WMT | 31 | 1.30% | 1.31% | 1 |
| XOM | 31 | 1.30% | 1.31% | 1 |

_Research experiment, not financial advice._
