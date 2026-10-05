# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (6 sessions, 180 name-forecasts; mean daily APE gain 0.002%, CS n=6 sessions (too few)).

**Pending (2026-10-06):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.002% | n=6 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.001% | n=6 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0012 | n=6 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.042% | n=6 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.045% | n=6 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.17%, free baseline 1.18%, prior close 1.13%; LLM 80% interval coverage 81.11%; names moved per session 0.7. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 1 | +0.018% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 0 | — | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 6 | 1.13% | 1.13% | 0 |
| ADBE | 6 | 1.36% | 1.36% | 0 |
| AMD | 6 | 1.30% | 1.30% | 0 |
| AMZN | 6 | 0.71% | 0.71% | 0 |
| AVGO | 6 | 2.00% | 2.00% | 0 |
| BAC | 6 | 1.09% | 1.09% | 0 |
| CAT | 6 | 1.16% | 1.16% | 0 |
| COST | 6 | 0.52% | 0.50% | 1 |
| CRM | 6 | 1.93% | 1.93% | 0 |
| CVX | 6 | 0.64% | 0.64% | 0 |
| DIS | 6 | 1.07% | 1.09% | 1 |
| GOOGL | 6 | 1.17% | 1.17% | 0 |
| HD | 6 | 0.84% | 0.88% | 1 |
| JNJ | 6 | 1.36% | 1.36% | 0 |
| JPM | 6 | 0.81% | 0.81% | 0 |
| KO | 6 | 0.62% | 0.62% | 0 |
| LLY | 6 | 0.78% | 0.78% | 0 |
| MA | 6 | 1.30% | 1.30% | 0 |
| META | 6 | 2.02% | 2.02% | 0 |
| MSFT | 6 | 0.79% | 0.79% | 0 |
| NFLX | 6 | 1.81% | 1.81% | 0 |
| NVDA | 6 | 1.04% | 1.04% | 0 |
| ORCL | 6 | 2.00% | 2.00% | 0 |
| PEP | 6 | 0.56% | 0.56% | 0 |
| PG | 6 | 1.23% | 1.23% | 0 |
| TSLA | 6 | 2.03% | 2.03% | 0 |
| UNH | 6 | 1.21% | 1.21% | 0 |
| V | 6 | 0.87% | 0.87% | 0 |
| WMT | 6 | 1.16% | 1.19% | 1 |
| XOM | 6 | 0.69% | 0.69% | 0 |

_Research experiment, not financial advice._
