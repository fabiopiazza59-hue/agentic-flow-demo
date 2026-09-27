# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (0 sessions, 0 name-forecasts; mean daily APE gain —, CS n=0 sessions (too few)).

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | — | n=0 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | — | n=0 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | None | n=0 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | None | n=0 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | — | n=0 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | — | n=0 sessions (too few) | ⏳ too few sessions |

MAPE — LLM —, free baseline —, prior close —; LLM 80% interval coverage —; names moved per session None. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| _no scored sessions yet_ | | | | |

_Research experiment, not financial advice._
