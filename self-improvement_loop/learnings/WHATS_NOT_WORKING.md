# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 22 of 31 fails are wrong-direction (71%), not magnitude. The model can't call the sign, which no amount of magnitude tuning fixes.
- Overall it's a coin flip at best — no single strategy clears a 26% hit rate. This is a directional prediction system that doesn't predict direction.
- Fails cluster on big-move days: worst APEs (0.13, 0.051, 0.042, 0.039, 0.037, 0.034) are all directional misses. The model gets steamrolled on volatile/gap sessions and never beats baseline meaningfully — it just tracks baseline.
- **Macro is the worst offender**: 11.8% hit rate, highest MAPE. Every macro-winning day here is either a fail or a directional miss (06-12, 07-13, 07-15, 07-30, 08-19). It should not be winning weight.

## Unreliable under these conditions
- **High-volatility / large-move days** (baseline_ape > ~0.02): near-universal directional misses regardless of strategy.
- **When macro takes the lead**: consistently loses. Same for momentum on trend-reversal days (06-12, 06-25, 07-08, 07-30).
- **Confidence is mildly miscalibrated on the upside**: the higher-confidence fails cluster at 0.5–0.62 (06-30, 07-13, 08-07, 08-19 @0.62, 08-20 @0.54). When the model is "sure," it's often wrong on direction — 08-19 and 08-20 are back-to-back confident directional misses.
- **Low confidence is not informative either**: passes and fails both scatter across 0.1–0.3, so confidence carries almost no signal.

## Fixes to try next
- Cap or drop **macro** weight; it's pure noise here. Redistribute toward news/technical (best MAPE), but don't expect miracles — all are weak.
- Build a **volatility regime filter**: on high-expected-move days, widen intervals or abstain rather than emit a confident point/direction.
- **Recalibrate confidence** against realized directional hits — current high-confidence outputs are anti-predictive; consider inverting or flattening above 0.5.
- Attack the sign problem directly: add a dedicated gap/overnight-drift feature and test a directional-only classifier separate from magnitude.