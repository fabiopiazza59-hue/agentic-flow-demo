# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure**: 22 of 32 misses (69%) are directional. The model calls the sign wrong far more than it mis-sizes moves. Point-estimate tuning won't fix this.
- **The ensemble is barely better than a coin flip on direction.** Best strategy (news) hits 27%, worst (macro) 12.5%. All five are below 30% hit rate — that's systematically *anti*-predictive on direction, not merely noisy.
- **Baseline beats us on most misses**: on failing days APE (2.61%) routinely exceeds baseline APE, meaning the model adds error vs. naive persistence rather than reducing it.
- **Large-move days are where damage concentrates**: 2026-07-31 (13.2% APE), 06-22 (5.1%), 07-23 (4.2%), 06-17/06-15/06-25/07-30 all ~3-5%. These high-magnitude days dominate total error and are almost all directional misses.

## Unreliable under these conditions
- **Macro-led days are the worst**: winning_strategy=macro (06-12, 07-13, 07-15, 07-30, 08-19, 08-25) is mostly failures, and macro has the lowest hit rate (12.5%). Macro should not be trusted as a tiebreaker.
- **News-heavy weighting (news ≥ 0.3) precedes many big misses** (06-25, 07-01, 07-13, 07-23, 07-24, 08-11) — news gets overweighted right before gap days it can't call.
- **Confidence is mildly miscalibrated on the high end**: only 2 flagged overconfident misses, but 08-19 (0.62), 08-20 (0.54), 08-31 (0.72), 06-30 (0.55) are confident failures. High confidence is NOT tracking accuracy — passes occur at conf 0.1 and fails at 0.6+.
- Errors cluster in **volatile/large-gap regimes**; in calm periods (early August) the model passes with small APE.

## Fixes to try next
- Since directional hit rate is <30% across all strategies, **test inverting the ensemble's directional signal** — it may be systematically contrarian to reality.
- **Down-weight or gate macro** (weight_hint already lowest; consider dropping it as a winning_strategy selector) and cap news weight to reduce pre-gap overreaction.
- **Decouple magnitude from direction**: predict direction with a separately calibrated classifier; use ensemble only for sizing.
- **Recalibrate confidence** against realized hit rate — current confidence carries no discriminative signal; shrink toward 0.5 until proven.
- Add a **volatility/large-move detector** and widen intervals (or abstain) on high-vol days where nearly all catastrophic misses occur.