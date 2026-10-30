# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the primary killer**: 30 of 54 fails are directional misses. When the sign is wrong, the model is consistently wrong — these aren't near-misses.
- **The system loses to the naive baseline on roughly half of failures** (beats_baseline=false on most fails), meaning on bad days we'd be better off predicting "no change."
- **Large-move days are systematically missed.** Nearly every APE > 3% (06-22, 07-23, 07-31, 10-19, 10-22, 10-29) is a failure. The ensemble is calibrated for quiet drift and gets run over on gap/volatility days.
- **Error clustering**: late June (06-12 → 06-29) and mid-Oct (10-06 → 10-19) are sustained losing streaks, indicating regime changes the model doesn't detect or adapt to.

## Unreliable under these conditions
- **macro** and **technical** are the weakest strategies (hit rates 11% and 12%, highest MAPE), yet still carry ~20% weight. macro-won days (06-12, 07-13, 07-15, 07-30, 08-19) are almost all large-error fails.
- **High-volatility / large-gap sessions**: all five strategies collapse together; no strategy provides a hedge when moves exceed ~2.5%.
- **Confidence is mildly miscalibrated on the upside**: 7 overconfident misses, concentrated in the later high-confidence era (08-19 @0.62, 09-03 @0.62, 10-08 @0.62, 10-19 @0.56, 10-29). Rising confidence (0.5–0.66) did NOT coincide with improved hit rate on fails.
- **contrarian** generates big one-sided errors on trending days (10-15, 10-19) even though it's a decent performer overall — it fights momentum at the worst time.

## Fixes to try next
- Add a **volatility/gap regime filter**: when expected range > ~2%, down-weight the ensemble toward baseline or widen the predicted band rather than committing to a point/direction.
- **Cut macro and technical weights** (or gate them to confirmation-only roles); lean harder on **news** (34% hit, lowest MAPE) and contrarian in calm regimes.
- **Recalibrate confidence** against realized hit rate — current 0.5–0.66 band shows no edge; clamp confidence down on large-predicted-move days.
- **Detect regime shifts** (rolling miss streak ≥3) and trigger reweighting rather than holding static ~0.2 blends.
- Separately track **directional accuracy** as a gating metric; suppress trades/calls when directional confidence is low even if magnitude looks tight.