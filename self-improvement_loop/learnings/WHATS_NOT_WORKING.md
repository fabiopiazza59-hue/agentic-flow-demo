# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate** (29 of 50 fails). The model gets the *direction* wrong more than it gets the magnitude wrong — this is a sign/trend problem, not just a sizing problem.
- **Baseline is often better.** A large share of fails have ape > baseline_ape (e.g. 06-12, 06-25, 06-26, 09-22, 10-15). The ensemble frequently adds noise over a naive carry-forward.
- **Big tail events wreck the mean.** 2026-07-31 (ape 13.2%) and a cluster of 3–5% misses (06-22, 06-25, 07-15, 08-19, 07-30) drive the 2.26% mean fail APE. These are gap/regime days the model has no answer for.
- **Confidence is weakly calibrated.** 7 overconfident misses, and recent high-confidence days (08-19 @0.62, 09-03 @0.62, 10-08 @0.62, 10-15 @0.57) still fail. Rising confidence (0.52→0.66 in Oct) is NOT matched by rising accuracy.

## Unreliable under these conditions
- **High-volatility / large-move days:** every >3% move day is a miss. The model reverts toward small moves and gets run over.
- **`macro` and `technical` as winning strategy:** hit rates 12.4% and 13.5% — barely better than noise. When macro wins the day it usually fails (07-13, 07-15, 08-19, 07-30).
- **`contrarian` in trending stretches:** 24.7% overall but clusters of consecutive fails in Oct (10-08, 10-12, 10-14, 10-15) show it fights persistent trends.
- **Recent regime (late Sep–Oct):** fail rate elevated despite highest confidence of the whole sample — calibration drift.

## Fixes to try next
- **Re-weight toward `news` (30% hit, lowest MAPE) and `contrarian`; cut `macro`/`technical` weight** given sub-14% hit rates.
- **Add a volatility/gap detector:** on high-expected-move days, widen the predicted magnitude or defer to baseline instead of mean-reverting.
- **Recalibrate confidence** against realized hit rate; current scores are inflated and uninformative — cap confidence until calibration is proven.
- **Fix the directional model first** (sign is wrong 58% of fails) — magnitude tuning is secondary.
- **Track beats_baseline as a gate:** if the ensemble can't consistently beat carry-forward, fall back to baseline on low-signal days.