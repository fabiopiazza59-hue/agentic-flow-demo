# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 17 of 25 fails are wrong-direction (68%), only 8 pure magnitude. The system can't call the sign, not just the size. This is the core problem.
- **We lose to the naive baseline constantly**: of the 25 fails, the vast majority also have `beats_baseline=false`. On many days we'd do better predicting "no change."
- **Big-move blindness**: on high-APE days (0.03–0.13) the model consistently under-reacts and picks the wrong side — 2026-06-22, 06-25, 07-15, 07-30, 07-31. It smooths through regime shifts.
- **No overconfident misses flagged (0), but calibration is inverted**: confidence is uniformly low (0.1–0.55) and barely tracks outcomes. Some passes came at conf 0.1 (07-27, 08-03) while misses sit at 0.5+ (06-30, 07-13, 07-20). Confidence carries almost no signal.

## Unreliable under these conditions
- **Macro-led days**: hit rate 0.132, worst MAPE (0.020). When `winning_strategy=macro` we usually miss (07-13, 07-15, 07-30). Macro should not lead.
- **Technical-led days**: hit rate 0.158 — nearly as bad. Heavy technical/momentum weighting (07-08, 06-17) precedes misses.
- **High-volatility / large-gap sessions**: all worst APEs cluster on trend days the model treats as mean-reverting.
- **Contrarian in trending tape**: contrarian "wins" often on days we still fail directionally — it flips us onto the wrong side during momentum runs.
- **News is the only relative bright spot** (hit rate 0.289, lowest MAPE) but still <30% — nothing is actually reliable.

## Fixes to try next
- Cut macro and technical weight hints; they're the worst performers. Lean the ensemble toward news.
- Add an explicit trend/regime filter so contrarian is suppressed when momentum is strong — stop fighting large moves.
- Recalibrate confidence against realized hit rate; current values are noise. Suppress trades when calibrated confidence is low.
- Build a volatility-aware magnitude scaler so big-move days aren't smoothed toward zero.
- Gate against baseline: if the ensemble can't beat "no change," default to baseline.