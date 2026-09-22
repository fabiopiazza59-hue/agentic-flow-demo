# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem**: 25 of 39 fails (64%) are directional misses, not magnitude. The model gets the size roughly right but picks the wrong sign — a coin-flip signal, not a calibration issue.
- **Barely beats the naive baseline**: on failed days mean APE (2.45%) is often ~equal to or worse than baseline_ape. When it misses, it misses no better than doing nothing (e.g. 06-22, 06-25, 07-20, 08-19, 09-17).
- **Fat-tail blowups**: rare huge-move days destroy the score (07-31 APE 13.2%, 06-22 5.1%, 07-23 4.2%). The model never anticipates large gaps — likely earnings/macro-shock days.
- **No strategy is reliable**: best hit rate is `news` at 31%; everything else is 13–23%. The whole ensemble hits well below 50% directionally.

## Unreliable under these conditions
- **`macro` is the worst offender** (12.9% hit, highest MAPE 0.0168) yet still wins the ensemble on several bad days (07-13, 07-15, 08-19, 08-25). It should almost never be the deciding strategy.
- **`technical` and `contrarian`** are also net-negative (16–17% hit) and repeatedly win on losing days.
- **High-volatility / large-move days**: every APE >3% is a directional miss. The system has no regime awareness for gap days.
- **Confidence is weakly calibrated**: overconfident misses are logged as only 4, but real damage is at conf 0.54–0.72 (08-19 0.62, 08-20 0.54, 09-03 0.62, 09-11 0.54, 09-17 0.54 all fail). High confidence does NOT predict success — recent confidence inflation (many 0.5–0.72) isn't earning its keep.

## Fixes to try next
- **Demote/gate `macro` and pure `technical`** as tie-breakers; lean the ensemble toward `news` (only sub-strategy beating baseline consistently).
- **Add a volatility/earnings-day flag**: widen intervals and cut position/confidence on expected large-move days to stop the 4–13% blowups.
- **Attack the sign problem directly**: build a separate directional classifier; current magnitude-first blend is guessing direction.
- **Recalibrate confidence** against realized hit rate — current 0.5–0.7 band has no edge over 0.4; shrink toward baseline until it earns trust.