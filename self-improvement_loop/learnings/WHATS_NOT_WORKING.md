# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- 55% fail rate (52/94) with directional misses (30) outnumbering magnitude misses (22) — the system is getting the *direction* wrong more often than it's just mis-sizing. Getting the sign right is the bigger problem.
- Big-move days are where it bleeds: APE spikes to 3–13% (2026-06-22, 07-23, 07-31, 10-19, 10-22) consistently coincide with fails. The model reverts to small moves and gets run over on gaps/volatility.
- On many fails the model barely differs from baseline (fail APE ≈ baseline APE, e.g. 06-15, 06-29, 09-11). It's adding no edge on those days — it's just tracking the random-walk and losing anyway.
- `macro` (11% hit, 0.167 MAPE) and `technical` (13% hit) are the worst-performing winning-strategy labels yet still get picked as the "winner" repeatedly — selection is routing to weak strategies.

## Unreliable under these conditions
- **High-volatility / gap days:** every APE>3% event is a fail; magnitude estimates are systematically too timid.
- **When `macro` or `technical` is the chosen strategy:** combined ~12% hit rate. macro-led fails (07-13, 07-15, 07-30, 08-19) are large-APE misses.
- **Rising confidence, flat accuracy:** recent window (conf 0.53–0.66, Oct) is still ~50% failing — 7 overconfident misses. Confidence climbed without accuracy following; calibration is drifting upward falsely. 08-31/09-09 fired conf 0.72 and failed.
- **contrarian** carries many directional *hits* but still fails on magnitude (08-04, 10-22 directional-right, huge APE) — right call, wrong size.

## Fixes to try next
- Demote/penalize `macro` and `technical` in selection; lean toward `news` (32% hit, lowest MAPE) and `contrarian` for direction.
- Add a volatility regime gate: widen predicted move magnitude when recent realized vol / gap risk is elevated instead of mean-reverting to small moves.
- Recalibrate confidence against realized hit-rate — current high-conf band (>0.6) is not earning its confidence; cap or shrink it until it predicts pass rate.
- Separate direction and magnitude models; directional misses dominate, so a dedicated sign classifier may beat the blended point estimate.
- Flag days where pred≈baseline and either abstain or defer to baseline rather than adding noise.