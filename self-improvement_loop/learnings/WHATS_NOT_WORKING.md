# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate** (29 of 49 fails, ~59%). The system isn't just sizing moves wrong — it's getting the *sign* wrong, which is a model-quality problem, not a calibration one.
- **Overall pass rate is a coin flip** (39/88, 44%) and most strategies lose to baseline on fails. Mean fail APE 2.29% means misses are expensive, not marginal.
- **Big-move days blow up the book.** Fails cluster on high-ape days (3–5%+, e.g. 2026-06-22, 07-15, 07-30, and the 13.2% blowup on 07-31). On jumps the model stays anchored near prior close and misses direction entirely.
- **Macro and technical are dead weight** — both at 12.5% hit rate, and macro has the worst MAPE (0.0163). Yet macro keeps getting picked as the "winning" strategy on losing days (07-13, 07-15, 07-30, 08-19) — it wins the ensemble on exactly the days it's wrong.
- **News is the only strategy pulling weight** (30.7% hit, lowest MAPE). Everything else dilutes it.

## Unreliable under these conditions
- **High-volatility / gap days (>2.5% baseline move):** directional accuracy collapses; this is where the worst APEs live.
- **When macro or technical is the winning strategy:** near-guaranteed loss — these drive confident wrong calls.
- **Recent high-confidence regime (Sept–Oct, conf 0.5–0.66):** confidence rose but pass rate didn't — 10-06 through 10-15 are mostly fails at 0.56–0.66 conf. Calibration is drifting: rising confidence on a flat/declining hit rate. The 7 overconfident misses concentrate here (08-19 @0.62, 09-03 @0.62, 09-09 @0.72, 10-08 @0.62).
- **Contrarian on trending days:** frequent sign errors when price keeps running (10-08, 10-15, 09-22).

## Fixes to try next
- **Cut or cap macro and technical weight** (both 12.5% hit); reallocate toward news. Stop letting macro "win" the ensemble.
- **Add a volatility/gap regime filter:** on expected large moves, widen the predicted magnitude and lean on news/momentum sign rather than anchoring to prior close.
- **Recalibrate confidence** — recent 0.6+ predictions are no more accurate than 0.4 ones; flatten or retrain the confidence head against realized hit rate.
- **Fix the directional model first** (59% of fails are sign errors); magnitude tuning is secondary until sign accuracy clears ~55%.
- **Gate contrarian** off when momentum/trend is strong to stop fighting the tape.