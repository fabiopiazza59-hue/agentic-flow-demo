# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure**: 22 of 31 misses are directional (71%). The model sizes moves roughly right but bets the wrong way. APE on fails averages 2.62% — big for daily AMZN.
- **Directional coin-flip overall**: hit rate is barely above 50% and no strategy clears 25%. This is a sign-prediction problem, not a calibration one.
- **Baseline-beating is weak**: a large share of fails also fail to beat baseline — the ensemble is adding noise, not edge, on hard days.
- **Confidence is mostly honest but flat**: only 1 overconfident miss flagged, yet the highest-confidence days (0.58, 0.62, 0.54, 0.55) are disproportionately misses. Confidence signal is nearly useless — high conf ≠ better outcome.

## Unreliable under these conditions
- **macro is the worst regime pick**: 13.5% hit rate, highest MAPE. When macro "wins" the ensemble (06-12, 07-13, 07-15, 07-30, 08-19), it repeatedly misses — often on high confidence (0.55, 0.62). Macro-driven days are red flags.
- **High-volatility days blow up**: the 07-31 miss (13.2% APE) plus clustered 3-5% APE fails (06-22, 07-23, 08-19) show the model has no handle on gap/large-move days.
- **technical wins credited but low reliability**: 17% hit rate despite second-highest weight_hint — overweighted relative to performance.
- **news-heavy weighting on fail days**: many misses carry news weight 0.30-0.42; heavy news tilt correlates with directional misses (07-23, 07-24, 08-10).

## Fixes to try next
- **Cap or gate macro**: down-weight macro when it's the lead strategy; it's actively harmful.
- **Rebuild confidence calibration** — current scores don't separate wins from losses; consider abstaining above a confidence threshold that historically underperforms.
- **Attack the sign problem directly**: add a regime filter (trend vs chop) and a volatility flag to widen/suppress predictions on gap days.
- **Rebalance weights toward hit rate, not just MAPE**: trim technical, test news/contrarian as leads only in low-vol regimes.
- **Flag large-move days for no-trade** rather than forcing a directional call.