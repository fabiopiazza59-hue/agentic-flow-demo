# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem**: 23 of 34 fails are directional misses (68%). We're not just off on magnitude — we're picking the wrong sign. When we're wrong, we're wrong about which way AMZN moves.
- **We barely beat the naive baseline**: dozens of fails also have `beats_baseline=false`, meaning the ensemble adds nothing over a persistence/last-close guess on those days. Many "passes" (e.g. 06-16, 07-14, 08-06) beat nothing.
- **Fails cluster on high-volatility days**: mean fail APE 2.55%, with tail blowups (07-31 at 13.2%, 06-22 at 5.1%, 07-23 at 4.2%). The model tracks calm drift fine, then gets run over on big-move days.
- **All strategies are weak; none is trustworthy.** Best hit rate is news at 27%, worst is macro at 12%. A 20% overall directional edge is near coin-flip.

## Unreliable under these conditions
- **Macro-led days are the worst**: 12% hit rate, highest MAPE (1.70%), and repeatedly the winning_strategy on big fails (07-13, 07-15, 08-19, 08-30). Macro leads exactly when it shouldn't.
- **Large-gap / news-shock days**: news wins the label on the biggest miss (07-31) yet still misses direction — news signal fires but points wrong. Contrarian also misfires on trending selloffs (06-26, 08-11, 08-20).
- **Rising-confidence regime (late Aug onward)**: confidence crept up to 0.5–0.72 while still failing (08-19 c0.62, 08-31 c0.72, 09-03 c0.62). Overconfident-miss count (3) is understated — calibration is drifting worse recently, not better.
- **Momentum/technical whipsaw in chop**: both lead many small-APE fails where price reversed intraday.

## Fixes to try next
- **Add a volatility gate**: when expected/realized vol is high, widen intervals, down-weight point bets, and flag "low-confidence regime" instead of guessing direction.
- **Cut macro weight hard** (or restrict it to confirmed macro-event days); its edge is negative. Redistribute to news/technical.
- **Recalibrate confidence** against the last ~20 days — current high-confidence outputs are not earning it; cap confidence until directional hit rate on high-conf days exceeds baseline.
- **Track a sign-only meta-model**: since magnitude is decent but direction fails, train a dedicated up/down classifier and let the ensemble set size only.
- **Require beats_baseline in backtest scoring** so "passes" that merely echo last close stop counting as wins.