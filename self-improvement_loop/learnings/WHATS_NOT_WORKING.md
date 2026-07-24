# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure.** 15 of 19 misses are directional (79%); only 4 are magnitude. The model is a coin-flip on up/down. Overall directional hit rate is ~52% (15/29) — basically random.
- **We routinely lose to a naive baseline on misses.** Most failing days also have `beats_baseline=false`, meaning on the days we're wrong we're worse than doing nothing.
- **Big-error days cluster and repeat.** APE >3% shows up again and again (06-15, 06-17, 06-22, 06-25, 06-26, 06-29, 07-15, 07-23) — these are gap/volatility days the model can't handle, always missing direction.
- **No overconfidence flag firing, but confidence is uninformative.** Confidence sits in a dead 0.40–0.55 band on both wins and losses. It doesn't separate good from bad predictions — it's miscalibrated by being flat, not by being high on misses.

## Unreliable under these conditions
- **`macro` is the worst strategy** (10% hit rate, highest MAPE 0.0188) yet still carries ~0.177 weight. Every day it "wins" the ensemble (06-12, 07-13, 07-15) it fails.
- **`momentum` and `contrarian` both underperform** (~17–21% hit rate). Momentum has the highest sum-APE. These fire during choppy/mean-reverting stretches and get whipsawed.
- **High-volatility / large-move days** (APE >3%): direction misses are near-universal regardless of which strategy wins.
- **`news` is the only reliable strategy** (34% hit rate, lowest MAPE) — still weak in absolute terms, but the rest are dragging it down via near-equal weighting.

## Fixes to try next
- **Cut or gate `macro`** (and demote momentum/contrarian); lean weight toward `news`. Equal-ish 0.18–0.22 weighting is diluting the one signal that works.
- **Add a volatility regime filter**: when expected daily range is high, widen intervals or abstain rather than forcing a directional call.
- **Recalibrate confidence** against realized hit rate — current outputs carry no information; a flat 0.4 is useless for sizing.
- **Track a direction-only accuracy metric** as the primary KPI, not APE, since magnitude is already decent when direction is right.