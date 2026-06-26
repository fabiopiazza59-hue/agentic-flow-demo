# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure.** 5 of 6 misses are directional (only 1 pure magnitude miss). The model gets the size roughly right but bets the wrong way — a sign-prediction problem, not a scaling problem.
- **Misses cluster and grow.** Failures bunch from 06-12 onward, and fail APE escalates (2.0% → 3.4% → 3.7% → 5.1% → 3.6%). Mean fail APE 3.41% vs tiny pass APEs (<0.5%) — when it's wrong, it's badly wrong, and it barely beats baseline on misses (beats_baseline false on 5/6 fails).
- **Momentum is the repeat offender on losses.** It "won" the blend on 3 of 6 fails (06-12, 06-16, 06-25), each a directional miss — momentum is leading the model into trend-chasing right at reversals.

## Unreliable under these conditions
- **Choppy / mean-reverting tape:** every momentum-led day flipped direction — classic late trend-following into a turn.
- **News-led days are a coin flip:** news wins 3 but also drives the two worst misses (06-15, 06-22 at 3.4%/5.1%). It overreacts to headlines without confirming price follow-through.
- **Contrarian and macro are dead weight:** hit rates 0.1 each, highest MAPE (2.33%/2.43%), yet still hold ~18% blend weight. They add noise, not signal.
- **Confidence is uninformative, not overconfident.** 0 overconfident misses, but confidence sits in a flat 0.38–0.58 band on both wins and losses — it carries no discriminating information.

## Fixes to try next
- **Cut momentum, contrarian, and macro weight** (especially momentum near volatility/reversal regimes); reallocate toward technical+news, which lead the few wins.
- **Add a reversal/regime filter** that suppresses momentum when recent returns whipsaw, since that's where directional misses concentrate.
- **Build a dedicated direction classifier** separate from magnitude — magnitude is fine, sign is broken.
- **Recalibrate confidence** against realized hit rate; current scores don't separate wins from losses and shouldn't be trusted for sizing.