# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not size, is the core failure.** 14 of 17 fails are directional misses (82%); only 3 are magnitude. The model consistently guesses the wrong side, not just the wrong distance.
- **We lose to baseline on the misses.** Almost every failed day has `beats_baseline: false` and fail-day APE (2.38% mean) running above baseline — the ensemble is adding noise on hard days, not value.
- **Big-move days destroy us.** The worst APEs (5.1% on 06-22, 3.7% on 06-17, 3.4% on 06-15, 3.5% on 06-25, 3.0% on 07-15) are all directional misses. When AMZN moves hard, we're on the wrong side.
- **Overall hit rate is weak.** No strategy clears 30% directional hit rate; macro is a coin-flip-losing 11%.

## Unreliable under these conditions
- **`macro` as winning strategy = red flag.** 3 wins / 27, 11% hit rate, worst MAPE (0.018). Every day macro "won" (06-12 region, 07-13, 07-15) it missed direction badly. It should not be steering.
- **`momentum`-led calls in choppy/reversal tape.** momentum hit rate 22%, MAPE 0.017; led several large misses (06-12, 06-25). It chases and gets whipsawed.
- **`news`-led high-volatility days.** news has the best relative record (30%) but still fronted the 5.1% 06-22 and 3.5% 06-25 disasters — it overreacts to headline days.
- **Confidence is flat, not calibrated.** 0 overconfident misses only because confidence never rises — everything sits 0.40–0.58 regardless of outcome. Confidence carries zero information; it's not discriminating good days from bad.
- **Low-signal near-equal weight days** (e.g. 06-15, 06-22 ~0.2 each) still miss — blending everything equally just averages into baseline error.

## Fixes to try next
- **Down-weight or gate macro** (and de-emphasize momentum) when volatility is elevated; let news/technical lead only on quiet days.
- **Add a directional confidence gate**: when strategies disagree on sign, shrink toward the naive/last-close baseline instead of committing to a side.
- **Recalibrate confidence** so it actually spreads (isotonic/Platt on historical hits); flat 0.4–0.5 is useless for sizing.
- **Build a volatility/regime detector**; suppress large predicted moves unless multiple strategies agree on direction.
- **Track sign-accuracy per strategy per regime**, not just MAPE — our failure is directional, so optimize for that.