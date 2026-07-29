# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer**: 16 of 20 fails are directional misses (80%). The model gets the size roughly right but bets the wrong way. Magnitude-only misses are just 4.
- **Baseline beats us on losses**: on nearly every fail, `beats_baseline` is false — we're not just wrong, we're worse than a naive carry-forward. The ensemble is adding noise, not signal.
- **No overconfidence flag firing, but confidence is uniformly low/flat** (~0.4). It's not miscalibrated-high; it's uninformative. Confidence barely moves and doesn't separate wins from losses (compare 0.55 fails on 07-13/06-30 vs 0.4 wins). Confidence has no discriminative power.
- **Big-error days cluster**: late-June (06-22 5.1%, 06-25 3.6%, 06-26 3.2%, 06-29 3.3%) and 07-15/07-23 (3–4% APE). These are directional whiffs during trending/volatile stretches — the model fades real moves.

## Unreliable under these conditions
- **macro-led days**: hit rate 9.4%, worst MAPE (1.76%). When macro wins the ensemble it's almost always wrong (06-12, 07-13, 07-15 all fails).
- **contrarian-led days**: 15.6% hit rate — contrarian is systematically fading moves that continue. It's mostly a directional-miss engine.
- **High-volatility / trending regimes** (the 3%+ APE clusters): the model mean-reverts into momentum and gets steamrolled.
- **news-led** is the least-bad (31% hits) but still loses more than it wins, and produces the worst single days when wrong (07-23 4.2%).

## Fixes to try next
- **Cut macro and contrarian weight hard** (both near coin-flip-down); they're dragging the ensemble. Redistribute to news/momentum.
- **Add a trend/volatility regime filter**: when recent move exceeds a threshold, suppress contrarian and stop fading — the late-June cluster shows we fight trends and lose.
- **Rebuild confidence to be directional-conviction based** and calibrate it; current flat ~0.4 output is useless for gating.
- **Since we lose to baseline on fails, add a "defer-to-baseline" gate** when ensemble disagreement is high or conviction low — that alone would recover the sub-baseline losses.