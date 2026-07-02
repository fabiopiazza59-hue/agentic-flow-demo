# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer**: 9 of 10 fails are directional misses (only 1 pure magnitude miss). The model gets the size roughly right but bets the wrong way. This is a sign flip problem, not a calibration-of-error problem.
- **We rarely beat the baseline**: of 14 scored, we beat baseline only ~4 times, and several "passes" (06-16, 06-24) still lost to baseline. Naive persistence is outperforming the ensemble.
- **Losing streak is structural**: the last 5 predictions (06-25 → 07-01) are all fails, all directional misses, all worse than baseline. Something regime-specific broke around late June.
- **Overconfidence is NOT the current issue**: 0 overconfident misses, confidences are clustered 0.38–0.58. The model is honestly unsure — the problem is it's unsure *and* wrong, not falsely bold.

## Unreliable under these conditions
- **News-driven days**: `news` is the most-picked winning strategy on fails (06-15, 06-22, 06-29, 07-01) yet only a 0.357 hit rate. It wins the internal contest then mispredicts direction — it's reactive/lagging on news.
- **Momentum and macro are broken**: `momentum` hit 0.214, `macro` hit 0.071 (1/14). Macro should be near-zero-weighted; it's actively harmful. Momentum whipsaws in choppy tape.
- **Contrarian is the worst by win rate** (0.143) yet gets ~0.21 weight hint — mispriced.
- **Larger absolute moves crush us**: fail APEs (2.7–5.1%) cluster on high-move days; passes are all sub-0.5% quiet days. We only "work" when nothing happens.

## Fixes to try next
- **Add a direction gate**: since magnitude is fine, build a separate up/down classifier and veto trades when strategies disagree on sign.
- **Reweight by hit rate, not APE**: cut `macro` to ~0, cut `contrarian`, cap `news` — its weight hints overstate its reliability.
- **Beat-baseline hard filter**: if ensemble can't beat persistence in backtest for the regime, default to baseline.
- **Detect the late-June regime break** (5-fail streak) and add a volatility/regime feature; the model is calibrated for quiet days only.