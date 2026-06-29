# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer.** 6 of 7 fails are directional misses; only 1 is a pure magnitude miss. The model gets the move size roughly right but bets the wrong way.
- **Failures cluster and snowball.** The recent run (06-22, 06-25, 06-26) and the 06-15→06-18 block are consecutive losses with rising APE (3.4%→5.1%→3.6%→3.2%), suggesting the model fails to adapt to a regime it's already inside.
- **We barely beat baseline.** Most fails also lose to the naive baseline (beats_baseline=false on nearly every miss), so the ensemble is actively destroying value on bad days, not just being unlucky.
- **No single reliable strategy.** Best hit rates are momentum/news at 27% — that's worse than a coin flip. Every strategy is below 50%.

## Unreliable under these conditions
- **News-led days are a trap on big moves.** News "wins" the weighting (06-15, 06-17, 06-22, 06-26 carry highest news weight) precisely on the largest-APE directional misses. High news weight + large realized move = wrong direction.
- **Momentum during reversals.** Momentum is the winning strategy on several fails (06-12, 06-25); it extrapolates trend into turns and gets whipsawed.
- **Macro is the weakest link** — 1 win in 11 (9% hit rate), highest MAPE (2.5%). It only "won" on the one easy pass.
- **Confidence is flat and uninformative, not overconfident.** 0 overconfident misses, but confidence sits ~0.40–0.58 on both wins and losses — it carries no signal. Misses aren't flagged; the system can't tell good days from bad.

## Fixes to try next
- Down-weight or cap **macro** (worst hit rate) and de-emphasize **news weighting on high-implied-move days**; news should inform magnitude, not direction.
- Add a **directional regime/volatility filter**: when consecutive misses or wide moves are detected, fall back toward the baseline instead of committing the ensemble.
- **Recalibrate confidence** against realized direction — current scores are near-constant and useless; force separation so low-confidence days trigger smaller/no positions.
- Build a **reversal detector** to override momentum on suspected turns; momentum is bleeding on whipsaws.