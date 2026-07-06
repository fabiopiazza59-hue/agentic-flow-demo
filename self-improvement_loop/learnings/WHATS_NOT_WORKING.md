# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the problem.** 9 of 10 fails are directional misses (only 1 pure magnitude miss). The model sizes moves roughly right but calls the sign wrong. Fixing APE tuning won't help; we're betting the wrong way.
- **We lose to the naive baseline on most fails.** 8 of 10 failures also had `beats_baseline=false` — on miss days the model adds negative value versus just persisting the prior close.
- **Directional hit rate is broadly awful across strategies:** best is news at 33%, worst is macro at 7% and contrarian at 13%. A coin flip (50%) would beat every strategy here. That's a signal-inversion or feature-lag problem, not a noise problem.
- **Clustered failure streak** from 2026-06-25 through 07-01 (6 straight misses) suggests a regime the model never adapted to, not random scatter.

## Unreliable under these conditions
- **Larger-move days.** Fail-day mean APE is 2.94%; the worst misses (06-22 5.1%, 06-17 3.7%, 06-25 3.6%) are all directional whiffs — the model won't commit to big moves and gets the sign wrong on them.
- **News- and momentum-led days are the biggest culprits by volume:** news won on 5 of the 10 fails, momentum on 3. News has the best hit rate overall yet still leads most losses — it's being trusted on the wrong days.
- **Macro and contrarian are dead weight** — 1 and 2 wins in 15, and macro-heavy weighting (06-12) produced a clean miss.
- **Confidence carries no information.** Fails span 0.38–0.58, passes span 0.4–0.5. `overconfident_misses=0` only because confidence is flat and low everywhere — it's uncalibrated/uninformative, not well-behaved.

## Fixes to try next
- **Investigate sign inversion:** with hit rates this far below 50%, test flipping the directional call on momentum/macro/contrarian and re-score offline.
- **Cut macro and contrarian weight to near zero;** they underperform baseline consistently.
- **Add a regime/volatility filter:** on high-expected-move days defer to baseline (persistence) rather than committing direction, since that's where we bleed.
- **Rebuild confidence** to actually track realized hit rate; current values are noise and can't gate anything.
- **Root-cause the 06-25→07-01 streak** — check for a stale feature or data lag introduced then.