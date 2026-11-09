# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- 56% fail rate (59/105). Failures split almost evenly: 31 directional misses vs 28 magnitude misses — this is BOTH a sign-problem and a size-problem, not one axis.
- Directional misses dominate the worst days. On large-move days (APE > 3%) the model repeatedly gets the sign wrong (2026-06-15, -22, -25, 07-15, 08-19, 10-19, 11-04). When the market actually moves, we flip the wrong way.
- Even "passes" frequently fail to beat baseline (many pass=true, beats_baseline=false). On quiet days we're barely better than a naive carry-forward, so our edge is thin where it's easy and absent where it's hard.
- Tail blowups: 2026-07-31 (13.2% APE), 10-30 (5.2%), 11-04 (6.7%), 10-19 (5.5%). These single-day disasters dwarf the mean fail APE of 2.45% and drag aggregate performance.

## Unreliable under these conditions
- High-volatility / gap days: every strategy's magnitude estimate collapses; macro and contrarian specifically pick the wrong direction into the move.
- macro and technical are the weakest engines (hit_rate 12% and 11%, highest MAPE ~0.017 and 0.016). Macro-won days are disproportionately big misses (07-15, 08-19, 10-30). Technical wins almost never.
- contrarian wins the most count (28) but is overrepresented in the biggest fails (10-19, 11-04, 10-08) — it fades real trends and gets run over.
- Confidence is mildly miscalibrated on the high end: only 7 overconfident misses, but note 08-19 (conf 0.62 fail), 08-31 (0.72 fail), 09-09 (0.72 fail), 10-06/10-08 (~0.66/0.62 fail). High confidence ≥0.6 does not reliably indicate a pass.

## Fixes to try next
- Cut or cap macro and technical weight; they add MAPE without hit rate. Lean on news (33% hit, lowest MAPE) and contrarian for quiet regimes only.
- Add a volatility regime gate: when expected daily range is high, suppress contrarian/macro sign bets and widen magnitude or abstain.
- Recalibrate confidence against realized pass rate; current ≥0.6 band is not earning its label. Isotonic/Platt recal on the last 100.
- Add outlier guardrails (earnings/gap flag) to prevent 5–13% single-day blowups that dominate loss.