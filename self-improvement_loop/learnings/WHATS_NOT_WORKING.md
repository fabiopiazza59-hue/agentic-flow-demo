# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **56% fail rate (61/109)** with mean failure APE 2.43% — misses are large, not marginal. Directional (32) and magnitude (29) misses are roughly balanced, so this is both a "wrong-way" and "wrong-size" problem.
- **Magnitude underestimation on big-move days** is the costliest pattern: 2026-07-31 (13.2% APE), 2026-11-04 (6.7%), 2026-10-19 (5.5%), 2026-10-30 (5.2%), 2026-06-22 (5.1%). The model systematically compresses toward small moves and gets run over on volatility spikes — several of these days it beat baseline yet still failed, i.e. the whole market gapped.
- **Contrarian is the chronic offender in the recent window.** It "wins" 30 times but a huge share of recent fails (10/19 through late Oct–Nov) are contrarian days — it keeps fading moves that continue (2026-11-04, 10-19, 10-15, 10-08, 11-11).
- **Barely beating baseline.** On many fails APE ≈ baseline_ape (e.g. 09-11, 08-10, 06-29), meaning the ensemble adds little over naive persistence.

## Unreliable under these conditions
- **High-volatility / gap days:** every APE > 3% is a directional *or* magnitude miss; model can't size large moves.
- **Contrarian-selected days in trending runs:** fading momentum during sustained directional stretches.
- **Technical and macro as winners are near-useless:** hit rates 11% and 13%; macro has the worst MAPE (1.70%). News (31%) and contrarian (28%) are the only strategies above chance.
- **Confidence is miscalibrated.** Recent block (0.55–0.66 confidence) is littered with fails: 08-19 (0.62), 08-20 (0.54), 10-08 (0.62), 10-19 (0.56), 09-03 (0.62). Confidence rose into the fall but accuracy didn't — 7 overconfident misses understates it because the whole scale drifted up.

## Fixes to try next
- **Add a volatility regime gate:** when realized/implied vol is elevated, widen predicted magnitude and suppress contrarian weight.
- **Demote/retire macro and technical as standalone winners;** cap their selection, lean on news.
- **Fix contrarian trigger** to require a confirmed exhaustion signal, not just a prior-day move — stop fading trends.
- **Recalibrate confidence** against realized hit rate (isotonic/Platt); current 0.5–0.65 band shows no edge.
- **Penalize "ties with baseline"** — flag predictions within ~0.3% of persistence as no-signal rather than counting them.