# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 30 of 52 fails are wrong-direction, not just magnitude. The model can't call sign on volatile days, and sign errors are the core problem, not calibration of size.
- **Big moves blow up APE**: nearly every fail with APE >3% (e.g. 2026-06-22, 07-15, 07-31 @13%, 10-19 @5.5%, 10-22 @4.5%) is a large-move day the model smooths over. Mean fail APE 2.37% vs pass days near 0.3-0.6%.
- **Barely beating baseline**: many fails have APE ≈ baseline_ape (e.g. 09-11, 10-08, 06-17), meaning the ensemble adds no edge on hard days — it's just tracking the naive forecast then eating the error.
- **macro and technical are dead weight**: macro hit_rate 11.6%, technical 12.6%, yet both still carry ~18-21% weight and are "winning_strategy" on several of the worst fails (07-15, 08-19, 07-30).

## Unreliable under these conditions
- **High-volatility / gap days**: clustered large-APE fails suggest earnings/macro-news spikes the model systematically underreacts to.
- **macro-led and contrarian-led days**: contrarian is winning_strategy on a disproportionate share of recent fails (10-08, 10-15, 10-19); macro leads the biggest misses.
- **Confidence is miscalibrated upward recently**: late-period fails run confidence 0.56-0.72 (08-19 @0.62, 08-31 @0.72, 09-09 @0.72, 10-06 @0.66, 10-08 @0.62). Rising confidence has NOT bought accuracy — overconfident misses concentrate in the high-conf regime.

## Fixes to try next
- Cut or floor macro and technical weights; shift mass to news (hit_rate 0.33, lowest MAPE).
- Add a volatility/earnings-day regime flag; widen intervals and down-weight confidence when expected move is large.
- Recalibrate confidence against realized hit-rate — current high-conf band (0.6+) is not earning its score.
- Build a dedicated sign/direction classifier; magnitude blending is masking that the real failure is directional.
- Flag "ensemble ≈ baseline" days and route them to a fallback rather than claiming edge.