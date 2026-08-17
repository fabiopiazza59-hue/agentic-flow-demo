# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 19 of 28 fails are direction misses (68%). The system isn't just sizing moves wrong — it's picking the wrong sign. This is the core problem, not magnitude.
- **Barely beating a naive baseline**: On most fails APE ≈ baseline_ape (e.g. 0.017/0.019, 0.032/0.031, 0.132/0.133). The model adds little over persistence and gets dragged down alongside it on big-move days.
- **News is the least-bad strategy but still weak**: highest hit rate (0.29) and lowest MAPE, yet it's the winning strategy on many of the worst fails (07-15, 07-23, 07-31). It wins by default, not by skill.
- **Macro and technical are dead weight**: macro hit rate 0.11, technical 0.16. Both routinely "win" the ensemble on losing days (macro on 07-13, 07-15, 07-30; technical on 06-17, 06-30).
- **Big-move blowups**: 07-31 (13% APE), 06-22 (5%), 07-23 (4.2%), 07-30 (3.9%). The model has no handle on gap/volatility days.

## Unreliable under these conditions
- **High-volatility / gap days**: every large-APE fail coincides with a large baseline_ape — the model can't anticipate regime shifts and just tracks yesterday.
- **When macro or technical wins the ensemble**: these are the lowest-hit-rate strategies; their "wins" correlate with directional misses.
- **Confidence is uninformative, not overconfident**: overconfident_misses=0, but confidence hovers 0.4 and shows no signal — passes happen at conf 0.1 (07-27, 08-03) and fails at conf 0.55 (06-30, 07-13). Calibration is flat/random, which is its own failure.
- **June cluster**: 06-12 through 06-29 was near-total failure (8 straight fails minus one). Sustained trending regime the model fought the whole way.

## Fixes to try next
- **Add a direction-first gate**: since misses are directional, optimize/select for sign accuracy separately from magnitude before blending.
- **Down-weight or bench macro & technical**; let news/momentum/contrarian carry, and stop letting sub-0.2-hit-rate strategies "win" days.
- **Volatility regime detector**: widen intervals and defer to baseline when expected move is large; the ensemble is worst exactly there.
- **Recalibrate confidence** against realized outcomes — current scores carry no predictive info; either fix it or stop reporting it.