# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **55% fail rate (56/101)** with mean fail APE 2.4% — magnitude on big-move days is the killer. Several fails are 3–5% APE, plus one blowout at 13.2% (2026-07-31) and 5.5% (2026-10-30).
- **Directional misses (30) slightly outnumber magnitude misses (26)** — the model is not just mis-sizing, it frequently gets the sign wrong. That's a core signal problem, not a calibration tweak.
- **A recurring pattern: directional_hit=true but still FAIL** (e.g. 07-23, 07-31, 10-22, 10-29, 10-14). Right direction, badly undersized move → systematic magnitude compression on volatile days.
- **macro and technical are dead weight**: both hit_rate 11.9% and the two highest MAPEs. macro wins the day only on big-error days (07-15, 07-30, 08-19, 10-30) — it "wins" by being least-wrong in a bad regime, not by being good.

## Unreliable under these conditions
- **High-volatility / large-move days**: whenever the realized move is large (baseline_ape >2%), the model tends to underclip and fail even when directionally right. Clustered late-June, late-July, mid-Oct, late-Oct.
- **macro-led and technical-led predictions**: unreliable almost always; when macro is the winning strategy it's usually a fail (07-15, 07-30, 08-19, 10-09, 10-30).
- **Mild overconfidence creep in Oct**: confidence drifted up to 0.56–0.66 while fail rate stayed high (10-06 through 10-19 mostly fails at conf ~0.57–0.66). Only 7 flagged overconfident misses, but the recent rising-confidence/flat-accuracy trend is the warning sign.
- **contrarian over-fires in choppy stretches** (10-08, 10-15, 10-19 fails) — it's the best hit-rate ex-news but produces the worst single tail errors.

## Fixes to try next
- **Cut or floor macro and technical weights** (both 11.9% hit); reallocate toward news (32.7%) and contrarian (26.7%).
- **Add a volatility-scaled magnitude term**: on high-ATR/large-gap days, widen predicted move — the model is chronically compressing big moves even when the sign is right.
- **Re-calibrate confidence downward in high-vol regimes**; cap confidence when macro/technical is the lead strategy since those lead on the worst days.
- **Build a regime gate**: detect large-move days (earnings, macro prints) and switch to a wider-interval mode rather than trusting the ensemble point estimate.
- **Audit the sign model separately** — 30 directional misses means an independent direction classifier may beat the current weighted blend.