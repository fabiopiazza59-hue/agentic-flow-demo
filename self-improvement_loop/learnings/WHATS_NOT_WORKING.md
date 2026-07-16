# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude.** 12 of 14 fails are directional misses (86%); only 2 are pure magnitude. The model gets the size roughly right but calls the sign wrong. This is the core problem.
- **Losing to baseline.** Nearly every fail also has `beats_baseline: false` — we're not just wrong, we're worse than naive persistence on the same days.
- **Big-error clustering.** Fails carry a mean APE of 2.58% and several 3–5% blowups (06-22 at 5.1%, 06-17 at 3.7%, 06-15 at 3.4%). These are large single-day moves the model never anticipates.
- **Confidence isn't the issue.** 0 overconfident misses; confidence is uniformly low (0.38–0.58). It's well-calibrated-by-being-timid — it never commits, so it never "over"-confidently misses, but it also adds no signal.

## Unreliable under these conditions
- **High-volatility / large-move days.** Every big directional whiff coincides with a large actual move; the model reverts toward small changes and gets run over.
- **`news`-led and `macro`-led calls on turbulent days.** News wins the ensemble on most fails (06-15, 06-22, 06-25→06-29, 07-01, 07-08/09) yet keeps missing direction; macro-led days (07-13, 07-15) also fail hard.
- **`contrarian` and `macro` are the weakest engines** (13% hit rate each), yet they still carry ~0.19 weight_hint. `technical` is barely better (17%). Only `news` (30%) and `momentum` (26%) clear even a coin-flip-adjacent bar — and none are above 50%.
- **Late-June cluster (06-22 to 06-30):** 5 straight/near-straight fails — a sustained regime the model never adapted to.

## Fixes to try next
- **Attack directional accuracy directly:** train/select on sign-hit, not just APE. A magnitude-fair model that flips sign is useless.
- **Add a volatility gate:** when expected move exceeds a threshold, widen predicted magnitude and down-weight mean-reverting engines instead of defaulting to small changes.
- **Cut dead weight:** demote `contrarian` and `macro` (13% hit) toward zero; concentrate on `news` + `momentum`, but only where they've historically hit.
- **Beat-baseline guardrail:** if ensemble disagrees with persistence on low-conviction days, fall back to baseline — we're currently losing to it.
- **Make confidence mean something:** current 0.4-ish flatline is noise; recalibrate so higher confidence actually predicts higher hit rate, then size accordingly.