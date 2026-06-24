# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate.** 4 of 5 failures got the *sign* wrong, not just the magnitude. This is a directional model failure, not a fine-tuning problem — we're betting the wrong way on multi-day moves.
- **The system loses to a naive baseline when it loses.** On 4 of 5 fails it failed to beat baseline_ape too. When wrong, we're worse than doing nothing.
- **Error magnitude is growing.** Fail APEs trend up across June (2.0% → 3.4% → 3.7% → 5.1%). Mean fail APE 3.38% vs sub-0.5% on passes — bimodal: we're either nearly perfect or badly off, no middle.
- **No single strategy is carrying.** Best is `news` at 37.5% hit rate; the other four sit at 12.5–25%. The whole ensemble is barely better than a coin flip on direction.

## Unreliable under these conditions
- **Late-June stretch (06/15–06/22) is a kill zone** — 4 straight directional misses with rising APE, suggesting a regime/trend shift the model never adapted to (likely a sustained move it kept fading).
- **`technical`, `contrarian`, `macro` are effectively dead weight** — 12.5% hit rate each, highest MAPEs (2.1–2.5%). When `momentum`/`macro` carry top weight (06/12, 06/15), results are poor.
- **Confidence is uninformative, not overconfident.** 0 overconfident misses, but confidence sits flat at 0.38–0.42 on nearly everything (pass AND fail). It's not miscalibrated high — it's miscalibrated *flat*. No signal value at all.

## Fixes to try next
- Add a **trend/regime filter**: when price is in a sustained directional run, stop letting contrarian/technical fade it. The 06/15–06/22 cluster screams unhandled regime.
- **Cut or shrink technical, contrarian, macro**; lean weight toward `news` (only strategy beating coin-flip). Re-test as a news+momentum core.
- **Rebuild confidence to actually vary** and gate trades — flat 0.4 confidence is useless. Suppress predictions when strategies disagree on sign.
- Investigate why we **lose to baseline when wrong** — possibly overfitting to short-term reversals; add a "defer to baseline" fallback under high disagreement.