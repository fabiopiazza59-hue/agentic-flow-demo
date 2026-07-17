# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer.** 12 of 15 fails are directional misses (only 3 pure magnitude). The model calls the sign wrong far more than it over/undershoots size.
- **Baseline beats us on the misses.** On failed days `mean_fail_ape` = 2.51% and we lose to naive baseline repeatedly (e.g. 06-12, 06-25, 06-26, 07-13). We're adding error, not skill, when it matters.
- **Big-move days are where we bleed.** The largest APEs (5.1% on 06-22, 3.7% on 06-17, 3.4% on 06-15, 3.6% on 06-25, 3.2% on 06-26/06-29, 3.0% on 07-15) all coincide with wrong direction — we get run over by real moves.
- **No strategy is reliable.** Best hit rate is `news` at 29%, everything else 12–25%. That's coin-flip-or-worse across the board; the ensemble has no genuine edge.
- **`macro` and `momentum` are the worst.** macro 12.5% hit / 1.81% MAPE, momentum 25% but highest sum_ape. macro-won days (07-13, 07-15) were both misses.

## Unreliable under these conditions
- **High-volatility / trend days (>2.5% real move):** consistently wrong sign, worst APEs cluster here.
- **When `news` or `momentum` is the winning strategy on a fail day:** these dominate the failed set (news won 5 fails, momentum 3) — the loud strategies steer us into the wrong direction.
- **Confidence is uselessly flat, not overconfident.** 0 overconfident misses, but nearly every prediction sits at 0.40–0.55 regardless of outcome. Confidence carries zero signal — it doesn't rise on wins or fall on misses. That's miscalibration by non-discrimination.
- **Late-June cluster (06-22 → 06-29):** 5 straight fails, all directional — suggests a regime the model never adapted to.

## Fixes to try next
- **Attack the sign problem directly:** train/evaluate a separate directional classifier; stop optimizing APE alone when direction hit-rate is 50%.
- **Add a volatility gate:** when expected move >2%, widen intervals or abstain rather than commit to a direction we can't call.
- **Cut macro weight, demote momentum on high-vol days;** they carry weight_hint ~0.18 despite worst hit rates.
- **Rebuild confidence to be discriminative** — right now it's constant noise; calibrate it against realized direction accuracy so low-confidence days can trigger abstain.
- **Beat-baseline as a hard gate:** if ensemble can't beat naive on backtest for a regime, default to baseline there.