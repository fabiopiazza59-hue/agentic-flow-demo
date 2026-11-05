# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- 56% fail rate (57/102) with directional misses (31) edging out magnitude misses (26) — the system gets the *sign* wrong more than the size, meaning this isn't just a calibration problem, the core signal is coin-flip on direction.
- Many fails barely beat or lose to baseline even when "passing," and several large-APE days (0.05–0.13) show the model has no handle on gap/vol days.
- Blended weights add little: `winning_strategy` rotates constantly, suggesting post-hoc attribution rather than a stable edge. The ensemble is averaging toward mush.
- Error magnitude on fails is high (mean fail APE 2.47%) — misses aren't near-misses, they're real blowups clustered on volatile sessions.

## Unreliable under these conditions
- **High-volatility / large-move days**: the worst APEs (2026-07-31 ~13%, 10-19 ~5.5%, 11-04 ~6.7%, 10-30 ~5.2%) are all big-move days the model under-reacted to. It systematically compresses toward small moves.
- **Macro strategy**: hit rate 11.8%, worst MAPE (1.74%). When macro "wins" it's usually on a blowup day (07-15, 07-30, 10-30). Unreliable and should carry minimal weight.
- **Technical strategy**: also 11.8% hit rate — nearly useless standalone; stop treating it as a 20% pillar.
- **Overconfidence**: 7 overconfident misses, concentrated in the Aug–Sep 0.6–0.72 confidence runs (08-19 conf .62 fail, 08-31 conf .72 fail, 09-03 conf .62 fail, 09-09 conf .72 fail). Rising confidence in late Aug/Sep did NOT track rising accuracy — calibration is drifting upward without justification.
- Recent Oct–Nov stretch: many fails still "beat baseline" directionally right but magnitude-wrong — the model lags trend changes.

## Fixes to try next
- Reweight ensemble by hit rate: lift **news** (0.32) and **contrarian** (0.27), cut **macro** and **technical** toward zero. News+contrarian carry nearly all the edge.
- Add a volatility regime flag; on high-vol/gap days widen predicted moves or abstain — the model's mean-reversion bias is killing it on big days.
- Recalibrate confidence against realized hit rate; cap confidence >0.6 until the 0.6–0.72 bucket proves out — currently those are disproportionately misses.
- Investigate directional sign logic directly: 31 direction misses means the problem is feature/signal, not sizing.