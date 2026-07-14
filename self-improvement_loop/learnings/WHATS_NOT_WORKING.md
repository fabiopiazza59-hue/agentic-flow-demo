# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core failure**: 11 of 13 misses are directional (85%), only 2 magnitude. The model predicts the wrong sign, not the wrong size. Getting APE down won't help; we're calling up/down wrong.
- Overall directional hit rate is barely coin-flip-ish and skews worse in the mid-June cluster (6/12–6/29): a long streak of consecutive directional misses with rising APE (peaking 5.1% on 6/22).
- No overconfident misses flagged, but confidence is **flat and useless** — nearly every prediction sits at 0.40–0.55 regardless of outcome. Confidence carries zero discriminating signal (misses at 0.55/0.58, hits at 0.40). It's miscalibrated by being inert, not by being cocky.
- Beating baseline and passing are correlated with hits, but on misses we usually **also lose to baseline** — the ensemble adds negative value on hard days.

## Unreliable under these conditions
- **The mid/late-June regime** (roughly 6/12–6/29): sustained directional misses, APE 2–5%, suggests a trend/volatility regime the model fought (likely a directional move it kept fading).
- **Macro strategy is the worst**: 9.5% hit rate, highest MAPE (0.0182). It "wins" the ensemble on losing days (6/12, 6/13) — a red flag that macro dominates exactly when it's wrong.
- **Contrarian (14%) and momentum (24%) hit rates are also sub-random**; contrarian gets weight ~0.20 despite being unreliable. Momentum "won" several outright misses (6/12, 6/25).
- **News is the least-bad** (33% hit, lowest sum_ape) but still not good, and it too anchors several misses (6/15, 6/22, 6/29).
- Days with empty/degenerate weights (6/30) still shipped predictions — pipeline gap.

## Fixes to try next
- Reframe scoring/optimization around **directional accuracy**, not APE — that's where we're bleeding.
- **Cut macro weight sharply** (it's near-zero skill) and down-weight contrarian; stop letting the worst strategies "win" the ensemble on high-error days.
- Add a **regime filter** (trend vs mean-revert): momentum/contrarian are being applied in the wrong regimes.
- **Rebuild confidence** so it actually separates hits from misses; current 0.4–0.55 band is meaningless. Suppress trades when signal disagreement is high.
- Add a **baseline guardrail**: if ensemble can't beat naive baseline in backtest for a regime, default to baseline.