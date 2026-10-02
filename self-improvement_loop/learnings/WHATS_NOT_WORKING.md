# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate failures: 27 of 42 (64%).** This is not a magnitude-calibration problem — the model is picking the wrong sign of the move. When it's wrong, it's usually wrong about direction, not just size.
- **Barely beats baseline.** Many fails have ape ~= baseline_ape (e.g., 06-15, 06-17, 06-22, 07-15, 08-20, 09-11). On the hard days the model adds no edge over naive persistence.
- **Macro is the weakest strategy by far** (11.5% hit rate, highest mape 0.0165) yet it keeps winning the vote on fail days (06-12 era, 07-13, 07-15, 07-30, 08-19). When macro wins, it tends to lose.
- **Technical is nearly as bad** (14% hit rate) despite the second-highest weight hint (0.209). It's over-weighted relative to its performance.
- **Big-tail blowups cluster around month-end / early-month** (07-31 ape 13%!, 06-22, 06-25, 07-30, 08-19). Likely earnings/macro-print regime shifts the model doesn't absorb.

## Unreliable under these conditions
- **High-volatility / large-move days:** mean fail ape 2.39% vs passes well under 1%. The model is fine in quiet tape and breaks when the move is large — exactly when it matters.
- **Macro- or technical-led votes:** combined ~13% hit rate. These two strategies are structurally unreliable, especially around macro events.
- **Confidence is weakly calibrated, trending to overconfident recently.** Only 4 flagged overconfident misses, but late-sample fails run confidence 0.52–0.72 (08-19 @0.62, 08-31 @0.72, 09-09 @0.72, 09-11 @0.54, 09-17 @0.54). Rising confidence is NOT buying accuracy. Early low-confidence (0.1–0.3) days were often passes — confidence and correctness are near-decoupled.

## Fixes to try next
- **Cut macro and technical weights hard** (both well below their current 0.18–0.21 hints); reallocate toward news (best hit rate 0.32, lowest mape) and contrarian.
- **Suppress/override on event days** (earnings, CPI/Fed): detect month-end and known print dates, widen intervals or defer to baseline, since that's where the 13% tail blew up.
- **Add a volatility regime gate:** when expected move is large, stop trusting macro/technical sign calls — they drive the directional misses.
- **Recalibrate confidence against realized hit rate;** current high-confidence (0.6–0.72) outputs miss too often. Flatten confidence or tie it to news-strategy agreement.
- **Build a directional-agreement check:** require ≥2 strategies to agree on sign before committing; the core failure is sign, not magnitude.