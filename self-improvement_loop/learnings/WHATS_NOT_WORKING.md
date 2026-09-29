# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate** (27 of 42 fails, 64%). This is a sign problem, not just a magnitude problem — the model repeatedly picks the wrong side, not merely the wrong size.
- Failures cluster in **high-volatility spikes**: APEs of 2–5%+ recur (06-22, 06-25, 06-29, 07-15, 07-30, 08-19) and the blowup on 07-31 (13.2% APE — likely earnings/gap event) where the model was helpless.
- On big-move days the model **barely tracks baseline** (fail mean APE 2.39% vs baseline often within ~0.1–0.5%). It adds little value precisely when it matters.
- **Macro is the worst engine**: 12% hit rate, highest MAPE (0.0168), yet still carries ~18% weight. Every day it "wins" the blend it tends to lose (06-12, 07-13, 07-15, 07-30, 08-19).
- **Technical also weak** (14.7% hit rate) despite the second-highest weight hint (0.21) — over-weighted relative to its accuracy.

## Unreliable under these conditions
- **Trend reversals / gap days**: momentum and technical keep extrapolating the prior move and get run over (07-31, 07-30, 08-19, 08-20).
- **Event-driven days** (earnings, macro prints): news wins the blend but still misses direction on the largest moves — it reacts too late/too small.
- **Confidence is mildly miscalibrated at the top end**: several highest-confidence calls (0.72 on 08-31, 09-09; 0.62 on 08-19, 09-03) are fails. Only 4 flagged "overconfident," but the pattern shows confidence ≥0.5 does NOT reliably indicate a pass.
- Best regime: quiet, low-range days with news/contrarian leading — those produce the sub-0.5% APE hits.

## Fixes to try next
- **Cut macro weight toward zero** and reallocate to news (best hit rate 0.32, lowest MAPE); trim technical's weight to match its poor hit rate.
- Add a **volatility/gap regime filter**: when expected range is high or an earnings/macro event is scheduled, widen intervals and lower confidence automatically.
- Attack the **directional error directly** — add a sign-accuracy penalty in strategy selection rather than optimizing APE alone.
- **Recalibrate confidence**: current high-confidence buckets pass at roughly coin-flip rates; fit confidence to realized hit rate and stop emitting 0.6+ on event days.
- Investigate the **07-31 outlier** as a distinct earnings-day model; a single 13% miss is dragging risk and signals no gap handling exists.