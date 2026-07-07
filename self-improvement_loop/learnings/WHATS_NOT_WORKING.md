# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude.** 9 of 10 failures are directional misses (only 1 pure magnitude). The model is calling the sign wrong, not just the size — that's the core problem.
- Overall directional hit rate is dismal: **7/16 (~44%), worse than a coin flip.** APE on misses averages ~2.9%, so when we're wrong, we're wrong big.
- We **rarely beat baseline on failures** — the naive baseline APE is near or below ours on almost every miss. We're adding noise, not signal.
- Best runs (6/08, 7/02, 7/06) coincide with **momentum/macro leading and low confidence**; blowups (6/22 at 5.1%, 6/25, 6/29) come when **news is the winning strategy**.

## Unreliable under these conditions
- **`macro` is broken as a standalone winner** — 1 win / 16, 6.25% hit rate, highest MAPE (2.11%). It only "worked" once (6/08 tie) and shouldn't lead.
- **`contrarian` is nearly as bad** — 12.5% hit rate, and consistently down-weighted (0.05–0.16), yet still steers losses (6/18, 6/26).
- **`news`-led days are volatile:** despite a 31% hit rate, the largest-APE failures (6/22, 6/25, 6/29, 7/01) all had news heavily weighted (0.22–0.34). News captures level but not direction.
- **Mid-June cluster (6/12–6/29) is a persistent failure zone** — 8 of 10 misses. Suggests a regime shift (post-6/08) the model never adapted to; APE trending up through 6/22.
- Confidence is **flat and low (0.38–0.58)** — no overconfident misses, but no discriminating power either. Confidence isn't tracking accuracy at all.

## Fixes to try next
- **Cut `macro` and `contrarian`** from the winner pool or hard-cap their weight; both are worse than random on direction.
- **Add a directional gate:** require momentum + news to agree on sign before committing; if they diverge, fall back to baseline (we lose to baseline anyway).
- **Detect regime shifts** — the mid-June cluster shows stale weights; add a rolling recalibration or volatility filter that widens uncertainty after 2 consecutive dir-misses.
- **Rebuild confidence to be predictive** — current 0.4-ish flatline is useless; calibrate against realized directional hit rate.
- Investigate the **6/22 5.1% APE outlier** and the news-led blowups as a group for a common catalyst (earnings/macro print mis-read).