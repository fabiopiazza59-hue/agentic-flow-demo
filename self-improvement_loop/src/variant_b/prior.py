"""Statistical prior for variant B — predict-raven's Elo/Monte-Carlo analog for a stock close.

Instead of asking LLMs to invent a number, the prior is a plain statistical estimate of today's
close distribution from historical daily log-returns, optionally re-centered on the pre-market
quote. Evidence (the pulse) may later shift it only within a hard-capped band, in units of the
prior's own sigma. Pure numpy, deterministic seed, fully testable offline.
"""

from __future__ import annotations

import numpy as np

from ..config import settings


def build_prior(history, features: dict) -> dict:
    """Return the prior close distribution for today.

    {center, sigma_pct, drift_pct, p10, p50, p90, centered_on}
    sigma_pct/drift_pct are fractions (0.01 == 1%). `history` is ascending daily OHLCV.
    """
    close = history["close"].astype(float)
    prev_close = float(close.iloc[-1])
    logret = np.log(close / close.shift(1)).dropna()

    drift = float(logret.tail(settings.B_DRIFT_WINDOW).mean()) if len(logret) else 0.0
    sigma = float(logret.tail(settings.B_SIGMA_WINDOW).std(ddof=1)) if len(logret) >= 2 else 0.01
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.01

    premarket = features.get("premarket_last")
    if premarket:
        center, centered_on = float(premarket), "premarket_quote"
    else:
        center, centered_on = prev_close * float(np.exp(drift)), "prev_close_drift"

    rng = np.random.default_rng(42)
    paths = center * np.exp(rng.normal(drift, sigma, settings.B_MC_PATHS))
    p10, p50, p90 = (float(x) for x in np.percentile(paths, [10, 50, 90]))

    return {
        "center": round(center, 2),
        "sigma_pct": round(sigma, 5),
        "drift_pct": round(drift, 5),
        "p10": round(p10, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "centered_on": centered_on,
        "prev_close": round(prev_close, 2),
    }
