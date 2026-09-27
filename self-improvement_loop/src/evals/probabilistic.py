"""Probabilistic forecasts and proper scoring rules.

The ±1% PASS rate is mostly a volatility thermometer (it measures P(|move| <= 1%) more than
skill), and a bare point forecast says nothing about how sure the desk is. State-of-the-art
forecast evaluation instead scores a full predictive distribution with a strictly proper
scoring rule (Gneiting & Raftery 2007, JASA): here the continuous ranked probability score
(CRPS), approximated by the average quantile (pinball) loss over 19 levels, plus the coverage of
central intervals and the Brier score of the implied P(close > prior close).

Every forecaster — both arms and the free baselines — is turned into a distribution the same way,
so their CRPS is comparable:

    close_q(τ) = center · exp(σ · z_τ)

  * center — the forecaster's point forecast (the random walk's is the prior close; the anchor
    baseline's is the latest pre-open trade).
  * σ — an EWMA volatility forecast of daily log returns (RiskMetrics, λ = 0.94) from sessions
    strictly before the target.
  * z_τ — symmetric quantiles from that forecaster's own past |standardized residuals|
    |ln(actual/center)|/σ: split-conformal order statistics, which give every central interval
    finite-sample coverage under exchangeability (Lei et al. 2018, "locally weighted"
    conformal), falling back to a unit-variance Student-t(5) until enough history exists.
    Calibration sets the width only; the center stays the forecaster's own point, so the
    calibration layer cannot smuggle in a trend bet learned from a drifting sample.

Only information available before each target session is used, so a distribution computed
after the fact for an old row is still an ex-ante forecast; such rows are flagged `posthoc`.
"""

from __future__ import annotations

import math
from functools import lru_cache

LEVELS: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))   # 0.05 .. 0.95
EWMA_LAMBDA = 0.94
CALIBRATION_WINDOW = 60      # most recent residuals used for the z-quantiles
MIN_CALIBRATION = 30         # below this, fall back to Student-t(5)
T_DF = 5


# ------------------------------------------------------------------------------ volatility

def log_returns(closes: list[float]) -> list[float]:
    out = []
    for a, b in zip(closes, closes[1:]):
        if a and b and a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def ewma_sigma(returns: list[float], lam: float = EWMA_LAMBDA, seed: int = 20,
               min_obs: int = 10) -> float | None:
    """RiskMetrics EWMA volatility of daily log returns, seeded with the first `seed` returns."""
    rets = [float(r) for r in returns if r is not None and math.isfinite(r)]
    if len(rets) < min_obs:
        return None
    head = rets[:seed]
    var = sum(r * r for r in head) / len(head)
    for r in rets[seed:]:
        var = lam * var + (1.0 - lam) * r * r
    return math.sqrt(var) if var > 0 else None


# ------------------------------------------------------------------- standardized quantiles

@lru_cache(maxsize=4)
def _student_t_unit_quantiles(df: int = T_DF) -> tuple[float, ...]:
    """Quantiles of a unit-variance Student-t at LEVELS (numeric CDF inversion, no scipy)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def pdf(x: float) -> float:
        return c * (1 + x * x / df) ** (-(df + 1) / 2)

    def cdf(x: float) -> float:      # Simpson's rule on [0, |x|]
        if x == 0:
            return 0.5
        n = 400
        h = abs(x) / n
        s = pdf(0) + pdf(abs(x)) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))
        area = s * h / 3
        return 0.5 + area if x > 0 else 0.5 - area

    scale = math.sqrt((df - 2) / df)
    out = []
    for tau in LEVELS:
        lo, hi = -50.0, 50.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if cdf(mid) < tau:
                lo = mid
            else:
                hi = mid
        out.append((lo + hi) / 2 * scale)
    return tuple(out)


def z_quantiles(residuals: list[float]) -> tuple[list[float], str]:
    """Symmetric standardized quantiles at LEVELS, and the method used.

    With >= MIN_CALIBRATION residuals: the central interval at level c = |2τ − 1| uses the
    split-conformal rank ceil((n + 1)·c) of the most recent CALIBRATION_WINDOW absolute
    residuals, which guarantees coverage >= c under exchangeability. Otherwise Student-t(5).
    """
    zs = [float(z) for z in residuals[-CALIBRATION_WINDOW:] if z is not None and math.isfinite(z)]
    n = len(zs)
    if n < MIN_CALIBRATION:
        return list(_student_t_unit_quantiles()), "student_t5"
    abs_z = sorted(abs(z) for z in zs)
    out = []
    for tau in LEVELS:
        c = abs(2.0 * tau - 1.0)
        if c == 0:
            out.append(0.0)
            continue
        q = abs_z[min(n, math.ceil((n + 1) * c)) - 1]
        out.append(q if tau > 0.5 else -q)
    return out, f"conformal_n{n}"


def predictive_quantiles(center: float, sigma: float, zq: list[float]) -> list[float]:
    return [round(center * math.exp(sigma * z), 4) for z in zq]


# ----------------------------------------------------------------------------- scoring

def prob_above(quantiles: list[float], threshold: float) -> float:
    """P(close > threshold) from the quantile function (linear interpolation, clipped tails)."""
    qs = list(quantiles)
    if threshold <= qs[0]:
        return 1.0 - LEVELS[0] / 2
    if threshold >= qs[-1]:
        return (1.0 - LEVELS[-1]) / 2
    for (q0, t0), (q1, t1) in zip(zip(qs, LEVELS), zip(qs[1:], LEVELS[1:])):
        if q0 <= threshold <= q1:
            frac = 0.0 if q1 == q0 else (threshold - q0) / (q1 - q0)
            return 1.0 - (t0 + frac * (t1 - t0))
    return 0.5


def prob_pass(quantiles: list[float], point: float, threshold: float = 0.01) -> float:
    """P(the ±threshold PASS event) under the forecast: |point − close| / close <= threshold.

    This is what the row's `confidence` claims to be; deriving it from the calibrated
    distribution replaces the old heuristic that scored worse than a constant (review F5).
    """
    lo, hi = point / (1.0 + threshold), point / (1.0 - threshold)
    return max(0.0, min(1.0, prob_above(quantiles, lo) - prob_above(quantiles, hi)))


def crps_from_quantiles(quantiles: list[float], actual: float) -> float:
    """CRPS ≈ 2·mean pinball loss over LEVELS, as a fraction of the actual close."""
    total = 0.0
    for q, tau in zip(quantiles, LEVELS):
        total += ((1.0 if actual < q else 0.0) - tau) * (q - actual)
    return 2.0 * total / len(LEVELS) / abs(actual)


def _interval(quantiles: list[float], lo_tau: float, hi_tau: float) -> tuple[float, float]:
    idx = {round(t, 2): i for i, t in enumerate(LEVELS)}
    return quantiles[idx[round(lo_tau, 2)]], quantiles[idx[round(hi_tau, 2)]]


def score_distribution(quantiles: list[float], actual: float, prior_close: float) -> dict:
    up = 1.0 if actual > prior_close else 0.0
    p_up = prob_above(quantiles, prior_close)
    lo50, hi50 = _interval(quantiles, 0.25, 0.75)
    lo80, hi80 = _interval(quantiles, 0.10, 0.90)
    lo90, hi90 = _interval(quantiles, 0.05, 0.95)
    return {
        "crps": crps_from_quantiles(quantiles, actual),
        "cover50": lo50 <= actual <= hi50,
        "cover80": lo80 <= actual <= hi80,
        "cover90": lo90 <= actual <= hi90,
        "p_up": p_up,
        "brier_up": (p_up - up) ** 2,
    }


# ------------------------------------------------------------ ledger-wide evaluation

def _returns_by_date(*ledgers: list[dict]) -> dict[str, float]:
    """Daily log return ln(close/prior close) per session, reconstructed from the ledgers."""
    out: dict[str, float] = {}
    for rows in ledgers:
        for r in rows:
            a, p = r.get("actual_close"), r.get("prior_close")
            if a and p and a > 0 and p > 0:
                out.setdefault(r["date"], math.log(float(a) / float(p)))
    return out


def sigma_before(date_iso: str, returns_by_date: dict[str, float]) -> float | None:
    past = [returns_by_date[d] for d in sorted(returns_by_date) if d < date_iso]
    return ewma_sigma(past)


def _row_sigma(r: dict, returns_by_date: dict[str, float]) -> float | None:
    return r.get("sigma_pct") or sigma_before(r["date"], returns_by_date)


def forecast_distribution(rows: list[dict], date_iso: str, point: float, sigma: float,
                          prior_close: float, is_clean=lambda r: True,
                          threshold: float = 0.01) -> dict:
    """Calibrated predictive distribution for a new forecast, from this arm's own history.

    Standardized residuals of the arm's clean scored rows before `date_iso` set the shape; with
    fewer than MIN_CALIBRATION of them the random walk's residuals (every scored day) stand in.
    """
    rets = _returns_by_date(rows)
    past = [r for r in rows if r.get("status") == "scored" and r.get("date", "") < date_iso
            and r.get("actual_close") and r.get("prior_close")]
    past.sort(key=lambda r: r["date"])
    model_z, rw_z = [], []
    for r in past:
        sig = _row_sigma(r, rets)
        if not sig:
            continue
        actual = float(r["actual_close"])
        rw_z.append(math.log(actual / float(r["prior_close"])) / sig)
        if not r.get("seed") and r.get("predicted_close") and is_clean(r):
            model_z.append(math.log(actual / float(r["predicted_close"])) / sig)
    zs = model_z if len(model_z) >= MIN_CALIBRATION else rw_z
    zq, method = z_quantiles(zs)
    quantiles = predictive_quantiles(point, sigma, zq)
    return {
        "quantiles": quantiles,
        "p_up": round(prob_above(quantiles, prior_close), 4),
        "p_pass": round(prob_pass(quantiles, point, threshold), 4),
        "calibration": method if zs is model_z else f"{method}_rw",
    }


def forecast_centers(row: dict) -> dict[str, float]:
    """Point forecasts of the row's forecaster and the free baselines it is judged against."""
    prior = float(row["prior_close"])
    anchor = float(row.get("anchor") or prior)
    return {"model": float(row["predicted_close"]), "random_walk": prior, "anchor": anchor}


def evaluate(rows: list[dict], returns_by_date: dict[str, float],
             is_clean=lambda r: True) -> list[dict]:
    """Per-row distribution scores for the model and both baselines, oldest first.

    Each row is scored with distributions built only from rows dated before it. The model's
    residual history uses clean (pre-open) rows only — late rows saw part of the tape and would
    make the intervals too narrow; the baselines' residuals are unaffected by timing.
    """
    scored = sorted((r for r in rows if r.get("status") == "scored" and not r.get("seed")
                     and r.get("actual_close") and r.get("prior_close")
                     and r.get("predicted_close")), key=lambda r: r["date"])
    history: dict[str, list[tuple[str, float]]] = {"model": [], "random_walk": [], "anchor": []}
    out = []
    for r in scored:
        d = r["date"]
        sigma = _row_sigma(r, returns_by_date)
        if not sigma:
            continue
        centers = forecast_centers(r)
        actual, prior = float(r["actual_close"]), float(r["prior_close"])
        entry = {"date": d, "clean": bool(is_clean(r)), "sigma": sigma}
        for name, center in centers.items():
            zs = [z for (dd, z) in history[name] if dd < d]
            if name == "model" and len(zs) < MIN_CALIBRATION:
                # too little clean model history: borrow the random walk's residual shape
                zs = [z for (dd, z) in history["random_walk"] if dd < d]
            zq, method = z_quantiles(zs)
            stored = r.get("quantiles") if name == "model" else None
            if stored and len(stored) == len(LEVELS):
                quantiles, method = [float(q) for q in stored], "stored"
            else:
                quantiles = predictive_quantiles(center, sigma, zq)
            s = score_distribution(quantiles, actual, prior)
            s["method"] = method
            entry[name] = s
        for name, center in centers.items():
            if name != "model" or entry["clean"]:
                history[name].append((d, math.log(actual / center) / sigma))
        out.append(entry)
    return out


def summarize(evals: list[dict], clean_only: bool = True) -> dict:
    """Mean CRPS / coverage / Brier per forecaster, plus the CRPS skill of the model."""
    rows = [e for e in evals if e["clean"] or not clean_only]
    if not rows:
        return {"n": 0}

    def mean(key, name):
        vals = [float(e[name][key]) for e in rows if name in e]
        return sum(vals) / len(vals) if vals else None

    out = {"n": len(rows)}
    for name in ("model", "random_walk", "anchor"):
        out[name] = {k: mean(k, name) for k in ("crps", "cover50", "cover80", "cover90",
                                                 "brier_up")}
    m, rw, an = (out[n]["crps"] for n in ("model", "random_walk", "anchor"))
    out["crps_skill_vs_random_walk"] = (1 - m / rw) if m is not None and rw else None
    out["crps_skill_vs_anchor"] = (1 - m / an) if m is not None and an else None
    # Direction: the calibrated distributions carry any drift in past residuals, so even the
    # random walk beats a coin flip in a trending tape. Skill is measured against it, not 0.5.
    b, b_rw = out["model"]["brier_up"], out["random_walk"]["brier_up"]
    out["direction_brier_skill"] = (1 - b / b_rw) if b is not None and b_rw else None
    return out
