"""End-to-end offline pipeline: predict -> score -> report, fully mocked, no network/keys."""

import numpy as np
import pandas as pd

import src.loop.run_daily as rd
from src.config import settings
from src.utils import read_jsonl


def _synthetic_history(n=80):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n).date
    prices = 180 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame(
        {"open": prices, "high": prices + 1, "low": prices - 1, "close": prices,
         "volume": rng.integers(1e6, 5e6, n)},
        index=list(dates),
    )
    return df


def _isolate(monkeypatch, tmp_path):
    """Point all writable paths at a temp dir so the test doesn't touch repo state."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data", raising=False)
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results", raising=False)
    monkeypatch.setattr(settings, "SITE_DIR", tmp_path / "results" / "site", raising=False)
    monkeypatch.setattr(settings, "LEARNINGS_DIR", tmp_path / "learnings", raising=False)
    monkeypatch.setattr(settings, "LEDGER_PATH", tmp_path / "data" / "predictions.jsonl", raising=False)
    monkeypatch.setattr(settings, "SCORECARDS_PATH", tmp_path / "learnings" / "scorecards.json", raising=False)
    monkeypatch.setattr(settings, "STRATEGY_PATH", tmp_path / "learnings" / "STRATEGY.md", raising=False)
    monkeypatch.setattr(settings, "RESULTS_CSV", tmp_path / "results" / "results.csv", raising=False)
    monkeypatch.setattr(settings, "METRICS_JSON", tmp_path / "results" / "metrics.json", raising=False)
    monkeypatch.setattr(settings, "RESULTS_MD", tmp_path / "RESULTS.md", raising=False)
    monkeypatch.setattr(settings, "SITE_DATA", tmp_path / "results" / "site" / "data.json", raising=False)
    settings.ensure_dirs()


def test_predict_then_score_offline(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    hist = _synthetic_history()
    last_date = hist.index[-1]
    actual_close = float(hist.iloc[-1]["close"])
    # predict targets `last_date`; its "prior" is the second-to-last session
    hist_for_predict = hist.iloc[:-1]

    monkeypatch.setattr(rd, "get_history", lambda *a, **k: hist_for_predict)
    monkeypatch.setattr(rd, "get_quote", lambda *a, **k: {
        "last": float(hist_for_predict.iloc[-1]["close"]), "prev_close": float(hist_for_predict.iloc[-1]["close"]),
        "source": "stooq", "symbol": "AMZN", "asof": hist_for_predict.index[-1].isoformat()})

    # offline (client=None) -> stub analysts + offline meta-judge
    rc = rd.main(["--mode", "predict", "--date", last_date.isoformat(), "--dry-run"])
    assert rc == 0
    rows = read_jsonl(settings.LEDGER_PATH)
    assert len(rows) == 1
    pred_row = rows[0]
    assert pred_row["status"] == "pending"
    assert pred_row["predicted_close"] > 0
    assert len(pred_row["analyst_predictions"]) == 5  # all stub analysts produced output

    # now score it: actual close available for last_date
    monkeypatch.setattr(rd, "get_actual_close", lambda sym, when: actual_close)
    rc = rd.main(["--mode", "score", "--dry-run"])
    assert rc == 0
    rows = read_jsonl(settings.LEDGER_PATH)
    scored = rows[0]
    assert scored["status"] == "scored"
    assert scored["actual_close"] == actual_close
    assert "pass" in scored and "ape" in scored
    assert scored["winning_strategy"] in pred_row["analyst_predictions"]

    # artifacts generated
    assert settings.RESULTS_CSV.exists()
    assert settings.METRICS_JSON.exists()
    assert settings.RESULTS_MD.exists()
    assert settings.SITE_DATA.exists()

    # idempotent re-score: no second scoring of same row, still one row
    rc = rd.main(["--mode", "score", "--dry-run"])
    assert len(read_jsonl(settings.LEDGER_PATH)) == 1
