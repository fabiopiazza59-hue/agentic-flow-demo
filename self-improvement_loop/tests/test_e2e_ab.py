"""End-to-end offline A/B pipeline: both arms predict from one snapshot, score, compare."""

import json

import src.loop.run_ab as ab
import src.loop.run_daily as rd
from src.config import settings
from src.utils import read_jsonl
from tests.test_e2e import _isolate, _synthetic_history


def _isolate_ab(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "LEDGER_B_PATH", tmp_path / "data" / "predictions_b.jsonl",
                        raising=False)
    monkeypatch.setattr(settings, "LEARNINGS_B_DIR", tmp_path / "learnings_b", raising=False)
    monkeypatch.setattr(settings, "AB_COMPARE_JSON", tmp_path / "results" / "ab_compare.json",
                        raising=False)
    settings.ensure_dirs()


def test_ab_predict_then_score_offline(monkeypatch, tmp_path):
    _isolate_ab(monkeypatch, tmp_path)
    hist = _synthetic_history()
    last_date = hist.index[-1]
    actual_close = float(hist.iloc[-1]["close"])
    hist_for_predict = hist.iloc[:-1]
    quote = {"last": float(hist_for_predict.iloc[-1]["close"]),
             "prev_close": float(hist_for_predict.iloc[-1]["close"]),
             "source": "stooq", "symbol": "AMZN", "asof": hist_for_predict.index[-1].isoformat()}

    monkeypatch.setattr(ab, "get_history", lambda *a, **k: hist_for_predict)
    monkeypatch.setattr(ab, "get_quote", lambda *a, **k: quote)

    rc = ab.main(["--mode", "predict", "--date", last_date.isoformat(), "--dry-run"])
    assert rc == 0

    rows_a = read_jsonl(settings.LEDGER_PATH)
    rows_b = read_jsonl(settings.LEDGER_B_PATH)
    assert len(rows_a) == 1 and len(rows_b) == 1
    # paired fairness: both arms saw the identical snapshot
    assert rows_a[0]["prior_close"] == rows_b[0]["prior_close"]
    assert rows_b[0]["variant"] == "b"
    assert rows_b[0]["prior"]["sigma_pct"] > 0
    assert rows_b[0]["predicted_close"] > 0
    # offline B abstains to its prior center
    assert rows_b[0]["adjustment_sigma"] == 0.0
    # pulse audit artifact written
    assert (settings.LEARNINGS_B_DIR / "pulse" / f"{last_date.isoformat()}.md").exists()

    # score both arms
    monkeypatch.setattr(rd, "get_actual_close", lambda *a: actual_close)
    monkeypatch.setattr("src.variant_b.runner.get_actual_close", lambda *a: actual_close)
    rc = ab.main(["--mode", "score", "--dry-run"])
    assert rc == 0
    scored_a = read_jsonl(settings.LEDGER_PATH)[0]
    scored_b = read_jsonl(settings.LEDGER_B_PATH)[0]
    assert scored_a["status"] == "scored" and scored_b["status"] == "scored"

    # comparison artifacts
    assert settings.AB_COMPARE_JSON.exists()
    cmp = json.loads(settings.AB_COMPARE_JSON.read_text(encoding="utf-8"))
    assert cmp["n_paired"] == 1
    md = settings.RESULTS_MD.read_text(encoding="utf-8")
    assert "A/B test" in md
