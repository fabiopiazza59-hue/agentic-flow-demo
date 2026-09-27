"""Failed predictions must produce a concentrated failure log + a what's-not-working diagnosis."""

import src.loop.run_daily as rd
from src.config import settings


def _isolate(monkeypatch, tmp_path):
    for attr, sub in [
        ("DATA_DIR", "data"), ("RESULTS_DIR", "results"), ("SITE_DIR", "results/site"),
        ("LEARNINGS_DIR", "learnings"),
    ]:
        monkeypatch.setattr(settings, attr, tmp_path / sub, raising=False)
    monkeypatch.setattr(settings, "LEDGER_PATH", tmp_path / "data" / "predictions.jsonl", raising=False)
    monkeypatch.setattr(settings, "SCORECARDS_PATH", tmp_path / "learnings" / "scorecards.json", raising=False)
    monkeypatch.setattr(settings, "STRATEGY_PATH", tmp_path / "learnings" / "STRATEGY.md", raising=False)
    monkeypatch.setattr(settings, "RESULTS_CSV", tmp_path / "results" / "results.csv", raising=False)
    monkeypatch.setattr(settings, "METRICS_JSON", tmp_path / "results" / "metrics.json", raising=False)
    monkeypatch.setattr(settings, "RESULTS_MD", tmp_path / "RESULTS.md", raising=False)
    monkeypatch.setattr(settings, "SITE_DATA", tmp_path / "results" / "site" / "data.json", raising=False)
    settings.ensure_dirs()


def test_failed_prediction_writes_failure_artifacts(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    # A pending prediction that will badly miss the actual close (forces a FAIL).
    rows = [{
        "date": "2026-06-09", "status": "pending", "prior_close": 100.0,
        "predicted_close": 100.0, "predicted_direction": "down", "confidence": 0.7,
        "weights": {"momentum": 0.6, "macro": 0.4},
        "analyst_predictions": {
            "momentum": {"predicted_close": 99.0, "direction": "down", "confidence": 0.7, "rationale": "x"},
            "macro": {"predicted_close": 101.0, "direction": "up", "confidence": 0.5, "rationale": "y"},
        },
        "actual_close": None,
    }]
    # actual close 110 -> 10% error -> FAIL, wrong direction
    monkeypatch.setattr(rd, "get_actual_close", lambda sym, when: 110.0)

    rows, scored = rd.do_score(rows, client=None)  # offline reflector/diagnosis
    assert scored["pass"] is False

    failures = settings.LEARNINGS_DIR / "FAILURES.md"
    wnw = settings.LEARNINGS_DIR / "WHATS_NOT_WORKING.md"
    assert failures.exists(), "FAILURES.md should be written on a miss"
    assert "2026-06-09 — FAIL" in failures.read_text()
    assert wnw.exists(), "WHATS_NOT_WORKING.md should be regenerated when a failure exists"
    body = wnw.read_text()
    assert "What keeps going wrong" in body
    # per-day post-mortem still written
    assert (settings.LEARNINGS_DIR / "2026-06-09.md").exists()


def test_pass_does_not_create_failure_log(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    rows = [{
        "date": "2026-06-09", "status": "pending", "prior_close": 100.0,
        "predicted_close": 100.5, "predicted_direction": "up", "confidence": 0.6,
        "weights": {}, "analyst_predictions": {
            "macro": {"predicted_close": 100.4, "direction": "up", "confidence": 0.6, "rationale": "z"},
        },
        "actual_close": None,
    }]
    monkeypatch.setattr(rd, "get_actual_close", lambda sym, when: 100.3)  # 0.2% -> PASS
    rows, scored = rd.do_score(rows, client=None)
    assert scored["pass"] is True
    assert not (settings.LEARNINGS_DIR / "FAILURES.md").exists()
