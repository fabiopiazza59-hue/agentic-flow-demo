"""extract_json robustness + ledger upsert."""

from src.utils import extract_json, upsert_ledger_row


def test_extract_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced():
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_extract_with_preamble():
    text = 'Here is my answer:\n{"predicted_close": 199.5, "direction": "up"}\nThanks!'
    assert extract_json(text)["predicted_close"] == 199.5


def test_extract_trailing_comma():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_upsert_replaces_same_date():
    rows = [{"date": "2024-01-02", "v": 1}]
    rows = upsert_ledger_row(rows, {"date": "2024-01-02", "v": 2})
    assert len(rows) == 1 and rows[0]["v"] == 2
    rows = upsert_ledger_row(rows, {"date": "2024-01-03", "v": 9})
    assert [r["date"] for r in rows] == ["2024-01-02", "2024-01-03"]
