"""Shared utilities: JSON extraction, JSONL IO, date/number helpers.

`extract_json` is adapted from trader-project/src/utils.py (markdown-fence + preamble tolerant),
with a tolerance for trailing commas commonly emitted by LLMs.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


def extract_json(text: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles raw JSON, ```json fenced blocks, surrounding prose, and trailing commas.
    Raises ValueError if no parseable object is found.
    """
    text = (text or "").strip()

    def _try(s: str) -> dict | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # tolerate trailing commas: {"a": 1,}  /  [1, 2,]
            repaired = re.sub(r",(\s*[}\]])", r"\1", s)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None

    direct = _try(text)
    if direct is not None:
        return direct

    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        parsed = _try(fence.group(1).strip())
        if parsed is not None:
            return parsed

    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1 and last > first:
        parsed = _try(text[first:last + 1])
        if parsed is not None:
            return parsed

    raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def iso_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


# --- JSONL ledger IO ---

def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(payload + ("\n" if rows else ""), encoding="utf-8")


def upsert_ledger_row(rows: list[dict], new_row: dict, key: str = "date") -> list[dict]:
    """Insert or replace a row matching new_row[key]; keep list sorted by key."""
    out = [r for r in rows if r.get(key) != new_row.get(key)]
    out.append(new_row)
    out.sort(key=lambda r: r.get(key, ""))
    return out
