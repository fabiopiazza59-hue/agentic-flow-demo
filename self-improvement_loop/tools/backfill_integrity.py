"""One-time migration: stamp `late_minutes` on historical ledger rows.

`late_minutes` is derived from each row's own `created_at` and session date, so this is a pure
re-derivation of data already in the ledger — it adds no new information and changes no
prediction. Rows without a usable timestamp (backfill seeds) are left alone.

Run:  python -m tools.backfill_integrity [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.evals import integrity
from src.utils import read_jsonl, write_jsonl


def migrate(path, dry_run: bool = False) -> int:
    rows = read_jsonl(path)
    changed = 0
    for row in rows:
        before = row.get("late_minutes")
        integrity.annotate(row)
        if row.get("late_minutes") != before:
            changed += 1
    late = [r for r in rows if integrity.is_late(r)]
    print(f"{path.name}: {len(rows)} rows, stamped {changed}, {len(late)} created after the open")
    for r in sorted((x for x in late if x.get("late_minutes") is not None),
                    key=lambda r: r.get("date", "")):
        print(f"    {r['date']}  +{r['late_minutes']} min after open  "
              f"({'post-close' if r['late_minutes'] > 390 else 'intraday'})")
    if not dry_run and changed:
        write_jsonl(path, rows)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for path in (settings.LEDGER_PATH, settings.LEDGER_B_PATH):
        if path.exists():
            migrate(path, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
