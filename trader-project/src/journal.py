"""Journal writer — audit trail for all pipeline outputs.

Every stage writes its output to journal/stage<N>/<date>_<run_id>.yaml.
Never overwrites. Append-only.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = PROJECT_ROOT / "journal"


def write_journal(stage: int, run_id: str, output: BaseModel) -> Path:
    """Write a stage output to the journal.

    Returns the path of the written file.
    """
    stage_dir = JOURNAL_DIR / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{run_id}.yaml"
    filepath = stage_dir / filename

    # Never overwrite — append a counter if file exists
    counter = 1
    while filepath.exists():
        filename = f"{date_str}_{run_id}_{counter}.yaml"
        filepath = stage_dir / filename
        counter += 1

    data = output.model_dump(mode="json")
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return filepath
