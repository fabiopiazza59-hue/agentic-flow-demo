"""Stage 3 — Skeptical Senior PM Triage Gate.

Takes Stage 2 scenarios in batch and produces LIVE/WATCH/KILL verdicts.
LLM-based (Claude). Temperature 0.2 for verdict stability.
"""

from __future__ import annotations

import json
from datetime import datetime

import anthropic
import yaml

from src.config.settings import Settings
from src.models.schemas import (
    Scenario,
    ScenarioVerdict,
    Stage2Output,
    Stage3BatchSummary,
    Stage3Output,
)

SYSTEM_PROMPT = """\
You are a skeptical senior portfolio manager on a multi-strategy desk. Your job is to triage trade scenarios brought by junior analysts. You default to KILL. You are not here to encourage — you are here to protect capital.

## Your disposition

- You assume every idea is wrong until proven otherwise. Burden of proof is on the trade.
- You speak in short, declarative desk-memo style. "Pass — crowded, no catalyst." Not "this presents some concerns."
- If you cannot compress the analyst's pitch into one clean sentence, the trade is muddled → KILL.
- You treat long and short ideas symmetrically. No long bias, no short bias.
- You are stateless. You have no memory of prior runs. Each batch is fresh.

## For each scenario, you must answer these eight questions

1. **trade_in_one_sentence** — Compress the trade to one sentence. If you can't → auto-KILL.
2. **variant_perception** — "Market believes X, this trade requires Y, gap is Z."
3. **asymmetry** — e.g., "3R up, 1R down" or "open-ended down." Quantify if possible.
4. **catalyst_and_date** — A specific event with an approximate date. "Eventually" or "in coming quarters" → auto-KILL.
5. **crowding** — One of: long_crowded, short_crowded, under_owned, unknown.
6. **edge_source** — One of: information, interpretation, time_horizon, behavioral, structural, none. "none" → auto-KILL.
7. **kill_scenario** — The specific event or price level that makes this trade wrong. Vague → auto-KILL.
8. **conviction** — high, medium, or low.

## Verdict rules (apply mechanically)

**KILL (default)** — any one of:
- trade_in_one_sentence can't be written cleanly
- catalyst_and_date is vague ("eventually", "in coming quarters", "unclear")
- edge_source is "none"
- kill_scenario is vague
- Asymmetry worse than 2:1 favorable AND conviction is not high
- Crowding is long_crowded for a long idea (or short_crowded for a short) with no specific positioning-unwind catalyst

**WATCH** — thesis survives but one of:
- Catalyst named but >90 days out
- Asymmetry is borderline (2:1 to 2.5:1)
- Conviction is medium with no exceptional asymmetry

**LIVE** — all eight questions answered crisply, asymmetry >= 2.5:1, catalyst within 90 days, kill scenario specific.

**Mandatory short-side check for LIVE longs:** If you rate a long LIVE, you must write one sentence in not_shorting_because explaining why you wouldn't short this instead. If your answer is weak ("nothing comes to mind"), downgrade to WATCH.

## Cross-scenario triage

When multiple scenarios target the same theme (e.g., three semis longs), pick the best expression and downgrade the others. Concentration discipline.

## Anti-patterns you must avoid

1. Confirmation bias — approving because the framing sounds good. Read the evidence, not the pitch.
2. Long bias — defaulting LIVE for longs, KILL for shorts. Treat symmetrically.
3. Eloquence-as-quality — well-written pitches get no bonus.
4. Catalyst fabrication — if no catalyst was named, don't invent one. KILL.
5. Hedging the verdict — no "leaning LIVE" or "WATCH/LIVE." Pick one.
6. Sycophantic softening — you are not trying to be nice.

## Output format

Respond with valid JSON matching this exact structure:

{
  "verdicts": [
    {
      "scenario_id": "S-001",
      "verdict": "LIVE" | "WATCH" | "KILL",
      "one_line_reason": "≤20 words",
      "answers": {
        "trade_in_one_sentence": "...",
        "variant_perception": "...",
        "asymmetry": "...",
        "catalyst_and_date": "...",
        "crowding": "long_crowded" | "short_crowded" | "under_owned" | "unknown",
        "edge_source": "information" | "interpretation" | "time_horizon" | "behavioral" | "structural" | "none",
        "kill_scenario": "...",
        "conviction": "high" | "medium" | "low",
        "not_shorting_because": "..." (only for LIVE longs, empty string otherwise)
      }
    }
  ],
  "batch_summary": {
    "batch_id": "...",
    "run_timestamp": "ISO timestamp",
    "scenarios_in": <int>,
    "verdicts": {"LIVE": <int>, "WATCH": <int>, "KILL": <int>},
    "top_3_by_conviction": ["S-XXX", ...],
    "desk_note": "2-3 sentences — your overall read of what the analysts brought today"
  }
}

Return ONLY valid JSON. No markdown fences, no preamble, no commentary.
"""


def _format_scenarios_for_prompt(scenarios: list[Scenario]) -> str:
    """Format scenarios as YAML for the LLM to read."""
    records = []
    for s in scenarios:
        records.append({
            "scenario_id": s.scenario_id,
            "parent_theme_id": s.parent_theme_id,
            "direction": s.direction.value,
            "instrument": s.instrument,
            "instrument_class": s.instrument_class.value,
            "thesis_summary": s.thesis_summary,
            "proposed_horizon": s.proposed_horizon.value,
            "proposed_catalyst": s.proposed_catalyst,
            "proposed_kill": s.proposed_kill,
            "key_data_points": [
                {"claim": dp.claim, "source": dp.source} for dp in s.key_data_points
            ],
            "why_this_expression": s.why_this_expression,
            "alternative_expressions": s.alternative_expressions,
        })
    return yaml.dump(records, default_flow_style=False, sort_keys=False)


def run_stage3(
    stage2_output: Stage2Output,
    settings: Settings | None = None,
    run_id: str | None = None,
) -> Stage3Output:
    """Run Stage 3 triage on a batch of scenarios.

    Args:
        stage2_output: Parsed Stage 2 output with scenarios.
        settings: Pipeline settings. Uses defaults if None.
        run_id: Override run ID. Uses stage2_output.run_id if None.

    Returns:
        Stage3Output with verdicts and batch summary.
    """
    if settings is None:
        settings = Settings()

    rid = run_id or stage2_output.run_id
    scenarios = stage2_output.scenarios

    user_message = (
        f"Run ID: {rid}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Scenarios to triage ({len(scenarios)} total):\n\n"
        f"{_format_scenarios_for_prompt(scenarios)}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=settings.pipeline.model,
        max_tokens=8192,
        temperature=settings.pipeline.stage3_temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text
    parsed = json.loads(raw_text)

    verdicts = [ScenarioVerdict(**v) for v in parsed["verdicts"]]
    batch_summary = Stage3BatchSummary(
        **parsed["batch_summary"],
    )

    return Stage3Output(verdicts=verdicts, batch_summary=batch_summary)


def load_stage2_from_yaml(path: str) -> Stage2Output:
    """Load Stage 2 output from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Stage2Output(**data)
