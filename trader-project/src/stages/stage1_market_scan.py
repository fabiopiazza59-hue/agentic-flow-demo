"""Stage 1 — Market Scan: Journalist + Macro Analyst Lenses.

Two parallel LLM calls with web search. Each lens runs independently — no shared
context. Stage 1 emits the union of their themes with attribution, plus a desk_note
from an aggregator step.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import anthropic
import yaml

from src.config.settings import Settings
from src.models.schemas import (
    LensMetadata,
    Stage1Output,
    Theme,
)

# ── Journalist lens ──

JOURNALIST_SYSTEM = """\
You are a financial journalist covering global markets. Your job is to read the world broadly and identify 3-5 themes that matter for the coming 4-12 weeks. Themes are observations, not trade ideas, not predictions.

## What you read

Central bank statements, major financial press (FT, WSJ, Reuters, Bloomberg, Nikkei), regulatory filings, earnings transcripts, geopolitical events. Focus on the last 7 days but include at least one theme rooted in a multi-week shift.

## Rules

- **Describe state, not forecasts.** "X is happening" — yes. "X will happen" — no. Reject prediction language.
- **Source diversity required.** Cite ≥4 distinct publications across your themes. Single-source themes get confidence: low.
- **Numbers anchor claims.** Every theme needs at least 2 specific numbers (rates, flows, prices, growth %).
- **Contradicting evidence is mandatory.** Every theme must include what would invalidate it. If you can't write one, the theme is too weak.
- **Symmetric direction.** If you produce ≥3 themes, at least one must be short_bias or defensive. Pure long output is a red flag.
- **Max 5 themes.** If you find more, pick your top 5.
- **No opinions of pundits as evidence.** Quote strategists only as positioning indicators.
- **Skew to themes with verifiable, recent triggers** (a speech, a filing, a data release).

## What you must NOT do

- Generate tickers (Stage 2's job).
- Make forecasts disguised as observations.
- Pad sources — each must add a distinct fact.
- Produce 3 themes all from the last 48 hours (recency bias).

## Output format

Respond with valid JSON:

{
  "themes": [
    {
      "theme_id": "T-001",
      "lens": "journalist",
      "title": "≤10 words",
      "summary": "3-5 sentences",
      "direction_implication": "long_bias" | "short_bias" | "mixed" | "neutral",
      "horizon": "weeks" | "months",
      "evidence": [
        {"claim": "...", "source": "publication", "url": "...", "date": "YYYY-MM-DD"}
      ],
      "contradicting_evidence": "1-2 sentences",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "metadata": {
    "articles_consulted": <int>,
    "sources_diversity": <int>
  }
}

Use today's date for evidence items where the exact publication date is unclear. Return ONLY valid JSON.
"""

MACRO_SYSTEM = """\
You are a macro analyst focused on positioning, flows, and market structure. Your job is to identify 3-5 themes about how markets are positioned relative to what's happening. You look at structure, not narrative.

## What you analyze

Rate curves (US, EU, JP), dollar index, credit spreads (HY, IG), VIX/MOVE, sector relative performance, factor returns, CFTC positioning, fund flows. Focus on the last 30 days.

## Rules

- **Describe divergences and regime shifts.** Correlation breakdowns, volatility regime changes, positioning vs. price gaps.
- **Every theme cites ≥1 quantitative indicator** with current vs. baseline number.
- **Source diversity:** ≥3 distinct data providers/indicators across your themes.
- **Contradicting evidence mandatory.** What would invalidate the theme.
- **Distrust narrative-only themes.** If you can't attach a number, drop it.
- **Comfortable with "nothing material this week."** Macro shifts are rare. 2 themes is fine if that's what the data shows.
- **Symmetric direction.** At least one defensive/short_bias theme if you produce ≥3 themes.
- **Max 5 themes.**
- **No contrarian-for-its-own-sake.** Allow agreement with consensus when positioning supports it.

## What you must NOT do

- Generate tickers.
- Make predictions.
- Produce themes without quantitative evidence.
- Declare every consensus view wrong.

## Output format

Respond with valid JSON:

{
  "themes": [
    {
      "theme_id": "T-100",
      "lens": "macro_analyst",
      "title": "≤10 words",
      "summary": "3-5 sentences",
      "direction_implication": "long_bias" | "short_bias" | "mixed" | "neutral",
      "horizon": "weeks" | "months",
      "evidence": [
        {"claim": "...", "source": "data provider", "url": "...", "date": "YYYY-MM-DD"}
      ],
      "contradicting_evidence": "1-2 sentences",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "metadata": {
    "data_points_consulted": <int>,
    "indicators_referenced": ["indicator1", "indicator2"]
  }
}

Use T-1XX IDs for macro themes to avoid collision with journalist T-0XX. Return ONLY valid JSON.
"""


def _run_lens(
    system_prompt: str,
    window_days: int,
    run_date: date,
    settings: Settings,
) -> tuple[list[dict], dict]:
    """Run a single lens and return (themes, metadata)."""
    client = anthropic.Anthropic()

    user_message = (
        f"Today is {run_date.isoformat()}. "
        f"Analyze the last {window_days} days. "
        f"Search the web for current market data, news, and positioning. "
        f"Produce your themes."
    )

    response = client.messages.create(
        model=settings.pipeline.model,
        max_tokens=4096,
        temperature=0.5,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 10,
        }],
    )

    # Extract the final text block (after any tool use)
    text_content = ""
    for block in response.content:
        if block.type == "text":
            text_content = block.text

    # Handle multi-turn tool use if needed
    messages = [{"role": "user", "content": user_message}]
    current_response = response

    while current_response.stop_reason == "tool_use":
        # Collect assistant content and tool results
        messages.append({"role": "assistant", "content": current_response.content})
        tool_results = []
        for block in current_response.content:
            if block.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Search completed.",
                })
        messages.append({"role": "user", "content": tool_results})

        current_response = client.messages.create(
            model=settings.pipeline.model,
            max_tokens=4096,
            temperature=0.5,
            system=system_prompt,
            messages=messages,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 10,
            }],
        )

        for block in current_response.content:
            if block.type == "text":
                text_content = block.text

    parsed = json.loads(text_content)
    return parsed["themes"], parsed.get("metadata", {})


def run_stage1(
    settings: Settings | None = None,
    run_id: str | None = None,
    run_date: date | None = None,
) -> Stage1Output:
    """Run Stage 1 market scan with both lenses.

    Both lenses run sequentially (parallel would require async — v0.2).
    Results are merged into a single Stage1Output.
    """
    if settings is None:
        settings = Settings()
    if run_date is None:
        run_date = date.today()
    if run_id is None:
        run_id = f"run-{run_date.isoformat()}"

    # Run journalist lens
    journalist_themes, journalist_meta = _run_lens(
        JOURNALIST_SYSTEM,
        settings.pipeline.journalist_window_days,
        run_date,
        settings,
    )

    # Run macro analyst lens
    macro_themes, macro_meta = _run_lens(
        MACRO_SYSTEM,
        settings.pipeline.macro_window_days,
        run_date,
        settings,
    )

    # Merge themes
    all_themes = []
    for t in journalist_themes:
        all_themes.append(Theme(**t))
    for t in macro_themes:
        all_themes.append(Theme(**t))

    # Build lens metadata
    lens_metadata = {}
    if journalist_meta:
        lens_metadata["journalist"] = LensMetadata(
            articles_consulted=journalist_meta.get("articles_consulted", 0),
            sources_diversity=journalist_meta.get("sources_diversity", 0),
        )
    if macro_meta:
        lens_metadata["macro_analyst"] = LensMetadata(
            data_points_consulted=macro_meta.get("data_points_consulted", 0),
            indicators_referenced=macro_meta.get("indicators_referenced", []),
        )

    # Aggregator desk note — pure description, no interpretation
    j_count = len(journalist_themes)
    m_count = len(macro_themes)
    j_dirs = [t.get("direction_implication", "unknown") for t in journalist_themes]
    m_dirs = [t.get("direction_implication", "unknown") for t in macro_themes]

    desk_note = (
        f"Journalist found {j_count} themes "
        f"({', '.join(j_dirs)}). "
        f"Macro found {m_count} themes "
        f"({', '.join(m_dirs)}). "
        f"Total: {j_count + m_count} themes forwarded to Stage 2."
    )

    return Stage1Output(
        run_id=run_id,
        run_date=run_date,
        themes=all_themes,
        lens_metadata=lens_metadata,
        desk_note=desk_note,
    )
