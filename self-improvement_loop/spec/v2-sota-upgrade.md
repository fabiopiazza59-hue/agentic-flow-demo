# v2 — from an agent that *talks* about learning to a forecasting system that *measures* it

**Status:** implemented (2026-09-27). **Builds on:**
[`improvements-from-2609.05663.md`](improvements-from-2609.05663.md) (the arXiv:2609.05663 review,
whose P0.1/P0.2 integrity fixes ship with this change).
**Reproduce every number:** `python -m src.loop.run_ab --mode report --no-commit`, then read
`results/metrics.json`; the timing/anchor numbers come from the Actions API and Yahoo 1-minute
bars, as described inline.

---

## 0. The short version

The loop never had the information, the timing, or the statistics to answer its own question
("can an LLM desk beat a random walk on AMZN's close?"):

| Finding | Evidence |
|---|---|
| **Most of the record is not a forecast.** 40 of 74 live arm-A rows were written after the 09:30 ET open — *every* row since 2026-08-27, median +99 min. | GitHub started the 11:30 UTC cron at 15:05–17:10 UTC (Actions API, runs 73–84); each run took 2–3 min. |
| **The "pre-market quote" was the previous close, 49 of 49 times.** Analysts were told `premarket_gap_pct: 0.0` — a fabricated "flat pre-market", not missing data. | `prior.center == prior.prev_close` on every arm-B row; Alpha Vantage's free `GLOBAL_QUOTE` has no extended-hours data. |
| **The strongest predictor was free and unused.** Over the last 60 sessions, "close = the 09:20 ET pre-market trade" scored MAPE **1.11%**; 04:30 ET: 1.27%; the prior close: 1.56%. Arm A scored 1.38% on the same days *even with* its post-open lookahead. | Yahoo 5-minute bars with `prepost=True`, 2026-07-02 → 2026-09-25. |
| **On clean rows there is no edge.** Arm A vs the random walk: +0.01% MAPE, anytime-valid 95% CS [−0.155%, +0.182%] (34 pre-open days). A vs B: undecided (26 clean paired days). No aggregation rule beats the plain mean. | `RESULTS.md` scoreboard, lab and A/B sections. |
| **Reading a p-value every day manufactures edges.** Under the null, a daily fixed-sample test "finds" significance within a year **40%** of the time; the confidence sequence used now: **3%**. | Simulation, 2,000 null paths × 250 daily looks (normal and t₃ noise). |
| **At one forecast a day, realistic edges are undetectable for years.** The daily APE gain has SD 0.305%; a true 0.05% edge needs ~357 clean forecasts (1.4 years) to confirm, a 0.02% edge ~10 years. | `cs_radius` in `src/evals/sequential.py` on the clean arm-A deltas. |

v2 fixes the first five in code. The sixth is an experiment-design decision for the owner (§4).

---

## 1. What changed, mechanism by mechanism

### 1.1 Timing: research in the evening, re-anchor in the morning — `src/loop/session.py`
GitHub's scheduler cannot be trusted near the bell, so work is split by urgency:

* **Research** (every LLM call) for session *T* runs in the window from the previous close
  (+30 min for the official close to settle) until 10 minutes before *T* opens. The evening run
  (21:41 UTC, plus a 01:17 UTC fallback) does it hours before the bell even when GitHub starts it
  late. The morning runs are a fallback.
* **Re-anchoring** (code only, no LLM cost): every later run that still lands before the open
  moves the pending forecast onto the latest after-hours / pre-market trade, keeping the models'
  view *relative to the anchor* (`reanchor_row` scales every price level by new/old anchor).
* A run during a session, or after the refresh cutoff, can only score and report. Research
  refuses to write a live row within 10 min of the open (`--allow-late` overrides and flags it).
* The session calendar is DST-, holiday- and early-close-aware (`market_calendar.session_times`):
  the open is 13:30 UTC in summer and **14:30 UTC in winter**, which the old fixed-13:30 check
  would have mislabelled from November on.
* Scoring waits until the close has settled, scores **every** pending row (not only the newest),
  and a forecast is only built once a provider actually has the previous session's close
  (`get_history_through`), so a lagging feed can no longer shift the prior close by a day.

### 1.2 The free baseline — `src/data/providers.py:get_anchor`
The anchor is the latest extended-hours trade after the previous close (Yahoo 1-minute bars,
pre/post-market; Finnhub if keyed), sanity-banded at ±35% (a real earnings gap on this name
exceeded 15% on 2026-07-31). With no trade, the anchor is the prior close and the features say
the gap is **unknown** rather than 0.0. Arm B's prior centers on a live anchor; arm A's gates
shrink toward it. Every row records `anchor`, `anchor_source`, `anchor_time`, `anchor_live`,
`anchored_at`, `anchor_updates`, and scoring adds `anchor_ape`. **The headline verdict now asks
whether the model beats this free baseline** — beating the prior close with information the
market already priced in is not skill.

### 1.3 Probabilistic forecasts and proper scores — `src/evals/probabilistic.py`
Every forecaster (both arms and both baselines) becomes a predictive distribution
`close_q(τ) = center · exp(σ · z_τ)`: σ from an EWMA of log returns (RiskMetrics, λ = 0.94),
z_τ from **symmetric split-conformal** quantiles of that forecaster's own past standardized
residuals (Student-t(5) until 30 exist). Calibration sets the width only; the center stays the
forecaster's point, so calibration cannot smuggle in a trend bet (an asymmetric version was tried
and inherited the sell-off's drift: P(up) 0.39 for an up-call). Scores: CRPS (strictly proper;
Gneiting & Raftery 2007), 50/80/90% interval coverage, Brier of P(close > prior close). A row's
`confidence` is now P(PASS) under this distribution — the old heuristic scored worse than a
constant (review F5). Old rows get distributions from information available before their
session only (tested: changing a later outcome never changes an earlier score).

### 1.4 Anytime-valid verdicts — `src/evals/sequential.py`
Every verdict (edge vs baselines, A vs B, CRPS, direction, lab decisions) reads an asymptotic
confidence sequence (Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas, *Annals of Statistics*
2024; arXiv:2103.06476) on the paired daily differences over **all-time pre-open** rows. It is
valid at every sample size simultaneously, so re-reading RESULTS.md daily keeps the 5% error
rate (simulated: 2.6–3.1% false alarms over a year of daily looks, 98% power for a 0.3-SD
effect). This replaces "pre-register a horizon" (review P1.2) with a method that needs none.
The fixed-sample bootstrap CI and sign test stay on the page as descriptive numbers.

### 1.5 Arm A v2: champion blend + bounded judge — `src/loop/run_daily.py:do_predict`
`analysts → champion aggregation rule (code) → judge's σ-sized adjustment, clamped to ±0.5σ in
code → guardrail gates → calibrated distribution`. The judge no longer writes the number
(review F1: that cost 0.14% MAPE against a plain mean over 60 days); it sizes a deviation, the
pattern arm B already used. Every stage is logged (`predicted_close_blend`,
`judge_adjustment_sigma_raw`, `predicted_close_pre_gates`, `predicted_close_raw`,
`predicted_close`), so each one's paired effect is measurable.

### 1.6 Self-improvement by measured selection — `src/evals/lab.py`
The loop's learning channel was text (post-mortems → prompts), which the review measured as
inert (F4) or harmful (F1). The lab makes it a mechanism:

* Six aggregation rules (mean, median, trimmed mean, inverse-MSE, best-recent, 50% shrinkage)
  are replayed over every pre-open day using only earlier days — free, no lookahead.
* A rule replaces the equal-weight mean only when its confidence sequence against the mean,
  Bonferroni-corrected across challengers, excludes zero. The mean is the right default: simple
  averages keep beating estimated weights (the *forecast combination puzzle*; Stock & Watson 2004,
  explained by weight-estimation error in Smith & Wallis 2009).
* The same test switches the judge or the gates **off** once their paired effect is shown to be
  negative. `learnings/lab_state.json` is rewritten every run and read by the next forecast.
* Today: no challenger has proven better (34 days); the v1 judge pipeline trails the mean by
  0.17% on clean days (CS [−0.41%, +0.06%], not yet conclusive).

### 1.7 Smaller integrity fixes
* News analyst and arm-B pulse record `web_results` — server-tool failures return as content,
  not exceptions, so an empty search used to look like research (review P1.4).
* Prompts name the target session and the forecast time, since research may run the evening
  before; features exclude the raw anchor dict from prompts.
* The Pages deploy runs only when a run committed something; charts are optional, so a blocked
  CDN no longer blanks the verdict.

---

## 2. What v2 deliberately does not do

* **No model upgrades.** The paper's model league found frontier models statistically
  indistinguishable at a 25× cost spread (§8.4), and switching models mid-record would confound
  every comparison. Model choice is a lab question (roadmap R4), not a default.
* **No more prompt context.** The review's §3 and the paper's §7 both point the other way.
* **No promotion without anytime-valid evidence**, including for changes this document
  recommends.

---

## 3. What to expect from the record now

* The first v2 row is written the evening after deploy; from then on, pre-open coverage should
  approach 100% (it was 0 of the last 20 sessions). RESULTS.md reports it.
* Arm-A and arm-B MAPE should drop toward the anchor baseline's (~1.1–1.3%) because both now see
  the pre-open price. That is information, not skill — which is why the verdict moved to the
  anchor baseline. The v1 record stays in the ledger, labelled, and still counts for the lab.

---

## 4. Roadmap — the owner's calls, ranked by information per dollar

| # | Change | Why | Cost |
|---|---|---|---|
| R1 | **Scale N: a panel of 20–50 liquid names**, each with the same pipeline and baselines; evaluate cross-sectionally (rank IC, CRPS skill per name). | The binding constraint is statistical power (§0). 50 names turn a 1.4-year wait for a 0.05% edge into weeks (less if names co-move — a market-neutral target helps). | ~N× analyst calls; mitigated by cheaper models (R4) and dropping null analysts (R6). |
| R2 | **Punctual trigger**: call `workflow_dispatch` at a fixed pre-open time from an external scheduler (a fine-grained token with `actions:write`). | Fixes the information set (same anchor time daily) instead of "whenever GitHub starts the run". | Free; needs a secret the owner creates. |
| R3 | **Null arm A′**: arm A again on the same snapshot with a different sample. | The A-vs-A′ spread is the noise floor any A/B effect must exceed (review P1.1). | +1 arm-A run per day. |
| R4 | **Model league on logged snapshots**: replay stored analyst outputs through cheaper judges; run cheaper analysts as a shadow arm. | Spend savings on N (R1), which is what the statistics lack. | Small, one-off. |
| R5 | **Distributional elicitation**: ask analysts for quantiles, not a point, and let the lab test quantile aggregation. | The scoring is already distributional; the models are not yet. | Prompt + parser change. |
| R6 | **Prune by evidence**: expire STRATEGY.md notes that never showed a measured effect (review P2.2); drop analysts whose leave-one-out effect is null (P2.3). | Less context, less cost, same accuracy. | Small. |
| R7 | **Change the target where predictability is documented**: volatility / range (well-established persistence), or event days (earnings), instead of the level of one close. | Returns are close to unpredictable: the live M6 competition (100 assets) reports "great difficulty" forecasting relative performance and consistently outperforming the market. | Experiment redesign. |

---

## 5. References

* Barton et al. (2026). *What LLM Trading Agents Actually Do in Production.* arXiv:2609.05663 —
  as applied in `improvements-from-2609.05663.md`.
* Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas (2024). Time-uniform central limit theory and
  asymptotic confidence sequences. *Annals of Statistics* 52(6). arXiv:2103.06476.
* Gneiting & Raftery (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*.
* Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). Distribution-free predictive inference for
  regression. *JASA* (split and locally weighted conformal prediction).
* Stock & Watson (2004). Combination forecasts of output growth in a seven-country data set.
  *Journal of Forecasting*. Smith & Wallis (2009). A simple explanation of the forecast
  combination puzzle. *Oxford Bulletin of Economics and Statistics* 71: 331–355.
* J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document* (EWMA volatility, λ = 0.94).
* Makridakis, Spiliotis, Hollyman, Petropoulos, Swanson & Gaba (2024). The M6 forecasting
  competition: Bridging the gap between forecasting and investment decisions. *International
  Journal of Forecasting*. arXiv:2310.13357.
* Glasserman & Lin (2023). Assessing look-ahead bias in stock return predictions generated by GPT
  sentiment analysis. arXiv:2309.17322. See also *Look-Ahead-Bench* (2026), arXiv:2601.13770.
* *LLM-based Agents for Forecasting and Prediction: Methods, Training, Evaluation, and
  Applications* (2026 survey), arXiv:2608.23058 — hybrid LLM + statistical designs, and
  ablations in which the LLM component does not help.
