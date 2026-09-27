"""
The Pydantic AI orchestrator.

One `Agent` owns two families of tools:

* fast tools   - answer inline (quotes, arithmetic, glossary)
* heavy tools  - dispatch a job to the compute backend and raise
                 `CallDeferred`. Pydantic AI then ends the run with a
                 `DeferredToolRequests` output; the harness persists the
                 message history and marks the run WAITING_FOR_JOBS.

When the JobManager reports that every job of a run is terminal it calls
`Orchestrator.resume_run`, which rebuilds `DeferredToolResults` from the
stored jobs and calls `agent.run(message_history=..., deferred_tool_results=...)`.
The loop repeats if the model defers again (bounded by `max_run_resumes`).
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import operator
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic_core import to_jsonable_python

from pydantic_ai import (
    Agent,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessagesTypeAdapter,
    RunContext,
    UsageLimits,
)
from pydantic_ai.models import Model

from .config import Settings
from .events import EventBus
from .jobs import JobManager
from .models import Job, JobKind, JobStatus, Run, RunStatus
from .security import signed_headers
from .store import RunStore

log = logging.getLogger("harness.orchestrator")


@dataclass
class HarnessDeps:
    run_id: str
    jobs: JobManager


INSTRUCTIONS = """You are the orchestrator of a quantitative-finance assistant.

Two kinds of tools are available.

Fast tools answer in milliseconds: get_stock_quote, calculate, explain_term.

Heavy tools run on Modal cloud containers and take seconds to minutes:
- run_monte_carlo: many-core CPU container, savings/retirement path simulations.
- price_option_gpu: GPU container, Monte Carlo option pricing.
- run_python_sandbox: isolated container that executes custom Python you write.

How heavy tools work: calling one pauses this conversation. The harness runs the job remotely and resumes you with the tool result attached. You do not need to poll, retry, or apologise for the delay. Call each heavy tool at most once per question unless the user explicitly asks for a comparison.

Prefer a dedicated heavy tool over run_python_sandbox. Use the sandbox only when custom code is genuinely required; the code must assign its answer to a variable named `result` and may use numpy as `np`.

When a heavy tool result arrives, report the key numbers, the compute device and elapsed time found in the result, and the assumptions used. If a job failed, say so plainly, quote the error, and suggest what to change. Keep answers concise and use markdown."""


# ---------------------------------------------------------------------------
# Fast tool helpers
# ---------------------------------------------------------------------------

_MOCK_QUOTES = {
    "NVDA": {"price": 140.50, "high": 142.00, "low": 138.00, "change_percent": 1.2},
    "AMD": {"price": 125.30, "high": 127.00, "low": 123.50, "change_percent": 0.8},
    "TSLA": {"price": 455.00, "high": 462.00, "low": 448.00, "change_percent": 1.5},
    "META": {"price": 595.00, "high": 602.00, "low": 588.00, "change_percent": 0.6},
    "AAPL": {"price": 232.10, "high": 234.00, "low": 229.80, "change_percent": 0.3},
    "GOOGL": {"price": 198.00, "high": 201.00, "low": 195.00, "change_percent": 0.4},
    "SPY": {"price": 610.00, "high": 612.00, "low": 607.00, "change_percent": 0.5},
}

_GLOSSARY = {
    "sharpe ratio": "Risk-adjusted return: (return - risk-free rate) / volatility. Above 1 is good, above 2 is very good.",
    "monte carlo": "Random sampling of many possible paths to estimate the distribution of outcomes.",
    "asian option": "An option whose payoff depends on the average price over the life of the option, not the final price.",
    "var": "Value at Risk: the maximum expected loss over a horizon at a given confidence level.",
    "vwap": "Volume-weighted average price, an intraday fair-value reference used by scalpers.",
    "rsi": "Relative Strength Index, a momentum oscillator; below 30 is oversold, above 70 overbought.",
    "compound interest": "Interest on principal plus previously accumulated interest: A = P(1 + r/n)^(nt).",
}

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(expression: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return _eval(ast.parse(expression, mode="eval"))


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(
    model: str | Model,
    *,
    settings: Settings,
    instrumentation: Any | None = None,
) -> Agent[HarnessDeps, str | DeferredToolRequests]:
    capabilities: list[Any] = []
    if instrumentation is not None:
        from pydantic_ai.capabilities import Instrumentation

        capabilities.append(Instrumentation(settings=instrumentation))

    model_settings = None
    is_anthropic = (isinstance(model, str) and model.startswith("anthropic:")) or (
        type(model).__name__ == "AnthropicModel"
    )
    if is_anthropic:
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        # Adaptive thinking is the default on Claude Opus 5, so `thinking` is left unset.
        # `anthropic_cache=True` caches the (stable) instructions + tool definitions prefix.
        model_settings = AnthropicModelSettings(max_tokens=8192, anthropic_cache=True)

    agent: Agent[HarnessDeps, str | DeferredToolRequests] = Agent(
        model,
        name="harness-orchestrator",
        deps_type=HarnessDeps,
        output_type=[str, DeferredToolRequests],
        instructions=INSTRUCTIONS,
        model_settings=model_settings,
        retries=2,
        capabilities=capabilities,
    )
    _register_tools(agent)
    return agent


async def _defer(ctx: RunContext[HarnessDeps], kind: JobKind, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a heavy job, then suspend the run. Dispatch failures return inline."""
    job = await ctx.deps.jobs.dispatch(
        run_id=ctx.deps.run_id,
        tool_call_id=ctx.tool_call_id or "",
        tool_name=ctx.tool_name or kind.value,
        kind=kind,
        payload=payload,
    )
    if job.status == JobStatus.FAILED:
        return {"status": "failed", "job_id": job.job_id, "error": job.error}
    raise CallDeferred(metadata={"job_id": job.job_id, "kind": kind.value, "backend": job.backend})


def _register_tools(agent: Agent[HarnessDeps, Any]) -> None:
    # -- fast tools -------------------------------------------------------
    @agent.tool_plain
    async def get_stock_quote(symbol: str) -> dict[str, Any]:
        """Get the latest price, high, low and % change for a ticker symbol."""
        symbol = symbol.upper().strip()
        api_key = os.environ.get("FINNHUB_API_KEY")
        if api_key:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get("https://finnhub.io/api/v1/quote", params={"symbol": symbol, "token": api_key})
                    data = resp.json()
                if data.get("c", 0) > 0:
                    return {"symbol": symbol, "price": data["c"], "high": data["h"], "low": data["l"],
                            "change_percent": data.get("dp"), "source": "finnhub"}
            except Exception as exc:  # noqa: BLE001 - fall back to mock data
                log.warning("finnhub lookup failed for %s: %s", symbol, exc)
        quote = _MOCK_QUOTES.get(symbol, {"price": 100.0, "high": 102.0, "low": 98.0, "change_percent": 0.0})
        return {"symbol": symbol, **quote, "source": "mock"}

    @agent.tool_plain
    def calculate(expression: str) -> dict[str, Any]:
        """Evaluate an arithmetic expression (numbers, + - * / ** %, parentheses)."""
        try:
            return {"expression": expression, "result": round(_safe_eval(expression), 6)}
        except Exception as exc:  # noqa: BLE001
            return {"expression": expression, "error": str(exc)}

    @agent.tool_plain
    def explain_term(term: str) -> dict[str, Any]:
        """Explain a finance term from the built-in glossary."""
        key = term.lower().strip()
        for name, definition in _GLOSSARY.items():
            if key == name or key in name or name in key:
                return {"term": name, "definition": definition}
        return {"term": term, "definition": None, "note": f"Not in glossary. Known: {', '.join(_GLOSSARY)}"}

    # -- heavy tools (deferred to Modal) ----------------------------------
    @agent.tool
    async def run_monte_carlo(
        ctx: RunContext[HarnessDeps],
        initial_value: float,
        annual_contribution: float,
        years: int,
        n_paths: int = 200_000,
        mean_return: float = 0.07,
        volatility: float = 0.15,
        target_value: float = 1_000_000,
    ) -> dict[str, Any]:
        """LONG-RUNNING (CPU container on Modal). Simulate a savings plan across many random return paths.
        Returns median / p10 / p90 final value and the probability of reaching target_value."""
        return await _defer(ctx, JobKind.CPU_MONTE_CARLO, {
            "initial_value": initial_value, "annual_contribution": annual_contribution, "years": years,
            "n_paths": n_paths, "mean_return": mean_return, "volatility": volatility, "target_value": target_value,
        })

    @agent.tool
    async def price_option_gpu(
        ctx: RunContext[HarnessDeps],
        spot: float,
        strike: float,
        rate: float = 0.04,
        volatility: float = 0.30,
        maturity_years: float = 1.0,
        option_type: Literal["call", "put"] = "call",
        n_paths: int = 2_000_000,
        steps: int = 252,
    ) -> dict[str, Any]:
        """LONG-RUNNING (GPU container on Modal). Price an arithmetic-average Asian option by Monte Carlo.
        Returns price, standard error, 95% confidence interval and the device used."""
        return await _defer(ctx, JobKind.GPU_PRICING, {
            "spot": spot, "strike": strike, "rate": rate, "volatility": volatility,
            "maturity_years": maturity_years, "option_type": option_type, "n_paths": n_paths, "steps": steps,
        })

    @agent.tool
    async def run_python_sandbox(ctx: RunContext[HarnessDeps], code: str, purpose: str) -> dict[str, Any]:
        """LONG-RUNNING (isolated sandbox container on Modal). Execute custom Python you wrote.
        The code must assign its answer to `result`; numpy is available as `np`. `purpose` is a one-line summary."""
        return await _defer(ctx, JobKind.SANDBOX_CODE, {"code": code, "purpose": purpose})


# ---------------------------------------------------------------------------
# Orchestrator: run / suspend / resume
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, agent: Agent[HarnessDeps, Any], store: RunStore, bus: EventBus,
                 jobs: JobManager, settings: Settings) -> None:
        self.agent = agent
        self.store = store
        self.bus = bus
        self.jobs = jobs
        self.settings = settings
        self._tasks: set[asyncio.Task[Any]] = set()
        jobs.set_resume_callback(self.resume_run)

    # -- public API -------------------------------------------------------
    async def start_run(self, query: str, callback_url: str | None = None) -> Run:
        run = Run(query=query, callback_url=callback_url)
        await self.store.create_run(run)
        await self.jobs.emit(run.run_id, "run.created", query=query)
        self._spawn(self._execute(run.run_id, user_prompt=query))
        return run

    async def resume_run(self, run_id: str) -> None:
        run = await self.store.get_run(run_id)
        if run is None or run.status != RunStatus.RESUMING:
            return
        if run.resume_count >= self.settings.max_run_resumes:
            await self._finish(run, RunStatus.FAILED,
                               error=f"exceeded max resumes ({self.settings.max_run_resumes})")
            return
        run.resume_count += 1
        await self.store.save_run(run)

        results = DeferredToolResults()
        fed: list[str] = []
        for job in await self.store.jobs_for_run(run_id):
            if job.consumed or not job.status.is_terminal:
                continue
            results.calls[job.tool_call_id] = _tool_result_for(job)
            job.consumed = True
            await self.store.save_job(job)
            fed.append(job.job_id)
        if not results.calls:
            await self._finish(run, RunStatus.FAILED, error="no job results available to resume with")
            return

        history = ModelMessagesTypeAdapter.validate_python(run.messages)
        await self._execute(run_id, message_history=history, deferred_results=results, fed_jobs=fed)

    async def cancel_run(self, run_id: str) -> Run | None:
        run = await self.store.get_run(run_id)
        if run is None or run.status.is_terminal:
            return run
        await self.jobs.cancel_jobs_for_run(run_id)
        await self._finish(run, RunStatus.CANCELLED, error="cancelled by client")
        return await self.store.get_run(run_id)

    # -- internals --------------------------------------------------------
    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute(self, run_id: str, *, user_prompt: str | None = None, message_history: Any = None,
                       deferred_results: DeferredToolResults | None = None, fed_jobs: list[str] | None = None) -> None:
        run = await self.store.get_run(run_id)
        if run is None:
            return
        resumed = deferred_results is not None
        run.status = RunStatus.RUNNING
        await self.store.save_run(run)
        if resumed:
            await self.jobs.emit(run_id, "run.resumed", resume_count=run.resume_count, jobs=fed_jobs or [])
        else:
            await self.jobs.emit(run_id, "run.started", model=self.settings.orchestrator_model)

        deps = HarnessDeps(run_id=run_id, jobs=self.jobs)
        try:
            result = await self.agent.run(
                user_prompt,
                message_history=message_history,
                deferred_tool_results=deferred_results,
                deps=deps,
                usage_limits=UsageLimits(request_limit=self.settings.max_model_requests_per_run),
            )
        except Exception as exc:  # noqa: BLE001 - any model/tool failure fails the run, never the server
            log.exception("agent run failed for %s", run_id)
            run = await self.store.get_run(run_id) or run
            await self._finish(run, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            return

        try:
            run = await self.store.get_run(run_id) or run   # refresh: jobs were added meanwhile
            run.messages = to_jsonable_python(result.all_messages())
            run.usage = _merge_usage(run.usage, _usage_dict(result))   # accumulate across resume cycles

            output = result.output
            if isinstance(output, DeferredToolRequests):
                pending = [
                    {"tool_call_id": call.tool_call_id, "tool": call.tool_name,
                     **output.metadata.get(call.tool_call_id, {})}
                    for call in output.calls
                ]
                run.status = RunStatus.WAITING_FOR_JOBS
                await self.store.save_run(run)
                await self.jobs.emit(run_id, "run.suspended", pending=pending)
                # Fast backends may already have finished: settle the race here.
                await self.jobs.check_resume(run_id)
                return

            await self._finish(run, RunStatus.COMPLETED, output=str(output))
        except Exception as exc:  # noqa: BLE001 - never leave a run stuck in RUNNING
            log.exception("post-processing failed for %s", run_id)
            run = await self.store.get_run(run_id) or run
            await self._finish(run, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

    async def _finish(self, run: Run, status: RunStatus, *, output: str | None = None, error: str | None = None) -> None:
        run.status = status
        run.output = output
        run.error = error
        await self.store.save_run(run)
        event = {RunStatus.COMPLETED: "run.completed", RunStatus.FAILED: "run.failed",
                 RunStatus.CANCELLED: "run.cancelled"}.get(status, "run.finished")
        await self.jobs.emit(run.run_id, event, output=output, error=error, resume_count=run.resume_count,
                             usage=run.usage)
        await self._notify_client(run)
        self.bus.close(run.run_id)

    async def _notify_client(self, run: Run) -> None:
        """Optional push to the client's own webhook (signed like the Modal -> harness one)."""
        if not run.callback_url:
            return
        body = json.dumps({"run_id": run.run_id, "status": run.status.value, "output": run.output,
                           "error": run.error, "query": run.query}).encode("utf-8")
        headers = signed_headers(self.settings.harness_webhook_secret, body)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(run.callback_url, content=body, headers=headers)
            await self.jobs.emit(run.run_id, "run.callback_sent", url=run.callback_url, status_code=resp.status_code)
        except Exception as exc:  # noqa: BLE001
            log.warning("client callback failed for %s: %s", run.run_id, exc)
            await self.jobs.emit(run.run_id, "run.callback_failed", url=run.callback_url, error=str(exc))


def _usage_dict(result: Any) -> dict[str, Any]:
    usage = result.usage
    if callable(usage):  # older Pydantic AI versions expose a method
        usage = usage()
    return {
        "requests": getattr(usage, "requests", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "tool_calls": getattr(usage, "tool_calls", None),
    }


def _merge_usage(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in set(previous) | set(current):
        a, b = previous.get(key), current.get(key)
        merged[key] = (a or 0) + (b or 0) if (a is not None or b is not None) else None
    return merged


def _tool_result_for(job: Job) -> dict[str, Any]:
    """The value handed back to the model for a deferred tool call."""
    base = {"job_id": job.job_id, "backend": job.backend, "delivered_via": job.delivered_via}
    if job.status == JobStatus.SUCCEEDED:
        return {"status": "succeeded", **base, "result": job.result}
    return {"status": job.status.value, **base, "error": job.error or job.status.value}
