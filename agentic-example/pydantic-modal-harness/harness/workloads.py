"""
Compute workloads.

Pure functions with no harness imports so the *same* code runs:
- inside Modal containers (CPU / GPU images), and
- in-process in local mode (dev + tests).

Every workload takes a JSON-serialisable payload and returns a
JSON-serialisable dict. Keep them deterministic when a `seed` is given.
"""

from __future__ import annotations

import io
import math
import time
from contextlib import redirect_stdout
from typing import Any, Callable


# ---------------------------------------------------------------------------
# CPU: vectorised retirement Monte Carlo (numpy)
# ---------------------------------------------------------------------------

def cpu_monte_carlo(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Simulate a savings plan under random annual returns.

    payload keys (all optional):
      initial_value, annual_contribution, years, n_paths,
      mean_return, volatility, target_value, seed
    """
    import numpy as np

    initial = float(payload.get("initial_value", 100_000))
    contribution = float(payload.get("annual_contribution", 12_000))
    years = int(payload.get("years", 30))
    n_paths = int(payload.get("n_paths", 200_000))
    mean_return = float(payload.get("mean_return", 0.07))
    vol = float(payload.get("volatility", 0.15))
    target = float(payload.get("target_value", 1_000_000))
    seed = payload.get("seed")

    n_paths = max(1_000, min(n_paths, 5_000_000))
    years = max(1, min(years, 80))

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    balance = np.full(n_paths, initial, dtype=np.float64)
    for _ in range(years):
        returns = rng.normal(mean_return, vol, n_paths)
        balance = balance * (1.0 + returns) + contribution

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "workload": "cpu_monte_carlo",
        "inputs": {
            "initial_value": initial, "annual_contribution": contribution, "years": years,
            "n_paths": n_paths, "mean_return": mean_return, "volatility": vol, "target_value": target,
        },
        "median_final_value": float(np.median(balance)),
        "mean_final_value": float(np.mean(balance)),
        "p10": float(np.percentile(balance, 10)),
        "p90": float(np.percentile(balance, 90)),
        "probability_reaching_target": float(np.mean(balance >= target)),
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# GPU: Monte Carlo Asian option pricing (torch, CUDA when available)
# ---------------------------------------------------------------------------

def gpu_pricing(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Price an arithmetic-average Asian option by Monte Carlo on the GPU.

    payload keys (all optional):
      spot, strike, rate, volatility, maturity_years, steps, n_paths,
      option_type ("call"|"put"), seed
    Falls back to CPU torch, then to numpy, so it also runs in local mode.
    """
    spot = float(payload.get("spot", 100.0))
    strike = float(payload.get("strike", 100.0))
    rate = float(payload.get("rate", 0.04))
    vol = float(payload.get("volatility", 0.30))
    maturity = float(payload.get("maturity_years", 1.0))
    steps = int(payload.get("steps", 252))
    n_paths = int(payload.get("n_paths", 2_000_000))
    option_type = str(payload.get("option_type", "call")).lower()
    seed = payload.get("seed")

    n_paths = max(10_000, min(n_paths, 50_000_000))
    steps = max(1, min(steps, 2_000))
    dt = maturity / steps
    drift = (rate - 0.5 * vol * vol) * dt
    diffusion = vol * math.sqrt(dt)
    discount = math.exp(-rate * maturity)

    started = time.perf_counter()
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(int(seed))
        # Chunk to keep memory bounded on big path counts
        chunk = 1_000_000 if device == "cuda" else 100_000
        payoff_sum = torch.zeros((), dtype=torch.float64, device=device)
        payoff_sq_sum = torch.zeros((), dtype=torch.float64, device=device)
        remaining = n_paths
        while remaining > 0:
            size = min(chunk, remaining)
            log_s = torch.full((size,), math.log(spot), dtype=torch.float32, device=device)
            avg = torch.zeros((size,), dtype=torch.float32, device=device)
            for _ in range(steps):
                z = torch.randn((size,), generator=gen, dtype=torch.float32, device=device)
                log_s = log_s + drift + diffusion * z
                avg = avg + torch.exp(log_s)
            avg = avg / steps
            payoff = torch.clamp(avg - strike, min=0.0) if option_type == "call" else torch.clamp(strike - avg, min=0.0)
            payoff = payoff.to(torch.float64)
            payoff_sum += payoff.sum()
            payoff_sq_sum += (payoff * payoff).sum()
            remaining -= size
        mean = (payoff_sum / n_paths).item()
        var = (payoff_sq_sum / n_paths).item() - mean * mean
        backend = f"torch:{device}"
        if device == "cuda":
            backend += f":{torch.cuda.get_device_name(0)}"
    except ImportError:
        import numpy as np

        rng = np.random.default_rng(seed)
        chunk = 100_000
        payoff_sum = 0.0
        payoff_sq_sum = 0.0
        remaining = n_paths
        while remaining > 0:
            size = min(chunk, remaining)
            log_s = np.full(size, math.log(spot))
            avg = np.zeros(size)
            for _ in range(steps):
                log_s = log_s + drift + diffusion * rng.standard_normal(size)
                avg += np.exp(log_s)
            avg /= steps
            payoff = np.maximum(avg - strike, 0.0) if option_type == "call" else np.maximum(strike - avg, 0.0)
            payoff_sum += float(payoff.sum())
            payoff_sq_sum += float((payoff * payoff).sum())
            remaining -= size
        mean = payoff_sum / n_paths
        var = payoff_sq_sum / n_paths - mean * mean
        backend = "numpy:cpu"

    price = discount * mean
    std_err = discount * math.sqrt(max(var, 0.0) / n_paths)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "workload": "gpu_pricing",
        "inputs": {
            "spot": spot, "strike": strike, "rate": rate, "volatility": vol, "maturity_years": maturity,
            "steps": steps, "n_paths": n_paths, "option_type": option_type,
        },
        "price": round(price, 4),
        "std_error": round(std_err, 5),
        "confidence_95": [round(price - 1.96 * std_err, 4), round(price + 1.96 * std_err, 4)],
        "device": backend,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Sandboxed code (local fallback). On Modal this kind runs in modal.Sandbox.
# ---------------------------------------------------------------------------

_BLOCKED_TOKENS = (
    "import os", "import sys", "import subprocess", "import socket", "import shutil",
    "import requests", "import urllib", "import http", "__import__", "open(", "eval(", "exec(",
    "compile(", "input(", "globals(", "locals(", "__builtins__", "__class__", "__subclasses__",
)


def _safe_globals() -> dict[str, Any]:
    import numpy as np
    import statistics
    import random
    import json as _json

    safe_builtins = {
        name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
        for name in (
            "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "int",
            "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed",
            "round", "set", "sorted", "str", "sum", "tuple", "zip",
        )
    }
    return {
        "__builtins__": safe_builtins, "np": np, "numpy": np, "math": math, "statistics": statistics,
        "random": random, "json": _json, "result": None,
    }


def sandbox_code_local(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Dev-only restricted `exec`. Real isolation happens on Modal
    (`modal.Sandbox` with no network, own filesystem, CPU/time limits).
    """
    code = str(payload.get("code", ""))
    if not code.strip():
        return {"workload": "sandbox_code", "success": False, "error": "empty code", "device": "local-exec"}
    lowered = code.lower()
    for token in _BLOCKED_TOKENS:
        if token in lowered:
            return {"workload": "sandbox_code", "success": False, "error": f"blocked token: {token}", "device": "local-exec"}

    started = time.perf_counter()
    scope = _safe_globals()
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(code, scope)  # noqa: S102 - dev-only path, see docstring
        value = scope.get("result")
        if hasattr(value, "tolist"):
            value = value.tolist()
        return {
            "workload": "sandbox_code", "success": True, "stdout": buffer.getvalue()[-8000:], "result": value,
            "device": "local-exec", "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:  # noqa: BLE001 - report to the model
        return {
            "workload": "sandbox_code", "success": False, "stdout": buffer.getvalue()[-8000:],
            "error": f"{type(exc).__name__}: {exc}", "device": "local-exec",
        }


# Program executed *inside* a modal.Sandbox. Reads code from stdin, prints JSON.
SANDBOX_DRIVER = r'''
import io, json, sys, time, traceback
from contextlib import redirect_stdout
code = sys.stdin.read()
scope = {"__name__": "__sandbox__", "result": None}
buf = io.StringIO()
started = time.perf_counter()
try:
    with redirect_stdout(buf):
        exec(code, scope)
    value = scope.get("result")
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        json.dumps(value)
    except TypeError:
        value = repr(value)
    out = {"success": True, "stdout": buf.getvalue()[-8000:], "result": value}
except Exception as exc:
    out = {"success": False, "stdout": buf.getvalue()[-8000:], "error": f"{type(exc).__name__}: {exc}",
           "traceback": traceback.format_exc()[-4000:]}
out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
print("__HARNESS_RESULT__" + json.dumps(out))
'''


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WORKLOADS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "cpu_monte_carlo": cpu_monte_carlo,
    "gpu_pricing": gpu_pricing,
    "sandbox_code": sandbox_code_local,
}


def execute_workload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        fn = WORKLOADS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown workload kind: {kind}") from exc
    return fn(payload)
