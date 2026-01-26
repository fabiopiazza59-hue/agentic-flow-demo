"""
Simple Python Sandbox for CodeAct Agents

Demo-grade sandboxed execution for financial calculations.
Allows: numpy, pandas, math, statistics
Blocks: file I/O, network, dangerous operations

For production, use E2B, Modal, or Docker-based isolation.
"""

import sys
import io
import traceback
import base64
from typing import Any
from dataclasses import dataclass, field
from contextlib import redirect_stdout, redirect_stderr
import signal


@dataclass
class CodeExecutionResult:
    """Result from code execution."""
    success: bool
    output: str = ""
    error: str = ""
    result: Any = None
    charts: list = field(default_factory=list)  # Base64 encoded images
    execution_time_ms: float = 0


# Allowed modules for import
ALLOWED_MODULES = {
    "numpy", "np",
    "pandas", "pd",
    "math",
    "statistics",
    "random",
    "datetime",
    "json",
    "collections",
    "itertools",
    "functools",
}

# Blocked patterns in code
BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import requests",
    "import urllib",
    "import http",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "file(",
    "input(",
    "raw_input(",
    "os.system",
    "os.popen",
    "os.spawn",
    "subprocess.",
    ".read(",
    ".write(",
    "globals()",
    "locals()",
    "__builtins__",
    "__code__",
    "__class__",
]


def check_code_safety(code: str) -> tuple[bool, str]:
    """
    Static analysis to block dangerous patterns.

    Returns:
        (is_safe, error_message)
    """
    code_lower = code.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            return False, f"Blocked operation detected: {pattern}"

    return True, ""


def create_safe_globals() -> dict:
    """Create a restricted globals dict for execution."""
    import numpy as np
    import pandas as pd
    import math
    import statistics
    import random
    import datetime
    import json
    from collections import defaultdict, Counter

    # Initialize numpy string formatting to avoid runtime errors
    np.set_printoptions(legacy='1.25')

    # Safe builtins
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }

    return {
        "__builtins__": safe_builtins,
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "math": math,
        "statistics": statistics,
        "random": random,
        "datetime": datetime,
        "json": json,
        "defaultdict": defaultdict,
        "Counter": Counter,
        # Pre-define result variable
        "result": None,
    }


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def execute_code(
    code: str,
    timeout_seconds: int = 30,
    context: dict = None
) -> CodeExecutionResult:
    """
    Execute Python code in a restricted sandbox.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time
        context: Optional dict of variables to inject

    Returns:
        CodeExecutionResult with output, errors, and result
    """
    import time
    start_time = time.time()

    # Safety check
    is_safe, error_msg = check_code_safety(code)
    if not is_safe:
        return CodeExecutionResult(
            success=False,
            error=error_msg
        )

    # Prepare execution environment (initializes numpy before redirect)
    safe_globals = create_safe_globals()

    # Force numpy to initialize its string formatter before stdout redirect
    import numpy as _np
    _ = str(_np.array([1]))

    # Inject context if provided
    if context:
        for key, value in context.items():
            if key not in BLOCKED_PATTERNS:
                safe_globals[key] = value

    # Capture stdout only (not stderr to avoid numpy issues)
    stdout_capture = io.StringIO()

    # Set timeout (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        with redirect_stdout(stdout_capture):
            exec(code, safe_globals)

        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        execution_time = (time.time() - start_time) * 1000

        # Get result variable if set
        result_value = safe_globals.get("result")

        # Convert pandas/numpy to serializable format
        if hasattr(result_value, "to_dict"):
            result_value = result_value.to_dict()
        elif hasattr(result_value, "tolist"):
            result_value = result_value.tolist()

        return CodeExecutionResult(
            success=True,
            output=stdout_capture.getvalue(),
            result=result_value,
            execution_time_ms=execution_time
        )

    except TimeoutError as e:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        return CodeExecutionResult(
            success=False,
            error=f"Execution timed out after {timeout_seconds} seconds"
        )
    except Exception as e:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        return CodeExecutionResult(
            success=False,
            error=f"{type(e).__name__}: {str(e)}",
            output=stdout_capture.getvalue()
        )


# Example usage
if __name__ == "__main__":
    # Test Monte Carlo simulation (np, pd are pre-loaded)
    test_code = '''
# Monte Carlo retirement simulation
initial_savings = 100000
annual_contribution = 12000
years = 30
num_simulations = 1000

# Simulate returns (mean 7%, std 15%)
final_values = []
for _ in range(num_simulations):
    balance = initial_savings
    for year in range(years):
        annual_return = np.random.normal(0.07, 0.15)
        balance = balance * (1 + annual_return) + annual_contribution
    final_values.append(balance)

final_values = np.array(final_values)

result = {
    "simulations": num_simulations,
    "years": years,
    "median_final_value": float(np.median(final_values)),
    "percentile_10": float(np.percentile(final_values, 10)),
    "percentile_90": float(np.percentile(final_values, 90)),
    "probability_over_1m": float(np.mean(final_values > 1000000) * 100),
}

print(f"Median final value: ${result['median_final_value']:,.0f}")
print(f"10th percentile: ${result['percentile_10']:,.0f}")
print(f"90th percentile: ${result['percentile_90']:,.0f}")
print(f"Probability of reaching $1M: {result['probability_over_1m']:.1f}%")
'''

    result = execute_code(test_code)
    print(f"\nSuccess: {result.success}")
    print(f"Output:\n{result.output}")
    print(f"Result: {result.result}")
    print(f"Time: {result.execution_time_ms:.0f}ms")
