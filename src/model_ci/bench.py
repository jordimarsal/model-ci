# src/model_ci/bench.py
"""bench — fleet bench plan and sequential execution."""

from __future__ import annotations

from typing import Any

import subprocess
from collections import OrderedDict
from typing import Callable


def _deterministic_order(engines: dict[str, dict[str, Any]]) -> list[str]:
    """Return engines in a fixed order: llamacpp first, then everything else sorted."""
    keys = list(engines.keys())
    if "llamacpp" not in keys:
        return sorted(keys)
    # llamacpp always goes first
    others = [k for k in keys if k != "llamacpp"]
    return ["llamacpp"] + sorted(others)


def plan(state: dict[str, Any]) -> list[dict[str, str]]:
    """Build one step per engine present in state["engines"].

    Each step is::

        {"engine": <name>, "model": <baseline model>, "reason": "drift"}

    Order is deterministic: llamacpp first, then the rest alphabetically.
    """
    engines: dict[str, dict[str, Any]] = state.get("engines", {})
    steps: list[dict[str, str]] = []
    for name in _deterministic_order(engines):
        info = engines[name]
        baseline = info.get("baseline", {})
        model = baseline.get("model") or ""
        steps.append({"engine": name, "model": model, "reason": "drift"})
    return steps


def run(steps: list[dict[str, str]], runner: Callable[[dict[str, str]], int]) -> dict[str, object]:
    """Execute *steps* sequentially via the injected *runner*.

    Stops at the first non-zero exit code (one failed engine must not lie about
    the rest).  Returns a summary with ran/failed lists and the last rc seen.
    """
    ran: list[str] = []
    failed: list[str] = []
    last_rc = 0

    for step in steps:
        engine_name = step["engine"]
        rc = runner(step)
        if rc == 0:
            ran.append(engine_name)
        else:
            failed.append(engine_name)
            last_rc = rc
            break  # stop on first failure

    return {"ran": ran, "failed": failed, "rc": last_rc}
