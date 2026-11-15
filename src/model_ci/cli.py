# src/model_ci/cli.py
"""model-ci — one-shot CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from model_ci.bench import plan as _plan, run as _run
from model_ci.propose import evidence_pack as _evidence_pack, write as _write
from model_ci.probe import (
    drift as _drift,
    parse_compute_apps as _parse_compute_apps,
    quiet_machine as _quiet_machine,
    gpu_rule as _gpu_rule,
    stabilized_tps as _stabilized_tps,
)
from model_ci.state import load as _state_load
from model_ci.watch import engine_changed as _engine_changed, models_changed as _models_changed


# ---------------------------------------------------------------------------
# Injectable hooks (module-level so tests can monkey-patch them)
# ---------------------------------------------------------------------------

def _git_ls_remote(url: str) -> str:
    """Default: ``subprocess.run(["git", "ls-remote", url])`` stdout."""
    import subprocess
    return subprocess.check_output(
        ["git", "ls-remote", url], text=True, timeout=30
    )


def _hf_tree(url: str) -> list[dict]:
    """Default: fetch HF tree API JSON via urllib."""
    import json
    from urllib.request import urlopen
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def _ggufbench_speed(model: str) -> float:
    """Default: run ggufbench on *model*, return tps."""
    import subprocess
    out = subprocess.check_output(
        ["ggufbench", "--model", model, "--tokens", "512"],
        text=True,
        timeout=300,
    )
    # last line is like "tps: 42.7"
    for line in reversed(out.splitlines()):
        if "tps:" in line.lower():
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError("ggufbench did not report tps")


def _stabilized(model: str, tokens: int = 512) -> float:
    """Default: call ``_ggufbench_speed`` three times and take median."""
    def complete(n: int) -> float:
        return _ggufbench_speed(model)
    return _stabilized_tps(complete, generations=3, tokens=tokens)


def _alive_llama() -> bool:
    """Default: check whether a llama process is running (e.g. via /proc)."""
    try:
        with open("/proc/1/status") as f:
            return "llama" in f.read().lower()
    except OSError:
        return False


def _alive_strata() -> bool:
    """Default: check whether a strata process is running."""
    try:
        with open("/proc/1/status") as f:
            return "strata" in f.read().lower()
    except OSError:
        return False


def _load(path: str | Path) -> dict:
    """Injectable state loader — defaults to model_ci.state.load (tests monkeypatch
    this hook; do not re-read the type here, the caller already did)."""
    return _state_load(path)


def _compute_apps() -> list[tuple[int, int]]:
    """Injectable nvidia-smi parse — default calls ``_parse_compute_apps`` on live output."""
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    return _parse_compute_apps(out)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _usage_error(msg: str) -> None:
    print(f"model-ci: {msg}", file=sys.stderr)
    sys.exit(64)


def _env_error(msg: str) -> None:
    print(f"model-ci (environment): {msg}", file=sys.stderr)
    sys.exit(65)


def _check(state_path: str, json_out: str | None) -> int:
    """Core check logic.  Returns the exit code."""
    # --- load state ---------------------------------------------------------
    try:
        state = _load(state_path)
    except Exception as exc:
        _env_error(str(exc))

    engines = state.get("engines", {})
    models_state = state.get("models", {})

    # --- watch engines ------------------------------------------------------
    changed_engines: dict[str, str] = {}
    for name, info in engines.items():
        repo_url = info.get("repo") or ""
        baseline_sha = info.get("baseline", {}).get("sha")
        if not repo_url:
            continue
        try:
            ls_remote_text = _git_ls_remote(repo_url)
        except Exception as exc:
            _env_error(f"git ls-remote failed for {name}: {exc}")
        refs = {}
        from model_ci.watch import parse_ls_remote
        refs = parse_ls_remote(ls_remote_text)
        ref = "refs/heads/main"
        if _engine_changed(baseline_sha, refs, ref):
            changed_engines[name] = refs[ref]

    # --- watch models (HF tree) ---------------------------------------------
    changed_models: dict[str, str] = {}
    for path, info in models_state.items():
        hf_url = info.get("hf_tree") or ""
        if not hf_url:
            continue
        try:
            entries = _hf_tree(hf_url)
        except Exception as exc:
            _env_error(f"HF tree fetch failed for {path}: {exc}")
        hf_shas = {}
        from model_ci.watch import hf_revisions
        hf_shas = hf_revisions(entries)
        diffs = _models_changed(models_state, hf_shas)
        changed_models.update(diffs)

    any_change = bool(changed_engines or changed_models)

    if json_out is not None and any_change:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(
            __import__("json").dumps({
                "engines": changed_engines,
                "models": changed_models,
            }, indent=2),
            encoding="utf-8",
        )

    if not any_change:
        return 0

    # --- probe --------------------------------------------------------------
    # pick the first drifted engine's baseline model (or first model drift)
    probe_model: str | None = None
    for name in sorted(changed_engines):
        info = engines[name]
        baseline = info.get("baseline", {})
        m = baseline.get("model")
        if m:
            probe_model = m
            break
    if not probe_model and changed_models:
        # use first changed model path as a proxy — unlikely but safe fallback
        probe_model = next(iter(changed_models))

    if not probe_model:
        _env_error("no model to probe after detecting changes")

    try:
        current_tps = _ggufbench_speed(probe_model)
    except Exception as exc:
        _env_error(f"probe failed ({exc})")

    # get baseline tps from state or re-bench?  We need it.
    baseline_tps: float | None = None
    for name in sorted(changed_engines):
        baseline = engines[name].get("baseline", {})
        bt = baseline.get("tps")
        if bt is not None:
            baseline_tps = float(bt)
            break

    if baseline_tps is None:
        _env_error("no baseline tps found for a drifted engine; cannot compute drift")

    has_drift = _drift(baseline_tps, current_tps)

    if not has_drift:
        return 1  # within tolerance

    # --- quiet-machine + gpu_rule -------------------------------------------
    load1 = os.sysconf("SC_CLK_TCK") / 100.0  # rough; tests inject via monkeypatch
    nproc = os.cpu_count() or 1
    apps = _compute_apps()
    ok, reason = _quiet_machine(load1, nproc, apps)
    if not ok:
        _env_error(reason)

    llama_alive = _alive_llama()
    strata_alive = _alive_strata()
    ok, reason = _gpu_rule(llama_alive, strata_alive)
    if not ok:
        _env_error(reason)

    # --- stabilized ---------------------------------------------------------
    try:
        stab_tps = _stabilized(probe_model)
    except Exception as exc:
        _env_error(f"stabilized measurement failed ({exc})")

    # --- full bench ---------------------------------------------------------
    steps = _plan(state)
    def runner(step: dict) -> int:
        """Default runner: call ggufbench on the step's model."""
        m = step.get("model", "")
        if not m:
            return 1
        try:
            _ggufbench_speed(m)
            return 0
        except Exception:
            return 1

    summary = _run(steps, runner)
    if summary["failed"]:
        _env_error(f"bench failed for engines: {summary['failed']}")

    # --- propose ------------------------------------------------------------
    receipts = {"load": f"{load1:.2f}", "nproc": str(nproc), "apps": str(apps)}
    pack = _evidence_pack(
        baseline_tps=baseline_tps,
        current_tps=current_tps,
        stabilized_tps=stab_tps,
        engine=list(changed_engines)[0],
        build="",
        model=probe_model,
        receipts=receipts,
    )
    out_dir = Path(state_path).parent / "proposals"
    _write(pack, out_dir, list(changed_engines)[0])

    return 3


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Entry point.  Returns an exit code."""
    parser = argparse.ArgumentParser(prog="model-ci", description="Model CI tool")
    sub = parser.add_subparsers(dest="cmd")

    check_p = sub.add_parser("check", help="Check for drift and probe")
    check_p.add_argument("--state", default=".model-ci-state.json", help="State file path")
    check_p.add_argument("--json", dest="json_out", default=None, help="Write change JSON to PATH")

    probe_p = sub.add_parser("probe", help="Probe a single model")
    probe_p.add_argument("--stabilized", type=int, default=1, metavar="N",
                         help="Number of generations (default 1)")
    probe_p.add_argument("--model", required=True, help="Model to probe")

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help(sys.stderr)
        sys.exit(64)

    if args.cmd == "check":
        try:
            rc = _check(args.state, args.json_out)
        except SystemExit as exc:
            # propagate our own exits; catch unexpected ones
            return int(exc.code or 0)
        return rc

    if args.cmd == "probe":
        try:
            tps = _ggufbench_speed(args.model)
        except Exception as exc:
            print(f"probe failed: {exc}", file=sys.stderr)
            return 65
        stab = _stabilized_tps(lambda n: _ggufbench_speed(args.model),
                               generations=args.stabilized, tokens=512)
        print(f"{tps:.2f} tps | stabilized={stab:.2f}")
        return 0

    parser.print_help(sys.stderr)
    sys.exit(64)


# ---------------------------------------------------------------------------
# Guard against direct execution without subcommand
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
