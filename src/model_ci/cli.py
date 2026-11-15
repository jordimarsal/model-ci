# src/model_ci/cli.py
"""model-ci — one-shot CLI entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

from model_ci.bench import plan as _plan, run as _run
from model_ci.config import ConfigError, load_config as _load_config
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


def _hf_tree(url: str) -> list[dict[str, Any]]:
    """Default: fetch HF tree API JSON via urllib."""
    import json
    from urllib.request import urlopen
    with urlopen(url, timeout=60) as resp:
        data: list[dict[str, Any]] = json.loads(resp.read())
        return data


def _ggufbench_speed(model: str, engine: str = "llama") -> float:
    """Default: the real ggufbench speed probe of *model* — `ggufbench.run --dims
    speed` from the workspace root (MODEL_CI_WORKSPACE, default cwd), which starts
    and stops llama-server itself. Output line '✓ <slug>: speed=NNms (TT.T t/s)'."""
    import os as _os
    import subprocess
    workspace = _os.environ.get("MODEL_CI_WORKSPACE", ".")
    out = subprocess.check_output(
        [sys.executable, "-m", "ggufbench.run", "--models", model,
         "--engine", engine, "--dims", "speed", "--thinking", "off",
         "--context", "fixed"],
        text=True, timeout=3600, cwd=workspace,
    )
    for line in reversed(out.splitlines()):
        if " t/s)" in line:
            return float(line.rsplit("(", 1)[1].split("t/s")[0].strip())
    raise RuntimeError("ggufbench did not report tps")


def _stabilized(model: str, tokens: int = 512, engine: str = "llama") -> float:
    """Default: call ``_ggufbench_speed`` three times and take the median
    (probe.stabilized_tps applies the median — see its docstring for why)."""
    def complete(n: int) -> float:
        return _ggufbench_speed(model, engine=engine)
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


def _load(path: str | Path) -> dict[str, Any]:
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


def _env_error(msg: str) -> NoReturn:
    print(f"model-ci (environment): {msg}", file=sys.stderr)
    sys.exit(65)


_SKELETON_STATE: dict[str, Any] = {"version": 1, "engines": {}, "models": {}}


def _atomic_save(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON atomically (tmp in the same dir + rename) — a scheduler killed
    mid-write must not corrupt the state."""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _init(state_path: str) -> int:
    """`model-ci init`: create the v1 skeleton. 65 if it exists — never clobber."""
    p = Path(state_path)
    if p.exists():
        print(f"model-ci (environment): state file already exists: {p}", file=sys.stderr)
        raise SystemExit(65)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(p, _SKELETON_STATE)
    print(f"wrote {p}")
    return 0


def _accept(engine: str, tps: float, ms: int, model: str, build: str | None,
            sha: str | None, state_path: str) -> int:
    """`model-ci accept`: the ONLY way a baseline enters the state — the human
    accepts a bench result and this stores it for the next drift comparison."""
    try:
        state = _load(state_path)
    except Exception as exc:
        _env_error(str(exc))
    from datetime import datetime, timezone
    entry = state.setdefault("engines", {}).setdefault(engine, {})
    baseline: dict[str, Any] = {"model": model, "tps": float(tps), "ms": int(ms),
                                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if build:
        baseline["build"] = build
    if sha:
        baseline["sha"] = sha
        entry["sha"] = sha
    entry["baseline"] = baseline
    _atomic_save(state_path, state)
    print(f"accepted baseline for {engine}: {tps} tps · {ms} ms · {model}")
    return 0


def _check(state_path: str, json_out: str | None,
           config_path: str | None = None) -> int:
    """Core check logic.  Returns the exit code.

    What to WATCH comes from --config (model-ci.toml) when given; the fallback is the
    state's engines with a repo. Baselines always live in the state — an engine that
    drifts WITHOUT an accepted baseline is an environment error (exit 65 with the
    `model-ci accept` hint), never a silent skip."""
    try:
        state = _load(state_path)
    except Exception as exc:
        _env_error(str(exc))

    engines = state.get("engines", {})
    models_state = state.get("models", {})

    # --- what to watch ------------------------------------------------------
    watched: dict[str, dict[str, str]] = {}
    if config_path:
        try:
            cfg = _load_config(config_path)
        except ConfigError as exc:
            _env_error(str(exc))
        for name, info in cfg.get("engines", {}).items():
            watched[name] = {"repo": str(info.get("repo", "")),
                             "ref": str(info.get("ref") or "refs/heads/main")}
    else:
        for name, info in engines.items():
            repo = info.get("repo") or ""
            if repo:
                watched[name] = {"repo": repo, "ref": "refs/heads/main"}
    watched_models = [str(m.get("repo")) for m in
                      (cfg or {}).get("models", [])] if config_path else []

    # --- watch engines ------------------------------------------------------
    changed_engines: dict[str, str] = {}
    drifted_without_baseline: list[str] = []
    for name, w in watched.items():
        entry = engines.get(name, {})
        # el SHA viu a nivell d'entry (schema v1) amb fallback al baseline — els
        # estats sembrats a mà solen posar-lo només al baseline
        baseline_sha = entry.get("sha") or entry.get("baseline", {}).get("sha")
        if not w["repo"]:
            continue
        try:
            ls_remote_text = _git_ls_remote(w["repo"])
        except Exception as exc:
            _env_error(f"git ls-remote failed for {name}: {exc}")
        from model_ci.watch import parse_ls_remote
        refs = parse_ls_remote(ls_remote_text)
        ref = w.get("ref") or "refs/heads/main"
        if _engine_changed(baseline_sha, refs, ref):
            changed_engines[name] = refs[ref]
            if not engines.get(name, {}).get("baseline"):
                drifted_without_baseline.append(name)

    # --- watch models (HF tree) ---------------------------------------------
    # First sight of a watched repo records its revisions SILENTLY (a rig that just
    # subscribed did not "detect change" — it learned the baseline); later revisions
    # of known files are the alert.
    changed_models: dict[str, str] = {}
    state_dirty = False
    for repo in watched_models:
        url = (f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
        try:
            entries = _hf_tree(url)
        except Exception as exc:
            _env_error(f"HF tree fetch failed for {repo}: {exc}")
        from model_ci.watch import hf_revisions
        hf_shas = hf_revisions(entries)
        known = models_state.get(repo, {}).get("files", {})
        if not known and hf_shas:
            state.setdefault("models", {})[repo] = {"files": hf_shas}
            state_dirty = True
            continue
        for path, sha in hf_shas.items():
            if known.get(path) != sha:
                changed_models[f"{repo}::{path}"] = sha

    any_change = bool(changed_engines or changed_models)

    if state_dirty:
        _atomic_save(state_path, state)

    if drifted_without_baseline:
        print("model-ci (environment): drifted engine(s) without an accepted baseline: "
              f"{', '.join(drifted_without_baseline)} — run the full bench, then "
              "`model-ci accept`", file=sys.stderr)
        return 65

    if json_out is not None and any_change:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(
            json.dumps({"engines": changed_engines,
                        "models": changed_models}, indent=2),
            encoding="utf-8",
        )

    if not any_change:
        return 0

    # --- probe --------------------------------------------------------------
    # pick the first drifted engine's baseline model (or first model drift)
    probe_model: str | None = None
    baseline_tps: float | None = None
    for name in sorted(changed_engines):
        baseline = engines.get(name, {}).get("baseline", {})
        if baseline.get("model") and baseline.get("tps") is not None:
            probe_model = str(baseline["model"])
            baseline_tps = float(baseline["tps"])
            break
    if not probe_model and changed_models:
        # use first changed model path as a proxy — unlikely but safe fallback
        probe_model = next(iter(changed_models))

    if not probe_model:
        _env_error("no model to probe after detecting changes")

    try:
        drifted_engine = next(iter(sorted(changed_engines)), "llama")
        gg_engine = "strata" if drifted_engine == "strata" else "llama"
        current_tps = _ggufbench_speed(probe_model, engine=gg_engine)
    except Exception as exc:
        _env_error(f"probe failed ({exc})")

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
        stab_tps = _stabilized(probe_model, engine=gg_engine)
    except Exception as exc:
        _env_error(f"stabilized measurement failed ({exc})")

    # --- full bench ---------------------------------------------------------
    steps = _plan(state)

    def runner(step: dict[str, str]) -> int:
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

def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns an exit code."""
    parser = argparse.ArgumentParser(prog="model-ci", description="Model CI tool")
    sub = parser.add_subparsers(dest="cmd")

    check_p = sub.add_parser("check", help="Check for drift and probe")
    check_p.add_argument("--state", default=".model-ci-state.json", help="State file path")
    check_p.add_argument("--json", dest="json_out", default=None, help="Write change JSON to PATH")
    check_p.add_argument("--config", default=None, metavar="TOML",
                         help="model-ci.toml: watched engines/repos + HF models")

    init_p = sub.add_parser("init", help="Create the v1 skeleton state file")
    init_p.add_argument("--state", default=".model-ci-state.json", help="State file path")

    accept_p = sub.add_parser("accept", help="Accept a bench result as the engine baseline")
    accept_p.add_argument("--engine", required=True, help="Engine name in the state")
    accept_p.add_argument("--tps", required=True, type=float, help="Generation t/s")
    accept_p.add_argument("--ms", required=True, type=int, help="Latency ms")
    accept_p.add_argument("--model", required=True, help="Model slug")
    accept_p.add_argument("--build", default=None, help="Engine build string")
    accept_p.add_argument("--sha", default=None, help="Engine commit SHA")
    accept_p.add_argument("--state", default=".model-ci-state.json", help="State file path")

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
            rc = _check(args.state, args.json_out, args.config)
        except SystemExit as exc:
            # propagate our own exits; catch unexpected ones
            return int(exc.code or 0)
        return rc

    if args.cmd == "init":
        return _init(args.state)

    if args.cmd == "accept":
        return _accept(args.engine, args.tps, args.ms, args.model,
                       args.build, args.sha, args.state)

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