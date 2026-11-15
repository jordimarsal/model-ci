# Design: model-ci — the fleet watch loop (2026-11-15)

> Feature 1 (and only feature) of this repo. The loop: watch → probe → decide drift →
> bench → propose → (human gate, outside this repo).

## Problem

The rig has a scoreboard (ggufbench), a proposer (debrief) and a gate (roles apply), but
nothing between "the world changed" and "someone noticed days later". Two real incidents
shaped this spec:

1. **2026-11-14**: llama.cpp moved 228 commits (`0.9.0-dev` → `1.0.0-dev`). The fleet's
   speed reference was compared by hand, one model per engine, in an interactive session —
   and even that nearly produced a false regression (the bench's micro-generation task on
   Strata swings ±20%; the first "−30%" reading was an artifact of a noisy task plus an
   unrelated background installer).
2. **The October pool**: rows measured under the 0.6.0 build sat unreproducible next to
   fresh rows until someone manually re-measured the whole fleet under 0.9.0. The
   scoreboard rots silently without a watcher.

## Non-goals (hard)

- No auto-apply of routing changes, no auto-prune of models, no writing to
  `models.yaml`/`retired.json`. The gate is `roles apply` (human).
- No measurement code. Bench = subprocess call to `python3 -m ggufbench.run` from the
  workspace root (the documented contract).
- No daemon. It runs as a **one-shot command** (`model-ci check`) that a scheduler calls —
  estol job, cron or systemd timer, all fine. Idempotent, exit codes machine-readable.
- No cloud. Everything local; the only network reads are the two git remotes and the HF
  API of *watched* repos.

## Architecture

```
model-ci check                 # the one-shot entry point
  ├─ watch engines             # git ls-remote of the two remotes (cheap, no clone)
  ├─ watch models              # HF tree API of watched repos (etag/sha compare)
  ├─ state: state.json         # last-seen engine SHA / HF revision / baseline tps
  ├─ if drift → probe          # ggufbench --dims speed, one reference model per engine
  │    └─ drift vs baseline?   # baseline = the tps stored with the last accepted bench
  │         └─ full bench      # ggufbench full dims, fleet, sequential (GPU rule!)
  │              └─ propose    # write evidence pack → call debrief routing_propose input
  └─ report to stdout          # machine-readable summary + exit code
```

Exit codes: `0` no change · `1` change detected, probe within tolerance · `2` drift,
full bench launched · `3` drift AND proposal written (gate open) · `64` usage · `65`
environment (ggufbench missing, GPU busy).

## GPU golden rule

llama.cpp and Strata never run simultaneously (shared VRAM). The full-bench phase is
strictly sequential per engine and the probe phase refuses to start if
`pgrep engine/strata` and a llama-server are both alive — it exits `65` with the reason
instead of fighting for VRAM. (Learned 2026-11-14: the OOM that killed the UD-Q3 control
was residual VRAM from the previous server.)

## Measurement discipline (the 11-14 lessons, encoded)

- The bench's speed task is a **micro-generation** (median of 3 reps of one ~4-token
  reply). For engine-version deltas it is only a *smoke signal*: drift declared on the
  speed task MUST be confirmed with a stabilized probe (≥512 generated tokens, median of
  3) before a full bench is worth the hours. `model-ci probe --stabilized N`.
- A measurement runs only on a **quiet machine**: load < threshold (default: 1-min load
  < nproc/2) and no foreign GPU compute processes (`nvidia-smi --query-compute-apps`
  empty or known-allowlist). Otherwise: defer, say so, exit `65`.
- Every probe/bench stores: engine build, model file, flags, host load at start — the
  receipt travels with the number (same rule as ggufbench).

## Files

| File | Role |
|---|---|
| `src/model_ci/cli.py` | arg parsing, exit codes, the `check`/`probe`/`watch` subcommands |
| `src/model_ci/watch.py` | `ls-remote` of the two engine repos; HF tree API of watched repos (stdlib `urllib`, etag cache) |
| `src/model_ci/state.py` | `state.json` read/write with schema versioning (last engine SHA, HF revisions, baselines) |
| `src/model_ci/probe.py` | subprocess ggufbench `--dims speed` (+`--stabilized N` direct generations), drift decision (tolerance default ±10%) |
| `src/model_ci/bench.py` | full fleet bench, engine-sequential, GPU-rule guard, quiet-machine guard |
| `src/model_ci/propose.py` | evidence pack (markdown + json) and the debrief hand-off |
| `tests/` | offline: fake git output, stubbed HF responses, state schema, drift math, exit codes. No network in tests. |

## State schema (v1)

```json
{
  "version": 1,
  "engines": {
    "llamacpp": {"sha": "a956a852c…", "build": "12387", "baseline": {"model": "<slug>", "tps": 160.9, "ms": 24, "ts": "2026-11-14"}},
    "strata":   {"sha": "c3b3f503…",  "version": "0.1.57", "baseline": {"model": "<slug>", "tps": 138.9, "ms": 36, "ts": "2026-11-05"}}
  },
  "models": {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": {"revision": "<sha>"}
  }
}
```

`baseline` is written ONLY when the human accepts a full-bench result (the tool prints
the command; the human reruns with `--accept` after the gate). This keeps "the number
we compare against" a human-blessed artifact, not an automatic one.

## T1-T6 (test-first)

- [ ] T1: state schema load/save/migrate; unknown version → clear error.
      refs: State schema.
- [ ] T2: watch — parse fake `git ls-remote` and fake HF tree JSON; detect new sha/revision.
      refs: Architecture.
- [ ] T3: probe — drift math (±10% tolerance), stabilized probe invocation, quiet-machine
      guard (fake load + fake compute-apps), GPU-rule guard.
      refs: Measurement discipline.
- [ ] T4: bench — engine-sequential plan, refusal to run when the other engine is alive.
      refs: GPU golden rule.
- [ ] T5: propose — evidence pack content (old baseline, new numbers, receipts, logs);
      debrief hand-off as a file drop (debrief's `bench_path` input), not an RPC.
      refs: Architecture.
- [ ] T6: CLI exit codes end-to-end with stubbed internals (0/1/2/3/64/65).
      refs: Architecture.

## Open questions (for the spec gate)

1. Watched-model list: file (`model-ci.toml`) or reuse ggufbench's `models/` dir listing?
   Proposal: `model-ci.toml`, explicit is better.
2. Notifications beyond stdout/exit code: estol already surfaces job logs. Skip email/tg
   for v1.
