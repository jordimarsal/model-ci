# model-ci

**Keep the local model fleet honest: watch engines and models, detect drift, bench, open a human gate.**

`model-ci` is the orchestration loop of a local LLM rig. It does NOT measure (that is
[ggufbench](https://github.com/jordimarsal/ggufbench)), it does NOT propose routing
(that is the debrief evidence pack), and it does NOT decide (that gate is always human).
It watches engines (llama.cpp, Strata) and models (watched Hugging Face repos); when the
world changes it probes for speed drift, runs the full bench when drift is confirmed,
and writes a drift-evidence pack for the human gate.

## Usage

```sh
# 0. declare what to watch (engines + HF models), and declare your ggufbench location
cat > model-ci.toml <<'TOML'
[engines.llamacpp]
repo = "https://github.com/ggml-org/llama.cpp"
ref = "refs/heads/main"
[engines.strata]
repo = "https://github.com/Niko1221/Strata"
ref = "refs/heads/main"
[[models]]
repo = "unsloth/Llama-3.2-1B-Instruct-GGUF"
TOML

# 1. seed the state (once)
model-ci init --state .model-ci-state.json

# 2. accept bench results as baselines — the ONLY way a baseline enters the state
model-ci accept --engine llamacpp --tps 160.9 --ms 24 \
  --model qwen3.6-35b-apex-i-compact --build 1.0.0-dev --sha a956a852c \
  --state .model-ci-state.json

# 3. the loop (one-shot; run it from estol, cron or a systemd timer)
model-ci check --config model-ci.toml --state .model-ci-state.json
```

Exit codes: `0` no change · `1` change detected, probe within tolerance (±10%) ·
`2` drift, full bench launched · `3` drift AND drift-evidence pack written ·
`64` usage · `65` environment (GPU golden-rule conflict, busy machine, missing
baseline, ggufbench missing).

Baseline discipline: a drift detected on the micro-bench is only a smoke signal — the
tool confirms it with a stabilized probe (median of ≥3 generations of 512 tokens) before
burning hours on the full bench. And a drifted engine WITHOUT an accepted baseline is an
environment error (`65`): measure first, accept, then let the loop judge.

## Status

Feature 1 (fleet watch loop, T1-T6) and feature 2 (runnable: strict mypy, init/accept
seeding, model-ci.toml, CI) implemented — 72 offline tests green, mypy strict clean.
Pending: first integration smoke on the rig (T12), scheduler examples (estol/cron).

## Why a separate repo

ggufbench stays a stdlib-only benchmark CLI; debrief stays read-only; estol stays a
scheduler. This repo is the one allowed to hold opinions about *when* to bench and *what*
counts as drift. One piece, one responsibility.
