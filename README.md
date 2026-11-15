# model-ci

**Keep the local model fleet honest: watch engines and models, detect drift, bench, open a human gate.**

`model-ci` is the orchestration loop of a local LLM rig. It does NOT measure (that is
[ggufbench](https://github.com/jordimarsal/ggufbench)), it does NOT propose routing
(that is [debrief](https://github.com/jordimarsal/debrief-mcp)), and it does NOT decide
(that gate is always human). It watches:

- **engines** — llama.cpp releases/`main` head, Strata releases — and on change runs the
  cheap speed probe of one reference model per engine against the stored baseline;
- **models** — watched Hugging Face repos (new revisions of quant packs you already run)
  and re-verifies what changed.

When drift is detected it launches the full bench, and hands the evidence to debrief's
`routing_propose` so the human decides at the `roles apply` gate. No auto-prune, no
auto-routing, ever.

## Status

Spec phase — see `docs/specs/2026-11-15-model-ci-design.md`. Nothing implemented yet.

## Why a separate repo

ggufbench stays a stdlib-only benchmark CLI; debrief stays read-only; estol stays a
scheduler. This repo is the one allowed to hold opinions about *when* to bench and *what*
counts as drift. One piece, one responsibility.

[![built with harness](https://raw.githubusercontent.com/jordimarsal/harness/main/assets/badge.svg)](https://github.com/jordimarsal/harness)
