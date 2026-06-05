# LRS1 — Stable-Rank Scaling Cross-Check on Qwen2.5

**Date filed:** 2026-05-17
**Status:** PRE-REGISTERED — runs same-day (planned replication, user-directed; no sleep gate)
**Owner:** Debanjan Basu

## One-line

Replicate the `the lazy-rudder paper` task-intrinsic stable-rank result on the
Qwen2.5 family: train DPO LoRA adapters on Qwen2.5-{0.5B, 1.5B, 3B} base and
test whether the stable rank of the adapter weight-update ΔW is flat across
model width — as it is on Pythia (GPT-NeoX), where lazy-rudder reports
srank ≈ 3.653 ± 0.289 RMS.

## Why this exists

The lazy-rudder "flat srank floor" is measured on a single architecture
(GPT-NeoX). It is also **contrarian**: Aghajanyan et al. (intrinsic
dimensionality) and Flat-LoRA argue larger models have *lower* intrinsic
dimension, and SR-GRPO suggests scale-dependence. The claim therefore needs an
independent cross-architecture test before it can be trusted. Qwen2.5 is
Llama-style (split q/k/v/o_proj, GQA, gated SwiGLU MLP, RMSNorm) — a clean
architecture contrast.

## What this feeds

This run is the **geometry baseline** for a separate role-provenance
workstream. It answers whether Qwen preference/instruction tuning forms a
genuine low-rank control surface (stable rank, layer/module concentration) —
the surface that role-provenance work's additive role vectors are hypothesised
to exploit. The consumer compares DPO-adapter geometry against role-adapter
geometry; it does **not** run role eval on these checkpoints. Role-specific
evals (`role_swap`, `constant_role`) are explicitly N/A here. The checkpoints
ship a geometry-self-describing manifest (see `H0.md` §10).

## Scope

Internal cross-check — **no manuscript, no microsite**. Outputs: artifact JSON,
per-step checkpoint manifests, `decision.json`, `postmortem.md`.

## Pointers

- Pre-reg main: `H0.md`
- Frozen τ: `tau.json` (locked with `sha256_self`)
- Power notes: `power.md`
- Post-run decision: `decision.json` (post-run only)
- Post-run write-up: `postmortem.md` (post-run only)
- Scripts: the full lazy-rudder-srank pipeline lives in `<monorepo>/...` (research monorepo); not part of this public sync
- Artifact: `<monorepo>/artifacts/lrs1_srank_scaling_qwen25_2026-05-17.json`
- Reference: `the lazy-rudder paper/` (Pythia DPO srank study)
