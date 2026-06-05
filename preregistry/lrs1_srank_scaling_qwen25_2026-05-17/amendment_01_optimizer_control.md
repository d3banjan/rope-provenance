# Amendment 01 — 3B optimizer fallback + optimizer control

**Date:** 2026-05-17 (post-H0-freeze; commit `845956c` froze H0/tau/power)
**Applies to:** `lrs1_srank_scaling_qwen25_2026-05-17`
**Trigger:** GPU OOM on the Qwen2.5-3B run, discovered after the prereg freeze.

## What happened

The pre-registered recipe (bf16 LoRA, r=128, the 7 Qwen modules, on a 12 GB
RTX 3060) trains 0.5B and 1.5B cleanly — peak VRAM 5.1 / 8.4 GB — but the 3B
run **OOMs by ~0.6 GB**: peak ~12.1 GB against 11.6 GB usable. Confirmed twice
(plain; and with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).

The pre-registered OOM fallback (H0 §9: `max_memory` CPU offload) **failed**:
peft LoRA backward + accelerate meta-device weight offload do not compose —
`RuntimeError: Function MmBackward0 returned an invalid gradient at index 1 —
expected device meta but got cuda:0`. CPU-offload of a partially-dispatched
base model is not a usable path for LoRA training here.

## Resolution

3B is trained with **8-bit Adam** (bitsandbytes `adamw_bnb_8bit`) instead of
32-bit `adamw_torch`. This quantizes the optimizer states (m, v) to 8-bit,
saving ~1.4 GB, and is the **minimal recipe deviation** that fits 3B in 12 GB:
it touches neither the base weights (unlike QLoRA) nor any adapter
hyperparameter (r, α, lr, steps, batch, β, dtype).

8-bit Adam is, however, a **size-correlated recipe change** if applied to 3B
alone — the confound class that ruled out QLoRA in planning. It is therefore
controlled, not assumed invisible.

## Optimizer control

A control run trains **Qwen2.5-0.5B with 8-bit Adam**, everything else
identical to the primary 32-bit 0.5B run (seed 42, data, recipe).
Output: the `qwen25_0.5b_adam8bit_ctrl` control checkpoint (`<monorepo>/...`, research monorepo).

- **Control statistic:** `Δ = |attn_qkv_srank(0.5B, 8-bit) − attn_qkv_srank(0.5B, 32-bit)|`
- **Control tolerance:** `Δ ≤ 0.25`. The prereg per-point precision on
  `attn_qkv_srank` is SEM ≈ ±0.12 (`power.md`); 0.25 ≈ 2× that. The two 0.5B
  runs share seed, data and every hyperparameter, so any Δ is purely the
  optimizer-precision effect.

## Amended decision rule

- **Control PASS (`Δ ≤ 0.25`):** the optimizer change is below measurement
  precision. Accept the mixed recipe. The primary scaling fit uses
  {0.5B 32-bit, 1.5B 32-bit, 3B 8-bit}; the H0 §3 decision rule applies
  unchanged. The control result is reported alongside the verdict.

- **Control FAIL (`Δ > 0.25`):** the optimizer change moves srank materially.
  3B is **excluded from the primary scaling fit** and reported as
  `optimizer-confounded exploratory`. The primary verdict degrades to a
  2-point (0.5B / 1.5B) band + spread check; the 3-model ΔAIC fit is not run.

## Manifest record

The 8-bit runs (3B primary, 0.5B control) carry a top-level `optimizer_control`
block (`optimizer`, `optimizer_control_for`, `control_pair`,
`primary_use_allowed_if`, `role`). The 32-bit runs record
`args.optimizer = "adamw_torch"` and `optimizer_control = null`.

## Postmortem carry-over

Recorded as methodology findings for `postmortem.md`: (1) bf16 LoRA at r=128 on
the 7 Qwen modules does not fit a 3B model in 12 GB; (2) accelerate
meta-device offload is incompatible with peft LoRA backward; (3) 8-bit Adam was
the minimal viable fallback.
