# Amendment 01 — RPCG7c1 step-count undertraining fix

**Date:** 2026-05-19 (post-H0-freeze; commit `5a34219` froze H0/tau)
**Applies to:** `rpcg7c1_freqbalanced_qwen25_2026-05-19`
**Trigger:** the first C1 run returned VOID — the gate did not install at all.

## What happened

The frozen C1 recipe (H0 §7) used batched DPO with
`n_steps = ceil(STEPS / BATCH_SIZE) = ceil(900 / 16) = 57` optimizer steps,
holding *pair-exposures* (≈ 900) constant with RPCG7a/b.

The run returned **VOID**: `in_dist_margin 0.0031 ≤ τ 0.0163` — the gate did not
install even in-distribution (RPCG7a/b installed it at ≈ 0.257). The gated-arm
loss trajectory is decisive: the **DPO component sat at ln 2 ≈ 0.69 for all 57
steps** (0.6916 → 0.6897, never trending) — ln 2 is the value when policy =
reference, i.e. the LoRA adapter never moved off its zero initialisation. The
anchor likewise did not lift `chosen` (trained-spec probes ≈ baseline).

## Diagnosis — step count, not a loss bug

The batched loss is correct: it *starts* at exactly ln 2 (the right
zero-adapter value). The failure is that **57 optimizer steps is far too few**.
RPCG7a/b/RPCG3/RPCG6 all needed ≈ 900 steps for the DPO+anchor to install the
gate; β = 0.1 makes the DPO gradient small, so many updates are required.
Holding *pair-exposures* constant (`n_steps = STEPS / BATCH_SIZE`) cut the
*optimizer-step* count 16× — and the optimizer-step count, not the
pair-exposure count, is what a from-scratch LoRA needs.

Broader finding: **batching does not speed up this workload.** The model is
small (0.5B) and the sequences short (~30 tokens), so each step is
overhead-bound, not compute-bound. Batching cannot reduce the number of
optimizer steps required; keeping the step count constant only adds total
compute (slower wall), and cutting it undertrains. Filling the GPU does not
reduce wall-time here.

## Resolution

- `n_steps` is fixed to `STEPS` (900) — the proven optimizer-step count, matched
  to RPCG7a/b. `batch_size` is decoupled and set to **1** for all ladder rungs
  (the RPCG7a/b recipe). C1's only intended delta from RPCG7a/b is the
  frequency-balanced pairs.
- `<monorepo>/scripts/dpo_perms.py` amended: `n_steps = STEPS` always; `batch_size` is a parameter
  (default 1) that changes per-step throughput only, never the step count.
- The batched code path is retained (correct, useful for a future large-model
  experiment) but is not used by the ladder.

## Amended recipe

C1 = RPCG7a/b recipe (batch 1, 900 optimizer steps, β 0.1, anchor λ 1.0,
LoRA r=8 q/k/v_proj, continued-pretrain reused) **plus** frequency-balanced
chosen/rejected pairs — the single C1 lever. The decision rule, τ, and the
`trained_sys_meanprob` diagnostic split (H0 §3) are unchanged.

The VOID first run is retained only as this amendment's evidence; the amended
run is the C1 result of record.
