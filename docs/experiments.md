# Experiment Tracker

Last updated: 2026-05-15T05:48:50+02:00.

This file is the active tracker. Every run gets one row. Do not encode active
state in README, ad hoc notes, or generated logs.

## Tracker Methodology

Run ids use this shape:

```text
<family>-<arm>-s<seed>
```

Examples: `cfv2-vanilla-s0`, `cfv2-rope-pi8-s0`,
`cfv2-rope-learnable-s0`.

Each row must include:

- status: `planned`, `running`, `completed`, `aborted`, or `blocked`.
- config path.
- output directory.
- W&B run id or `none`.
- checkpoint policy.
- decision gate or next action.

When a run completes, copy stable metrics into [results.md](results.md). Keep
transient progress here only if it affects an operational decision.

## Active Run

| Run id | Status | Config | Output dir | W&B | Checkpoints | Notes |
|---|---|---|---|---|---|---|
| `cfv2-vanilla-s0` | completed | `src/rope_prov/configs/vanilla_counterfactual_v2.yaml` | `runs/vanilla_counterfactual_v2_online-seed0` | `n3b2ajjb` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. |
| `cfv2-zeroed-s0` | completed | `src/rope_prov/configs/vanilla_zeroed_counterfactual_v2.yaml` | `runs/vanilla_zeroed_P8_counterfactual_v2_online-seed0` | `vp7rso3y` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. |
| `cfv2-rope-pi8-s0` | completed | `src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml` | `runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0` | `y0033rou` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP completed. |
| `cfv2-rope-prew-pi8-smoke-s0` | completed | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_smoke-seed0` | `rw38jp7x` | every 200 steps, keep 2 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-400`, `checkpoint-600`. |

W&B URLs:

- `cfv2-vanilla-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/n3b2ajjb`
- `cfv2-zeroed-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/vp7rso3y`
- `cfv2-rope-pi8-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/y0033rou`
- `cfv2-rope-prew-pi8-smoke-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/rw38jp7x`

Completed state for `cfv2-vanilla-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.603, step 400 = 1.841, step 600 = 1.529,
  step 1000 = 1.282, step 1600 = 1.198, step 2200 = 1.184,
  step 2600 = 1.182, step 2800 = 1.178.
- final train loss: 1.6655.
- runtime: 46.4 min.
- throughput: 33.35 examples/sec.
- SEP: -0.135, with instruction execution 0.155 and data execution 0.290.

Gate read: passed. Vanilla v2 has stable utility loss and improves SEP over the
old Alpaca vanilla baseline by +0.085, so the remaining ablations are
informative.

Completed state for `cfv2-zeroed-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.604, step 400 = 1.836, step 600 = 1.531,
  step 1000 = 1.288, step 1600 = 1.189, step 2200 = 1.186,
  step 2600 = 1.181, step 2800 = 1.180.
- final train loss: 1.6655.
- runtime: 47.4 min.
- throughput: 32.67 examples/sec.
- SEP: -0.125, with instruction execution 0.150 and data execution 0.275.

Gate read: passed. Zeroing the lowest-frequency RoPE pairs again behaves like a
cheap control at this context length, with stable utility loss and SEP within
0.010 of the vanilla counterfactual baseline.

Completed state for `cfv2-rope-pi8-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 3.003, step 400 = 2.259, step 600 = 1.860,
  step 1000 = 1.605, step 1600 = 1.530, step 2200 = 1.524,
  step 2600 = 1.526, step 2800 = 1.522.
- SEP: -0.275, with instruction execution 0.020 and data execution 0.295.
- delta-of-deltas vs vanilla: -0.080.

Gate read: failed. Fixed post-projection pi/8 does not exploit the aligned v2
training signal. It mostly suppresses instruction-slot execution while leaving
DATA-slot execution near the vanilla/zeroed level.

Completed state for `cfv2-rope-prew-pi8-smoke-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.666, step 400 = 2.379, step 600 = 2.336.
- final train loss over the 600-step smoke: 1.7671.
- runtime: 10.7 min.
- throughput: 29.94 examples/sec overall.
- SEP: not run; this was a placement/utility smoke.

Gate read: mixed. Pre-W starts less disruptive than post-projection pi/8 at
step 200 (2.666 vs 3.003), but it does not catch up by step 400 or 600 on eval
loss. The concerning number is the step-400 to step-600 decrease: vanilla drops
0.312, zeroed drops 0.305, and pre-W drops only 0.043. That supports the
"algebraically available" part of the transport-filter hypothesis but weakly
undercuts the "SGD finds the near-commutant quickly" part. This does not answer
SEP, and a matched 2901-step pre-W run remains the clean way to distinguish
slow convergence from failure to find the transport path.

## Counterfactual v2 Matrix

| Run id | Status | Hypothesis | Config | Output dir |
|---|---|---|---|---|
| `cfv2-vanilla-s0` | completed | data-only baseline | `src/rope_prov/configs/vanilla_counterfactual_v2.yaml` | `runs/vanilla_counterfactual_v2_online-seed0` |
| `cfv2-zeroed-s0` | completed | capacity/zeroed-dim control | `src/rope_prov/configs/vanilla_zeroed_counterfactual_v2.yaml` | `runs/vanilla_zeroed_P8_counterfactual_v2_online-seed0` |
| `cfv2-rope-pi8-s0` | completed | fixed small post-projection role rotation | `src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml` | `runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0` |
| `cfv2-rope-prew-pi8-smoke-s0` | completed | pre-W role rotation smoke; tests placement correction | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_smoke-seed0` |
| `cfv2-rope-learnable-s0` | queued | model chooses role-angle gap | `src/rope_prov/configs/rope_prov_learnable_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_counterfactual_v2_online-seed0` |
| `cfv2-rope-prew-pi8-full-s0` | queued | budget-vs-findability check for pre-W transport | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_online-seed0` |
| `cfv2-rope-learnable-pi8-unfreeze-s0` | queued | nonzero learnable gap tests whether optimizer drives angle back to zero | `src/rope_prov/configs/rope_prov_learnable_pi8_unfreeze_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_pi8_unfreeze_counterfactual_v2_online-seed0` |
| `cfv2-rope-independent-angles-s0` | queued | rotational steelman against equal-frequency collapse | `src/rope_prov/configs/rope_prov_independent_angles_counterfactual_v2.yaml` | `runs/rope_prov_P8_independent_angles_counterfactual_v2_online-seed0` |
| `cfv2-best-rope-s1` | conditional | seed variance calibration | duplicate best rope config with seed 1 | TBD |

Run vanilla first. If vanilla final eval loss is unstable or above 2.0, revise
the counterfactual curriculum before spending GPU on the other arms. If vanilla
does not move SEP relative to the old vanilla baseline, the data did not teach
role-conditioning to the architecture-free baseline.

The vanilla and zeroed gates passed on 2026-05-14. The fixed pi8 training run
completed on 2026-05-15 and failed the SEP gate. The pre-W smoke also completed
on 2026-05-15. The final rotational-cell batch is queued in this order:
standard learnable, full-budget pre-W pi8, pi8-initialized learnable with
200-step freeze, and independent fixed per-pair angles. A seed-1 rerun is only
needed if one rope arm clears the positive/marginal decision threshold.

The pi8 SEP failure is asymmetric rather than merely null: INSTRUCTION-slot
execution collapses while DATA-slot execution is unchanged. That makes the
learnable-angle arms more diagnostic. If standard learnable angles converge near
zero, run a nonzero-initialized/unfrozen variant before treating "zero angle" as
only a bandwidth result.

## Commands

Train or resume:

```bash
uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/vanilla_counterfactual_v2.yaml \
  --output-dir runs/vanilla_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest
```

Next arms:

```bash
uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/vanilla_zeroed_counterfactual_v2.yaml \
  --output-dir runs/vanilla_zeroed_P8_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest

uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml \
  --output-dir runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest

uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2_smoke.yaml \
  --output-dir runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_smoke-seed0 \
  --resume-from-checkpoint latest

uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/rope_prov_learnable_counterfactual_v2.yaml \
  --output-dir runs/rope_prov_P8_learnable_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest

uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2.yaml \
  --output-dir runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest

uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/rope_prov_learnable_pi8_unfreeze_counterfactual_v2.yaml \
  --output-dir runs/rope_prov_P8_learnable_pi8_unfreeze_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest

uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/rope_prov_independent_angles_counterfactual_v2.yaml \
  --output-dir runs/rope_prov_P8_independent_angles_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest
```

SEP after training:

```bash
uv run python -m rope_prov.eval_sep \
  --model-path runs/vanilla_counterfactual_v2_online-seed0/final \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --variant vanilla \
  --sep-json /tmp/sep_repo/SEP_dataset/SEP_dataset.json \
  --output results/sep/vanilla_counterfactual_v2.json \
  --batch-size 8 \
  --progress-every 10
```

For `rope_prov` arms, pass `--variant rope_prov --prov-dim 8` and the matching
`--role-angles`; for learnable arms also pass `--learnable-angles`. For
independent per-pair angles, pass the flattened role rows, e.g.
`--role-angles 0 0 0 0 0.19634954 0.26179939 0.39269908 0.52359878`.

SEP evaluation intentionally uses manual no-cache decoding so `role_ids` are
supplied on every forward pass. Use `--batch-size` for throughput; this batches
multiple no-cache generations without relying on HF cached generation plumbing.

## Pre-Registered Gates

Primary metric:

```text
[SEP(rope_prov_cf) - SEP(rope_prov_alpaca)]
-
[SEP(vanilla_cf) - SEP(vanilla_alpaca)]
```

Positive: at least +0.05 absolute SEP on the same SEP subset.

Marginal: +0.02 to +0.05.

No architectural SEP gain: within +/-0.02, unless the learnable arm shows a
clear role-angle movement.

Learnable-angle interpretation:

| Learned gap | Interpretation |
|---|---|
| above 5 degrees plus SEP gain | architectural channel becomes useful under aligned training |
| 1-3 degrees | marginal channel, probably not load-bearing |
| near zero with stable training | post-projection rotational placement structurally inadequate |
| unstable/fluctuating | data or optimization inconclusive |

Before claiming a positive result, run the best rope arm with a second seed and
estimate SEP measurement variance.

## Historical Runs

| Run | Status | Use |
|---|---|---|
| `sskfecco` | completed and synced | counterfactual v1 vanilla smoke run; generator too narrow for publishable inference |
| `65k0k7cy` | aborted and deleted | first online v2 vanilla attempt; no checkpoint existed before deletion |
