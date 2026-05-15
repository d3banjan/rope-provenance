# Experiment Tracker

Last updated: 2026-05-15T12:28:38+02:00.

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

## Run Ledger

| Run id | Status | Config | Output dir | W&B | Checkpoints | Notes |
|---|---|---|---|---|---|---|
| `cfv2-vanilla-s0` | completed | `src/rope_prov/configs/vanilla_counterfactual_v2.yaml` | `runs/vanilla_counterfactual_v2_online-seed0` | `n3b2ajjb` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. |
| `cfv2-zeroed-s0` | completed | `src/rope_prov/configs/vanilla_zeroed_counterfactual_v2.yaml` | `runs/vanilla_zeroed_P8_counterfactual_v2_online-seed0` | `vp7rso3y` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. |
| `cfv2-rope-pi8-s0` | completed | `src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml` | `runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0` | `y0033rou` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP completed. |
| `cfv2-rope-prew-pi8-smoke-s0` | completed | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_smoke-seed0` | `rw38jp7x` | every 200 steps, keep 2 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-400`, `checkpoint-600`. |
| `cfv2-rope-learnable-s0` | completed | `src/rope_prov/configs/rope_prov_learnable_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_counterfactual_v2_online-seed0` | `mn858blz` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP completed. |
| `cfv2-rope-prew-pi8-full-s0` | completed | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_online-seed0` | `vmgck3dr` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP completed. |
| `cfv2-rope-learnable-pi8-unfreeze-s0` | completed | `src/rope_prov/configs/rope_prov_learnable_pi8_unfreeze_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_pi8_unfreeze_counterfactual_v2_online-seed0` | `kkrlei1m` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP completed. |
| `cfv2-rope-independent-angles-s0` | completed | `src/rope_prov/configs/rope_prov_independent_angles_counterfactual_v2.yaml` | `runs/rope_prov_P8_independent_angles_counterfactual_v2_online-seed0` | `i4dwm9tf` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP completed. |
| `cfv2-rope-prew-learnable-s0` | completed | `src/rope_prov/configs/rope_prov_pre_w_learnable_counterfactual_v2.yaml` | `runs/rope_prov_pre_w_P8_learnable_counterfactual_v2_online-seed0` | `zqfwyd7o` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP completed. |

W&B URLs:

- `cfv2-vanilla-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/n3b2ajjb`
- `cfv2-zeroed-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/vp7rso3y`
- `cfv2-rope-pi8-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/y0033rou`
- `cfv2-rope-prew-pi8-smoke-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/rw38jp7x`
- `cfv2-rope-learnable-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/mn858blz`
- `cfv2-rope-prew-pi8-full-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/vmgck3dr`
- `cfv2-rope-learnable-pi8-unfreeze-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/kkrlei1m`
- `cfv2-rope-independent-angles-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/i4dwm9tf`
- `cfv2-rope-prew-learnable-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/zqfwyd7o`

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

Superseded read: the matched full-budget pre-W run converged to the vanilla
utility band. Treat this smoke as a useful usability warning, not as a stable
negative result about pre-W convergence.

Completed state for `cfv2-rope-learnable-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.606, step 400 = 1.841, step 600 = 1.532,
  step 1000 = 1.283, step 1600 = 1.196, step 2200 = 1.180,
  step 2600 = 1.179, step 2800 = 1.186.
- final train loss: 1.6656.
- runtime: about 58 min.
- final learned angle gap: about -0.0096 rad, or -0.55 degrees.
- SEP: -0.130, with instruction execution 0.155 and data execution 0.285.

Gate read: failed positive threshold. Learnable post-projection angles preserve
utility by staying near zero, but they do not open a useful role channel under
the aligned counterfactual curriculum.

Completed state for `cfv2-rope-prew-pi8-full-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.603, step 400 = 1.845, step 600 = 1.526,
  step 1000 = 1.287, step 1600 = 1.193, step 2200 = 1.182,
  step 2600 = 1.179, step 2800 = 1.183.
- final train loss: 1.6654.
- runtime: 51.3 min.
- throughput: 30.19 examples/sec.
- SEP: -0.125, with instruction execution 0.160 and data execution 0.285.

Gate read: failed positive threshold, but passed the utility/comparability
check. Pre-W pi/8 avoids the post-projection pi8 instruction-compliance
collapse and converges like vanilla/zeroed, yet its SEP is still in the
architecture-free band.

Completed state for `cfv2-rope-learnable-pi8-unfreeze-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- final eval loss: step 2800 = 1.5296.
- final train loss: 1.7438.
- runtime: 58.1 min.
- final learned angle gap: about 0.3463 rad, or 19.84 degrees.
- SEP: -0.290, with instruction execution 0.025 and data execution 0.315.

Gate read: failed. Freezing a nonzero post-projection pi/8-like gap for the
first 200 steps traps the arm in the same damaging family as fixed pi/8. When
unfrozen, the optimizer does not return to the zero-gap basin within budget.
The result confirms that nonzero post-projection role rotation is actively
harmful, not merely unused.

Completed state for `cfv2-rope-independent-angles-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 3.132, step 400 = 2.246, step 600 = 1.866,
  step 1000 = 1.625, step 1600 = 1.562, step 2200 = 1.560,
  step 2600 = 1.555, step 2800 = 1.555.
- final train loss: 1.7686.
- runtime: 54.5 min.
- throughput: 28.39 examples/sec.
- SEP: -0.240, with instruction execution 0.010 and data execution 0.250.

Gate read: failed. Independent fixed angles avoid the equal-frequency
single-phase bottleneck, but they do not rescue role separation. Utility remains
well above the vanilla/pre-W band, and SEP shows the same instruction-execution
collapse pattern as the post-projection fixed-angle family.

Completed state for `cfv2-rope-prew-learnable-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 2800 = 1.1822.
- final train loss: 1.6653.
- runtime: 53.3 min.
- throughput: 29.05 examples/sec.
- final learned angle gap: about -0.0081 rad, or -0.46 degrees.
- SEP: -0.115, with instruction execution 0.160 and data execution 0.275.

Gate read: failed positive threshold. Pre-W learnable preserves the utility and
compliance recovery of fixed pre-W, but it keeps the learned role-angle gap near
zero and does not produce positive role separation. This is the cleanest
pre-W null: when given both upstream placement and angle freedom, the optimizer
closes the rotational channel rather than using it.

## Counterfactual v2 Matrix

| Run id | Status | Hypothesis | Config | Output dir |
|---|---|---|---|---|
| `cfv2-vanilla-s0` | completed | data-only baseline | `src/rope_prov/configs/vanilla_counterfactual_v2.yaml` | `runs/vanilla_counterfactual_v2_online-seed0` |
| `cfv2-zeroed-s0` | completed | capacity/zeroed-dim control | `src/rope_prov/configs/vanilla_zeroed_counterfactual_v2.yaml` | `runs/vanilla_zeroed_P8_counterfactual_v2_online-seed0` |
| `cfv2-rope-pi8-s0` | completed | fixed small post-projection role rotation | `src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml` | `runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0` |
| `cfv2-rope-prew-pi8-smoke-s0` | completed | pre-W role rotation smoke; tests placement correction | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_smoke-seed0` |
| `cfv2-rope-learnable-s0` | completed | model chooses role-angle gap | `src/rope_prov/configs/rope_prov_learnable_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_counterfactual_v2_online-seed0` |
| `cfv2-rope-prew-pi8-full-s0` | completed | budget-vs-findability check for pre-W transport | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_online-seed0` |
| `cfv2-rope-learnable-pi8-unfreeze-s0` | completed | nonzero learnable gap tests whether optimizer drives angle back to zero | `src/rope_prov/configs/rope_prov_learnable_pi8_unfreeze_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_pi8_unfreeze_counterfactual_v2_online-seed0` |
| `cfv2-rope-independent-angles-s0` | completed | rotational steelman against equal-frequency collapse | `src/rope_prov/configs/rope_prov_independent_angles_counterfactual_v2.yaml` | `runs/rope_prov_P8_independent_angles_counterfactual_v2_online-seed0` |
| `cfv2-rope-prew-learnable-s0` | completed | upstream placement plus learned gap tests whether pre-W freedom creates a usable channel | `src/rope_prov/configs/rope_prov_pre_w_learnable_counterfactual_v2.yaml` | `runs/rope_prov_pre_w_P8_learnable_counterfactual_v2_online-seed0` |
| `cfv2-best-rope-s1` | conditional | seed variance calibration | duplicate best rope config with seed 1 | TBD |

Run vanilla first. If vanilla final eval loss is unstable or above 2.0, revise
the counterfactual curriculum before spending GPU on the other arms. If vanilla
does not move SEP relative to the old vanilla baseline, the data did not teach
role-conditioning to the architecture-free baseline.

The vanilla and zeroed gates passed on 2026-05-14. The fixed pi8 training run
completed on 2026-05-15 and failed the SEP gate. The pre-W smoke also completed
on 2026-05-15. The final rotational-cell batch then completed in this order:
standard learnable, full-budget pre-W pi8, pi8-initialized learnable with
200-step freeze, independent fixed per-pair angles, and pre-W learnable angles.
No rope arm clears the positive or marginal decision threshold, so a seed-1
rerun is not required for a positive claim.

The pi8 SEP failure is asymmetric rather than merely null: INSTRUCTION-slot
execution collapses while DATA-slot execution is unchanged. The standard
learnable arm avoids this by keeping the angle gap near zero. The
pi8-initialized freeze/unfreeze arm keeps a large nonzero gap and reproduces the
damage. Independent per-pair fixed angles do not rescue the rotational channel.
Pre-W learnable also keeps its angle gap near zero, showing that upstream
placement plus angle freedom still does not make the channel useful.

## Commands

Train or resume:

```bash
uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/vanilla_counterfactual_v2.yaml \
  --output-dir runs/vanilla_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest
```

Training command catalog:

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
