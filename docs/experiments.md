# Experiment Tracker

Last updated: 2026-05-15T04:41:41+02:00.

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
| `cfv2-rope-pi8-s0` | completed | `src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml` | `runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0` | `y0033rou` | every 500 steps, keep 3 | Completed cleanly. Final artifacts in `final/`; retained checkpoints: `checkpoint-2000`, `checkpoint-2500`, `checkpoint-2901`. SEP intentionally deferred until after pre-W smoke. |
| `cfv2-rope-prew-pi8-smoke-s0` | running | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_smoke-seed0` | `rw38jp7x` | every 200 steps, keep 2 | 600-step GPU smoke queued immediately after fixed pi8 completed, before pi8 SEP. Dataset counts match other v2 arms; early role sanity shows both role ids. |

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

## Counterfactual v2 Matrix

| Run id | Status | Hypothesis | Config | Output dir |
|---|---|---|---|---|
| `cfv2-vanilla-s0` | completed | data-only baseline | `src/rope_prov/configs/vanilla_counterfactual_v2.yaml` | `runs/vanilla_counterfactual_v2_online-seed0` |
| `cfv2-zeroed-s0` | completed | capacity/zeroed-dim control | `src/rope_prov/configs/vanilla_zeroed_counterfactual_v2.yaml` | `runs/vanilla_zeroed_P8_counterfactual_v2_online-seed0` |
| `cfv2-rope-pi8-s0` | completed, SEP pending | fixed small post-projection role rotation | `src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml` | `runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0` |
| `cfv2-rope-prew-pi8-smoke-s0` | running | pre-W role rotation smoke; tests placement correction | `src/rope_prov/configs/rope_prov_pre_w_pi8_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_pi8_counterfactual_v2_smoke-seed0` |
| `cfv2-rope-learnable-s0` | planned | model chooses role-angle gap | `src/rope_prov/configs/rope_prov_learnable_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_counterfactual_v2_online-seed0` |
| `cfv2-best-rope-s1` | conditional | seed variance calibration | duplicate best rope config with seed 1 | TBD |

Run vanilla first. If vanilla final eval loss is unstable or above 2.0, revise
the counterfactual curriculum before spending GPU on the other arms. If vanilla
does not move SEP relative to the old vanilla baseline, the data did not teach
role-conditioning to the architecture-free baseline.

The vanilla and zeroed gates passed on 2026-05-14. The fixed pi8 training run
completed on 2026-05-15; SEP is deferred until after the pre-W smoke so the
placement correction gets a cheap GPU check before post-hoc pi8 analysis.

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
```

SEP after training:

```bash
uv run python -m rope_prov.eval_sep \
  --model-path runs/vanilla_counterfactual_v2_online-seed0/final \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --variant vanilla \
  --sep-json /tmp/sep_repo/SEP_dataset/SEP_dataset.json \
  --output results/sep/vanilla_counterfactual_v2.json
```

For `rope_prov` arms, pass `--variant rope_prov --prov-dim 8` and the matching
`--role-angles`; for the learnable arm also pass `--learnable-angles`.

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
