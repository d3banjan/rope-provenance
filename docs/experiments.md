# Experiment Tracker

Last updated: 2026-05-14T19:08:56+02:00.

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
| `cfv2-vanilla-s0` | running | `src/rope_prov/configs/vanilla_counterfactual_v2.yaml` | `runs/vanilla_counterfactual_v2_online-seed0` | `n3b2ajjb` | every 500 steps, keep 3 | Restarted clean after deleting aborted run `65k0k7cy` locally and from W&B. First checkpoint exists at `checkpoint-500`. |

W&B URL: `https://wandb.ai/d3banjan/rope-provenance/runs/n3b2ajjb`

Known early state for `cfv2-vanilla-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in early batches.
- eval loss: step 200 = 2.603, step 400 = 1.841, step 600 = 1.529.
- early throughput: about 29-36 examples/sec.

These are plumbing and early-learning signals only. They are not final
research results.

## Counterfactual v2 Matrix

| Run id | Status | Hypothesis | Config | Output dir |
|---|---|---|---|---|
| `cfv2-vanilla-s0` | running | data-only baseline | `src/rope_prov/configs/vanilla_counterfactual_v2.yaml` | `runs/vanilla_counterfactual_v2_online-seed0` |
| `cfv2-zeroed-s0` | planned | capacity/zeroed-dim control | `src/rope_prov/configs/vanilla_zeroed_counterfactual_v2.yaml` | `runs/vanilla_zeroed_P8_counterfactual_v2_online-seed0` |
| `cfv2-rope-pi8-s0` | planned | fixed small role rotation | `src/rope_prov/configs/rope_prov_pi8_counterfactual_v2.yaml` | `runs/rope_prov_P8_pi8_counterfactual_v2_online-seed0` |
| `cfv2-rope-learnable-s0` | planned | model chooses role-angle gap | `src/rope_prov/configs/rope_prov_learnable_counterfactual_v2.yaml` | `runs/rope_prov_P8_learnable_counterfactual_v2_online-seed0` |
| `cfv2-best-rope-s1` | conditional | seed variance calibration | duplicate best rope config with seed 1 | TBD |

Run vanilla first. If vanilla final eval loss is unstable or above 2.0, revise
the counterfactual curriculum before spending GPU on the other arms. If vanilla
does not move SEP relative to the old vanilla baseline, the data did not teach
role-conditioning to the architecture-free baseline.

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
