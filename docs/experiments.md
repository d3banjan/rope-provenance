# Experiment Tracker

Last updated: 2026-05-17T13:09:27+02:00.

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
| `cfv2-rope-prew-rezero-pi8-smoke-s0` | completed | `src/rope_prov/configs/rope_prov_pre_w_rezero_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_rezero_counterfactual_v2_smoke-seed0` | `j38ih52f` | every 200 steps, keep 2 | Completed cleanly. 600-step staged-gate smoke. SEP completed. Note: gates were converted to bf16 in this first smoke; fixed for later runs. |
| `cfv2-rope-prew-rezero-independent-smoke-s0` | completed | `src/rope_prov/configs/rope_prov_pre_w_rezero_independent_angles_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_rezero_independent_angles_counterfactual_v2_smoke-seed0` | `q8l31kt7` | every 200 steps, keep 2 | Completed cleanly. 600-step staged-gate smoke with fixed independent per-pair target angles and fp32 gates. SEP completed. |
| `cfv3-vanilla-role-contrast-smoke-s0` | completed | `src/rope_prov/configs/vanilla_counterfactual_v3_role_contrast_smoke.yaml` | `runs/vanilla_counterfactual_v3_role_contrast_smoke-seed0` | `17yp3vws` | every 200 steps, keep 2 | Completed cleanly. 600-step curriculum smoke. SEP completed. |
| `toy-add-role-hidden-correct-s0` | completed | CLI: `scripts/toy_role_provenance.py --hide-tags --role-control correct --answer-loss-weight 4 --batch-size 1024` | `results/toy/role_embedding_hidden_tags_bs1024_s0.json` | `10oi0jn4` | result JSON only | Simple-template positive control. Final SEP 1.000. Useful tooling smoke, but not strong evidence because the constant-role control also learns the generator. |
| `toy-add-role-hidden-constant-s0` | completed | CLI: same toy script with `--hide-tags --role-control constant --answer-loss-weight 4 --batch-size 1024` | `results/toy/role_embedding_hidden_tags_constant_bs1024_s0.json` | `2l445kb2` | result JSON only | Simple-template causal control. Final SEP 0.953, confirming that the initial generator is too shortcutable without role variation. |
| `toy-add-role-hidden-swapped-s0` | planned | CLI: same toy script with `--hide-tags --role-control swap_instr_data --answer-loss-weight 4 --batch-size 1024` | `results/toy/role_embedding_hidden_tags_swapped_bs1024_s0.json` | planned | result JSON only | Directionality control for the simple-template generator. Lower priority after constant-role shortcut was observed. |
| `toy-add-role-hidden-diverse-correct-s0` | completed | CLI: same toy script with `--template-mode diverse --hide-tags --role-control correct --answer-loss-weight 4 --batch-size 1024` | `results/toy/role_embedding_hidden_tags_diverse_bs1024_s0.json` | `8itmpe7b` | result JSON only | Harder positive-control target with varied templates, field labels, field order, and heldout template families. Final SEP 0.836. |
| `toy-add-role-hidden-diverse-constant-s0` | completed | CLI: same toy script with `--template-mode diverse --hide-tags --role-control constant --answer-loss-weight 4 --batch-size 1024` | `results/toy/role_embedding_hidden_tags_diverse_constant_bs1024_s0.json` | `ub86g90v` | result JSON only | Main shortcut control for the diverse generator. Final SEP 0.680. Correct roles beat this by +0.156 final SEP and +0.328 at step 500. |
| `toy-add-role-hidden-diverse-evalswap-s0` | stopped | CLI: `--template-mode diverse --role-control correct --eval-role-control swap_instr_data` | `results/toy/role_embedding_hidden_tags_diverse_traincorrect_evalswap_bs1024_s0.partial.json` | `jq4g488w` | partial JSON only | Correct directionality control: train on correct hidden roles, evaluate with instruction/DATA roles swapped. Stopped after step 1000 because eval-swapped SEP stayed ~0, enough to show the diverse positive result is role-direction sensitive. |
| `toy-gate-pretrain-block512-bs512-s0` | completed | CLI: `--template-mode gate_pretrain --vocab-template-modes gated --hide-tags --role-control correct --block-size 512 --batch-size 512 --answer-loss-weight 4 --fail-on-truncation` | `results/toy/toy_gate_pretrain_block512_bs512_s0.json` | `nl3dkyrm` | result JSON only | Corrected gated prerequisite run after discovering `block_size=256` truncated answer regions. 731,656-param scratch char model fit training loss but finished with heldout exact-match SEP 0.000, with only transient tiny hits. |
| `toy-gate-pretrain-syntax-reg-bs512-s0` | completed | CLI: `--template-mode gate_pretrain --gate-kinds no_not question --vocab-template-modes gated --hide-tags --role-control correct --block-size 512 --batch-size 512 --answer-loss-weight 4 --weight-decay 0.1 --dropout 0.05 --embedding-dropout 0.05 --label-smoothing 0.02 --fail-on-truncation` | `results/toy/toy_gate_pretrain_syntax_reg_bs512_s0.json` | `yy39ct44` | result JSON only | Cheap scratch kill-test after semantic gate failed. Final SEP 0.195, best intermediate SEP 0.211. Above the <0.10 hard-kill line but below the >=0.50 pass gate, so scratch gated-role remains blocked for the paper. |
| `toy-add-role-hidden-gated-correct-s0` | blocked | CLI: `--template-mode gated --hide-tags --role-control correct --answer-loss-weight 4 --block-size 512` | `results/toy/role_embedding_hidden_tags_gated_bs512_s0.json` | planned | result JSON only | Stronger toy test: paired examples keep visible text identical or near-identical and vary only hidden role assignment; blocked until the model has a reliable linguistic-gate prior. |
| `toy-add-role-hidden-gated-constant-s0` | planned | CLI: `--template-mode gated --hide-tags --role-control constant --answer-loss-weight 4 --batch-size 1024` | `results/toy/role_embedding_hidden_tags_gated_constant_bs1024_s0.json` | planned | result JSON only | Text-only contradiction control for the gated generator. |
| `toy-add-role-hidden-gated-evalswap-s0` | planned | CLI: `--template-mode gated --hide-tags --role-control correct --eval-role-control swap_instr_data --answer-loss-weight 4 --batch-size 1024` | `results/toy/role_embedding_hidden_tags_gated_traincorrect_evalswap_bs1024_s0.json` | planned | result JSON only | Directionality control for the gated generator. |
| `slm-qwen25-0.5b-base-gate-pretrain-raw-s0` | invalid | CLI: `scripts/slm_gate_provenance.py --model /mnt/expansion/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --template-mode gate_pretrain --hide-tags --batch-size 16 --eval-batch-size 16 --lora-rank 8 --lr 2e-4 --prompt-format raw` | `results/slm/qwen25_0_5b_base_gate_pretrain_s0.json` | `yx5fus24` | adapter checkpoint ignored, result JSON tracked | Invalidated by unshifted-label bug in the first SLM harness. Earlier batch-32 attempt `9i1k8sjr` OOMed after step 1. |
| `slm-qwen25-0.5b-base-gate-pretrain-answer-s0` | invalid | CLI: same script/model with `--prompt-format answer --batch-size 16 --eval-batch-size 16 --lora-rank 8 --lr 2e-4` | `results/slm/qwen25_0_5b_base_gate_pretrain_answer_s0.json` | `g1ewvphj` | adapter checkpoint ignored, result JSON tracked | Invalidated by unshifted-label bug in the first SLM harness. Rerun with corrected next-token loss before interpreting Qwen base capability. |
| `slm-qwen25-0.5b-base-gate-pretrain-answer-shift-s0` | completed | same script/model with corrected next-token loss and `--prompt-format answer --batch-size 16 --eval-batch-size 16 --lora-rank 8 --lr 2e-4` | `results/slm/qwen25_0_5b_base_gate_pretrain_answer_shift_s0.json` | `fd4scfjm` | adapter checkpoint ignored, result JSON tracked | Valid Qwen base capability rerun. Final exact-match 0.188, below the >=0.50 pass gate; train loss saturated, but heldout outputs were partial copies, wrong-field copies, or repetitions. |
| `slm-qwen25-0.5b-base-gated-role-s0` | blocked | pending script/config | pending | planned | LoRA/full-adapter output plus SEP JSON | Blocked because corrected Qwen base capability stayed below the >=0.50 pass gate. |
| `slm-qwen25-0.5b-instruct-chat-zero-s0` | completed | CLI: same script with `--model /mnt/expansion/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --prompt-format chat --steps 0` | `results/slm/qwen25_0_5b_instruct_chat_zero_s0.json` | `kcawfngi` | result JSON tracked | Confounded eval-only capability probe. Exact-match 0.172: nontrivial but not solved. This unblocks one short instruct LoRA chat fine-tune as a reachability check. |
| `slm-qwen25-0.5b-instruct-gated-probe-s0` | invalid | same script/model with `--prompt-format chat --steps 500 --batch-size 16 --eval-batch-size 16 --lora-rank 8 --lr 2e-4` | `results/slm/qwen25_0_5b_instruct_gate_pretrain_s0.json` | `fiteqspq` | adapter checkpoint ignored, result JSON tracked | Invalidated by unshifted-label bug in the first SLM harness. The zero-shot instruct result remains valid because it did not train. |
| `slm-qwen25-0.5b-instruct-gated-probe-shift-s0` | completed | same script/model with corrected next-token loss, `--prompt-format chat --steps 500 --batch-size 16 --eval-batch-size 16 --lora-rank 8 --lr 2e-4` | `results/slm/qwen25_0_5b_instruct_gate_pretrain_shift_s0.json` | `8vpjaili` | adapter checkpoint ignored, result JSON tracked | Confounded reachability check. Final exact-match 1.000 from step 100 onward; samples are exact witness copies. Shows the task is reachable with instruction-posttraining priors, not causal evidence for provenance. |
| `toy-rope-pos-add-role-s0` | planned | pending script option | `results/toy/rope_pos_add_role_s0.json` | planned | result JSON only | RoPE-native toy bridge: replace learned absolute positions with RoPE while keeping additive role embeddings. |
| `toy-rope-role-rotation-s0` | planned | pending script option | `results/toy/rope_role_rotation_s0.json` | planned | result JSON only | From-scratch rotational test: checks whether RoPE-style role rotation fails only as adaptation to pretrained SmolLM2 or also fails when present from initialization. |

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
- `cfv2-rope-prew-rezero-pi8-smoke-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/j38ih52f`
- `cfv2-rope-prew-rezero-independent-smoke-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/q8l31kt7`
- `cfv3-vanilla-role-contrast-smoke-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/17yp3vws`
- `toy-add-role-hidden-correct-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/10oi0jn4`
- `toy-add-role-hidden-constant-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/2l445kb2`
- `toy-add-role-hidden-diverse-correct-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/8itmpe7b`
- `toy-add-role-hidden-diverse-constant-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/ub86g90v`
- `toy-add-role-hidden-diverse-evalswap-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/jq4g488w`
- `toy-gate-pretrain-block512-bs512-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/nl3dkyrm`
- `toy-gate-pretrain-syntax-reg-bs512-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/yy39ct44`
- `slm-qwen25-0.5b-base-gate-pretrain-raw-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/yx5fus24`
- `slm-qwen25-0.5b-base-gate-pretrain-raw-s0-batch32-oom`: `https://wandb.ai/d3banjan/rope-provenance/runs/9i1k8sjr`
- `slm-qwen25-0.5b-base-gate-pretrain-answer-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/g1ewvphj`
- `slm-qwen25-0.5b-base-gate-pretrain-answer-shift-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/fd4scfjm`
- `slm-qwen25-0.5b-instruct-chat-zero-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/kcawfngi`
- `slm-qwen25-0.5b-instruct-gated-probe-shift-s0`: `https://wandb.ai/d3banjan/rope-provenance/runs/8vpjaili`

Cached model queue on `/mnt/expansion/huggingface/hub`:

- `Qwen/Qwen2.5-0.5B`: `models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987`
- `Qwen/Qwen2.5-0.5B-Instruct`: `models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- `HuggingFaceTB/SmolLM2-360M`: `models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49`
- `HuggingFaceTB/SmolLM2-360M-Instruct`: `models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951`
- `microsoft/Phi-3.5-mini-instruct`: `models--microsoft--Phi-3.5-mini-instruct/snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77`
- `microsoft/Phi-4-mini-instruct`: `models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083`

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

Completed state for `cfv2-rope-prew-rezero-pi8-smoke-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.6605, step 400 = 2.3726,
  step 600 = 2.3369.
- final train loss over the 600-step smoke: 1.7668.
- runtime: 11.1 min.
- throughput: 29.04 examples/sec overall.
- final gate max_abs: 0.1250.
- SEP: -0.155, with instruction execution 0.085 and data execution 0.240.

Gate read: failed smoke threshold. ReZero staging opens the pi/8 pre-W channel,
but it does not improve role separation over vanilla or fixed pre-W. DATA-slot
execution remains higher than INSTRUCTION-slot execution. This first smoke used
gates that were converted with the attention module to bf16, so treat it as an
operational result rather than a clean precision-controlled ablation.

Completed state for `cfv2-rope-prew-rezero-independent-smoke-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.7760, step 400 = 2.4834,
  step 600 = 2.4485.
- final train loss over the 600-step smoke: 1.7668.
- runtime: 11.7 min.
- throughput: 27.47 examples/sec overall.
- final gate max_abs: 0.1505.
- SEP: -0.170, with instruction execution 0.075 and data execution 0.245.

Gate read: failed. This is not a cheaper-perturbation test: the independent
per-pair target angles are being opened from scratch through learned ReZero
gates. It tests whether extra fixed phase bandwidth plus staged gates gives SGD
a route to a usable channel. The answer at smoke scale is no: utility is worse
than the shared-angle ReZero smoke, gates open, and SEP remains below vanilla.

Completed state for `cfv3-vanilla-role-contrast-smoke-s0`:

- dataset: train 30957, eval 600, counterfactual train 12000, eval 400.
- counterfactual variant: `role_contrast_v3`.
- role sanity: both role ids present in train/eval batches.
- eval loss: step 200 = 2.5865, step 400 = 2.2839,
  step 600 = 2.2472.
- final train loss over the 600-step smoke: 1.7664.
- runtime: 9.7 min.
- throughput: 33.12 examples/sec overall.
- SEP: -0.160, with instruction execution 0.065 and data execution 0.225.

Gate read: failed. The stricter curriculum is trainable and suppresses external
SEP DATA-slot execution relative to v2 vanilla, but it suppresses
INSTRUCTION-slot execution too. It does not fix the baseline role-confusion
problem at smoke scale, so ReZero or another rope arm is not yet justified on
this curriculum.

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
| `cfv2-rope-prew-rezero-pi8-smoke-s0` | completed | staged gate opens fixed pre-W pi/8 channel after early-layer adaptation | `src/rope_prov/configs/rope_prov_pre_w_rezero_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_rezero_counterfactual_v2_smoke-seed0` |
| `cfv2-rope-prew-rezero-independent-smoke-s0` | completed | staged gate opens fixed independent pre-W angle channels after early-layer adaptation | `src/rope_prov/configs/rope_prov_pre_w_rezero_independent_angles_counterfactual_v2_smoke.yaml` | `runs/rope_prov_pre_w_P8_rezero_independent_angles_counterfactual_v2_smoke-seed0` |
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

ReZero-gated pre-W smokes were added on 2026-05-17 to test the hypothesis that
the full-feature SFT runs failed because the rotational channel was introduced
too abruptly. Both shared pi/8 and independent fixed-angle gates opened, but
neither improved SEP. The independent-angle ReZero smoke was especially
diagnostic because it used fp32 gates and a richer fixed angle pattern; it still
landed below the vanilla/pre-W band.

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
