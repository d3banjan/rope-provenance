# Reproducing the Policy-Rail Ladder

This file lists the exact commands that produced each result JSON under
`results/slm/`. Every row of the
[microsite](https://policy-rails-microsite.pages.dev/) rung table is backed by
one of the runs below. Numbers cited in the [Overview], [Explainer], and
[Technical] pages of the microsite are taken from the JSON `metrics` block of
the corresponding result file.

[Overview]: https://policy-rails-microsite.pages.dev/
[Explainer]: https://policy-rails-microsite.pages.dev/explainer
[Technical]: https://policy-rails-microsite.pages.dev/technical

## Setup

```bash
# uv-managed Python environment is canonical here.
git clone https://github.com/d3banjan/rope-provenance.git
cd rope-provenance
uv sync

# The default HF cache assumed by the scripts is /mnt/expansion/huggingface/hub.
# Override with --cache-dir <path> on every invocation, or pre-populate.

# Sanity check that the environment is wired up.
uv run pytest -q
```

GPU expectations: every Qwen2.5-0.5B-Instruct rung below fits on a 12GB
consumer GPU. The 1.5B replication (PR9) needs the same envelope at smaller
batch sizes; configure with `--batch-size 8 --eval-batch-size 4` if memory is
tight.

## Per-Rung Commands

The unified driver is `scripts/slm_policy_vector.py`, dispatched via two
flags:

- `--dataset-kind` chooses the corpus shape (single-span vs grid vs multi-span
  vs SEP projection vs SEP paired).
- `--permission-rail` chooses the compiler (`off` = raw policy bits, `oracle`
  = software-compiled lookup, `binder` = small learned compiler).

Rung 1 (source rail) and Rung 2 (source + operation rail) use the simpler
dedicated drivers `slm_source_rail.py` and `slm_policy_rail.py`.

### Rung 1 — source rail (Qwen2.5-0.5B-Instruct)

```bash
uv run python scripts/slm_source_rail.py \
  --output results/slm/qwen25_0_5b_instruct_source_rail_s0.json
```

Expected: strict exact 1.000 with correct source ids, 0.305 with source
constant, 0.000 with source swapped. Trains 5,376 parameters.

### Rung 2 — source + operation rail

```bash
uv run python scripts/slm_policy_rail.py \
  --output results/slm/qwen25_0_5b_instruct_source_operation_rail_s0.json
```

Expected: strict exact 1.000 across trusted OBEY, untrusted-OBEY suppression,
DATA-USE, DATA-QUOTE. Source swap 0.000, OBEY/USE swap 0.438. Trains 9,856
parameters.

### Rung 3a — raw policy bits (kill)

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind policy_vector \
  --permission-rail off \
  --output results/slm/qwen25_0_5b_instruct_policy_vector_s0.json
```

Expected: loss 12.02 → 1.96; strict exact stays at 0.048 (seen 0.056,
held-out 0.000). 15,232 additive embedding parameters trained. The kill
diagnostic.

### Rung 3b — compiled permission rail (works)

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind policy_vector \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --output results/slm/qwen25_0_5b_instruct_permission_rail_s0.json
```

Expected: strict exact 1.000 on seen and held-out OBEY+QUOTE masks. Only the
3 × 896 permission table is trained — 2,688 parameters.

### PR4 — 4-cell compositional grid

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind source_policy_grid \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --output results/slm/qwen25_0_5b_instruct_pr4_permission_grid_s0.json
```

Expected: 1.000 in all four cells (C1–C4); constant-policy trap 0.444;
invert-policy trap 0.000.

### PR5 — SEP projection (initial early kill)

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind sep_projection \
  --permission-rail oracle \
  --load-adapter <PR4 adapter checkpoint> \
  --output results/slm/qwen25_0_5b_instruct_pr5_sep_projection_eval_s0.json
```

Expected: denied SEP exact 0.900 (below the 0.95 transfer gate). Recorded as
the early-kill diagnostic that motivated PR5b.

### PR5b — paired SEP-surface adaptation

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind sep_paired \
  --permission-rail oracle \
  --load-adapter <PR4 adapter checkpoint> \
  --output results/slm/qwen25_0_5b_instruct_pr5b_sep_paired_s0.json
```

Expected: paired SEP exact 1.000, OPEN_OBEY 1.000, DECLINE_OBEY 1.000 after
300 adaptation steps.

### PR6 — learned operation detector (early kill)

```bash
uv run python scripts/slm_operation_detector.py \
  --output results/slm/qwen25_0_5b_instruct_pr6_operation_detector_s0.json
```

Expected: held-out template detection 0.615; shuffled-label trap 0.380. Below
the gate to feed the permission rail.

### PR7 — tiny binder on the simple lookup (positive)

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind policy_vector \
  --permission-rail binder \
  --binder-hidden-size 32 \
  --output results/slm/qwen25_0_5b_instruct_pr7_tiny_binder_s0.json
```

Expected: exact 1.000 on seen masks and the held-out `101` mask;
constant-policy 0.429; invert-policy 0.000. 29,792 trained parameters.

### PR7b — same binder on the PR4 grid (template fragility)

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind source_policy_grid \
  --permission-rail binder \
  --binder-hidden-size 32 \
  --output results/slm/qwen25_0_5b_instruct_pr7_binder_grid_s0.json
```

Expected: C1 / C3 = 1.000; C2 = 0.448; C4 = 0.438. The boundary that ruled
out the learned compiler as the strongest available.

### PR8 — multi-span oracle compiler (initial boundary, 300 steps)

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind multi_span_grid \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --steps 300 \
  --output results/slm/qwen25_0_5b_instruct_pr8_multispan_oracle_s0.json
```

Expected: exact 0.965; C2 = 0.938; C4 = 0.917; invert-policy 0.003.

### PR8b — multi-span fix (200-step protocol, clears gate)

```bash
uv run python scripts/slm_policy_vector.py \
  --dataset-kind multi_span_grid \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --steps 200 \
  --output results/slm/qwen25_0_5b_instruct_pr8b_multispan_oracle_step200_s0.json
```

Expected: exact 0.989; C1 = 1.000; C2 = 0.982; C3 = 1.000; C4 = 0.969;
constant-policy 0.444; invert-policy 0.002; zero distractor errors.

### PR9 — first scale replication on Qwen2.5-1.5B-Instruct (boundary)

```bash
uv run python scripts/slm_policy_vector.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-kind multi_span_grid \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --steps 200 \
  --batch-size 8 --eval-batch-size 4 \
  --output results/slm/qwen25_1_5b_instruct_pr9_multispan_oracle_step200_s0.json
```

Expected: exact 0.948; C1 = 0.979; C2 = 0.917; C3 = 0.979; C4 = 0.917;
invert-policy 0.017; zero distractor errors. First scale boundary — the
rail is causal and span-bound, but held-out-template OPEN_OBEY is the
weak axis. See `docs/pr9_postmortem.md` for the diagnosis.

### PR9b — extended-step regression (kill)

```bash
uv run python scripts/slm_policy_vector.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-kind multi_span_grid \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --steps 400 \
  --batch-size 8 --eval-batch-size 4 \
  --output results/slm/qwen25_1_5b_instruct_pr9b_multispan_oracle_step400_s0.json
```

Expected: matches PR9 through step 200, then regresses at step 300 onward.
Final: exact 0.882; C2 = 0.766; C4 = 0.760; held-out-template 0.764;
invert-policy 0.000; distractor error 0.000. Failures are mostly `other`
formatting errors — copying the full held-out carrier phrase instead of
the bare value. **More sample exposure does not fix scale transfer; it
overfits the model to template format.** Motivates PR9c.

### PR9c — value-delimited surface (1.5B scale rung clears)

```bash
uv run python scripts/slm_policy_vector.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-kind multi_span_grid \
  --template-family value_delimited \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --steps 200 \
  --batch-size 8 --eval-batch-size 4 \
  --output results/slm/qwen25_1_5b_instruct_pr9c_multispan_value_delimited_s0.json
```

Expected: **exact 1.000 across all four cells, every OPEN/DECLINE
primitive at 1.000, every error-type at 0.000, constant-policy 0.444,
invert-policy 0.000.** Same rail architecture, same step budget,
candidate-value surface explicitly delimited (`VALUE=...`, `[ ... ]`,
`<value>...</value>`). The 0.5B → 1.5B scale boundary was an output-
extraction ambiguity, not a capacity ceiling. PR10 risk-domain rails
are now unblocked.

### PR10 — risk rail on top of value-delimited permission stack

Training uses only the correct control and saves the adapter; the all-control
trap eval reloads the best adapter.

```bash
uv run python scripts/slm_policy_vector.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-kind risk_multispan_grid \
  --template-family value_delimited \
  --risk-rail embedding \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --steps 200 --eval-every 100 \
  --train-pairs 512 --eval-pairs 2 \
  --eval-use-heldout-values \
  --eval-controls correct \
  --batch-size 8 --eval-batch-size 16 \
  --output results/slm/qwen25_1_5b_instruct_pr10_risk_value_delimited_train_s0.json \
  --save-adapter results/slm/qwen25_1_5b_instruct_pr10_risk_value_delimited_s0.pt

uv run python scripts/slm_policy_vector.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-kind risk_multispan_grid \
  --template-family value_delimited \
  --risk-rail embedding \
  --permission-rail oracle \
  --no-policy-bit-embeddings --no-source-embeddings --no-operation-embeddings \
  --steps 0 \
  --train-pairs 16 --eval-pairs 4 \
  --eval-use-heldout-values \
  --eval-controls correct constant_policy invert_policy constant_risk invert_risk \
  --batch-size 8 --eval-batch-size 16 \
  --load-adapter results/slm/qwen25_1_5b_instruct_pr10_risk_value_delimited_s0.best.pt \
  --output results/slm/qwen25_1_5b_instruct_pr10_risk_value_delimited_s0.json
```

Expected: correct exact 0.995, C1/C2/C3/C4 all >= 0.993, risk-allow
0.988, risk-refuse 1.000, permission-decline 1.000, and distractor
error 0.000. Traps: constant-policy 0.444, invert-policy 0.005,
constant-risk 0.773, invert-risk 0.444. The risk rail is causal, but
this is still a synthetic rung, not a HarmBench/JailbreakBench/XSTest
result.

## Selfcheck Variants

Several `*_selfcheck_s0.json` files exist alongside the scored runs (Rung 1,
Rung 2, Rung 3a, PR4). They are produced by adding `--steps 10
--train-pairs 32 --eval-pairs 16` to the corresponding command and verify
that the pipeline initialises and emits a parseable result file. Use these
as a wiring check before any longer run.

## Where to Look After Reproducing

- Rung-by-rung narrative + analogies: [Explainer].
- Per-rung tables (numbers ↔ controls ↔ kill logic): [Technical].
- Forward proposals (provable output mask) and blind avenues: see the
  [Overview]'s `#blind-avenues` section.
- Connection to prior work (RepE, instruction hierarchy, Spotlighting,
  StruQ, ASIDE, AIR, role confusion, BIPIA): the microsite's
  [Literature](https://policy-rails-microsite.pages.dev/literature) tab.
- Plan and ladder status notes: `docs/policy_ir_ladder.md`.
- Stable result snapshots and interpretation: `docs/results.md`.
- Active run tracker: `docs/experiments.md`.

If a re-run on your machine produces numbers more than a few hundredths
away from the values above and the seed is unchanged, that is worth
opening an issue against this repo — the rail's strict-exact metrics
are tight by design.
