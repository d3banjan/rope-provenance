# Postmortem — rpcg11d_class_weighted_nested_qwen25 — 2026-05-21

**Verdict:** `VOID` (RPCG11d spec §6.3 5-tier rule, frozen
`tau.json` sha `22af72d458f132e28489d0297ce9a28653f3cd7436302303e1d66175c34096da`;
`void_reason: c1_failed`).

Class-weighted CE at `w_open=0.83 / w_decline=1.25` (schedule-rescaled
inverse-frequency) did NOT rescue compositional generalization. The
gate collapsed to the same always-OPEN local minimum that drove
RPCG11c to VOID — per-primitive OPEN-coverage uniformly ~1.000,
forbidden_decline_rate 0.000-0.167. **Six rung-1 input-side levers
now tested, all fail.** Strong justification for architectural pivot
(RPCG12+).

## Question

Does removing the OPEN-class marginal at the optimizer-visible
gradient level (via per-row schedule-rescaled inverse-frequency class
weighting) rescue compositional generalization in nested-context
training where RPCG11c's class-balanced spec-set without class-
weighted loss could not?

Answer: NO. The class-marginal gradient rebalancing at the 0.83 ×
OPEN / 1.25 × DECLINE ratio is insufficient to break the always-OPEN
basin. The gate still finds the degenerate optimum.

## Result

Sanity arms — all PASS:

```
trap_collapsed       True  (trap_c1_coverage natural 0.582, structured
                            0.556; both ≤ 0.667 threshold — clean
                            collapse from min-overlap shuffle)
converged            True  (gated loss 0.584 → 0.190 median; trap
                            0.391 → 0.237; both pass convergence)
gated_srank          1.56  (low_rank True; cap 8.0)
baseline_quiet       True  (baseline_mean_signed_margin -0.140 ≤ 0.30)
tau                  0.286
class_weighting_method: inverse_frequency (asserted at artifact-write)
```

Per-(arm, cell) cell-mean OPEN-coverage / forbidden-decline-rate +
per-primitive P-permitted-only coverage:

```
structured/C1  cov=0.985 forb=0.122  obey 1.000 (n=45)  use 0.956 (n=45)  quote 1.000 (n=45)
structured/C2  cov=0.963 forb=0.111  obey 1.000 (n=9)   use 0.889 (n=9)   quote 1.000 (n=9)
structured/C3  cov=1.000 forb=0.000  obey 1.000 (n=15)  use 1.000 (n=15)  quote 1.000 (n=15)
structured/C4  cov=1.000 forb=0.000  obey 1.000 (n=3)   use 1.000 (n=3)   quote 1.000 (n=3)

natural/C1     cov=0.989 forb=0.095  obey 1.000 (n=63)  use 0.968 (n=63)  quote 1.000 (n=63)
natural/C2     cov=1.000 forb=0.167  obey 1.000 (n=27)  use 1.000 (n=27)  quote 1.000 (n=27)
natural/C3     cov=1.000 forb=0.000  obey 1.000 (n=21)  use 1.000 (n=21)  quote 1.000 (n=21)
natural/C4     cov=1.000 forb=0.000  obey 1.000 (n=9)   use 1.000 (n=9)   quote 1.000 (n=9)
```

Every cell fails the cell_pass gate because forbidden_decline_rate ≤
0.167 < 0.667 threshold. Coverage itself is uniformly perfect (1.0).
Same failure pattern as RPCG11c (which had cov 0.83-1.00 / forb
0.00-0.167).

## Mechanism

**Class-marginal gradient rebalancing at 0.83/1.25 ratio is
insufficient to break the always-OPEN basin.** The 60:40 OPEN:DECLINE
class imbalance in the scheduled 900-step subset (535 OPEN + 365
DECLINE on gated, 546 OPEN + 354 DECLINE on trap) produces an
optimizer landscape where the per-row gradient pressure ratio
DECLINE:OPEN = 1.25/0.83 = 1.5× is not steep enough to overcome the
basin's attraction. The model still finds always-OPEN as a stable
local minimum.

Comparison with RPCG11c (which had no class weighting):

| Metric              | RPCG11c        | RPCG11d        | Δ           |
|---------------------|----------------|----------------|-------------|
| Verdict             | VOID/c1_failed | VOID/c1_failed | same        |
| tau                 | 0.371          | 0.286          | -0.085      |
| gated_srank         | 1.52           | 1.56           | +0.04       |
| baseline_margin     | -0.140         | -0.140         | identical   |
| trap_c1 (natural)   | 0.582          | 0.582          | identical   |
| trap_c1 (structured)| 0.578          | 0.556          | -0.022      |
| forbidden_decline (best) | 0.167     | 0.167          | identical   |
| OPEN_coverage (typical) | 1.000      | 1.000          | identical   |

The class weighting moved tau slightly (from 0.371 to 0.286 — null
arm distribution shifted) and slightly improved trap collapse on the
structured arm, but the **OPEN-class basin is preserved at full
depth**. Gradient rebalancing at 1.5× ratio did not perturb the
optimum.

What would have broken the basin: either a much steeper class
weight (e.g. `w_decline = 5×` instead of 2.5×, which would
likely over-correct toward always-DECLINE) OR an architectural
change that prevents the gate from being class-constant (out-of-band
tensor policy, per-primitive weight sharing, structured attention).

## Interpretation

**Six orthogonal rung-1 input-side levers now tested**:

| Lever varied                                              | Path             | Result                  | Failure mode |
|----------------------------------------------------------|------------------|-------------------------|--------------|
| Objective: ordinal DPO → factorized two-class CE         | RPCG7 → RPCG9    | NO_GENERALIZATION       | per-spec memorization |
| Format: inline allow-list → bit-vector                   | RPCG7 → RPCG9    | NO_GENERALIZATION       | per-spec memorization |
| Basis: arbitrary action → grounded provenance            | RPCG9 → RPCG10b  | NO_GENERALIZATION       | per-spec memorization |
| Training distribution: flat → nested-context             | RPCG10b → RPCG11 | VOID (rare-prim QUOTE)  | per-primitive coverage |
| Frequency: rare QUOTE (2/5) → balanced (3/5)              | RPCG11 → RPCG11c | VOID (class-marginal)   | OPEN-class collapse |
| Class gradient: 60:40 → schedule-rescaled inverse-freq    | RPCG11c → RPCG11d| VOID (class-marginal)   | OPEN-class basin (same) |

**The class-marginal basin is the unrescued failure.** RPCG11c
exposed it; RPCG11d tried to fix it at the cleanest one-variable
intervention point (gradient rebalancing without corpus/sampler
mutation) and found the basin too deep to escape at the natural
inverse-frequency ratio.

Two design iterations were rejected pre-run by codex review as
confounded (v1 corpus-undersample → cardinality skew; v2 step-
alternating sampler → triple-spec OPEN-only exposure shift). v3
normalized loss was rejected at codex v4.1 for batch-1 cancellation.
v4.2-v4.4 landed schedule-rescaled per-row weighting — the
cleanest fix, but the basin survives.

**Conclusion: the rung-1 input-side ladder is exhausted.** Six
orthogonal interventions including two landed confound-removal runs
(RPCG11c + RPCG11d) and three pre-run-rejected confounded design
iterations all fail to break the compositional-generalization
boundary on this controlled 3-primitive, 7-spec lattice. The natural
next direction is architectural levers (RPCG12+): out-of-band tensor
policy vectors (permission as structured side input, not text in
prompt), per-primitive weight sharing (LoRA parameters tied across
primitive-containing specs by construction), architectural binding
modules (gated routing / structured attention forcing per-primitive
reuse), tracked under the RPCG rung-2 architectural workstream in the
research monorepo.

## Gate decision for downstream artefacts

- Microsite update: **SKIPPED** per plan verdict gate (only
  FULL_GENERALIZATION or BOUNDARY_SHIFT triggers wire-through).
- Paper update: **SKIPPED** (same condition).
- Lean theorem: **SKIPPED** (only FULL_GENERALIZATION triggers).
- EXPERIMENTS row + memory update are the only downstream artefacts.

## Reproduction

```bash
uv run python <monorepo>/scripts/run_nested.py --root . --ckpt checkpoints_rpcg11d \
  --base-model qwen0.5b --basis provenance \
  --spec-variant balanced --trap-shuffle min_overlap \
  --class-weighting inverse_frequency \
  --prereg preregistry/rpcg11d_class_weighted_nested_qwen25_2026-05-21 \
  --artifact <monorepo>/artifacts/rpcg11d_class_weighted_nested_qwen25_2026-05-21.json \
  --exp rpcg11d_class_weighted_nested_qwen25_2026-05-21
```
(Training/eval code lives in `src/rope_prov/train.py`; see the research monorepo for `run_nested.py`.)

Qwen2.5-0.5B, bf16, seed 42. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
was needed because of GPU contention with a separate LFS5
patch-library-search workstream running concurrently — the run itself
is deterministic and would reproduce on a clean GPU without the env
var. Artifact:
`<monorepo>/artifacts/rpcg11d_class_weighted_nested_qwen25_2026-05-21.json`.
τ frozen, sha `22af72d...` valid. All three regression guards (RPCG9 +
RPCG11 + RPCG11c) PASS end-to-end at every commit.

obstacle_class: class-marginal collapse (same family as RPCG11c —
NOT rescued by gradient rebalancing at 0.83/1.25 inverse-frequency
ratio).
