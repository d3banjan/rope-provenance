# Postmortem — rpcg5_layer_localization_pythia70m — 2026-05-19

**Verdict:** `DIFFUSE`
**Decision metric:** `k_crit` — greedy-cumulative layers ablated to reach the
RPCG3 gate-removed threshold (margin ≤ 0.0059).

## Result

| quantity | value |
|---|---|
| **k_crit** | **5 / 6** |
| margin(no ablation) | 0.2229 — reproduces RPCG3 exactly (sanity ✓) |
| margin(all 6 ablated) | 0.0000 — adapter fully removed = baseline ✓ |
| monotonic greedy chain | True — `RoleGateReachability.logit_mono` reaffirmed |

Per-layer single-ablation margin drop (depth profile):

```
layer   margin after ablating it alone   drop from 0.2229
L0      0.0744                           0.1486   (dominant)
L3      0.1596                           0.0634
L1      0.1664                           0.0565
L2      0.1818                           0.0411
L4      0.1977                           0.0253
L5      0.2049                           0.0180   (smallest)
```

Greedy cumulative chain (layers ablated in descending single-effect order):

```
k=0  []                  0.2229
k=1  [0]                 0.0744
k=2  [0,3]               0.0363
k=3  [0,3,1]             0.0267
k=4  [0,3,1,2]           0.0146
k=5  [0,3,1,2,4]         0.0038   <- first <= tau 0.0059  => k_crit = 5
k=6  [0,3,1,2,4,5]       0.0000
```

`k_crit = 5 ≥ diffuse_k_min 5` → **DIFFUSE**.

## Interpretation (per feedback_research_loop_order)

**The role gate is low-rank but depth-diffuse.** RPCG3 already showed the gate is
rank-compact — the gated adapter's mean stable rank is 1.30, a near-rank-1 update
*within each layer's* QKV projection. RPCG5 shows the orthogonal fact: that
rank-compact update is **spread across all six layers**. No single layer, and no
layer pair, carries the gate — removing it needs 5 of 6 layers ablated.

The depth profile is **front-loaded but not localised**. Ablating L0 alone
removes 67% of the margin (0.149 of 0.223), far more than any other single layer
— the earliest layer does the most gate work. But the residual 0.074 is still an
order of magnitude above the gate-removed threshold, and it is distributed:
L1/L2/L3 each contribute a further 0.04–0.06, L4/L5 a tail. The gate is a
depth-distributed computation with an early-layer emphasis, not a single-site
circuit.

"Low-rank within a layer, diffuse across layers" is a coherent picture: the DPO
contrastive signal recruits a thin direction in every layer's QKV map and
composes them down the stack. This answers the spec's escalation rung 3 ("locate
the gate to layers/heads"): at layer granularity the answer is *diffuse,
L0-weighted*; a head-level localisation could refine the L0 finding but is not
required by this result.

## Lean co-evolution

No new theorem (`H0` §11). RPCG5 **empirically instantiates**
`lean/RoleGateReachability.lean`: each layer is one ablation
component; `logit_mono` predicts the greedy cumulative margin is non-increasing.
The observed chain `0.2229 → 0.0744 → 0.0363 → 0.0267 → 0.0146 → 0.0038 → 0.0000`
is strictly decreasing — the descriptive theorem's monotone-reachability
prediction matches data, and full ablation reaches the un-gated base
(`logit_full_ablation`: margin 0.0000). The theorem stands reaffirmed; no
theorem tracker (research monorepo) change.

## Reproduction

`ablate_localize.py --ckpt checkpoints_rpcg3` — no training, ablates and
re-probes the frozen RPCG3 gated adapter. Pythia-70m, fp32, RTX 3060, wall ≈ 1
min. Artifact:
`<monorepo>/artifacts/rpcg5_layer_localization_pythia70m_2026-05-19.json`.

obstacle_class: NEW
