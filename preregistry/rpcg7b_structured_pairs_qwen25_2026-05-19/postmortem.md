# Postmortem — rpcg7b_structured_pairs_qwen25 — 2026-05-19

**Verdict:** `NO_GENERALIZATION`
**Rung:** B of the RPCG7 compositional-generalization ladder.

## Result — near-identical to rung A

| quantity | rung B (structured) | rung A (plain) |
|---|---|---|
| verdict | NO_GENERALIZATION | NO_GENERALIZATION |
| in_dist_margin | 0.2567 | 0.2575 |
| τ | 0.0370 | 0.0203 |
| auditor_margin {exec,sys} | 0.2360, sign 2/3 | 0.2360, sign 2/3 |
| singleton_margin {sys} | −0.1956, sign 1/3 | −0.2025, sign 1/3 |
| trap_collapsed | True | True |
| gated srank | 1.35 | 1.37 |
| compositionality_score | 0.623 | 0.646 |

The structured-pair intervention (role/list-decorrelation pairs) **moved
nothing**. Held-out probe, gated arm:

```
auditor {exec,sys}   exec 0.997   net 0.002   sys 0.000   (sys permitted, crushed)
janitor {sys}        exec 0.001   net 0.973   sys 0.026   (sys permitted, crushed; forbidden net promoted)
net_sys  (trained)   exec 0.000   net 0.972   sys 0.024   (sys underopens in-distribution)
```

## Diagnosis — the structured arm addressed the wrong axis

Rung B's hypothesis was that the model rides the *role name* instead of the
inline list; the structured pairs decorrelate them. The null result falsifies
that hypothesis: decorrelation changed nothing because the failure was never
role-vs-list. It is a **per-primitive failure specific to `sys`** — the `sys`
gate never learned to open as a positive, list-driven output, even
*in-distribution* (`{net,sys}` → sys 0.024).

Root cause, sharpened from rungs A+B: two compounding effects.

1. **Frequency imbalance.** Across the 5 trained specs, `sys` is a *chosen*
   primitive in only 2 specs ({net,sys}, {exec,net,sys}); `exec` in 3, `net` in
   4. `sys` is the globally rarest chosen primitive — the RPCG1/2 marginal-
   frequency failure mode, here induced by the held-out split (holding out
   {exec,sys} and {sys} left the trained specs `sys`-poor).
2. **DPO is ordinal.** A DPO pair only needs `chosen > rejected`. In a
   multi-permitted spec like {net,sys} the model satisfies both `net>exec` and
   `sys>exec` by putting net ≈ 0.97 and sys ≈ 0.02 — sys barely above the
   rejected primitive. DPO never requires sys to be *high*, only above the
   out-of-set primitive. So the prior-favoured primitive absorbs the mass and
   the other permitted primitive gets scraps.

## Implication for rung C — the planned regularizer is mistargeted

Rung C as planned adds a **cross-spec per-primitive consistency** regularizer
(penalize a primitive's open-prob varying across specs that permit it). But
`sys` is *already consistent* — consistently ≈ 0.02 across {net,sys} and
{exec,net,sys}. A consistency regularizer is near-satisfied by a consistently
*wrong* gate; it has no gradient to lift `sys`. Rung C as specified would likely
return NO_GENERALIZATION for a reason that is an artifact of the regularizer
being the wrong tool — a misleading kill.

The actual fixes the A+B diagnosis points to: (a) **per-primitive
frequency-balanced** chosen/rejected pairs, and (b) a within-spec
**cross-primitive mass-sharing** pressure (each permitted primitive in a spec
should open comparably, not just beat the rejected one). Rung C should be
redesigned around these before it is run — recorded as an open decision, not
auto-escalated.

## Lean co-evolution

None — NO_GENERALIZATION.

## Reproduction

`<monorepo>/scripts/run_perms.py --rung b --base-model qwen0.5b --ckpt checkpoints_rpcg7`.
Qwen2.5-0.5B, bf16, seed 42. Continued-pretrain reused from rung A. Wall ≈ 15 min.
Artifact: `<monorepo>/artifacts/rpcg7b_structured_pairs_qwen25_2026-05-19.json`.

obstacle_class: NEW
