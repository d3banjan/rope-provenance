# Postmortem — rpcg9_bitvector_bce_qwen25 — 2026-05-19

Two runs. **Run 1** (per-primitive BCE) — `VOID_DESIGN_FLAW`, retained below as
evidence. **Run 2** (amendment_02 — candidate-specific two-class CE) —
**`NO_GENERALIZATION`**, the scientific result; see the final section.

## Run 1 — VOID_DESIGN_FLAW

**Verdict:** `VOID` — reclassified **`VOID_DESIGN_FLAW`**.
**Not** a scientific result; not evidence for or against the factorized
permission-vector idea. The objective as implemented is mis-specified.

## Result (the invalid run)

`decide_perms` returned VOID: `in_dist_margin −0.034 ≤ τ 1.014` (in-distribution
gate not installed; τ enormous).

## Diagnosis — the per-primitive BCE evacuates open-token probability

The gated-arm probe is decisive — open-token probabilities collapse to ≈ 0 on
nearly every spec:

```
exec          exec 0.32->0.00  net 0.35->0.00  sys 0.33->0.00
net           exec 0.32->0.00  net 0.35->0.00  sys 0.32->0.00
exec_net      exec 0.32->0.00  net 0.35->0.00  sys 0.33->0.00
net_sys       exec 0.32->0.00  net 0.35->0.00  sys 0.33->0.00
exec_net_sys  exec 0.32->0.13  net 0.36->0.35  sys 0.32->0.06
exec_sys (held-out)  exec 0.002  net 0.007  sys 0.000
```

The null arm collapses identically (every cell ≈ 0.00).

**Root cause.** The objective is `binary_cross_entropy_with_logits` on the
*isolated raw vocabulary logit* of each open token. That optimizes
`sigmoid(raw_logit_P) ≈ target_P` — but (1) `sigmoid ≈ 1` only needs a raw logit
of ≈ +5, which does **not** make the token dominate a 151k-vocab softmax, and
(2) the forbidden targets drive their logits toward −∞. The net effect is the
model moves probability mass *off all three open tokens* — exactly the
probability-evacuation failure class that RPCG3 hit with bare single-token DPO
(RPCG3 `amendment_01`). The objective optimizes a quantity the probe (and the
gate semantics) does not read: it controls an isolated sigmoid, not the
softmax probability.

**The τ = 1.01 is downstream of the same evacuation, not an independent
null-arm issue.** The null arm sits at ≈ 0.00 on every cell while the baseline
is ≈ 0.33; the per-cell signed shifts are therefore ≈ ±0.33, SD ≈ 0.34, so
τ = 3 × 0.34 ≈ 1.01. The initial hypothesis (null-arm random-target scrambling)
named a real symptom but the cause is the objective itself — fixing only the
null arm would leave the gated arm degenerate and the run would VOID again.

## What this does NOT establish

Nothing about whether explicit factorized supervision induces a compositional
gate. The bit-vector format + per-primitive-binary idea is untested — the
objective implementation never produced a probe-visible gate, even
in-distribution. RPCG9 must be re-run with a corrected objective before any
interpretation. obstacle_class: I (methodology).

## Amendment required (see `amendment_01_objective.md`)

The per-primitive objective must be redesigned so it does not evacuate
probability — e.g. a per-primitive two-class cross-entropy between the open
token and a designated decline/anchor token (allowed → open token, forbidden →
decline), or a BCE on a renormalized open-token probability with an
absolute-mass anchor. The null arm is then a non-scrambling control
(label-shuffled-but-structured, or frozen-base probe noise). Re-run after.

## Reproduction (invalid run, retained as evidence)

`run_bce.py --root . --ckpt checkpoints_rpcg9 --base-model qwen0.5b`.
Qwen2.5-0.5B, bf16, seed 42. Artifact:
`<monorepo>/artifacts/rpcg9_bitvector_bce_qwen25_2026-05-19.json`.

obstacle_class: I

---

# Run 2 — amendment_02 (candidate-specific two-class CE) — NO_GENERALIZATION

**Verdict:** `NO_GENERALIZATION`. A genuine scientific result — the redesigned
objective is sound, and factorized supervision does **not** induce a
compositional gate.

## The objective is valid (the v1 flaw is closed)

- In-distribution gate fully installed: **15/15** train cells clear τ, mean
  signed margin **+5.90** (τ = 0.234). No probability evacuation — the
  open-vs-decline margin is a logit difference, and the two-class CE trained it
  directly. Smoke-verified on Pythia-70m first (separation −0.15 → +7.0).
- Trap collapsed: shuffled-map arm reads −0.239 ≤ τ on the true map.
- Gate low-rank (srank 1.32 over 72 layers), converged, baseline ≈ 0 (−0.075 —
  continued-pretrain taught the format, not the map).

This is `NO_GENERALIZATION`, not `VOID` — the verdict is interpretable.

## Held-out specs do not compose

Per-cell signed margin (positive = correct; allowed→open, forbidden→decline):

```
auditor  exec_sys {exec,sys}   exec +5.63 ✓   net  -1.94 ✗   sys -1.21 ✗   -> 1/3
janitor  sys      {sys}        exec -4.99 ✗   net  -4.65 ✗   sys +1.43 ✓   -> 1/3
```

On the held-out `auditor` combination the model opens forbidden `net` and
declines permitted `sys`; on the held-out singleton `janitor` it strongly opens
both forbidden primitives (−4.99, −4.65). On unseen specs the gate defaults to
**opening** — consistent with the 9-permitted : 6-forbidden training imbalance
(a prior toward open). The held-out decision template gives the same verdict.

Per-primitive margin spread across the trained specs that permit it — exec
2.01, net 2.55, sys 0.36 (mean 1.64): a primitive's open-margin varies by ~2
logits with the surrounding spec. The gate is **spec-conditioned**, not a clean
per-primitive function.

## Interpretation — sharpens the RPCG7 negative

RPCG7 found compact set-list DPO does not compose; that negative could be
blamed on the *ordinal* DPO objective (winner-take-all) or on the *set-parsing*
burden. RPCG9 amendment_02 removes **both** — independent per-candidate binary
decisions (no ordinal competition) over a **bit-vector** format (no set to
parse) — and still gets `NO_GENERALIZATION`. The non-compositionality is robust
to objective and format: this model memorizes per-spec gates and does not
induce a factorized role→permission binding layer. Consistent with RPCG7c2 (a
mass-sharing regularizer also failed). Factorizing the *supervision* is not
enough; the open lever is the *representation* — the provenance-operation
primitive basis of RPCG10.

## Reproduction (Run 2)

`run_binding.py --root . --ckpt checkpoints_rpcg9_amended --base-model qwen0.5b`
(smoke-gate `binding_perms.py --smoke` first). Qwen2.5-0.5B, bf16, seed 42.
Artifact: `<monorepo>/artifacts/rpcg9_bitvector_bce_qwen25_2026-05-19_amended.json`.
τ frozen, sha `9a8132a0…` valid.

obstacle_class: compositional-generalization negative (same family as RPCG7).

> Note: bare `*.py` module names in this document name scripts in the research
> monorepo (training/eval pipeline); they are not part of this public repository.
