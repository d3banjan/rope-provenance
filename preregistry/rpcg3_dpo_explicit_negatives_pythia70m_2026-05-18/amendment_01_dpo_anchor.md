# Amendment 01 — RPCG3 DPO probability-evacuation fix

**Date:** 2026-05-18 (post-H0-freeze; commit `d5e3b6f` froze H0/tau)
**Applies to:** `rpcg3_dpo_explicit_negatives_pythia70m_2026-05-18`
**Trigger:** pipeline degeneracy discovered on the first RPCG3 run.

## What happened

The pre-registered RPCG3 recipe — bare single-token DPO at the open-token slot,
`loss = -logσ(β·(π − ref))`, β = 0.1 — **degenerated**. The first run trained
cleanly (DPO loss dropped 0.69 → 0.49 on the gated arm) but every probe cell read
**p(open_token) = 0.000** for all three primitives in all 9 cells, in both the
gated and trap arms.

Diagnosis: a bare DPO loss constrains only the chosen−rejected *gap*. It does not
anchor absolute probability. Each primitive token is `chosen` in some pairs and
`rejected` in others; the optimizer reduced the loss by pushing probability mass
*off the primitive tokens entirely* onto unrelated vocabulary — the β = 0.1
reference-KL term is far too weak to hold the absolute mass. This is the
well-documented DPO failure mode (DPO can drive down the likelihood of the chosen
response); it is acute here because the objective is a single token with no
sequence-level NLL anchoring.

The resulting verdict (VOID via Probe-2 fail) is a **pipeline degeneracy**, not a
valid test of whether DPO installs a role gate.

## Resolution

Add a **chosen-NLL anchor** — the standard DPO+SFT (RPO) objective:

```
loss = -logσ(β·(π − ref))  +  λ · ( -logπ(chosen) )
```

with `λ = ANCHOR_LAMBDA = 1.0`. The anchor is exactly the open-token SFT loss on
the *chosen* token; it holds the chosen primitive's absolute probability up while
the DPO term supplies the contrastive role signal. This is the minimal recipe
change that fixes the degeneracy: it touches neither β, the LoRA hyperparameters,
the step count, the arms, nor the decision rule.

This is a pipeline-bug fix, analogous to LRS1 `amendment_01_optimizer_control`
(post-freeze GPU-OOM fallback). The RPCG3 *intent* — a contrastive objective that
penalises the marginal — is unchanged; the anchor only prevents probability
evacuation.

## Amended recipe

`<monorepo>/scripts/dpo_role.py`: `dpo_loss()` returns `(total, dpo_component, anchor_component)`;
`total = dpo + λ·anchor`. Manifests record `args.anchor_lambda` and the per-step
`dpo` / `anchor` component means. β, LoRA r=8 α=16, 900 steps, 3 arms, the probe,
`<monorepo>/scripts/decide.py`, and every τ are unchanged.

## Decision rule

Unchanged. The H0 §3 verdicts and τ apply as frozen. The amended run is the
RPCG3 result of record; the degenerate first run is retained only as this
amendment's evidence.
