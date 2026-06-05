# Postmortem — rpcg1_capability_gate_pythia70m — 2026-05-18

**Verdict:** `VOID`
**Decision metric:** gate margin Δ (mean expected-sign-corrected per-cell shift, 9 cells)

## Result

| quantity | value |
|---|---|
| margin_gated | 0.0084 |
| τ (3× null SD) | 0.0137 |
| margin_trap | 0.0095 |
| margin_null | 0.0002 |
| sign_correct | 6 / 9 |
| trap_collapsed | True |
| probe1_forbidden_drop | False |
| probe2_permitted_fit | **False** |
| gated SFT converged (loss) | True |
| gated adapter mean srank | 1.74 |
| held-out-frame margin | 0.0092 |

Frozen decision rule (`H0.md` §3): the gate did not pass (`margin_gated 0.0084 ≤ τ
0.0137`); with `probe2_permitted_fit = False` the rule routes to **VOID** — "SFT did
not fit the permitted set". The verdict is the frozen-rule output, not a
discretionary call.

## Why VOID is correct here — and what it means

VOID is *not* a pipeline bug in this run (the trap collapsed cleanly, all four
probes ran, the corpus χ²/count self-checks passed). It is the §7-crux rule
declining to over-claim: we cannot promote a null margin to `NO_GATE` (a true
kill) because the SFT never demonstrably fit a *role-conditional* permitted set.

The probe outputs say exactly what happened:

- **Baseline (post-pretrain):** every one of the 9 cells sits at ≈ 0.33 open-token
  probability. Continued-pretrain on all 9 (role × primitive) cells in equal
  proportion taught the model a strong prior that **role is independent of
  primitive** — after any role+frame prefix all three open tokens are equiprobable.
- **gated and trap arms are near-identical** — mean |Δ| between them = **0.0044**.
  This is the decisive diagnostic. The `gated` and `trap` SFT sets are built from
  *different* role→permission maps but, by construction, have the *same marginal
  primitive frequency* (exec 2 / net 2 / sys 1 cells). Both arms shifted every
  role the same way (net ↑ ≈ 0.38, sys ↓ ≈ 0.29, exec flat ≈ 0.33). The model
  learned the **SFT marginal** and ignored the role entirely.
- `operator/sys` is *permitted* for `operator`, yet its probability *dropped*
  (0.330 → 0.288) — a role-conditioning model would have raised it. It fell
  because `sys` is the globally rare primitive in the SFT marginal.

So the model did not install a role-conditioned gate. But this is **not** a
`NO_GATE` finding, because the failure is upstream of the mechanism question:
positive-only LM-loss LoRA-QKV SFT did not even *fit* a role-conditional target.

## Root cause (per feedback_research_loop_order — model the result)

Two compounding causes, both about the **SFT signal**, not the role-gate mechanism:

1. **LM loss does not force role-conditioning.** Each SFT doc is ≈ 25 tokens; the
   open-token slot is one of them. The body tokens dominate the cross-entropy. The
   model can drive the SFT loss down (it converged: 0.16, stable) by fitting body
   text and the *marginal* open-token frequency — it is never *required* to use the
   role to reduce loss. Convergence on LM loss ≠ fitting the role-conditional map.

2. **Pretrain installed a role⊥primitive prior.** All 9 cells in equal proportion
   is required so every primitive is reachable from every role (else the gate is
   baked by pretrain, not alignment) — but equal proportion also teaches *certainty*
   that the role does not predict the primitive. Positive-only LoRA-QKV SFT, a weak
   intervention, cannot reverse that prior; it only nudges the marginal.

The marginal-matched `gated`/`trap` pair turned out to be an excellent accidental
control: identical marginals mean any role-conditioning would surface as a
gated≠trap difference. There is none (0.0044). The role was ignored, decisively.

## What the run does and does not establish

- **Does establish:** positive-only LM-loss LoRA-QKV SFT on Pythia-70m, on top of
  an all-9-cells-equal pretrain, installs *no* role-conditioning — it reproduces
  the SFT marginal uniformly across roles.
- **Does not establish:** anything about whether a role-conditioned capability gate
  *exists* or is installable by a stronger alignment signal. The NEW-class
  hypothesis (`H0` §1) was **not reached** — the SFT stage hit a methodology wall
  first. This is a Class I (methodology / setup) outcome, not a mechanism result.

## Escalation (recommended; not auto-run — awaiting direction)

The escalation ladder rung-1 (DPO) is gated on `INCONCLUSIVE_WEAK_SIGNAL`, which
this run did not produce. The correct next step for a VOID-by-SFT-weakness is to
**strengthen the SFT signal so it is forced to fit the role-conditional map**,
then re-run. Options, cheapest first:

1. **Open-token-weighted SFT loss** — mask the LM loss to the open-token slot (or
   up-weight it heavily). Forces the gradient through the role→primitive decision.
   Smallest change; keeps positive-only.
2. **DPO with explicit negatives** — in-role open token as chosen, out-of-role as
   rejected, at the decision slot. This is ladder rung-1's intent; it supplies the
   suppression signal positive-only SFT lacks.
3. **Full-FT SFT** instead of LoRA-QKV — removes the capacity bottleneck as a
   confound (LoRA-QKV r=8 may simply be too small a surface to overwrite the
   pretrained prior).

A design question also surfaced: the all-9-cells-equal pretrain installs a
role⊥primitive prior that any alignment stage must *reverse*. A future revision
could pretrain with the role token *masked* in the action region, or with a
mild role→primitive correlation, so alignment installs rather than reverses.

## Lean co-evolution

None. `H0` §11: a VOID outcome adds no theorem. The monotone-reachability theorem
(gate confirmed) and the role-gate KILL theorem (true `NO_GATE`) both stay
unwritten — neither verdict was reached.

## Reproduction

`src/rope_prov/train.py` (see the research monorepo for `<monorepo>/run_all.py`; commit recorded in
the experiment tracker). Pythia-70m, fp32, seed 42, RTX 3060. Wall ≈ 4 min.
Artifact: `<monorepo>/artifacts/rpcg1_capability_gate_pythia70m_2026-05-18.json`.

obstacle_class: I
