# Postmortem — rpcg2_open_token_weighted_sft_pythia70m — 2026-05-18

**Verdict:** `VOID` (via trap-not-collapsed)
**Decision metric:** gate margin Δ (mean expected-sign-corrected per-cell shift, 9 cells)
**Escalation:** rung 1 of RPCG1 — SFT loss masked to the open-token slot.

## Result

| quantity | RPCG2 | RPCG1 |
|---|---|---|
| margin_gated | 0.0241 | 0.0084 |
| τ (3× null SD) | 0.0192 | 0.0137 |
| margin_trap | **0.0234** | 0.0095 |
| margin_null | 0.0006 | 0.0002 |
| sign_correct | 6 / 9 | 6 / 9 |
| trap_collapsed | **False** | True |
| probe1_forbidden_drop | True | False |
| probe2_permitted_fit | True | False |
| gated adapter mean srank | 2.13 | 1.74 |
| gated-vs-trap mean \|Δ\| | 0.0208 | 0.0044 |

Frozen decision rule: `trap_collapsed = False` (`margin_trap 0.0234 > τ 0.0192`)
routes to **VOID**. Note that the gate would not have passed anyway —
`sign_correct = 6 < 8`.

## What the open-token-masked loss did, and did not, do

The masked loss had a real effect. Probe-1 and Probe-2 now pass, the gated margin
roughly tripled (0.008 → 0.024), and `sys` open-token probability was driven from
≈ 0.33 down to ≈ 0.22. **But this is all marginal sharpening, not role
conditioning.** The probe table is decisive:

```
cell           base   gated   trap
analyst/exec   0.330  0.386  0.370
operator/exec  0.331  0.386  0.373     <- exec ~identical across roles
retriever/exec 0.335  0.395  0.380
analyst/net    0.343  0.374  0.412
operator/net   0.339  0.368  0.407     <- net ~identical across roles
retriever/net  0.343  0.390  0.405
analyst/sys    0.327  0.240  0.217
operator/sys   0.330  0.246  0.220     <- sys ~identical across roles
retriever/sys  0.321  0.214  0.215
```

Within the `gated` arm every role receives the *same* primitive distribution
(exec ≈ 0.39, net ≈ 0.38, sys ≈ 0.22). `operator/sys` is permitted for `operator`
yet sits at 0.246 — a role-conditioning model would have raised it well above the
other roles' `sys`. It did not. The role is still ignored.

## Why the trap did not collapse — and why that is correct, not a bug

`H0` §6 says "trap Δ > τ → halt, pipeline bug." Here trap Δ > τ is **not** a bug —
it is the trap working exactly as designed, catching a non-role-specific margin.

The `gated` and `trap` SFT sets, by construction, have the *same* marginal
primitive frequency (exec 2 / net 2 / sys 1 cells). The model learned that
marginal in both arms. A uniform marginal shift "net ↑ δ, sys ↓ δ, exec flat"
produces, under the true-map expected signs, a **spurious positive** gate margin
of +2δ/9 — because `net` happens to be permitted for 2 of 3 roles, so the
marginal is partially correlated with the true map. Both `gated` and `trap` carry
the identical marginal, so both produce the same spurious +0.023–0.024 margin.
The trap arm refusing to collapse is the pre-registered control correctly
reporting: **the gated margin is marginal-driven, not role-conditional.** The
gated-vs-trap mean |Δ| (0.021) is itself a uniform-across-roles shift, not a
role-specific one.

## Root cause (per feedback_research_loop_order)

The open-token-masked loss removed RPCG1's "body tokens dominate the loss"
problem, but a second wall remains: **the marginal solution is a local optimum of
the masked loss, and positive-only LoRA-QKV (r=8) does not escape it.**

For the gated arm, predicting the pooled marginal (exec .4 / net .4 / sys .2)
gives loss ≈ −ln(0.4) ≈ 0.92 on every doc. The role-conditional solution
(analyst→exec→1.0; operator→{net,sys}→0.5/0.5) gives strictly lower loss — but
reaching it requires the model to attend to the `[role: …]` token and route on
it, a harder basin. The gated SFT loss converged to ≈ 1.0–1.17, i.e. it sat in
the marginal basin. A rank-8 QKV adapter under a positive-only objective supplies
no force pushing it out: positive-only demonstration never directly penalises the
marginal, it only fails to maximally reward it.

## What the run establishes

- **Establishes:** an open-token-masked positive-only LoRA-QKV SFT still installs
  no role-conditioning on Pythia-70m — it sharpens the SFT marginal uniformly
  across roles. Confirmed by gated ≈ trap and by flat-across-roles probe rows.
- **Does not establish:** anything about whether a role gate is installable by a
  *contrastive* objective. The NEW-class hypothesis remains unreached. Class I
  (methodology — the positive-only objective admits the marginal as a local
  optimum the weak adapter does not escape).

## Escalation → RPCG3 (DPO, explicit negatives)

Auto-escalation authorised by the user. The next rung supplies the missing force:
**DPO with explicit negatives** — at the decision slot, in-role open token =
chosen, out-of-role open token = rejected. A contrastive objective *directly*
penalises the marginal: predicting the marginal scores chosen and rejected
equally, so the DPO loss is only reduced by making chosen > rejected, which
requires role conditioning. If RPCG3 also fails, RPCG4 (full-FT SFT) tests
whether the rank-8 QKV adapter is itself the capacity bottleneck.

## Lean co-evolution

None. `H0` §11: VOID adds no theorem. `role_gate_monotone_reachability` stays
blocked in the theorem tracker (research monorepo).

## Reproduction

`<monorepo>/scripts/run_all.py --sft-loss open_masked --ckpt checkpoints_rpcg2`. Pythia-70m, fp32,
seed 42, RTX 3060. Wall ≈ 4 min. Artifact:
`<monorepo>/artifacts/rpcg2_open_token_weighted_sft_pythia70m_2026-05-18.json`.

obstacle_class: I
