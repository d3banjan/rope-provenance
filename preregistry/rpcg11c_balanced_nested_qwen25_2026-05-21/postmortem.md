# Postmortem — rpcg11c_balanced_nested_qwen25 — 2026-05-21

**Verdict:** `VOID` (RPCG11c spec §4.4 5-tier rule, frozen `tau.json` sha
`5854075aeb04b355436eead988740f98e5e75accde7d95d4a99b1e4674224975`;
`void_reason: c1_failed`).

The VOID is *strict-rule-valid* and surfaces a **completely different failure
mode from RPCG11**: balancing per-primitive frequency rescued the rare-primitive
coverage problem (every primitive now opens at 1.000 coverage everywhere
including held-out) but the gate collapsed to the **OPEN-class marginal**
(forbidden-decline-rate 0.000–0.167 vs the 0.667 cell-pass threshold). The
trained gate is "always OPEN", not the role→permission map. RPCG11c isolates
a 5th orthogonal failure mode and tells us the per-primitive frequency prior
was *one* confound; the class-imbalance prior (60:40 OPEN:DECLINE) is the
load-bearing remaining lever.

## Question

Does balanced per-primitive frequency in a nested-context corpus with
latent-candidate two-class CE training induce compositional generalization
where RPCG11 (rare-primitive QUOTE) failed?

## Result

Sanity arms — all green:

```
trap_collapsed       True  (trap_c1_coverage natural 0.582, structured 0.578;
                            both ≤ 0.667 threshold — clears collapse cleanly
                            after min-overlap shuffle, unlike RPCG11's
                            knife-edge 0.639)
converged            True  (gated loss 0.576 -> 0.180 median)
gated_srank          1.52  (low_rank True; cap 8.0)
baseline_quiet       True  (baseline_mean_signed_margin -0.140; |·| ≤ 0.30)
tau                  0.371
```

Per-(arm, cell) cell-mean OPEN-coverage / forbidden-decline-rate and
per-primitive P-permitted-only coverage:

```
structured/C1  cov=0.993 forb=0.167  obey 1.000 (n=45)  use 0.978 (n=45)  quote 1.000 (n=45)
structured/C2  cov=0.963 forb=0.167  obey 1.000 (n=9)   use 0.889 (n=9)   quote 1.000 (n=9)
structured/C3  cov=1.000 forb=0.000  obey 1.000 (n=15)  use 1.000 (n=15)  quote 1.000 (n=15)
structured/C4  cov=1.000 forb=0.000  obey 1.000 (n=3)   use 1.000 (n=3)   quote 1.000 (n=3)

natural/C1     cov=1.000 forb=0.159  obey 1.000 (n=63)  use 1.000 (n=63)  quote 1.000 (n=63)
natural/C2     cov=1.000 forb=0.167  obey 1.000 (n=27)  use 1.000 (n=27)  quote 1.000 (n=27)
natural/C3     cov=1.000 forb=0.000  obey 1.000 (n=21)  use 1.000 (n=21)  quote 1.000 (n=21)
natural/C4     cov=1.000 forb=0.000  obey 1.000 (n=9)   use 1.000 (n=9)   quote 1.000 (n=9)
```

Every cell fails because `forbidden_decline_rate ≤ 0.167 < 0.667`. Coverage
itself is uniformly perfect (1.0) — the gate opens every primitive on every
prompt, including those whose outer spec FORBIDS them. This is the
"always OPEN" marginal, not a gate.

## Mechanism

**Class-imbalance marginal collapse.** Balanced per-primitive training
exposes the binding model to OPEN:DECLINE rows in a 60:40 ratio (each spec
has 3 P-cells; each primitive permitted in 3 of 5 trained specs = 9 OPEN
cells / 6 DECLINE cells per primitive; binding rows uniform across cells).
The model learns the marginal — always answer OPEN — because that's the
class-majority strategy and the two-class CE loss does not penalize it more
than penalizing the wrong gate on individual rows.

RPCG11's rare-primitive QUOTE failure mode masked this. QUOTE-permitted
specs (only 2 of 5) drove the model toward declining QUOTE specifically,
which incidentally tilted away from always-OPEN. With QUOTE balanced
(3/5), the always-OPEN strategy is locally optimal.

This is the **exact opposite confound** to RPCG11: instead of QUOTE
collapsing to 0 coverage in held-out, every primitive is at 1.0 coverage
everywhere — and the decline rate collapses instead. Same root: the
training objective doesn't force role-conditional behavior when the
marginal already explains most rows.

**Trap collapse is now clean** — min-overlap shuffle dropped trap_c1
from RPCG11's 0.639 to RPCG11c's ~0.58 in both arms (well below the 0.667
threshold). The trap discipline pre-registered in tau.json worked.

**Baseline_quiet is barely clearing** — baseline_mean_signed_margin
-0.140 < 0.30 threshold. Negative because forbidden P-cells contribute
margin × -1 (sign for forbidden); a small forbidden-bias in continued-pretrain
shows up here. Below the threshold so the gate fires non-VOID, but worth
noting that the baseline isn't perfectly decorrelated.

## Interpretation

**Five orthogonal levers now tested on the rung-1 binding-layer program**:

| Lever varied                                              | Path             | Result                  | Failure mode |
|----------------------------------------------------------|------------------|-------------------------|--------------|
| Objective: ordinal DPO → factorized two-class CE         | RPCG7 → RPCG9    | NO_GENERALIZATION       | per-spec memorization |
| Format: inline allow-list → bit-vector                   | RPCG7 → RPCG9    | NO_GENERALIZATION       | per-spec memorization |
| Basis: arbitrary action → grounded provenance            | RPCG9 → RPCG10b  | NO_GENERALIZATION       | per-spec memorization |
| Training distribution: flat → nested-context outermost-wins | RPCG10b → RPCG11 | VOID (rare-prim QUOTE)  | per-primitive coverage |
| Frequency: rare QUOTE (2/5) → balanced (3/5) under nested | RPCG11 → RPCG11c  | VOID (class-marginal)   | OPEN-class collapse |

**Each input-side lever has a distinct failure mode**. RPCG11c is not just
"NO_GENERALIZATION with different numbers" — it's a categorically new
failure (the gate becomes degenerate, not just non-compositional). The
class-imbalance lever is the obvious unexpended next experiment, but the
sequence of failure modes — per-spec memorization → per-primitive coverage
collapse → class-marginal collapse — suggests every input-side lever
exposes a different local optimum the two-class CE can settle into.

**What this does NOT establish**: it does not refute the nested-context
direction. Trap collapse is clean, in-distribution coverage is perfect,
sanity arms green. The pipeline is sound.

**What this DOES establish**: the per-primitive frequency prior was *one*
input-side confound; the class-imbalance prior (60:40 OPEN:DECLINE) is the
load-bearing remaining input-side lever. A future RPCG11d would
class-balance binding rows (undersample OPEN or upsample DECLINE so the
two-class CE has no marginal solution). If RPCG11d also produces a new
failure mode rather than composition, the input-side ladder is genuinely
exhausted — every direction unlocks a new local optimum the gate can hide
in. That justifies the move to architectural levers (RPCG12+).

## Gate decision for downstream artefacts

- Microsite update: **SKIPPED** per plan verdict gate (only
  FULL_GENERALIZATION or BOUNDARY_SHIFT triggers wire-through).
- Paper update: **SKIPPED** (same condition).
- Lean theorem: **SKIPPED** (only FULL_GENERALIZATION triggers).
- EXPERIMENTS row + memory update are the only downstream artefacts.
- Future direction: RPCG11d (class-balanced nested corpus) is the natural
  next experiment if input-side levers are to be exhausted exhaustively;
  otherwise jump to RPCG12+ architectural levers.

## Reproduction

```bash
uv run python <monorepo>/scripts/run_nested.py --root . --ckpt checkpoints_rpcg11c \
  --base-model qwen0.5b --basis provenance \
  --spec-variant balanced --trap-shuffle min_overlap \
  --prereg preregistry/rpcg11c_balanced_nested_qwen25_2026-05-21 \
  --artifact <monorepo>/artifacts/rpcg11c_balanced_nested_qwen25_2026-05-21.json \
  --exp rpcg11c_balanced_nested_qwen25_2026-05-21
```
(Training/eval code lives in `src/rope_prov/train.py`; see the research monorepo for `<monorepo>/scripts/run_nested.py`.)

Qwen2.5-0.5B, bf16, seed 42. Artifact:
`<monorepo>/artifacts/rpcg11c_balanced_nested_qwen25_2026-05-21.json`.
τ frozen, sha valid. RPCG9 + RPCG11 regression guards PASS end-to-end at
every commit through Phase 2; both still PASS after the scored run.

obstacle_class: class-marginal collapse (NEW failure mode, distinct from
RPCG7/9/10b per-spec memorization and RPCG11 per-primitive coverage
collapse). Same family as RPCG3's bare-DPO probability evacuation in a
structural sense: the objective has a degenerate solution the model
prefers over the role-conditional one.
