# Postmortem — rpcg11_nested_context_qwen25 — 2026-05-21

**Verdict:** `VOID` (RPCG11 spec §3 5-tier rule, frozen `tau.json` sha
`5d01d788d85ab410511e957331af637ecc3efaa6fa6836c1b99d13510dee3753`).

The VOID is *strict-rule-valid* and **methodologically clean**: trap collapsed,
training converged, gate low-rank, baseline ≈ 0. The mechanism that fails C1
is the per-primitive frequency prior on the rare primitive (QUOTE) — the
exact pattern RPCG10b documented on SYS. Nested-context supervision did
**not** rescue it. The held-out cells decisively fail too (QUOTE coverage
0.000 in C3 and C4), so this is informative on the central question: the
nested lever does not induce compositional generalization on the rare
primitive.

## Question

Does a recursive nested-context corpus with latent-candidate two-class CE
training — outer role wrapper around `<|context|>`-wrapped inner attempts —
induce compositional generalization where RPCG7/9/10b (flat-corpus,
candidate-explicit) did not? Two arms (structured-tag injection +
natural-language imperative injection); 4-cell crossed held-out grid; 5-tier
verdict.

## Result

Sanity arms — all green:

```
trap_collapsed     True  (trap C1 coverage 0.639 ≤ τ-gate 0.667; both arms)
converged          True
gated_srank        1.36  (low_rank True; cap 8.0)
baseline           ≈ 0   (continued-pretrain map untaught by 50/50 decorrelation)
tau                0.357
```

Per-(arm, cell, primitive) OPEN coverage (P-permitted rows only;
**OPEN iff margin > τ = 0.357**) and per-cell forbidden-decline rate:

```
structured/C1  cov=0.833 forb=1.000 pass=False    obey 1.000 (n=45)  use 1.000 (n=60)  quote 0.500 (n=30)
structured/C2  cov=0.833 forb=1.000 pass=False    obey 1.000 (n=9)   use 1.000 (n=12)  quote 0.500 (n=6)
structured/C3  cov=0.500 forb=0.667 pass=False    obey 1.000 (n=15)  use   N/A         quote 0.000 (n=30)
structured/C4  cov=0.500 forb=0.667 pass=False    obey 1.000 (n=3)   use   N/A         quote 0.000 (n=6)

natural/C1     cov=0.833 forb=1.000 pass=False    obey 1.000 (n=63)  use 1.000 (n=84)  quote 0.500 (n=42)
natural/C2     cov=0.833 forb=1.000 pass=False    obey 1.000 (n=27)  use 1.000 (n=36)  quote 0.500 (n=18)
natural/C3     cov=0.500 forb=0.667 pass=False    obey 1.000 (n=21)  use   N/A         quote 0.000 (n=42)
natural/C4     cov=0.500 forb=0.667 pass=False    obey 1.000 (n=9)   use   N/A         quote 0.000 (n=18)
```

The cell-mean coverages are clean — every C1/C2 cell sits at 0.833 with
forbidden-decline 1.000 — but the spec's `cell_pass` requires **every
non-N/A per-primitive coverage > 0.667**, and QUOTE at 0.500 in every
in-distribution cell drives all four cells to `cell_pass = False`. Per the
verdict precedence (VOID dominates), C1's failure short-circuits the 5-tier
verdict to VOID.

## Mechanism

The per-primitive frequency prior. QUOTE is permitted in exactly 2 of the
5 trained outer specs — the same structural position SYS held in RPCG10b
(also 2/5 trained, and also the primitive that broke composition there).
The same imbalance, the same outcome: the model learns OBEY and USE
reliably (each in 3-4 of the 5 trained specs), and QUOTE gets only enough
gradient pressure to land at ~50% coverage in-distribution. On held-out
specs, QUOTE collapses to 0% (model defaults to declining the rarely-seen
permitted-QUOTE configuration). USE is N/A in C3/C4 (neither held-out spec
permits USE — by construction, since the held-out specs are `{obey,quote}`
and `{quote}`).

The trap-collapse margin is **tight** — trap_c1_coverage of 0.639 is
just below the 0.667 collapse threshold. The cyclic-shuffled trap arm
partly learned the wrong map well enough to still satisfy ~64% of true-map
cells on coincidence; with the rare-primitive imbalance, this is plausibly
not coincidence but the same OBEY+USE-dominate-everything pattern (a trap
that fits OBEY's and USE's gating cells correctly because those primitives
are gating-trivial — present-permitted-everywhere — will look ~⅔ correct on
the true map regardless of the shuffled-map structure). Worth flagging for
the next rung.

## Interpretation

**Four orthogonal input-side levers now exhausted on the rare-primitive
frequency-prior pattern**:

| Lever varied                                              | Path           | Result                  |
|----------------------------------------------------------|----------------|-------------------------|
| Objective: ordinal DPO → factorized two-class CE         | RPCG7 → RPCG9  | NO_GENERALIZATION       |
| Format: inline allow-list → bit-vector                   | RPCG7 → RPCG9  | NO_GENERALIZATION       |
| Basis: arbitrary action → grounded provenance            | RPCG9 → RPCG10b | NO_GENERALIZATION      |
| Training distribution: flat → nested-context outermost-wins | RPCG10b → RPCG11 | VOID (rare-prim fails) |

The nested-context lever joins the other three as an input-side variation
that does **not** rescue the per-primitive frequency-prior failure. The
gate's failure mode on the rare primitive is robust to all four input-side
dimensions tried. RPCG7c1's frequency-balanced lesson (already in the
program) is the obvious natural next intervention; RPCG11 leaves that lever
unspent.

**What this does NOT establish**: it does not refute the nested-context
direction outright. The C1 cell-mean coverage (0.833) and the
forbidden-decline rate (1.000) show the latent-candidate two-class CE *is*
installing a gate that reads the inner span. The failure is specifically
on the rare primitive; a frequency-balanced nested corpus (RPCG11c, future
work) could land a defensible verdict.

**What this DOES establish**: the per-primitive frequency prior is robust
to four orthogonal input-side levers, including the nested-context training
distribution. Any rescue likely needs to attack the prior directly
(frequency-balancing or per-primitive weight sharing), not the input
format.

## Gate decision for downstream artefacts

- Microsite + paper update: **skipped** per the plan's verdict gate (only
  FULL_GENERALIZATION or BOUNDARY_SHIFT triggers wire-through).
  EXPERIMENTS row + memory update are the only downstream artefacts.
- Lean theorem: **skipped** (only FULL_GENERALIZATION triggers).
- Memory: updated to record RPCG11 VOID and the four-lever-exhaustion frame.
- Future direction: a frequency-balanced nested corpus (RPCG11c) is the
  obvious next experiment; it isolates the nested-context-lever question
  from the rare-primitive prior.

## Reproduction

```bash
uv run python <monorepo>/scripts/run_nested.py --root . --ckpt checkpoints_rpcg11 \
  --base-model qwen0.5b --basis provenance \
  --prereg preregistry/rpcg11_nested_context_qwen25_2026-05-21 \
  --artifact <monorepo>/artifacts/rpcg11_nested_context_qwen25_2026-05-21.json \
  --exp rpcg11_nested_context_qwen25_2026-05-21
```
(Training/eval code lives in `src/rope_prov/train.py`; see the research monorepo for `run_nested.py`.)

Qwen2.5-0.5B, bf16, seed 42. Artifact:
`<monorepo>/artifacts/rpcg11_nested_context_qwen25_2026-05-21.json`.
τ frozen, sha `5d01d788…` valid. RPCG9 regression guard PASS end-to-end at
every commit through Phase 1 (deterministic; the action basis remained
byte-identical throughout).

obstacle_class: per-primitive frequency-prior failure (same family as
RPCG7c1's pre-balancing rungs and RPCG10b's SYS slot).
