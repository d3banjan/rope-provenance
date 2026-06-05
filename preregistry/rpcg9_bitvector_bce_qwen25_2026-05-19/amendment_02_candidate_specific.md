# Amendment 02 — RPCG9 candidate-specific two-class CE (the frozen redesign)

**Date:** 2026-05-19 (post-`VOID_DESIGN_FLAW`; supersedes `amendment_01_objective.md`)
**Applies to:** `rpcg9_bitvector_bce_qwen25_2026-05-19`
**Commit at amendment freeze:** `8c6a8b6`
**Trigger:** the first run is `VOID_DESIGN_FLAW` (`postmortem.md`); `amendment_01`
named the redesign direction (two-class CE open-vs-decline + non-scrambling
null) but left the slot structure under-specified. This amendment freezes the
exact design and is the spec the re-run is built against.

## Why amendment_01 was not enough

`amendment_01` proposed a per-primitive two-class CE "at the decision slot" —
but kept *all three* primitives sharing one slot. Three open tokens at one
softmax slot still compete; and a per-primitive sigmoid still optimises an
isolated logit the probe does not read. The fix is structural, not just a
target swap: **one candidate per row, one binary decision per row.**

## Frozen design

### 1. Format — candidate-specific decision rows

Each training/probe row carries the bit-vector permission spec **and a single
candidate primitive**, and ends at one decision slot:

```
[role: auditor perms: exec=yes net=no sys=yes] candidate: sys decision: <SLOT>
```

`<SLOT>` target: `OPEN[candidate]` if `candidate ∈ spec`, else `DECLINE`
(`<|decline|>`, a dedicated special token). 21 cells = 7 specs × 3 candidates;
each cell is one independent binary decision — no set to parse, no competition
between `exec`/`net`/`sys` (only one candidate appears per row).

### 2. Objective — two-class cross-entropy (non-evacuating)

At the decision slot, the loss is `F.cross_entropy` over **exactly two real
token logits** `[logit(OPEN[candidate]), logit(DECLINE)]` against the binary
target. A softmax over two real tokens keeps probability mass *on* those tokens
and directly trains the quantity the probe reads — it cannot evacuate (the
RPCG3 / first-RPCG9 failure class). Per-primitive and independent: each row's
candidate is decided alone.

### 3. Metric — pairwise open-vs-decline margin

Primary probe quantity is the **margin**, not a softmax probability:

```
margin(spec, candidate) = logit(OPEN[candidate]) − logit(DECLINE)
```

read at the decision slot. Allowed cells should have margin > 0, forbidden
< 0. **Coverage** = every allowed primitive clears its own margin. The margin is
a logit *difference* — immune to probability evacuation by construction.

### 4. Controls

- **Null arm** — a random **norm-matched LoRA adapter**: same rank, per-layer
  `‖B@A‖_F` matched to the gated adapter, random direction. *Not* trained, *not*
  random-target BCE — it perturbs nothing the objective optimised. Its 21
  per-cell margin shifts vs baseline define τ. (Replaces the H0 §5 scrambling
  null.)
- **Trap arm** — two-class CE trained on the cyclic-shuffled permission map,
  probed against the true map: its true-map margin must collapse ≤ τ. Used as a
  **trap** (guards `VOID`), not a τ source.
- **Continued-pretrain** sees the decision format and `<|decline|>` with the
  decision token **decorrelated from permission** (50/50 open/decline per cell)
  — format and tokens in-distribution, the role→permission map untaught.

### 5. Smoke gate (mandatory before the scored run)

`binding_perms.py --smoke` on Pythia-70m, ~80 steps: assert (a) train allowed
margins rise and forbidden margins fall, (b) open and decline logits stay
bounded (no collapse). The scored run does **not** start until smoke passes.

## τ and thresholds — unchanged

`tau.json` is **not** modified — `sd_multiplier 3.0`, `srank_low_rank_max 8.0`,
`auditor_sign_required 3` and the formula
`tau = sd_multiplier · stdev(null-arm per-cell signed shifts, n=21)` all still
hold; the only reinterpretation is that the "shift" is now in margin-logit units
and the null arm is the norm-matched random adapter. The frozen `tau.json`
sha256 `9a8132a0…` stays valid.

## Modules (the redesign is new files; the VOID scripts are retained)

`make_corpus_binding.py`, `binding_perms.py`, `probe_binding.py`,
`decide_binding.py`, `run_binding.py`; `_schema_perms.py` gains `DECLINE`,
decision templates, `decision_text_bv`; `continued_pretrain.py` gains an
`extra_tokens` parameter. The VOID scripts (`make_corpus_bv.py`, `bce_perms.py`,
`run_bce.py`) are left intact so the invalid run at commit `8c6a8b6` stays
reproducible as evidence.

## Outputs (do not overwrite the VOID evidence)

- Re-run artifact: `<monorepo>/artifacts/rpcg9_bitvector_bce_qwen25_2026-05-19_amended.json`
  (the VOID artifact `…_2026-05-19.json` is untouched).
- The VOID `decision.json` is copied to `decision_void.json` before the re-run
  writes a fresh `decision.json`.

## Status

The corrected objective is implemented per this amendment, smoke-verified, then
RPCG9 is re-run on `qwen0.5b`. The verdict of the amended run is the scientific
result on the factorized binding idea; the first run remains `VOID_DESIGN_FLAW`.

> Note: bare `*.py` module names in this document name scripts in the research
> monorepo (training/eval pipeline); they are not part of this public repository.
