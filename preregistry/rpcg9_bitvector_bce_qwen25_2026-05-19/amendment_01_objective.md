# Amendment 01 — RPCG9 objective redesign (probability-evacuation fix)

**Date:** 2026-05-19 (post-H0-freeze; commit `94f5f82` froze H0/tau)
**Applies to:** `rpcg9_bitvector_bce_qwen25_2026-05-19`
**Trigger:** the first run is `VOID_DESIGN_FLAW` — see `postmortem.md`.

## What is wrong

The H0 §7 objective — `binary_cross_entropy_with_logits` on the isolated raw
vocabulary logit of each open token — **evacuates probability off the open
tokens**. Both the gated and null arms collapse every open-token probability to
≈ 0.00. `sigmoid(raw_logit)` is not the softmax probability the probe reads;
optimizing it does not install a probe-visible gate, and it lets the model dump
mass onto unrelated vocabulary. (Same failure class as RPCG3 `amendment_01`'s
bare single-token DPO.) The inflated τ ≈ 1.01 is a downstream symptom: the null
arm at 0.00 vs a 0.33 baseline yields per-cell signed shifts ≈ ±0.33.

The bit-vector format and the *factorized per-primitive* idea are sound and
retained. Only the objective's coupling to the probe is wrong.

## Resolution — corrected objective (two parts)

1. **Non-evacuating per-primitive objective.** Replace the isolated-logit BCE
   with a per-primitive **two-class cross-entropy** between the open token and a
   designated decline/anchor token: for each primitive P, target = open-token-P
   if P is allowed, else the decline token. This keeps probability *on* the open
   tokens for allowed primitives (the CE target is a real token), routes
   forbidden primitives to "decline", and is read directly by the
   softmax-probability probe. It stays per-primitive and independent — no
   ordinal competition. (Alternative: BCE on a renormalized 2-token softmax
   `[open_P, decline]` per primitive — equivalent.)

2. **Non-scrambling null arm.** The null must estimate background noise without
   perturbing the probed logits. Use a structured label-shuffled control — the
   null arm's targets are valid bit-vectors drawn from a random spec per doc
   (role↔permission destroyed, the {0,1} structure and marginal preserved) — so
   the model converges to the role-independent marginal rather than a collapsed
   or chaotic state. (Alternatives considered: frozen-base probe noise;
   matched-norm random adapter.)

## Status

The corrected objective is an amendment-level redesign of `bce_perms.py`, not a
one-line null swap. RPCG9 is **not re-run under this amendment until the
corrected objective is designed and smoke-verified** (verification: a per-arm
probe that is *not* all-zero; a null arm whose per-cell probabilities stay near
the baseline so τ is small). The invalid first run is retained as the evidence
above; RPCG9 remains the in-progress next rung of the binding-layer program.

> Note: bare `*.py` module names in this document name scripts in the research
> monorepo (training/eval pipeline); they are not part of this public repository.
