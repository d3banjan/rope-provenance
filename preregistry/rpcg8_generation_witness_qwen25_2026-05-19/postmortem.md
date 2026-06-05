# Postmortem — rpcg8_generation_witness_qwen25 — 2026-05-19

**Verdict:** `GENERATION_GATE_CONFIRMED`
**Paper-gap experiment 4 of 4** — the Class-K generation-path witness. Run of
record is the Amendment-01 recipe; the first run VOIDed on a control-metric
definition bug (see `amendment_01_control_metric.md`).

## Result — the gate reaches generation behavior

| checkpoint | trained-spec forbidden-exec | allowed-exec |
|---|---|---|
| rpcg7_base (control) | 0.458 (forbidden-bearing specs) | — |
| **rpcg7c1_gated** | **0.000** (0 / 240 samples) | **1.000** |
| **rpcg7c2_gated** | **0.000** | **1.000** |
| rpcg6_base (role schema) | 0.44 | — |
| **rpcg6_gated** | **0.000** (0 / 144) | **1.000** |

In free, sampled generation (no forced token) the gated checkpoints execute a
forbidden primitive **zero times** on trained specs — across the RPCG7 perms
schema (c1, c2) *and* the RPCG6 role schema — while the un-gated base executes
forbidden actions 44–46 % of the time. **The logit-level role gate is not
Class-K-decoupled: its forbidden-action suppression reaches generation behavior
fully.** This is the pre-registered "strong positive".

## But — two honest qualifications

**1. Within-permitted coverage collapses.** The gate *blocks* forbidden actions
perfectly, but the model does not *exercise* the full permitted set. The
RPCG7 winner-takes-all carries into generation: `exec_net` → exec 47 / net 1;
`net_sys` → sys 47 / net 1; `exec_net_sys` → exec 31 / net 15 / sys 2. So the
claim is precisely "the gate reliably blocks forbidden actions" — **not** "the
model uses its whole permission set". `allowed_exec_rate = 1.000` only means
*some* permitted primitive; it is not coverage.

**2. The `boundary` annotation is `inconsistent` — and that is a metric
artifact, not a contradiction of RPCG7.**

- Held-out combination `{exec,sys}`: forbidden-exec **0.0** — but only because
  the gated model collapses to `exec`-only (RPCG7c1 logit probe: exec 1.00).
  `exec` is permitted, `net` simply never wins, so 0 forbidden actions occur
  *by winner-takes-all luck* — the model is not respecting `{exec,sys}`, it is
  doing `exec` and ignoring `sys`.
- Held-out singleton `{sys}`: forbidden-exec **0.44** — the gated model still
  samples forbidden `net` ~half the time (RPCG7 logit probe had `sys` as the
  top but with substantial `net` mass). This is honest leakage.

So `forbidden_exec_rate` alone is **gamed by winner-takes-all** whenever the
winner happens to be a permitted primitive. The held-out picture does not
contradict RPCG7's logit boundary — it shows the single-metric is insufficient.
A future generation witness needs a **per-primitive coverage** metric (does the
model exercise *every* permitted primitive, not just one).

## What this establishes — and the paper spine

- **The role gate's forbidden-suppression is behavioral**, not a logit artifact:
  0 forbidden executions in free generation on trained specs, two schemas. The
  paper's central claim strengthens from "logit-level role gate" to "role gate
  whose forbidden-action suppression reaches generation behavior."
- The compositional boundary (RPCG7) stands; RPCG8 adds the behavioral readout
  and a methodology caveat (forbidden-exec is winner-takes-all-gameable;
  coverage needs its own metric).
- With RPCG8 the four paper-gap legs are complete: scale (RPCG6),
  layer-localization (RPCG5), novel-role generalization (RPCG7 ladder),
  generation-path witness (RPCG8).

## Lean co-evolution

None new. `lean/RoleGateReachability.lean` gains a behavioral anchor (the suppression
gate reaches generation) — a descriptive note, no theorem. The Class-K predicate
`BiasGenDecoupling.lean` (research monorepo) is *not* invoked as a limit here: RPCG8 shows the
suppression gate is *not* bias/generation-decoupled on trained specs.

## Reproduction

`<monorepo>/scripts/gen_witness.py` (Amendment-01 control metric). Frozen checkpoints, no training.
Qwen2.5-0.5B, bf16, seed 42 (sampled generation, deterministic under the seed).
Artifact: `<monorepo>/artifacts/rpcg8_generation_witness_qwen25_2026-05-19.json`.

obstacle_class: K
