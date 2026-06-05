# Postmortem — rpcg7c1_freqbalanced_qwen25 — 2026-05-19

**Verdict:** `NO_GENERALIZATION`
**Rung:** C1 of the RPCG7 ladder (rung-C redesign). Run of record is the
Amendment-01 recipe (batch 1, 900 optimizer steps); the first run VOIDed on a
step-count bug — see `amendment_01_step_count.md`.

## Result

| quantity | C1 (freq-balanced) | RPCG7a/b |
|---|---|---|
| verdict | NO_GENERALIZATION | NO_GENERALIZATION |
| in_dist_margin | 0.2572 (gate installs) | 0.257 |
| τ | 0.0490 | 0.020 / 0.037 |
| auditor_margin {exec,sys} | 0.2372, sign **2/3** | 0.236, 2/3 |
| **singleton_margin {sys}** | **0.2289, sign 3/3 ✓** | **−0.20, 1/3 ✗** |
| trained_sys_meanprob | 0.4938 | (sys ≈ 0.02) |

## Frequency-balancing fixed cause (1)

The held-out **singleton `{sys}` now passes — margin 0.229, 3/3 sign**, against
−0.20 / 1/3 in RPCG7a/b. The `sys` gate, which never opened in A/B (even
in-distribution `{net,sys}` gave sys ≈ 0.02), now opens: singleton `{sys}` →
sys 0.66, trained `{net,sys}` → sys 0.95. Equalising the per-primitive chosen
frequency removed the marginal-frequency prior. **Cause (1) — frequency
imbalance — is confirmed and repaired.**

## But cause (2) is now isolated and undeniable

The gated probe, every multi-permitted spec:

```
exec_sys  (auditor, held-out)  exec 1.00  net 0.00  sys 0.00   exec wins, sys starved
net_sys   (trained)            exec 0.00  net 0.04  sys 0.95   sys wins, net starved
exec_net_sys (trained)         exec 0.84  net 0.13  sys 0.03   exec wins, sys starved
sys       (singleton, held-out) exec 0.01 net 0.33  sys 0.66   one permitted -> it gets the mass
```

In **every** spec with ≥ 2 permitted primitives the model dumps ≈ all probability
on **one** permitted primitive and starves the rest — and not even consistently
(`net_sys` picks `sys`, `exec_net_sys` and `exec_sys` pick `exec`). This is the
**ordinal-DPO weakness** (cause 2): a DPO pair only requires `chosen > rejected`,
so within a multi-permitted spec a winner-takes-all distribution satisfies every
pair (each permitted primitive that wins beats the forbidden one; the starved
permitted primitive is simply never the `chosen` whose ranking is checked).

`trained_sys_meanprob = 0.4938` is a *win/loss average* — `sys` 0.95 in
`{net,sys}` and 0.03 in `{exec,net,sys}`. It exceeds the pre-registered 0.10
threshold, so per `H0` §3 the verdict routes to: **sys opens in-distribution but
the held-out specs still fail → repaired per-primitive priors, not compositional
gating → escalate to C2.**

The singleton-passes / pair-fails split localises the residual failure exactly:
it is **not** a per-primitive gate problem any more (singletons work); it is a
**within-spec mass-allocation** problem confined to multi-permitted specs.
`auditor`={exec,sys} fails for the same reason `{net,sys}` and `{exec,net,sys}`
mis-allocate — one permitted primitive eats the mass.

## What this establishes

- C1 cleanly isolates the two causes the ladder hypothesised. Frequency-balancing
  (the single C1 lever) repairs the per-primitive `sys` gate — held-out singleton
  generalizes. The residual failure is purely cause (2), the ordinal-DPO
  within-spec mass-dumping, now demonstrated on three independent multi-permitted
  specs.
- The auditor `{exec,sys}` 2/3 is `sys` starved by `exec`, not a `sys`-gate
  failure.

## Escalation → rung C2 (rpcg7c2)

`NO_GENERALIZATION`, `trained_sys_meanprob > 0.10` → rung C2: frequency-balanced
pairs **+ a within-spec mass-sharing regularizer** — penalise the permitted
primitives of a spec having unequal open-probability, forcing each permitted
primitive *high* rather than just above the forbidden one. This targets cause (2)
directly. Per the methodology directive, C2 reports that C1 (frequency alone)
was necessary but not sufficient, and C2 adds the equality-style pressure.

## Lean co-evolution

None — NO_GENERALIZATION.

## Reproduction

`<monorepo>/scripts/run_perms.py --rung c1 --base-model qwen0.5b --ckpt checkpoints_rpcg7`
(Amendment-01 recipe: batch 1, 900 steps, freq-balanced pairs). Qwen2.5-0.5B,
bf16, seed 42. Continued-pretrain reused. Artifact:
`<monorepo>/artifacts/rpcg7c1_freqbalanced_qwen25_2026-05-19.json`.

obstacle_class: NEW
