# Postmortem — rpcg7a_plain_dpo_qwen25 — 2026-05-19

**Verdict:** `NO_GENERALIZATION`
**Rung:** A of the RPCG7 compositional-generalization ladder.

## Naming clarification (read first)

"Plain DPO" here labels the **pair construction** (plain whole-set chosen/rejected
pairs), NOT the objective. The objective is the proven RPCG3/RPCG6 DPO **+
chosen-NLL anchor** — *not* the bare anchorless DPO that degenerated in RPCG3
`amendment_01`. Confirmed by the result: the gate installed cleanly
in-distribution with no probability evacuation. Rung B will change the pairs to
"structured"; the objective stays the same DPO+anchor across all three rungs.

## Result

| quantity | value |
|---|---|
| **verdict** | **NO_GENERALIZATION** |
| in_dist_margin (5 trained specs) | 0.2575 — gate installs |
| τ (3× null SD, 21 cells) | 0.0203 |
| auditor_margin {exec,sys} | 0.2360, sign **2/3** |
| singleton_margin {sys} | **−0.2025**, sign 1/3 |
| trap_collapsed | True |
| gated adapter srank | 1.37 |
| compositionality_score | 0.646 |
| converged | True |

`in_dist_margin 0.2575 > τ` (gate installed) but auditor sign 2/3 < 3 →
`NO_GENERALIZATION`.

## What happened — held-out probe

```
auditor {exec,sys}   exec 0.321->0.996 ✓   net 0.355->0.001 ✓   sys 0.321->0.000 ✗
janitor {sys}        exec 0.319->0.000 ✓   net 0.361->0.984 ✗   sys 0.319->0.016 ✗
```

Auditor: `exec` correctly promoted, `net` correctly suppressed — but `sys`
(permitted) was crushed to zero. Janitor: `sys` (the *only* permitted primitive)
crushed, and forbidden `net` promoted to 0.98.

## Diagnosis (per feedback_research_loop_order)

**The `sys` primitive's gate never learned to open as a list-driven positive.**
The model learned per-primitive *priors* — exec opens readily, net opens, `sys`
≈ never — rather than a list-conditional per-primitive gate. Evidence: even
*in-distribution* the trained spec `{net,sys}` probes net 0.97 / sys 0.03, and
`{exec,net,sys}` probes exec 0.89 / net 0.10 / sys 0.01 — `sys` underopens even
where it is permitted and trained.

Why plain pairs allow this: in the 5 trained specs, `sys` is permitted only in
`{net,sys}` and `{exec,net,sys}` — always co-occurring with `net`, never the
sole or dominant permitted primitive. A whole-set DPO pair only needs *some*
in-set primitive ranked over an out-set one; ranking `net` over the out-set
primitive satisfies every `sys`-containing trained pair without ever opening
`sys`. So the trained-spec gate is satisfiable by the prior `exec>net>sys`
composed with the list — the model never has to read the list to gate `sys`
specifically. The held-out specs `{exec,sys}` and `{sys}`, where `sys` must be a
primary permitted primitive, expose the missing per-primitive gate; `{sys}` even
falls back to the prior's second-favourite, forbidden `net`.

This is the `NO_GENERALIZATION` the ladder anticipated: plain DPO installs a
role/list-shaped gate in-distribution but it is **not compositional** — it does
not factor into reusable per-primitive gates, so an unseen combination that
needs `sys` fails.

## What this establishes

- The RPCG3/6 DPO+anchor recipe **installs** the gate in-distribution under
  format C (role name + inline list) — `in_dist_margin 0.2575`, trap collapses,
  srank 1.37 (low-rank, consistent with RPCG3/6 ~1.3).
- It does **not generalize compositionally** on its own. Plain whole-set pairs
  do not force the model to read the list and gate each primitive positively;
  it settles on per-primitive priors instead.

## Escalation → rung B (rpcg7b)

`NO_GENERALIZATION` → rung B: structured DPO pairs. Single-primitive-flip
negatives and role/list-decorrelation pairs are designed to force exactly the
missing behavior — each gradient step isolates one primitive's listed-status, so
`sys` must be opened when (and only when) listed. If rung B still fails → rung C
(compositionality regularizer).

## Lean co-evolution

None (`H0` §11) — NO_GENERALIZATION adds no theorem.

## Reproduction

`<monorepo>/scripts/run_perms.py --rung a --base-model qwen0.5b --ckpt checkpoints_rpcg7`.
Qwen2.5-0.5B, bf16, seed 42, RTX 3060. Wall ≈ 30 min. Artifact:
`<monorepo>/artifacts/rpcg7a_plain_dpo_qwen25_2026-05-19.json`.

obstacle_class: NEW
