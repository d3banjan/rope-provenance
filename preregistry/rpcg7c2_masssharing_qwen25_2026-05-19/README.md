# RPCG7c2 — compositional ladder rung C2 (+ within-spec mass-sharing)

Final rung of the RPCG7 ladder. Plan:
`<monorepo>/docs/plans/2026-05-19-rpcg7-compositional-gate-ladder.md`.

RPCG7c1 (frequency-balanced DPO) repaired the per-primitive `sys` gate — the
held-out singleton `{sys}` generalized — but every multi-permitted spec still
dumped all mass on one permitted primitive (ordinal-DPO winner-takes-all), so
held-out `auditor`={exec,sys} failed. C2 adds a **within-spec mass-sharing
regularizer**: penalize the variance of a spec's permitted primitives'
open-log-probs, forcing each permitted primitive high. C1's frequency-balancing
is retained. Built on the C1 amendment recipe (batch 1, 900 steps).

## Run

```bash
uv run python <monorepo>/scripts/run_perms.py --rung c2 --base-model qwen0.5b \
  --root . --ckpt checkpoints_rpcg7 \
  --prereg preregistry/rpcg7c2_masssharing_qwen25_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg7c2_masssharing_qwen25_2026-05-19.json \
  --exp rpcg7c2_masssharing_qwen25_2026-05-19
```

Final rung — `GENERALIZES` closes RPCG7 positive; `NO_GENERALIZATION` closes it
as an instrumented negative result. See `H0.md`.
