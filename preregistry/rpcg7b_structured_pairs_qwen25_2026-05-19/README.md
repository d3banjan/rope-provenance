# RPCG7b — compositional ladder rung B (structured pairs)

Rung B of the RPCG7 ladder. Plan:
`<monorepo>/docs/plans/2026-05-19-rpcg7-compositional-gate-ladder.md`.

RPCG7a (plain whole-set DPO pairs) returned NO_GENERALIZATION — the model rode
per-primitive priors instead of reading the inline list. Rung B changes only the
DPO pair construction to **structured**: adds role/list-decorrelation pairs (same
inline list under a different role name) that force the model to read the list.
Continued-pretrain reused from rung A; objective, τ, decision rule unchanged.

## Run

```bash
uv run python <monorepo>/scripts/run_perms.py --rung b --base-model qwen0.5b \
  --root . --ckpt checkpoints_rpcg7 \
  --prereg ../preregistry/rpcg7b_structured_pairs_qwen25_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg7b_structured_pairs_qwen25_2026-05-19.json \
  --exp rpcg7b_structured_pairs_qwen25_2026-05-19
```

`GENERALIZES` → ladder stops. `NO_GENERALIZATION` → rung C. See `H0.md`.
