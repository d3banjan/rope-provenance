# RPCG7c1 — compositional ladder rung C1 (frequency-balanced DPO)

Rung C1 of the RPCG7 ladder. Plan:
`<monorepo>/docs/plans/2026-05-19-rpcg7-compositional-gate-ladder.md`.

RPCG7a/b (plain / structured pairs) both returned NO_GENERALIZATION — the `sys`
gate never opened (frequency imbalance: `sys` chosen in only 2 of 5 trained
specs). The planned rung C (consistency regularizer) was mistargeted, so rung C
is split into two single-lever rungs. **C1 = frequency-balanced pairs only** —
chosen/rejected pairs replicated so every primitive is a `chosen` primitive
equally often. No regularizer. Tests whether the marginal-frequency prior was
the whole story. Batched DPO (`BATCH_SIZE` 16) — see `H0.md` §7 comparability
note. C2 (+ mass-sharing regularizer) runs only if C1 leaves `sys` weak.

## Run

```bash
uv run python <monorepo>/scripts/run_perms.py --rung c1 --base-model qwen0.5b \
  --root . --ckpt checkpoints_rpcg7 \
  --prereg ../preregistry/rpcg7c1_freqbalanced_qwen25_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg7c1_freqbalanced_qwen25_2026-05-19.json \
  --exp rpcg7c1_freqbalanced_qwen25_2026-05-19
```

`GENERALIZES` → ladder stops. `NO_GENERALIZATION` → rung C2. See `H0.md` for the
frozen kill criteria.
