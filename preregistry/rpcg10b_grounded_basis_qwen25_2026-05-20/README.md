# RPCG10b — grounded-basis compositional re-test

Rung 1, stage b of the binding-layer program (spec
`<monorepo>/docs/specs/2026-05-19-rpcg10-grounded-basis-design.md`). RPCG9
amendment_02's exact pipeline, basis swapped to OBEY/USE/QUOTE. Gated on the
RPCG10a `PASS` verdict (commit `25cb274`).

## Run

```bash
uv run python <monorepo>/scripts/run_binding.py --root . --ckpt checkpoints_rpcg10b \
  --base-model qwen0.5b --basis provenance \
  --prereg ../preregistry/rpcg10b_grounded_basis_qwen25_2026-05-20 \
  --artifact <monorepo>/artifacts/rpcg10b_grounded_basis_qwen25_2026-05-20.json \
  --exp rpcg10b_grounded_basis_qwen25_2026-05-20
```

The deterministic refactor guard (`rpcg9_regression_guard.py`, see the research monorepo) confirms the
`--basis action` path still reproduces RPCG9 amendment_02 bit-identically;
this run uses `--basis provenance`.

`GENERALIZES` → the grounded basis fixes composition (RPCG7/9 was a
basis-mismatch, not a fundamental limit). `NO_GENERALIZATION` → even a
grounded basis does not fix composition; the lever is somewhere else.
