# RPCG11 — nested-context compositional gate

Rung 2 of the binding-layer program; spec
`<monorepo>/docs/superpowers/specs/2026-05-21-rpcg11-nested-context-compositional-gate-design.md`.

## Run

```bash
uv run python <monorepo>/scripts/run_nested.py --root . --ckpt checkpoints_rpcg11 \
  --base-model qwen0.5b --basis provenance \
  --prereg preregistry/rpcg11_nested_context_qwen25_2026-05-21 \
  --artifact <monorepo>/artifacts/rpcg11_nested_context_qwen25_2026-05-21.json \
  --exp rpcg11_nested_context_qwen25_2026-05-21
```

Smoke-gate first (`python <monorepo>/scripts/binding_perms.py --smoke --basis provenance
--corpus nested` style — see Task 8 of the plan). The deterministic RPCG9 regression guard (`rpcg9_regression_guard.py`, see the research monorepo) must remain
PASS end-to-end through every commit.

`FULL_GENERALIZATION` → nested-context supervision induces composition where
flat supervision did not (rung-1 ladder closed); paper + microsite + Lean
updates land. `BOUNDARY_SHIFT` → partial — both single-axis OOD passes; the
crossed axis does not. `WEAK_GENERALIZATION` → only the pattern axis
generalizes. `NO_GENERALIZATION` / `VOID` → the nested lever joins
objective/format/basis as another input-side dimension that does not rescue
composition.
