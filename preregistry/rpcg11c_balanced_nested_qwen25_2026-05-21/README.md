# RPCG11c — frequency-balanced nested-context compositional gate

Rung-1 ladder, fifth input-side lever; spec
`<monorepo>/specs/2026-05-21-rpcg11c-balanced-nested-context-compositional-gate-design.md`.

## Run

```bash
uv run python <monorepo>/scripts/run_nested.py --root . --ckpt checkpoints_rpcg11c \
  --base-model qwen0.5b --basis provenance \
  --spec-variant balanced --trap-shuffle min_overlap \
  --prereg ../preregistry/rpcg11c_balanced_nested_qwen25_2026-05-21 \
  --artifact <monorepo>/artifacts/rpcg11c_balanced_nested_qwen25_2026-05-21.json \
  --exp rpcg11c_balanced_nested_qwen25_2026-05-21
```

Smoke-gate first (Pythia-70m petri-dish + Qwen tokenizer smoke — see
plan Tasks 11 + 12). Both regression guards
(`<monorepo>/scripts/rpcg9_regression_guard.py` AND
`<monorepo>/scripts/rpcg11_regression_guard.py`) must
remain PASS end-to-end through every commit.

## Verdict gates for downstream artefacts

| Verdict             | Postmortem + EXPERIMENTS + memory | Microsite | Paper | Lean |
|---------------------|-----------------------------------|-----------|-------|------|
| FULL_GENERALIZATION | ✓                                 | ✓         | ✓     | ✓    |
| BOUNDARY_SHIFT      | ✓                                 | ✓         | ✓     | (descriptive lemma) |
| WEAK_GENERALIZATION | ✓                                 | -         | -     | (descriptive lemma) |
| NO_GENERALIZATION   | ✓                                 | -         | -     | -    |
| VOID                | ✓                                 | -         | -     | -    |

## Interpretation map

- `FULL_GENERALIZATION` → nested-context lever rehabilitated; per-primitive
  frequency prior was the confound; rung-1 input-side ladder reopens.
- `BOUNDARY_SHIFT` → both single-axis OOD compose; full-OOD breaks.
- `WEAK_GENERALIZATION` → only the pattern axis composes; novel outer spec
  doesn't.
- `NO_GENERALIZATION` → composition fails beyond frequency prior. Closes
  the input-side ladder cleanly; pushes to architectural / out-of-band
  tensor levers (RPCG12+).
- `VOID` (any `void_reason`) → methodological collapse; investigate before
  reading scientific signal.
