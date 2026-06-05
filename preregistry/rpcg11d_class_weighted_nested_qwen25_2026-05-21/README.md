# RPCG11d — class-weighted nested-context compositional gate

Rung-1 ladder closer, sixth input-side lever and second landed
confound-removal run; spec
`<monorepo>/docs/specs/2026-05-21-rpcg11d-class-balanced-nested-context-compositional-gate-design.md`
(commit `29acacb`, v4.4).

## Run

```bash
uv run python <monorepo>/scripts/run_nested.py --root . --ckpt checkpoints_rpcg11d \
  --base-model qwen0.5b --basis provenance \
  --spec-variant balanced --trap-shuffle min_overlap \
  --class-weighting inverse_frequency \
  --prereg preregistry/rpcg11d_class_weighted_nested_qwen25_2026-05-21 \
  --artifact <monorepo>/artifacts/rpcg11d_class_weighted_nested_qwen25_2026-05-21.json \
  --exp rpcg11d_class_weighted_nested_qwen25_2026-05-21
```
(Training/eval code lives in `src/rope_prov/train.py`; see the research monorepo for `run_nested.py`.)

Smoke-gate first (Pythia-70m petri-dish + Qwen tokenizer smoke — see
the research-monorepo plan Tasks 6 + 7). All three regression guards
(`<monorepo>/scripts/rpcg9_regression_guard.py` +
`<monorepo>/scripts/rpcg11_regression_guard.py` +
`<monorepo>/scripts/rpcg11c_regression_guard.py`) must remain PASS
end-to-end through every commit.

## Verdict gates for downstream artefacts

| Verdict             | Postmortem + EXPERIMENTS + memory | Microsite | Paper | Lean |
|---------------------|-----------------------------------|-----------|-------|------|
| FULL_GENERALIZATION | ✓                                 | ✓         | ✓     | ✓    |
| BOUNDARY_SHIFT      | ✓                                 | ✓         | ✓     | (descriptive lemma) |
| WEAK_GENERALIZATION | ✓                                 | -         | -     | (descriptive lemma) |
| NO_GENERALIZATION   | ✓                                 | -         | -     | -    |
| VOID                | ✓                                 | -         | -     | -    |

## Interpretation map

- FULL_GENERALIZATION → rung-1 input-side ladder reopens with a
  qualified constructive positive (scope caveat per §9 of H0.md).
- BOUNDARY_SHIFT → both single-axis OOD compose; full-OOD breaks.
- WEAK_GENERALIZATION → only the pattern axis composes.
- NO_GENERALIZATION → composition fails beyond all five input-side
  confounds tested. Strong justification for RPCG12+ architectural
  pivot.
- VOID (any void_reason) → methodological collapse OR new local
  minimum surfaced; investigate before reading scientific signal.
