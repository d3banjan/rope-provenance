# RPCG8 — generation-path witness

Paper-gap experiment 4 of 4. The Class-K behavioral readout: RPCG3/6/7 measured
the role gate at the logit level; RPCG8 asks whether it survives into *free
generation* — does a gated checkpoint avoid *executing* forbidden primitives, or
does the gate stop at the logits?

No training — frozen checkpoints only (`rpcg7_base`, `rpcg7c1_gated`,
`rpcg7c2_gated`, and `rpcg6_base`/`rpcg6_gated` if present). Each role/permission
prompt is generated normally (sampled, no forced token); completions are scored
by the first primitive open-token emitted. Primary metric: forbidden-action
execution rate, gated vs base.

## Run

```bash
uv run python <monorepo>/scripts/gen_witness.py --root . \
  --prereg ../preregistry/rpcg8_generation_witness_qwen25_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg8_generation_witness_qwen25_2026-05-19.json
```

Verdicts: `GENERATION_GATE_CONFIRMED` / `CLASS_K_FAILURE` / `VOID`, plus a
boundary annotation (does free-gen match RPCG7's logit-level
singleton-gates / combination-fails boundary). See `H0.md`.
