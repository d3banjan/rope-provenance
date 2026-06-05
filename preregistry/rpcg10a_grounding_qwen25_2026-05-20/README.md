# RPCG10a — grounding pre-check

Rung 1, stage a of the binding-layer program (spec
`<monorepo>/specs/2026-05-19-rpcg10-grounded-basis-design.md`). Frozen
Qwen2.5-0.5B, no LM training: does the provenance basis OBEY/USE/QUOTE occupy
more orthogonal, factored directions than the action basis exec/net/sys?

## Run

```bash
uv run python <monorepo>/scripts/run_grounding.py --root . --ckpt checkpoints_rpcg10a \
  --prereg ../preregistry/rpcg10a_grounding_qwen25_2026-05-20 \
  --artifact <monorepo>/artifacts/rpcg10a_grounding_qwen25_2026-05-20.json \
  --exp rpcg10a_grounding_qwen25_2026-05-20
```

PASS -> run RPCG10b on OBEY/USE/QUOTE. WEAK -> 10b runs as "alternate basis".
FAIL -> amend the basis before 10b. INCONCLUSIVE -> the metric is artifact-prone.
