# RPCG5 — role-gate layer localization (Pythia-70m)

Follow-up probe of RPCG3 (`GATE_INSTALLED`). Asks **where in depth** the
role-conditioned capability gate lives.

Takes the frozen RPCG3 gated DPO adapter (LoRA on the fused `query_key_value`
across 6 layers), ablates its rank-1 update layer-by-layer, re-runs the
primitive-open-token probe, and recomputes the gate margin. No new training.

Decision: `k_crit` = how many layers (greedy by individual effect) must be
ablated to drop the margin to the RPCG3 gate-removed threshold. `≤2` LOCALIZED,
`≥5` DIFFUSE, else INTERMEDIATE. Monotonicity of the greedy chain is a trap and
empirically instantiates `RoleGateReachability.logit_mono`.

## Run

```bash
uv run python <monorepo>/scripts/ablate_localize.py \
  --ckpt checkpoints_rpcg3 \
  --baseline checkpoints_rpcg3/probes/probe_baseline.json \
  --probes probe_prompts/probe_prompts.jsonl \
  --prereg ../preregistry/rpcg5_layer_localization_pythia70m_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg5_layer_localization_pythia70m_2026-05-19.json
```

See `H0.md` for the frozen decision rule.
