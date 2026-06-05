# RPCG6 — scale check, Qwen2.5-0.5B

Paper-gap experiment 2 of 4. Re-runs the **identical RPCG3 recipe** (DPO +
chosen-NLL anchor, 3 arms) on Qwen2.5-0.5B — a Llama-style architecture at ~7×
the Pythia-70m parameter count, and the LRS1 geometry-baseline model. Tests
whether the role-conditioned capability gate is a Pythia-70m artifact or
replicates across architecture and scale.

Only the base model, the LoRA target modules (`q_proj,k_proj,v_proj` vs Pythia's
fused `query_key_value`), and the dtype (bf16) change. A model registry
(`<monorepo>/scripts/_model.py`) carries those three; corpus, objective, hyperparameters,
probe, decision rule, and τ are unchanged from RPCG3.

## Run

```bash
uv run python <monorepo>/scripts/run_all.py --root . \
  --prereg ../preregistry/rpcg6_scale_qwen25_0p5b_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg6_scale_qwen25_0p5b_2026-05-19.json \
  --method dpo --base-model qwen0.5b --ckpt checkpoints_rpcg6 \
  --exp rpcg6_scale_qwen25_0p5b_2026-05-19
```

See `H0.md` for the frozen decision rule and the OOM contingency (8-bit Adam
amendment if the full-FT continued-pretrain exceeds 12 GB).
