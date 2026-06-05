# RPCG7a — compositional ladder rung A (plain DPO)

Rung A of the RPCG7 compositional-generalization ladder. Plan:
`<monorepo>/docs/plans/2026-05-19-rpcg7-compositional-gate-ladder.md`.

Format-C prompts (`[role: name allow: primitives]`). 7 permission specs, 5
DPO-trained, 2 held out (`auditor`={exec,sys}, `janitor`={sys}). Rung A uses
plain whole-set DPO pairs — tests whether the gate generalizes to the held-out
specs *on its own*, no structured pairs and no regularizer.

## Run

```bash
uv run python <monorepo>/scripts/run_perms.py --rung a --base-model qwen0.5b \
  --root . --ckpt checkpoints_rpcg7 \
  --prereg ../preregistry/rpcg7a_plain_dpo_qwen25_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg7a_plain_dpo_qwen25_2026-05-19.json \
  --exp rpcg7a_plain_dpo_qwen25_2026-05-19
```

Verdict `GENERALIZES` → ladder stops. `NO_GENERALIZATION` → rung B. See `H0.md`.
