# RPCG3 — DPO explicit-negatives escalation (Pythia-70m)

Escalation rung 2 of RPCG1. RPCG1 (plain SFT) and RPCG2 (open-token-masked SFT)
both VOIDed: a positive-only objective admits the SFT marginal as a local
optimum, and a rank-8 QKV adapter does not escape it.

RPCG3 changes the objective to **DPO with explicit negatives**. At the open-token
decision slot, an in-role primitive open token is `chosen`, an out-of-role one is
`rejected`. The contrastive loss directly penalises the marginal — predicting the
marginal scores chosen and rejected equally, so the loss drops only by genuine
role-conditioning. Reference model via PEFT `disable_adapter()`. Corpus,
continued-pretrain, 3-arm structure, probe, decision rule, τ unchanged from RPCG1.

- RPCG1/2: `preregistry/rpcg{1,2}_*_2026-05-18/`
- Code: `src/rope_prov/train.py` (see the research monorepo for `dpo_role.py`)

## Run

```bash
uv run python <monorepo>/scripts/run_all.py --root . \
  --prereg ../preregistry/rpcg3_dpo_explicit_negatives_pythia70m_2026-05-18 \
  --artifact <monorepo>/artifacts/rpcg3_dpo_explicit_negatives_pythia70m_2026-05-18.json \
  --method dpo --ckpt checkpoints_rpcg3 \
  --exp rpcg3_dpo_explicit_negatives_pythia70m_2026-05-18
```

See `H0.md`: under a contrastive objective a VOID/NO_GATE reads as a capacity wall
(rank-8 QKV), routing to RPCG4 (full-FT SFT).
