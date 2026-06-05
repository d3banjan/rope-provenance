# RPCG2 — open-token-weighted SFT escalation (Pythia-70m)

Escalation rung 1 of RPCG1. RPCG1 returned VOID — a plain causal-LM SFT loss let
the model satisfy the objective via the SFT marginal without using the role.

RPCG2 changes **only the SFT loss**: cross-entropy is masked to the open-token
slot (positions whose causal label is a primitive open token), forcing the
gradient through the role→primitive decision. Corpus, continued-pretrain, 3-arm
structure (gated / role-swap trap / tag-only null), probe, decision rule, and all
τ values are identical to RPCG1.

- RPCG1: `preregistry/rpcg1_capability_gate_pythia70m_2026-05-18/`
- Spec: `<monorepo>/docs/specs/2026-05-18-role-provenance-capability-gate-design.md`
- Code: `src/rope_prov/train.py` (`--loss open_masked`)

## Run

```bash
uv run python <monorepo>/scripts/run_all.py --root . \
  --prereg ../preregistry/rpcg2_open_token_weighted_sft_pythia70m_2026-05-18 \
  --artifact <monorepo>/artifacts/rpcg2_open_token_weighted_sft_pythia70m_2026-05-18.json \
  --sft-loss open_masked --ckpt checkpoints_rpcg2 \
  --exp rpcg2_open_token_weighted_sft_pythia70m_2026-05-18
```

See `H0.md` for the frozen decision rule and the interpretation amendment (a VOID
under the masked loss reads as a capacity wall, not a recipe weakness).
