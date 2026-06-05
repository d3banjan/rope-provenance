# RPCG9 — bit-vector format + factorized permission binding

First rung of the role-to-boundary binding program; direct attack on the RPCG7
negative (ordinal set-list DPO does not induce compositional gates). The prompt
carries a **bit-vector** permission spec (`exec=yes net=no sys=yes` — no set to
parse) and a factorized per-primitive alignment objective.

## Two runs

**Run 1 — `VOID_DESIGN_FLAW`** (objective: per-primitive `binary_cross_entropy`
on the isolated raw open-token logit). The objective evacuated probability off
all three open tokens — see `postmortem.md`. Not a scientific result. Modules:
`make_corpus_bv.py`, `bce_perms.py`, `run_bce.py`. Retained, reproducible at
commit `8c6a8b6`.

**Run 2 — amendment_02 redesign** (objective: candidate-specific two-class
cross-entropy). One candidate primitive per row, ending at one decision slot;
the loss is `cross_entropy` over exactly two real token logits —
`[logit(OPEN[candidate]), logit(DECLINE)]` — which cannot evacuate. Primary
metric is the open-vs-decline **margin** (a logit difference). See
`amendment_02_candidate_specific.md` for the frozen design. Modules:
`make_corpus_binding.py`, `binding_perms.py`, `probe_binding.py`,
`decide_binding.py`, `run_binding.py`. Runs on Qwen2.5-0.5B.

## Run (amendment_02 — the current run)

```bash
uv run python <monorepo>/scripts/run_binding.py --root . --ckpt checkpoints_rpcg9_amended \
  --base-model qwen0.5b \
  --prereg ../preregistry/rpcg9_bitvector_bce_qwen25_2026-05-19 \
  --artifact <monorepo>/artifacts/rpcg9_bitvector_bce_qwen25_2026-05-19_amended.json \
  --exp rpcg9_bitvector_bce_qwen25_2026-05-19
```

Smoke-gate first (`python <monorepo>/scripts/binding_perms.py --smoke` — a full Pythia-70m
petri-dish mini-run; must show the gate install with no logit collapse).

`GENERALIZES` → factorized supervision installs a compositional gate (the
constructive counterpart to RPCG7's negative). See `H0.md` §0 for the framing
and `amendment_02_candidate_specific.md` for the objective + metric.

## Files

`H0.md` · `tau.json` (frozen, sha valid) · `amendment_01_objective.md` (named
the redesign direction) · `amendment_02_candidate_specific.md` (the frozen
design) · `decision_void.json` (Run 1 evidence) · `decision.json` (Run 2) ·
`postmortem.md`.

> Note: bare `*.py` module names in this document name scripts in the research
> monorepo (training/eval pipeline); they are not part of this public repository.
