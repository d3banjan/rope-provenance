# RPCG1 — role-provenance capability-gate (Pythia-70m)

Tests whether positive-only alignment SFT installs a role-conditioned gate that
suppresses out-of-role primitive logits.

- Design spec: `<monorepo>/docs/specs/2026-05-18-role-provenance-capability-gate-design.md` (research monorepo)
- Implementation plan: `<monorepo>/docs/plans/2026-05-18-role-provenance-capability-gate.md` (research monorepo)
- Implementation: `src/rope_prov/train.py` (see the research monorepo for the full pipeline)
- Geometry baseline: LRS1 (`preregistry/lrs1_srank_scaling_qwen25_2026-05-17/`)

Three arms (gated / role-swap trap / tag-only null) share one synthetic 9-cell
corpus and one continued-pretrain; they differ only in SFT doc selection. Verdict
via gate margin Δ vs a null-derived τ. See `H0.md` for the frozen decision rule,
`tau.json` for the frozen multipliers, `power.md` for the thin-DOF caveat.

Outputs after the run: `decision.json`, `postmortem.md`,
`<monorepo>/artifacts/rpcg1_capability_gate_pythia70m_2026-05-18.json`.
