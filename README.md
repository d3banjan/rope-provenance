# RoPE-Provenance

First **per-layer rotational** attempt at the out-of-band role channel
called for by Hines et al. (2024) and mechanistically motivated by Ye et
al. (2026)'s *role confusion* analysis. We test whether a sub-block of
RoPE rotations can encode token role (instruction vs data) by phase
rather than learned content. Mechanism feasibility study on SmolLM2-135M.

## Motivation

Hines et al. (Spotlighting, MSR 2024) explicitly call for what we build:

> "Out-of-band signaling has many advantages over in-band signaling…
> we need to devise a multi-channel analog for LLMs… control tokens
> would be passed to model in a separate 'channel' from the data tokens…
> With current architectures… this is not feasible in any straightforward
> way. Nonetheless, this premise is compelling and future work remains."

Ye et al. (Prompt Injection as Role Confusion, MIT 2026) characterize why
existing architectural role tags fail: model role-perception is
*style-dominated* in latent space, and architectural tags are overwhelmed.

> "Security is defined at the interface but authority is assigned in
> latent space."

This work is the future work Hines pointed at, evaluated against the
diagnostic frame Ye established.

## Design-space placement

Our work targets the missing cell of a 2×2:

|              | Input-layer            | Per-layer                |
|--------------|------------------------|--------------------------|
| **Additive** | ISE (Wu '24)           | AIR (Kariyappa & Suh '25)|
| **Rotational** | ASIDE (Zverev '26)   | **ours**                 |

AIR (NVIDIA, arxiv:2505.18907) is the per-layer **additive** analog and
reports 1.6×–9.2× ASR reduction. Our negative result with the
**rotational** analog at the same per-layer placement, plus the
architectural diagnosis of *why* rotation fails where addition succeeds,
fills out this design-space comparison.

## Method

Standard RoPE rotates each `(coord i, coord i + head_dim/2)` pair by
`position_id · inv_freq[i]`. We split head_dim into:

- **Positional subspace** = pair indices `[0, half_p)` (highest-frequency
  pairs) — receives standard RoPE.
- **Provenance subspace** = pair indices `[half_p, half_h)` (lowest-frequency
  pairs) — receives a per-token rotation by `role_angles[role_ids]`.

For SmolLM2-135M (`head_dim=64`) with `prov_dim=8`, this sacrifices the 4
lowest-frequency pairs (12.5% of positional pairs). The provenance
rotation is applied **after** the learned `W_Q, W_K` projections, at
every transformer layer.

See `src/rope_prov/rotary.py` for the kernel and
`src/rope_prov/model.py` for the `LlamaAttention` patching.

## Four findings

### 1. (1 − cos θ) is the per-layer rotation cost law

SmolLM2-135M, Alpaca-cleaned, 3 epochs, seed=0, prov_dim=8.

| angle θ        | cos θ | 1 − cos θ | Δ eval_loss | slope |
|----------------|-------|-----------|-------------|-------|
| vanilla        | —     | —         | 0           | —     |
| vanilla_zeroed | 1.000 | 0.000     | −0.001      | —     |
| π/8            | 0.924 | 0.076     | +0.103      | 1.36  |
| π/6            | 0.866 | 0.134     | +0.168      | 1.25  |
| π/4            | 0.707 | 0.293     | +0.372      | 1.27  |
| π/2            | 0.000 | 1.000     | +1.537      | 1.54  |

Reference: vanilla eval_loss = 1.629. Slopes tight in 1.25–1.54.
Empirical predictor for the utility cost of this per-layer
post-projection rotational intervention family at similar scale.

### 2. T2b is empirically loose at this scale

Zeroing the lowest-frequency 8 of 64 head_dim coords (cos=1, sin=0)
costs 0.0005 in eval loss at 1024-token training/eval length. This means
the tested short-context setup barely depends on positional rotation in
those pairs; it does not prove those coordinates are globally unused.
This complements Chiang & Yogatama's theoretical result on the
*high*-frequency end, but the long-context scaling still needs to be
measured.

### 3. Post-projection rotation cannot be compensated

Per-layer rotation `R(θ)` applied after learned `W_Q, W_K` projections
cannot be absorbed by the projections (their training happens upstream).
ASIDE-style rotation on input embeddings *is* absorbed; AIR's additive
per-layer signal interacts with projections by superposition rather than
composition. The rotational/per-layer cell tested here fails specifically
due to this ordering. The 30-layer stack compounds the disruption. This is
a diagnosis of the tested placement, not a proof that all rotational
provenance channels must fail.

### 4. SmolLM2-135M is below the SEP discrimination floor

Vanilla SEP score = **−0.22**: the model executes data-slot probes 3×
more often than instruction-slot probes. There is no role-discrimination
substrate for architectural interventions to selectively modulate at
this scale. Methodological caveat: SEP is a discriminating metric only
above a certain capability floor.

## SEP results (n=200)

| variant                     | exec_instr | exec_data | SEP    |
|-----------------------------|------------|-----------|--------|
| vanilla                     | 0.090      | 0.310     | −0.22  |
| vanilla_zeroed              | 0.105      | 0.300     | −0.195 |
| rope_prov π/8 (fixed)       | 0.005      | 0.285     | −0.28  |
| rope_prov π/2 (fixed)       | 0.000      | 0.020     | −0.02  |
| rope_prov learnable (gap 0.32°) | 0.105  | 0.305     | −0.20  |

At π/8 the model becomes less responsive in both slots (94% drop in
exec_instr, 8% drop in exec_data) — generic degradation, not selectivity.

**Learnable-angle diagnostic (v2.5).** Replacing the fixed `θ_role` with
two `nn.Parameter` scalars (init=0) and letting SFT choose the angles:
the model autonomously drives the gap to **0.32°** (θ_I = −0.00279, θ_D
= +0.00279), 70× smaller than the smallest fixed angle (π/8 = 22.5°).
Trajectory plateaus by step ~300 at peak LR; final eval and SEP score
are indistinguishable from `vanilla_zeroed`. Through Ye et al.'s lens
this is the model telling us the architectural role channel cannot
compete with its style-dominated role perception at this scale and
mechanism location. Outcome (1) of the three predicted: channel dead.

Footnote: witness-substring detector matches the model's literal
witness; on 10 manually-reviewed examples, automatic and manual
labeling agreed 7–8/10. Bias is toward false negatives on data-slot
detection (model partial-executes probe with wrong content; no literal
witness substring). Bias direction strengthens the negative-baseline
finding.

## Related work

**Defense lineage (architectural):**
- Hines et al. 2024 (Spotlighting, MSR) — calls for the out-of-band
  channel direction.
- Chen et al. 2025 (StruQ) — structured queries via fine-tuning.
- Wu et al. 2024 (ISE) — input-layer additive segment embeddings.
- Zverev et al. 2026 (ASIDE, ICLR) — input-layer rotational.
- Kariyappa & Suh 2025 (AIR) — per-layer additive; the direct foil for
  our work.
- This work — per-layer rotational.

**Mechanistic grounding:**
- Ye, Cui, Hadfield-Menell 2026 (Role Confusion, MIT) — role perception
  is style-dominated; architectural tags compete and lose.
- Arditi et al. 2024, Wollschläger et al. ICLR 2025 — refusal direction
  geometry; multi-dimensional concept cones.
- Zhou et al. 2024, Aug 2025 follow-ups — safety encoding concentrated
  in a small subset of attention heads.

**RoPE modification landscape (methodology peers, not competitors):**
- Movahedi et al. 2026 (Selective RoPE, ICLR) — input-dependent
  learnable rotation; mostly linear/recurrent attention.
- Chiang & Yogatama 2025 — RoPE dimension inefficiency (high-frequency
  end).
- Yu et al. 2025 (ComRoPE), Ostmeier et al. 2024 (LieRE), Heo et al.
  2024 (Mixed RoPE), Veisi et al. 2025 (CARoPE), Wang et al. (TAPA).

**Orthogonal-mechanism defenses (cite for completeness):**
- Anonymous 2026 (PromptArmor, ICLR) — LLM-as-filter preprocessor.
- Chen et al. 2025 (DefensiveTokens, ICML/AISec) — test-time defense.

## Future work

Three independent reasons v1 fails, each suggesting a v2 direction:

- **v2a — pre-`W_Q,W_K` rotation.** Architectural fix for the
  compensation issue identified in Finding 3. Restores learned-weight
  ability to absorb the rotation, analogous to how AIR's additive
  per-layer signal composes with projections.
- **v2b — head-targeted rotation.** Concentrate signal where safety
  computation happens (small subset of attention heads per Zhou et al.).
  Uniform allocation across heads dilutes the signal.
- **v2c — layer-targeted rotation.** Concentrate at mid-early layers
  where role-relevant processing is denser (Arditi et al., Li et al.).

A v3 angle is the "test-time architecture modification" cell of an
orthogonal taxonomy (Test-time × Architecture vs Test-time × Input vs
Training-time × Architecture).

## Reproducibility

```
# Train a variant
uv run python -m rope_prov.train \
    --config src/rope_prov/configs/vanilla.yaml

# SEP evaluation
uv run python -m rope_prov.eval_sep \
    --model-path runs/vanilla-seed0/final \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --sep-json /path/to/SEP_dataset.json \
    --output results/sep/vanilla.json

# Tests
uv run pytest -v
```

W&B project: `d3banjan/rope-provenance`. Trajectories saved at
`results/training/*_trajectory.json`. SEP outputs at `results/sep/`.
