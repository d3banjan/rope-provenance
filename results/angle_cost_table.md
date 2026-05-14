# Per-Layer Role-Rotation Utility Cost

SmolLM2-135M, Alpaca-cleaned, 3 epochs, paired-seed, prov_dim=8.
Reference: vanilla eval_loss = 1.629.

| angle θ | cos(θ) | 1 − cos(θ) | Δ eval_loss | Δ / (1-cos θ) |
|---|---|---|---|---|
| 0       | 1.000  | 0.000      | -0.001 (vanilla_zeroed) | — |
| π/8     | 0.924  | 0.076      | +0.103              | 1.36 |
| π/6     | 0.866  | 0.134      | +0.168              | 1.25 |
| π/4     | 0.707  | 0.293      | +0.372              | 1.27 |
| π/2     | 0.000  | 1.000      | +1.537              | 1.54 |

## Relationship

Δ eval_loss tracks (1 − cos θ) approximately linearly; per-unit cost
climbs as θ → π/2 where cross-role content matching is fully rotated to
the perpendicular component. The (1 − cos θ) cost model gives a
back-of-envelope predictor for **any per-layer post-projection rotational
intervention** at this scale.

## Mechanism reminder

Per-layer rotation R(θ) applied *after* learned W_Q/W_K cannot be
compensated by learned weights (their projection happens upstream).
ASIDE-style rotation on input embeddings is absorbed by W_Q/W_K and pays
no such cost. The 30-layer SmolLM2-135M stack compounds the per-layer
disruption.

## SEP selectivity at π/8 (n=200)

| metric          | vanilla | rope_prov π/8 | Δ            |
|-----------------|---------|---------------|--------------|
| exec_rate_instr | 0.090   | 0.005         | -0.085 (-94%)|
| exec_rate_data  | 0.310   | 0.285         | -0.025 (-8%) |
| SEP score       | -0.22   | -0.28         | worse        |

**Outcome class (2): generic degradation, not selectivity.** exec_instr
falls 94% while exec_data barely moves — the model became less
responsive in both slots rather than more selective. Per-layer
post-projection rotation suppresses generation broadly. v1 architecture
not viable for selectivity at this scale; v2 (input-layer ASIDE-style
rotation absorbed by W_Q/W_K) required.

## Learnable-angle diagnostic (v2.5)

Replaced fixed `θ_role` with two `nn.Parameter` scalars (init=0, shared
across all 30 layers). 3 epochs SFT, all other hyperparameters identical
to fixed-angle runs.

| metric          | v2.5 result | vanilla_zeroed | Δ      |
|-----------------|-------------|----------------|--------|
| final eval_loss | 1.6285      | 1.6283         | +2e-4  |
| exec_rate_instr | 0.105       | 0.105          | 0      |
| exec_rate_data  | 0.305       | 0.300          | +0.005 |
| SEP score       | −0.20       | −0.195         | −0.005 |

Learned angles: θ_I = −0.00279, θ_D = +0.00279. **Gap = 0.32°**, 70×
smaller than π/8 = 22.5°. Trajectory plateaus by step ~300 of 1776 at
peak LR — gradient signal on the role channel saturates immediately.

This is the clearest possible "outcome 1": given freedom to choose the
angle, the model picks effectively zero. Channel-dead at this mechanism
location (per-layer, post-projection) is not an artifact of the manual
angle sweep — it's the model's revealed preference.
