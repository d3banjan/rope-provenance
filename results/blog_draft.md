---
title: "RoPE-Provenance: A First Architectural Attempt at the Out-of-Band Role Channel"
date: 2026-05-14
author: Debanjan Basu
---

Hines et al. (Microsoft Research, 2024), in their Spotlighting paper,
explicitly pointed at the direction this work explores:

> "Out-of-band signaling has many advantages over in-band signaling… we
> need to devise a multi-channel analog for LLMs. In this approach,
> control tokens would be passed to model in a separate 'channel' from
> the data tokens, and the model would (somehow) only react to
> instructive tokens from the control layer. With current architectures
> of common language models, however, this is not feasible in any
> straightforward way. Nonetheless, this premise is compelling and
> future work remains to be done in this area."

Ye, Cui, and Hadfield-Menell (MIT, 2026) describe *why* this matters at
the mechanistic level: the architectural role-tag boundary is overwhelmed
by *style* in latent space. Their one-line framing:

> "Security is defined at the interface but authority is assigned in
> latent space."

This work tests whether a stronger architectural signal — per-layer
rotation on a subspace of RoPE — can compete with the style-dominated
role perception Ye et al. characterize. At SmolLM2-135M scale we find it
cannot, but the failure is informative: we derive a (1 − cos θ) cost law
for per-layer rotational interventions, identify post-projection
placement as the specific mechanism that prevents compensation, and
complete a 2×2 design-space comparison that places our work as
complement (not competitor) to recent additive analogs.

## The 2×2 design space

A clean lineage of architectural instruction/data separation defenses
exists in 2024–2026 work:

|              | Input-layer            | Per-layer                |
|--------------|------------------------|--------------------------|
| **Additive** | ISE (Wu 2024)          | AIR (Kariyappa & Suh 2025) |
| **Rotational** | ASIDE (Zverev 2026)  | **this work**            |

AIR (NVIDIA, arxiv:2505.18907) is the per-layer additive analog — they
inject layer-specific trainable embeddings at intermediate
representations and report 1.6×–9.2× attack success rate reduction. Our
work tests the rotational version of the same per-layer placement. The
asymmetric result (additive succeeds, rotational fails) is the core
contribution.

## The mechanism in one paragraph

Standard RoPE rotates each `(coord i, coord i + head_dim/2)` pair by
`position_id · inv_freq[i]`. Inv_freq is geometric in `i`: pair index 0
rotates fastest, pair index `head_dim/2 − 1` rotates slowest. We split
`head_dim = 64` into 28 positional pairs (the highest-frequency ones)
and 4 provenance pairs (the lowest-frequency ones), then apply an
additional rotation by `θ_role` on the provenance pairs at every layer.
INSTRUCTION tokens get `θ = 0`; DATA tokens get `θ = π/2` (or the other
angles in the sweep).

The intuition: cross-role attention scores get rotated by
`(θ_b − θ_a)`, which scales their content-match component by
`cos(θ_b − θ_a)`. At `θ = π/2`, content matching between instruction
and data tokens gets rotated 90° out of phase.

## Finding 1: (1 − cos θ) is the utility-cost predictor

Across the angle sweep, Δ eval_loss tracks `(1 − cos θ)` approximately
linearly:

| angle θ        | cos θ | 1 − cos θ | Δ eval_loss | slope |
|----------------|-------|-----------|-------------|-------|
| vanilla_zeroed | 1.000 | 0.000     | −0.001      | —     |
| π/8            | 0.924 | 0.076     | +0.103      | 1.36  |
| π/6            | 0.866 | 0.134     | +0.168      | 1.25  |
| π/4            | 0.707 | 0.293     | +0.372      | 1.27  |
| π/2            | 0.000 | 1.000     | +1.537      | 1.54  |

Slopes 1.25–1.54. This relationship is a quotable predictor for the
utility cost of any per-layer post-projection rotational intervention at
similar scale.

## Finding 2: the lowest-frequency RoPE pairs are dimensionally unused

Zeroing the lowest-frequency 8 of 64 head_dim coords trains to within
0.001 of vanilla eval loss. The bottom of the RoPE frequency ladder is
empirically unused at 135M / 1024-seq scale.

This is complementary to Chiang & Yogatama (2025), which establishes
theoretical underutilization at the *high*-frequency end (early-index
pairs, which wrap many times in long contexts). Our finding is at the
opposite end — low-frequency, late-index pairs at short contexts barely
rotate, so they *could* encode position information if the model used
them; it just doesn't. Two ends of the RoPE spectrum, both underutilized
at this scale, for different reasons.

## Finding 3: post-projection rotation is uncompensatable

At `θ = π/2`, eval loss balloons from 1.629 (vanilla) to 3.166 (Δ =
+1.537). With 3 epochs of full fine-tuning the model cannot recover.

The mechanism is direct. Per-layer rotation `R(θ)` applied *after* the
learned `W_Q, W_K` projections means the rotation operates on
already-projected Q, K. The projections were trained on un-rotated Q, K;
gradient cannot push them to compensate because the rotation is
downstream of where they act. Three contrasts:

- **ASIDE-style rotation** on input embeddings is upstream of `W_Q,
  W_K` — the projections see rotated input and absorb the rotation.
- **AIR's additive per-layer signal** interacts with the projections by
  superposition: the projection of a sum is the sum of projections, so
  the additive role channel propagates through learned weights
  cleanly.
- **Our per-layer rotational signal** is multiplicative on
  already-projected Q, K. Multiplication does not commute with the
  preceding projection in a way the projection can compensate.

The 30-layer SmolLM2 stack compounds this: each layer's role rotation
is a fresh disruption. The π/2 eval loss is the compounded version.

## Finding 4: 135M is below the SEP discrimination floor

The SEP benchmark (Should it be Executed or Processed) measures whether
a model executes injected probes more often when they appear in the
instruction slot vs the data slot. A well-aligned model scores high
positive; a model that ignores slot structure scores near zero.

Vanilla SmolLM2-135M scores **−0.22**: it executes data-slot probes 3×
more often than instruction-slot probes. The model has no role-aware
substrate for our mechanism to selectively modulate. The SEP signal is
negative because at this scale the model defaults to "execute whatever
looks like a directive."

This is a methodological caveat for the field. SEP is a discriminating
metric only above a certain capability floor. Defense interventions on
sub-floor models cannot be evaluated for selectivity; they can only be
evaluated for utility cost.

## The selectivity attempt and what it told us

At π/8 (the smallest angle preserving most utility), SEP results:

| variant       | exec_rate_instr | exec_rate_data | SEP   |
|---------------|-----------------|----------------|-------|
| vanilla       | 0.090           | 0.310          | −0.22 |
| rope_prov π/8 | 0.005           | 0.285          | −0.28 |

exec_rate_instr falls by 94% (0.090 → 0.005). exec_rate_data falls by
8%. This is generic degradation, not selectivity. The mechanism makes
the model less responsive in both slots — actually *more* suppressed in
the slot we wanted to preserve. SEP score gets *worse*.

Through Ye et al.'s lens this makes sense: the model's role perception
is style-dominated. Our architectural signal had to compete against
that, and at 135M scale it didn't compete well enough to be selective —
it only added enough noise to suppress responsiveness uniformly.

### Cross-check: let the model choose the angle

To rule out "we picked the wrong angle in the sweep," we ran a
diagnostic with two learnable `nn.Parameter` scalars in place of fixed
`θ_role`, both initialized at 0. After 3 epochs the model autonomously
drove the gap to **0.32°** (`θ_I = −0.00279`, `θ_D = +0.00279`) — 70×
smaller than the smallest fixed angle in the sweep (π/8 = 22.5°). The
trajectory plateaus by step ~300 *at peak LR*, not late in training as
LR decays; gradient signal on the role channel saturates almost
immediately and the rest of training is drift. Final eval loss (1.6285)
and SEP score (−0.20) are indistinguishable from `vanilla_zeroed`
(1.6283, −0.195) — i.e., the model behaves as if the role channel
weren't there at all.

This is a clean empirical version of Ye et al.'s framing. Given the
freedom to use the architectural role signal, the model declines: at
this scale and mechanism location, style-based role inference dominates
to the point where any architectural signal is computationally not
worth using.

## Why publish a negative result

The four findings together save the next researcher's time:

- **Finding 1** is a quantitative cost predictor.
- **Finding 2** says you can repurpose ~12% of RoPE coords for free at
  this scale (and complements Chiang's high-frequency observation).
- **Finding 3** locates *where* a rotational role signal needs to go
  (pre-projection, not post). A v2 architecture should start here, not
  from scratch.
- **Finding 4** is a methodological note for the field about SEP's
  effective scale floor.

This also fills the missing cell of the 2×2 design space — a clean
contribution that's hard to argue with regardless of the eval outcome.

## Future work

Three v2 directions, each motivated by an independent failure mode:

- **v2a: pre-`W_Q,W_K` rotation.** Architectural fix for the compensation
  problem identified in Finding 3. The model's projections see the
  rotated input and learn to absorb it.
- **v2b: head-targeted rotation.** Zhou et al. and follow-up work show
  safety is concentrated in a small subset of attention heads. Uniform
  allocation of `prov_dim` across all heads dilutes the signal. Probe
  for safety-critical heads, apply rotation only there.
- **v2c: layer-targeted rotation.** Arditi et al. and the "mid-early
  safety layer" hypothesis (Li et al.) suggest role-relevant computation
  is concentrated in early-middle layers. Modifying every layer dilutes
  across irrelevant ones.

A v3 angle, motivated by DefensiveTokens (test-time input modification)
and the broader 2×2 of {test-time, training-time} × {input,
architecture}: the test-time architectural-modification cell is open.
Can role-aware rotation be applied only at inference? Probably hard
given the negative result, but worth flagging.

## Code

GitHub: <https://github.com/debanjan-basu/rope-provenance>.
W&B project: <https://wandb.ai/d3banjan/rope-provenance>.
All training configs in `src/rope_prov/configs/`. SEP eval reads the
local SEP dataset (Zverev et al.'s repo) via the path in
`results/sep/*.json`.

```
uv sync --extra dev
uv run python -m rope_prov.train --config src/rope_prov/configs/vanilla.yaml
uv run python -m rope_prov.eval_sep \
    --model-path runs/vanilla-seed0/final \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --sep-json /path/to/SEP_dataset.json \
    --output results/sep/vanilla.json
```

## Acknowledgements

The reframing of this work — from "we tried a thing" to "first
architectural attempt at the out-of-band channel direction identified by
Hines and grounded by Ye" — came out of a literature survey conducted
after v1 experiments completed. Hines et al.'s future-work paragraph
named the direction; Ye et al.'s role-confusion paper named the failure
mode; Kariyappa & Suh's AIR established the per-layer-vs-input-layer
diagnosis for the additive case. Our contribution is the rotational
cell of the 2×2, with an architectural diagnosis of why it fails where
the additive analog succeeds.
