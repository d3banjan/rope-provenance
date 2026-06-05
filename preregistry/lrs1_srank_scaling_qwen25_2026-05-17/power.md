# Power notes — lrs1_srank_scaling_qwen25 — 2026-05-17

## Design

- 3 width points: Qwen2.5-0.5B/1.5B/3B (d_model 896 / 1536 / 2048).
- 1 seed (42), DPO only.
- Per size, per (layer × module) adapter: stable rank from the O1 kernel.
- Decision metric `attn_qkv_srank` per size = mean over the q/k/v_proj
  adapters = n_layers × 3 = 72 (0.5B) / 84 (1.5B) / 108 (3B) samples.
- `global_srank` (all 7 modules; 168 / 196 / 252 samples) is a reported
  secondary, not the decision metric — see H0 §2 metric note.

## Within-size precision

`attn_qkv_srank` is a mean over 72–108 per-module stable ranks. The per-module
RMS dispersion within a single Pythia run is ≈ 1.08 (lazy-rudder, 410M). With
~90 samples the standard error of the mean is ≈ 1.08 / √90 ≈ 0.114 — so each
`attn_qkv_srank` point is precise to roughly ±0.12. Within-size precision is
not the limiting factor.

## The thin-DOF problem (the real limit)

The **scaling fit has only n = 3 points** and each candidate model has k = 1
parameter → residual DOF = n − k = 2. Consequences:

- AIC differences are high-variance. A single noisy `global_srank` can flip
  `ΔAIC` past the decisive threshold of 2. **AIC here ranks models; it does
  not test them.**
- The honest expected outcome, absent large curvature, is
  `INCONCLUSIVE_THIN_DOF` on the AIC criterion alone.
- Therefore the **primary verdict is the robust band criterion** (H0 §3): all
  three `global_srank ∈ [2.5, 5.0]` and max−min spread `< 1.5`. This is a
  direct, low-variance reading of "flat floor" that does not depend on fitting
  a curve to 3 points.

## Detectable effect

- The lazy-rudder Pythia QKV-module spread across 4 sizes is max−min ≈
  3.93 − 3.13 ≈ 0.80 (DPO). A genuine width-scaling effect on Qwen of
  comparable or larger size would push the spread past `srank_spread_max = 1.5`
  or a point outside `[2.5, 5.0]` — detectable by the band criterion.
- A `c/√d` law over 896→2048 predicts a srank ratio of √(2048/896) ≈ 1.51 —
  i.e. the smallest model's srank ~1.5× the largest. Against a floor of ~3.6
  that is a spread of ~1.8, which the spread criterion would catch.

## Escalation

If the AIC criterion is inconclusive but the band criterion is met, the verdict
is `REPLICATES_FLAT_FLOOR` on the robust reading. If both are inconclusive,
the pre-registered escalation is a **4th width point — Qwen2.5-7B** — added to
the fit (n=4 matches the lazy-rudder design and roughly halves the AIC
variance). 7B training needs cloud/A100 or 4-bit QLoRA; not run in lrs1.

## Compute budget

- CPU prep + smoke + V1: ~1 h.
- GPU: ~2.5–3.5 h for the 3 DPO runs serial; final analysis is CPU, minutes.

## Risks to power

- **Single seed:** no within-architecture sampling error on the scaling fit.
  Mitigation — bootstrap `c` over the per-(layer,module) srank population and
  report a CI on each fit; if a verdict sits on a τ boundary, add seed 117 at
  one size before concluding.
- **bf16 vs fp16:** a recipe confound. srank is invariant to a positive scalar
  on ΔW, so dtype cannot move the floor by construction; noted, not corrected.
- **GQA k/v modules:** `d_out ≤ r` caps their srank low; they pull the mean
  down uniformly across sizes, so they bias the absolute floor but not the
  *scaling* — the per-module-type breakdown isolates this.
