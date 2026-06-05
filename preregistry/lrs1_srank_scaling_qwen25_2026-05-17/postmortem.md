# Postmortem — lrs1_srank_scaling_qwen25 — 2026-05-17

**Verdict:** `REPLICATES_FLAT_FLOOR`
**Decision metric:** `attn_qkv_srank` (mean stable rank over q/k/v_proj LoRA adapters)

## Result

DPO LoRA adapters trained on Qwen2.5-{0.5B, 1.5B, 3B} base, recipe identical to
`the lazy-rudder paper` (r=128, α=256, lr 5e-6, 800 steps, hh-rlhf 2000 pairs,
seed 42) except dtype bf16 and — for 3B only — 8-bit Adam (see Amendment 01).

| size | d_model | attn_qkv_srank | global_srank (7-mod) |
|------|---------|----------------|----------------------|
| 0.5B | 896  | 3.435 | 4.600 |
| 1.5B | 1536 | 3.963 | 4.426 |
| 3B   | 2048 | 3.939 | 4.163 |

- All three `attn_qkv_srank` lie in the pre-registered band [2.5, 5.0].
- Max−min spread = **0.527** < `srank_spread_max` 1.5.
- ΔAIC discrimination (n=3): constant fit **decisively best** — `c/√d` worse by
  ΔAIC +7.73, `c/d^⅓` worse by +6.06. `ΔAIC(constant) = 0 < 2`.
- Constant fit: **c = 3.779 ± 0.243 RMS**. lazy-rudder Pythia reference:
  3.653 ± 0.289 RMS. The two architectures agree within ~0.13.

Both the band/spread criterion (primary) and the ΔAIC test (secondary) point
the same way. The flat task-intrinsic srank floor **survives the GPT-NeoX →
Llama-style architecture change** (split q/k/v/o_proj, GQA, gated SwiGLU,
RMSNorm). H1 holds; H0 (architecture-sensitive scaling) is rejected.

## Optimizer control (Amendment 01)

3B did not fit bf16 LoRA on the 12 GB card (OOM ~0.6 GB; the pre-registered
CPU-offload fallback failed — peft LoRA backward is incompatible with
accelerate meta-device offload). 3B was trained with 8-bit Adam
(`adamw_bnb_8bit`); a 0.5B 8-bit-Adam control quantifies the deviation:

- 0.5B 32-bit AdamW: attn_qkv_srank 3.435
- 0.5B 8-bit  AdamW: attn_qkv_srank 3.511
- **Δ = 0.076 ≤ tol 0.25 → control PASS.**

The optimizer change moves srank by 0.076 — below the per-point precision
(±0.12) and far below the spread that would matter. The mixed recipe is
accepted; 3B is used in the primary scaling fit. Had the control failed, 3B
would have been demoted to optimizer-confounded-exploratory and the verdict
would have rested on the 0.5B/1.5B 2-point check (which is itself flat — see
table — so the headline would not have changed).

## Kernel validation (V1)

The O1 QR-reduce stable-rank kernel reproduced lazy-rudder's recorded
Pythia-410M srank to rel 0.001% (`query_key_value` 3.9244, 4-module global
5.1968). V1 also corrected the decision metric: lazy-rudder's published floor
3.653 is the `query_key_value` module only, not the 7-module global mean —
hence `attn_qkv_srank`, not `global_srank`, is the comparison metric.

## Secondary observations

- The 7-module `global_srank` is also fairly flat (4.60 / 4.43 / 4.16) — mildly
  *decreasing* with width, dominated by the high-srank MLP/o_proj modules; not
  the lazy-rudder-comparable metric but recorded for the consumer.
- Within each run, srank **descends over training steps** toward the floor
  (3B: 4.21 → 3.94; 1.5B: 4.17 → 3.96; 0.5B: 3.56 → 3.44) — the floor behaves
  as an attractor, not a random endpoint, consistent with "task-intrinsic".
- Random-adapter null (analytic baseline): srank 30–80 across modules/sizes —
  the measured ~3.8 is an order of magnitude below noise.

## Methodology findings (carry-over)

1. bf16 LoRA at r=128 on the 7 Qwen modules does not fit a 3B model in 12 GB
   (peak ~12.1 GB) — the 152k-vocab logits + 239M-param adapter optimizer
   states are the cost.
2. accelerate meta-device CPU-offload is incompatible with peft LoRA backward
   (`MmBackward0 ... expected device meta but got cuda:0`) — not a usable
   training fallback. The pre-registered fallback (H0 §9) was wrong; corrected
   in Amendment 01.
3. 8-bit Adam was the minimal viable, controlled fallback.

## Interpretation (per feedback_research_loop_order)

The task-intrinsic stable-rank floor is **architecture-robust**: DPO
preference-tuning forms a low-rank (srank ≈ 3.8, flat across 4× width) update
geometry on the attention QKV projections of both GPT-NeoX (Pythia) and
Llama-style (Qwen2.5) models. The floor is set by preference-learning
complexity, not parameter count or attention-block design.

For the downstream role-provenance workstream: **yes** — Qwen preference tuning
forms a genuine low-stable-rank QKV control surface, and it does so
consistently across scale. This supports the interpretation that additive role
vectors operate on an *existing* low-rank alignment control surface rather than
constructing one. The per-step checkpoint manifests ship the geometry payload
(`geometry_ready`) for the direct DPO-adapter vs role-adapter subspace
comparison.

No Lean co-evolution: this is a measurement/replication probe, no new mechanism.

## Escalation

Not required — the verdict is decisive. The pre-registered escalation (a 4th
width point, Qwen2.5-7B) is available if a future rung wants to tighten the
n=3 ΔAIC fit, but is not warranted by this result.

obstacle_class: replication-probe
