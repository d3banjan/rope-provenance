"""Phase 2 wired test: ``apply_role_aware_rotary_paired`` plumbed into
``LlamaAttention.forward``.

Architectural note (see PLAN.md Phase 6, T2 wording): with the pair-aware
split, "uniform-role logits == vanilla logits" is **not** the right
invariant — the highest-frequency RoPE pairs are *replaced* with role
rotation, so patched ≠ vanilla even at uniform role (positional info on
those pairs is lost regardless of role assignment).

The correct invariant — and the one the model architecture actually
guarantees — is: **with uniform role_ids, the patched logits are invariant
under changes to the per-role angle.** That is what cancels in Q·K^T.

Three checks:

1. ``prov_dim=0`` ⇒ patched ≡ vanilla bit-for-bit (independent of role_ids).
2. With ``prov_dim=8`` and uniform role_ids, logits under role_angles=[0.0]
   match logits under role_angles=[0.7] within fp32 noise. (T2 operational.)
3. Mixed role_ids at angles [0, π/2] produce logits that differ measurably
   from the uniform-role baseline — the role signal is not a silent no-op.

Slow (downloads ~270MB if not cached):

    uv run pytest -q -m slow tests/test_phase2_wired.py
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.slow


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
BATCH = 2
SEQLEN = 64
SEED = 0


def _fresh_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    return model


def _common_inputs(vocab_size: int):
    torch.manual_seed(SEED)
    return torch.randint(0, vocab_size, (BATCH, SEQLEN))


def _logits(model, input_ids, role_ids=None):
    with torch.no_grad():
        kwargs = {"input_ids": input_ids}
        if role_ids is not None:
            kwargs["role_ids"] = role_ids
        return model(**kwargs).logits.detach().clone()


def test_prov_dim_zero_matches_vanilla_bitwise():
    from rope_prov.model import patch_model_with_role_aware_attention

    model = _fresh_model()
    input_ids = _common_inputs(model.config.vocab_size)
    ref_logits = _logits(model, input_ids)

    patch_model_with_role_aware_attention(model, prov_dim=0, role_angles=None)
    role_ids = torch.zeros(BATCH, SEQLEN, dtype=torch.long)
    out_logits = _logits(model, input_ids, role_ids=role_ids)

    max_abs_diff = (out_logits - ref_logits).abs().max().item()
    print(f"[wired prov_dim=0] max_abs_diff={max_abs_diff:.3e}")
    assert max_abs_diff < 1e-5


def test_uniform_role_logits_invariant_under_angle_change():
    """T2 operational: uniform-role rotation cancels in Q·K^T, so patched
    logits with the same role_ids should not depend on the per-role angle
    chosen for that role. (They will differ from vanilla because high-freq
    RoPE pairs are replaced by role rotation; that's a different concern.)"""
    from rope_prov.model import patch_model_with_role_aware_attention

    role_ids = torch.zeros(BATCH, SEQLEN, dtype=torch.long)  # all role 0

    model_a = _fresh_model()
    input_ids = _common_inputs(model_a.config.vocab_size)
    patch_model_with_role_aware_attention(
        model_a, prov_dim=8, role_angles=[0.0, math.pi / 2]
    )
    logits_a = _logits(model_a, input_ids, role_ids=role_ids)

    model_b = _fresh_model()
    patch_model_with_role_aware_attention(
        model_b, prov_dim=8, role_angles=[0.7, math.pi / 2]
    )
    logits_b = _logits(model_b, input_ids, role_ids=role_ids)

    max_abs_diff = (logits_a - logits_b).abs().max().item()
    print(f"[wired T2 operational] max_abs_diff={max_abs_diff:.3e}")
    # 30-layer fp32 accumulation; cancellation is exact in principle, so
    # the gap should sit at numerical noise.
    assert max_abs_diff < 1e-3, (
        f"Uniform-role logits depended on role angle (diff={max_abs_diff:.3e}); "
        "the role-rotation cancellation in Q·K^T (Theorem T2) is broken."
    )


def test_mixed_role_logits_differ_from_uniform_baseline():
    """Half INSTRUCTION (angle 0) and half DATA (angle π/2). Cross-role
    pairs pick up a cos(π/2) = 0 factor on the provenance subspace, so
    logits cannot match the uniform-role baseline."""
    from rope_prov.model import patch_model_with_role_aware_attention

    model = _fresh_model()
    input_ids = _common_inputs(model.config.vocab_size)
    patch_model_with_role_aware_attention(
        model, prov_dim=8, role_angles=[0.0, math.pi / 2]
    )

    uniform_role_ids = torch.zeros(BATCH, SEQLEN, dtype=torch.long)
    mixed_role_ids = uniform_role_ids.clone()
    mixed_role_ids[:, SEQLEN // 2 :] = 1

    uniform_logits = _logits(model, input_ids, role_ids=uniform_role_ids)
    mixed_logits = _logits(model, input_ids, role_ids=mixed_role_ids)

    max_abs_diff = (mixed_logits - uniform_logits).abs().max().item()
    mean_abs_diff = (mixed_logits - uniform_logits).abs().mean().item()
    print(
        f"[wired mixed-vs-uniform] max_abs_diff={max_abs_diff:.3e} "
        f"mean_abs_diff={mean_abs_diff:.3e}"
    )
    # The role rotation has to perturb the attention pattern enough that
    # Phase 4 training can pick it up. A two-layer fp32 model wouldn't move
    # much; SmolLM2 has 30 layers, so the signal should be clearly above
    # noise.
    assert max_abs_diff > 1e-2, (
        f"Mixed roles produced near-zero logit perturbation "
        f"({max_abs_diff:.3e}); rotation is a silent no-op."
    )
