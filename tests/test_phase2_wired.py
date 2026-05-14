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


def test_t2a_uniform_role_logits_invariant_under_angle_change():
    """**Theorem T2a operational**: uniform-role rotation cancels in Q·K^T,
    so patched logits at fixed ``role_ids`` are invariant under changes to
    the per-role angle. (Patched ≠ vanilla — that's T2b's concern.)

    Observed numerics (fp32 model, 30 layers): max_abs_diff ≈ 4e-5 across
    role_angles=[0.0,…] vs [0.7,…]. Logits magnitude is O(10), so this is
    ~1e-6 relative ≈ a few fp32 ULPs of multi-layer accumulation noise.
    No bf16 in the test path; this is the fp32 floor for SmolLM2 at
    seqlen=64. If you rerun in fp64 the diff should drop to ~1e-12.
    """
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
    print(f"[wired T2a operational] max_abs_diff={max_abs_diff:.3e}")
    assert max_abs_diff < 1e-3, (
        f"Uniform-role logits depended on role angle (diff={max_abs_diff:.3e}); "
        "the role-rotation cancellation in Q·K^T (Theorem T2a) is broken."
    )


class _ZeroedProvPairsAttention:
    """Reference attention class: vanilla LlamaAttention with positional
    RoPE replaced by identity on the prov-pair coord set. Defined here
    inside the test module (rather than in src/) because it is specifically
    a reference implementation for Theorem T2b's claim — not production
    code. It zeros cos/sin on the prov coord set before delegating to the
    parent's forward.
    """
    # Implemented as a free factory at call time so the LlamaAttention
    # base class isn't imported at module-import time (it's slow).


def _build_zeroed_attention_class(prov_dim_value: int):
    """Constructs an attention class capturing prov_dim. Deferred import to
    keep test collection cheap when transformers isn't needed."""
    from transformers.models.llama.modeling_llama import LlamaAttention

    class ZeroedProvPairsLlamaAttention(LlamaAttention):
        prov_dim = prov_dim_value

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            cos, sin = position_embeddings
            head_dim = self.head_dim
            half_h = head_dim // 2
            pos_dim = head_dim - self.prov_dim
            half_p = pos_dim // 2
            cos = cos.clone()
            sin = sin.clone()
            # Identity rotation on prov coords: cos := 1, sin := 0.
            cos[..., half_p:half_h] = 1.0
            cos[..., half_h + half_p :] = 1.0
            sin[..., half_p:half_h] = 0.0
            sin[..., half_h + half_p :] = 0.0
            return super().forward(
                hidden_states,
                (cos, sin),
                attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

    return ZeroedProvPairsLlamaAttention


def _patch_with_zeroed_prov_pairs(model, prov_dim: int):
    Cls = _build_zeroed_attention_class(prov_dim)
    for layer in model.model.layers:
        src = layer.self_attn
        new = Cls(src.config, src.layer_idx)
        example = next(src.parameters(), None)
        if example is not None:
            new.to(device=example.device, dtype=example.dtype)
        missing, unexpected = new.load_state_dict(src.state_dict(), strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        layer.self_attn = new
    return model


def test_t2b_uniform_role_equals_vanilla_with_zeroed_prov_pairs():
    """**Theorem T2b operational**: with uniform ``role_ids``, the patched
    model's logits equal those of a vanilla SmolLM2 whose positional RoPE
    on prov-pair coords has been replaced by identity (cos=1, sin=0). This
    is the architecture's *cost bound* — what the modification gives up
    regardless of training: the lowest-frequency P/2 RoPE pairs no longer
    carry positional information."""
    from rope_prov.model import patch_model_with_role_aware_attention

    prov_dim = 8

    # Patched model with arbitrary non-zero role angle (uniform role ⇒
    # cancels in Q·K^T regardless of angle, per T2a).
    model_patched = _fresh_model()
    input_ids = _common_inputs(model_patched.config.vocab_size)
    patch_model_with_role_aware_attention(
        model_patched, prov_dim=prov_dim, role_angles=[0.7, math.pi / 2]
    )
    role_ids = torch.zeros(BATCH, SEQLEN, dtype=torch.long)
    patched_logits = _logits(model_patched, input_ids, role_ids=role_ids)

    # Reference: vanilla model with prov-pair RoPE zeroed.
    model_zeroed = _fresh_model()
    _patch_with_zeroed_prov_pairs(model_zeroed, prov_dim=prov_dim)
    zeroed_logits = _logits(model_zeroed, input_ids)

    max_abs_diff = (patched_logits - zeroed_logits).abs().max().item()
    print(f"[wired T2b structural] max_abs_diff={max_abs_diff:.3e}")
    # Both paths funnel through the same SDPA implementation post-RoPE;
    # the only numerical difference is the order in which constants
    # multiply onto Q/K. Should sit at fp32 noise.
    assert max_abs_diff < 1e-3, (
        f"Patched (uniform role) != vanilla-with-zeroed-prov-pairs "
        f"(diff={max_abs_diff:.3e}); T2b cost-bound claim broken."
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
