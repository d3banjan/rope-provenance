import math

import pytest
import torch

from rope_prov.rotary import Role, apply_role_aware_rotary, rotate_half


def _standard_rope(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def _pos_cos_sin(B, T, head_dim, base=10000.0, dtype=torch.float64):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype) / half))
    t = torch.arange(T, dtype=dtype)
    freqs = torch.outer(t, inv_freq)              # [T, half]
    emb = torch.cat((freqs, freqs), dim=-1)       # [T, head_dim]
    cos = emb.cos()[None, None, :, :].expand(B, 1, T, head_dim)
    sin = emb.sin()[None, None, :, :].expand(B, 1, T, head_dim)
    return cos, sin


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(0)


def test_identity_when_prov_dim_zero(rng):
    B, H, T, D = 2, 4, 8, 16
    q = torch.randn(B, H, T, D, generator=rng, dtype=torch.float64)
    k = torch.randn(B, H, T, D, generator=rng, dtype=torch.float64)
    cos, sin = _pos_cos_sin(B, T, D)

    q_ref, k_ref = _standard_rope(q, k, cos, sin)
    q_out, k_out = apply_role_aware_rotary(
        q, k, cos, sin,
        role_ids=None, prov_dim=0,
        role_angles=torch.empty(0, dtype=torch.float64),
    )

    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=1e-12)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=1e-12)


def test_equal_role_preserves_attention_scores(rng):
    """When every token has the same role, role rotation is global and
    cancels inside Q K^T. So attention scores match the prov_dim=0 case
    (same standard RoPE on the leading subspace, no rotation on trailing)."""
    B, H, T, D, P = 1, 2, 6, 16, 4
    pos_dim = D - P
    q = torch.randn(B, H, T, D, generator=rng, dtype=torch.float64)
    k = torch.randn(B, H, T, D, generator=rng, dtype=torch.float64)

    cos_full, sin_full = _pos_cos_sin(B, T, D)
    cos_pos, sin_pos = cos_full[..., :pos_dim], sin_full[..., :pos_dim]

    role_angles = torch.tensor([0.3, 1.2, -0.7], dtype=torch.float64)

    for role in (0, 1, 2):
        role_ids = torch.full((B, T), role, dtype=torch.long)
        q_rot, k_rot = apply_role_aware_rotary(
            q, k, cos_pos, sin_pos, role_ids, prov_dim=P, role_angles=role_angles,
        )
        scores = q_rot @ k_rot.transpose(-1, -2)

        # Reference: standard RoPE on the positional slice only; no rotation
        # on the provenance slice. Equal-role rotation cancels in Q K^T.
        q_pos_ref, k_pos_ref = _standard_rope(
            q[..., :pos_dim], k[..., :pos_dim], cos_pos, sin_pos,
        )
        scores_ref = (
            q_pos_ref @ k_pos_ref.transpose(-1, -2)
            + q[..., pos_dim:] @ k[..., pos_dim:].transpose(-1, -2)
        )

        torch.testing.assert_close(scores, scores_ref, rtol=1e-10, atol=1e-10)


def test_cross_role_phase_offset_predicted_score():
    """With matched q=k on the provenance slice and zero positional slice,
    Q K^T on a (role_a, role_b) pair equals |v|^2 * cos(angle_a - angle_b)."""
    B, H, T, D, P = 1, 1, 2, 4, 2
    pos_dim = D - P

    # Zero out positional subspace so it doesn't affect the score.
    v = torch.tensor([1.0, 0.0], dtype=torch.float64)
    qk = torch.zeros(B, H, T, D, dtype=torch.float64)
    qk[..., pos_dim:] = v

    cos_full, sin_full = _pos_cos_sin(B, T, D, dtype=torch.float64)
    # Zero positional rotation contribution by zeroing the positional part —
    # already zero, so cos_pos / sin_pos values are irrelevant for those dims.
    cos_pos, sin_pos = cos_full[..., :pos_dim], sin_full[..., :pos_dim]

    angles = [0.0, math.pi / 2, math.pi / 3, -math.pi / 4]
    role_angles = torch.tensor(angles, dtype=torch.float64)

    for a_idx, angle_a in enumerate(angles):
        for b_idx, angle_b in enumerate(angles):
            role_ids = torch.tensor([[a_idx, b_idx]], dtype=torch.long)
            q_rot, k_rot = apply_role_aware_rotary(
                qk.clone(), qk.clone(), cos_pos, sin_pos,
                role_ids, prov_dim=P, role_angles=role_angles,
            )
            score = (q_rot[0, 0, 0] @ k_rot[0, 0, 1]).item()
            expected = (v @ v).item() * math.cos(angle_a - angle_b)
            assert score == pytest.approx(expected, abs=1e-12), (
                f"roles=({angle_a:.3f},{angle_b:.3f}) got {score}, want {expected}"
            )


def test_role_enum_matches_default_angles():
    """Sanity: Role.INSTRUCTION=0 (angle 0), Role.DATA=1 (angle pi/2)."""
    assert Role.INSTRUCTION == 0
    assert Role.DATA == 1
