import math

import pytest
import torch

from rope_prov.rotary import (
    Role,
    apply_hidden_role_rotation_paired,
    apply_role_aware_rotary,
    apply_role_aware_rotary_paired,
    rotate_half,
)


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


def test_t2b_uniform_role_equals_pos_rope_on_pos_slice_only(rng):
    """**Theorem T2b (contiguous formulation)**: with uniform ``role_ids``,
    the patched attention scores equal those of a model in which standard
    RoPE is applied to the positional slice and the provenance slice
    contributes the *unrotated* ``Q_prov · K_prov^T`` dot product.

    Equivalent statement: uniform role rotation cancels in `Q·K^T`, so the
    provenance slice contribution is invariant to the chosen role angle,
    and equals the "identity rotation" reference. This bounds what the
    architecture gives up — the operational counterpart on SmolLM2 is in
    ``tests/test_phase2_wired.py::test_t2b_uniform_role_equals_vanilla_with_zeroed_prov_pairs``."""
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


def test_t2a_uniform_role_logits_invariant_under_angle_change(rng):
    """**Theorem T2a (contiguous formulation)**: with uniform ``role_ids``,
    attention scores are invariant under arbitrary changes to the
    per-role angle vector. Cycle three distinct angle assignments and
    assert all produce bit-identical (modulo fp64 noise) Q·K^T."""
    B, H, T, D, P = 1, 2, 6, 16, 4
    pos_dim = D - P
    q = torch.randn(B, H, T, D, generator=rng, dtype=torch.float64)
    k = torch.randn(B, H, T, D, generator=rng, dtype=torch.float64)

    cos_full, sin_full = _pos_cos_sin(B, T, D)
    cos_pos, sin_pos = cos_full[..., :pos_dim], sin_full[..., :pos_dim]

    angle_vectors = [
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),       # all zero
        torch.tensor([0.7, -1.3, 2.4], dtype=torch.float64),
        torch.tensor([math.pi, math.pi / 7, -math.pi / 3], dtype=torch.float64),
    ]
    reference_scores = None
    for ang in angle_vectors:
        # Pin every token to role 0 — same uniform assignment across runs;
        # only the angle attached to that role varies.
        role_ids = torch.zeros((B, T), dtype=torch.long)
        q_rot, k_rot = apply_role_aware_rotary(
            q, k, cos_pos, sin_pos, role_ids, prov_dim=P, role_angles=ang,
        )
        scores = q_rot @ k_rot.transpose(-1, -2)
        if reference_scores is None:
            reference_scores = scores
        else:
            torch.testing.assert_close(
                scores, reference_scores, rtol=1e-10, atol=1e-10
            )


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


def test_paired_per_pair_role_angles_score():
    """Independent per-pair phases add separate cosine terms in QK."""
    B, H, T, D, P = 1, 1, 2, 8, 4
    half_h = D // 2
    half_p = (D - P) // 2

    qk = torch.zeros(B, H, T, D, dtype=torch.float64)
    qk[..., half_p:half_h] = torch.tensor([1.0, 2.0], dtype=torch.float64)

    cos, sin = _pos_cos_sin(B, T, D, dtype=torch.float64)
    role_angles = torch.tensor(
        [
            [0.0, 0.0],
            [math.pi / 2, math.pi / 3],
        ],
        dtype=torch.float64,
    )
    role_ids = torch.tensor([[0, 1]], dtype=torch.long)

    q_rot, k_rot = apply_role_aware_rotary_paired(
        qk.clone(),
        qk.clone(),
        cos,
        sin,
        role_ids,
        prov_dim=P,
        role_angles=role_angles,
    )
    score = (q_rot[0, 0, 0] @ k_rot[0, 0, 1]).item()
    expected = 1.0**2 * math.cos(math.pi / 2) + 2.0**2 * math.cos(math.pi / 3)
    assert score == pytest.approx(expected, abs=1e-12)


def test_role_enum_matches_default_angles():
    """Sanity: Role.INSTRUCTION=0 (angle 0), Role.DATA=1 (angle pi/2)."""
    assert Role.INSTRUCTION == 0
    assert Role.DATA == 1


def test_hidden_role_rotation_paired_preserves_pair_norms(rng):
    """Pre-W role rotation should be an orthogonal transform on each pair."""
    B, T, D, P = 2, 5, 16, 4
    hidden = torch.randn(B, T, D, generator=rng, dtype=torch.float64)
    role_ids = torch.tensor(
        [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        dtype=torch.long,
    )
    role_angles = torch.tensor([0.0, math.pi / 3], dtype=torch.float64)

    out = apply_hidden_role_rotation_paired(hidden, role_ids, P, role_angles)

    half_h = D // 2
    half_p = (D - P) // 2
    before = (
        hidden[..., half_p:half_h].pow(2)
        + hidden[..., half_h + half_p :].pow(2)
    )
    after = (
        out[..., half_p:half_h].pow(2)
        + out[..., half_h + half_p :].pow(2)
    )
    torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)

    # Non-provenance coordinates are untouched.
    torch.testing.assert_close(out[..., :half_p], hidden[..., :half_p])
    torch.testing.assert_close(
        out[..., half_h : half_h + half_p],
        hidden[..., half_h : half_h + half_p],
    )


def test_hidden_role_rotation_paired_accepts_per_pair_angles(rng):
    B, T, D, P = 2, 5, 16, 4
    hidden = torch.randn(B, T, D, generator=rng, dtype=torch.float64)
    role_ids = torch.tensor(
        [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        dtype=torch.long,
    )
    role_angles = torch.tensor(
        [[0.0, 0.0], [math.pi / 8, math.pi / 3]],
        dtype=torch.float64,
    )

    out = apply_hidden_role_rotation_paired(hidden, role_ids, P, role_angles)

    half_h = D // 2
    half_p = (D - P) // 2
    before = (
        hidden[..., half_p:half_h].pow(2)
        + hidden[..., half_h + half_p :].pow(2)
    )
    after = (
        out[..., half_p:half_h].pow(2)
        + out[..., half_h + half_p :].pow(2)
    )
    torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)
