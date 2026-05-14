"""Role-aware rotary position embedding.

Splits head_dim into a positional subspace (standard RoPE on position_ids)
and a provenance subspace (fixed rotation per role_id).
"""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor


class Role(IntEnum):
    INSTRUCTION = 0
    DATA = 1


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_role_cos_sin(theta: Tensor, prov_dim: int) -> tuple[Tensor, Tensor]:
    """theta: [B, T] -> cos, sin: [B, 1, T, prov_dim]."""
    cos = theta.cos()[:, None, :, None].expand(-1, 1, -1, prov_dim)
    sin = theta.sin()[:, None, :, None].expand(-1, 1, -1, prov_dim)
    return cos, sin


def apply_role_aware_rotary_paired(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    role_ids: Tensor | None,
    prov_dim: int,
    role_angles: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pair-aware role-aware RoPE for HF Llama-style attention.

    HF Llama pairs coords (i, i + head_dim//2) for RoPE. To avoid scrambling
    those pairs when carving out a provenance subspace, this variant assigns:

      - **Positional subspace** = the *lowest-frequency* half-pairs.
        Coords {0..half_p-1, half_h..half_h+half_p-1}, where
        half_h = head_dim // 2 and half_p = pos_dim // 2. Receives standard
        RoPE rotation by ``position_ids`` (via the supplied full-length
        ``cos`` / ``sin``).
      - **Provenance subspace** = the *highest-frequency* half-pairs.
        Coords {half_p..half_h-1, half_h+half_p..head_dim-1}. Receives a
        per-token rotation by ``role_angles[role_ids]``.

    Sacrifices the highest-frequency positional info to encode role. For a
    pretrained model this is the right trade — the high-freq pairs carry
    fine-grained position info that's least costly to repurpose during SFT.

    q, k:        [B, H, T, head_dim]
    cos, sin:    full-length cos/sin from the model's
                 ``LlamaRotaryEmbedding``, shape broadcastable to
                 [B, T, head_dim] or [B, 1, T, head_dim].
    role_ids:    [B, T] int, indexes into ``role_angles``.
    prov_dim:    number of trailing head_dim coords (in pair terms) carved
                 out for role rotation. Must be even and <= head_dim.
    role_angles: [num_roles] float.
    """
    head_dim = q.shape[-1]
    if prov_dim == 0 or role_ids is None:
        # Pure RoPE on full head_dim.
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

    half_h = head_dim // 2
    pos_dim = head_dim - prov_dim
    half_p = pos_dim // 2

    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # Standard RoPE applied to the full tensor; we overwrite prov coords
    # below. Negligible extra cost vs. computing only on pos coords.
    q_rope = (q * cos) + (rotate_half(q) * sin)
    k_rope = (k * cos) + (rotate_half(k) * sin)

    # Role rotation on prov pair (half_p+j, half_h+half_p+j) for j in [0, half_v).
    work_dtype = (
        torch.float32 if q.dtype in (torch.bfloat16, torch.float16) else q.dtype
    )
    theta = role_angles.to(q.device, work_dtype)[role_ids]  # [B, T]
    cos_r = theta.cos()[:, None, :, None]   # [B, 1, T, 1]
    sin_r = theta.sin()[:, None, :, None]
    if cos_r.dtype != q.dtype:
        cos_r = cos_r.to(q.dtype)
        sin_r = sin_r.to(q.dtype)

    def _rotate_prov(x: Tensor) -> tuple[Tensor, Tensor]:
        first = x[..., half_p:half_h]                  # [B, H, T, half_v]
        second = x[..., half_h + half_p :]             # [B, H, T, half_v]
        first_rot = first * cos_r - second * sin_r
        second_rot = second * cos_r + first * sin_r
        return first_rot, second_rot

    q_first_rot, q_second_rot = _rotate_prov(q)
    k_first_rot, k_second_rot = _rotate_prov(k)

    q_out = q_rope.clone()
    q_out[..., half_p:half_h] = q_first_rot
    q_out[..., half_h + half_p :] = q_second_rot
    k_out = k_rope.clone()
    k_out[..., half_p:half_h] = k_first_rot
    k_out[..., half_h + half_p :] = k_second_rot
    return q_out, k_out


def apply_role_aware_rotary(
    q: Tensor,
    k: Tensor,
    cos_pos: Tensor,
    sin_pos: Tensor,
    role_ids: Tensor | None,
    prov_dim: int,
    role_angles: Tensor,
) -> tuple[Tensor, Tensor]:
    """Contiguous-split role-aware RoPE.

    The leading ``head_dim - prov_dim`` coordinates receive standard RoPE
    (caller pre-slices ``cos_pos`` / ``sin_pos`` to this width). The
    trailing ``prov_dim`` coordinates receive a per-role rotation indexed
    by ``role_ids``.

    Note: this formulation assumes the embedding pair structure can be
    re-paired within the contiguous slice (i.e., training from scratch
    or with an embedding layout that aligns with the slice). For patching
    a pretrained HF Llama model — whose RoPE pairs coords (i, i+head_dim/2)
    — use :func:`apply_role_aware_rotary_paired` instead.

    The mathematical unit tests in ``tests/test_rotary.py`` exercise this
    contiguous formulation; the operational integration tests in
    ``tests/test_phase2_wired.py`` exercise the paired formulation.

    q, k:        [B, H, T, head_dim]
    cos_pos,
    sin_pos:     broadcastable to [B, 1, T, pos_dim] (full head_dim if prov_dim==0)
    role_ids:    [B, T] int; ignored when prov_dim == 0
    role_angles: [num_roles] float, indexed by role_id
    """
    head_dim = q.shape[-1]
    pos_dim = head_dim - prov_dim

    if prov_dim == 0:
        q_rot = (q * cos_pos) + (rotate_half(q) * sin_pos)
        k_rot = (k * cos_pos) + (rotate_half(k) * sin_pos)
        return q_rot, k_rot

    if role_ids is None:
        raise ValueError("role_ids required when prov_dim > 0")

    q_pos, q_prov = q[..., :pos_dim], q[..., pos_dim:]
    k_pos, k_prov = k[..., :pos_dim], k[..., pos_dim:]

    q_pos = (q_pos * cos_pos) + (rotate_half(q_pos) * sin_pos)
    k_pos = (k_pos * cos_pos) + (rotate_half(k_pos) * sin_pos)

    # Compute cos/sin in at least fp32 then cast back: bf16/fp16 cosine of a
    # small angle loses meaningful precision (≈1e-3 ULPs). For fp64 q, stay
    # in fp64 — promoting *down* to fp32 would silently degrade test precision.
    work_dtype = (
        torch.float32 if q.dtype in (torch.bfloat16, torch.float16) else q.dtype
    )
    theta = role_angles.to(q.device, work_dtype)[role_ids]
    cos_r, sin_r = build_role_cos_sin(theta, prov_dim)
    if cos_r.dtype != q.dtype:
        cos_r = cos_r.to(q.dtype)
        sin_r = sin_r.to(q.dtype)
    q_prov = (q_prov * cos_r) + (rotate_half(q_prov) * sin_r)
    k_prov = (k_prov * cos_r) + (rotate_half(k_prov) * sin_r)

    return torch.cat((q_pos, q_prov), dim=-1), torch.cat((k_pos, k_prov), dim=-1)
