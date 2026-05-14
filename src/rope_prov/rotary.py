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


def apply_role_aware_rotary(
    q: Tensor,
    k: Tensor,
    cos_pos: Tensor,
    sin_pos: Tensor,
    role_ids: Tensor | None,
    prov_dim: int,
    role_angles: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply standard RoPE to the leading head_dim - prov_dim coordinates,
    and a fixed per-role rotation to the trailing prov_dim coordinates.

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

    theta = role_angles.to(q.device, q.dtype)[role_ids]
    cos_r, sin_r = build_role_cos_sin(theta, prov_dim)
    q_prov = (q_prov * cos_r) + (rotate_half(q_prov) * sin_r)
    k_prov = (k_prov * cos_r) + (rotate_half(k_prov) * sin_r)

    return torch.cat((q_pos, q_prov), dim=-1), torch.cat((k_pos, k_prov), dim=-1)
