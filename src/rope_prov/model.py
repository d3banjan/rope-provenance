"""Role-aware LlamaAttention wiring for SmolLM2 / Llama models.

Subclasses ``LlamaAttention`` and replaces the single ``apply_rotary_pos_emb``
call at the RoPE site with a call to :func:`rope_prov.rotary.apply_role_aware_rotary`,
which splits ``head_dim`` into a positional subspace (standard RoPE) and a
provenance subspace (fixed per-role rotation indexed by ``role_ids``).

When ``prov_dim == 0`` or ``role_ids is None``, falls back to vanilla
``apply_rotary_pos_emb`` — generation without role markup remains valid, and
``prov_dim=0`` configs reproduce the original model bit-for-bit.

Targets transformers 4.49.x; the forward body mirrors
``transformers.models.llama.modeling_llama.LlamaAttention.forward`` and will
need refresh if upstream changes the RoPE call shape (unlikely within the
4.45–4.49 window we pin in pyproject.toml).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .rotary import apply_role_aware_rotary_paired


class RoleAwareLlamaAttention(LlamaAttention):
    """``LlamaAttention`` with a role-aware RoPE call site.

    Args added beyond the base class:
        prov_dim:    number of trailing head_dim coordinates reserved for the
                     provenance rotation. ``0`` ⇒ behave as vanilla.
        role_angles: 1D tensor of fixed angles indexed by role id. Stored as a
                     non-persistent buffer (not part of the saved state dict).
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        prov_dim: int = 0,
        role_angles: Optional[Sequence[float]] = None,
    ):
        super().__init__(config, layer_idx)
        if prov_dim < 0:
            raise ValueError(f"prov_dim must be >= 0, got {prov_dim}")
        if prov_dim > self.head_dim:
            raise ValueError(
                f"prov_dim={prov_dim} exceeds head_dim={self.head_dim}"
            )
        if prov_dim % 2 != 0:
            raise ValueError(
                f"prov_dim must be even (rotate_half pairs), got {prov_dim}"
            )
        self.prov_dim = prov_dim
        if role_angles is None:
            role_angles = [0.0]
        angles = torch.as_tensor(list(role_angles), dtype=torch.float32)
        # Non-persistent so it doesn't pollute the model state dict; the
        # config (prov_dim, role_angles list) is the source of truth.
        self.register_buffer("role_angles", angles, persistent=False)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings  # [B, T, head_dim]

        if self.prov_dim == 0 or role_ids is None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        else:
            # Pair-aware split keeps the pretrained pair structure
            # (coord i paired with coord i + head_dim//2) on the positional
            # subspace, sacrificing only the *lowest-frequency* pairs
            # (highest-indexed in inv_freq) to provenance. See
            # rotary.apply_role_aware_rotary_paired docstring for the
            # convention + coordinate diagram.
            query_states, key_states = apply_role_aware_rotary_paired(
                query_states,
                key_states,
                cos,
                sin,
                role_ids,
                self.prov_dim,
                self.role_angles,
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if (
                self.config._attn_implementation == "sdpa"
                and kwargs.get("output_attentions", False)
            ):
                pass  # eager fallback for output_attentions under sdpa
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------- #
def _rebuild_attention_module(
    src: LlamaAttention,
    prov_dim: int = 0,
    role_angles: Optional[Sequence[float]] = None,
) -> RoleAwareLlamaAttention:
    """Construct a ``RoleAwareLlamaAttention`` initialized from ``src`` weights."""
    new = RoleAwareLlamaAttention(
        src.config, src.layer_idx, prov_dim=prov_dim, role_angles=role_angles
    )
    example_param = next(src.parameters(), None)
    if example_param is not None:
        new.to(device=example_param.device, dtype=example_param.dtype)
    # role_angles is non-persistent, so it's absent from src.state_dict() and
    # missing/unexpected lists stay clean.
    missing, unexpected = new.load_state_dict(src.state_dict(), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return new


def patch_model_with_role_aware_attention(
    model: nn.Module,
    prov_dim: int = 0,
    role_angles: Optional[Sequence[float]] = None,
) -> nn.Module:
    """Replace every ``model.model.layers[i].self_attn`` with a
    :class:`RoleAwareLlamaAttention`, preserving weights. In place; returns
    the same ``model`` for chaining.

    ``prov_dim=0`` (default) preserves vanilla behavior — the override exists
    solely to accept and ignore a ``role_ids`` kwarg.
    """
    for layer in model.model.layers:
        layer.self_attn = _rebuild_attention_module(
            layer.self_attn, prov_dim=prov_dim, role_angles=role_angles
        )
    return model
