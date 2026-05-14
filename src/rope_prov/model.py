"""Phase 2 smoke-test scaffolding: a minimal LlamaAttention subclass that
accepts a `role_ids` kwarg in `forward(...)`, ignores it, and delegates to
the parent. Includes a helper to patch a loaded SmolLM2 / Llama model in
place, swapping every decoder layer's `self_attn` with the subclass while
preserving its weights and other state.

This module deliberately does NOT wire `role_ids` into the rotary embedding
yet — that is the next sub-step. The goal here is only to derisk that the
kwarg threads cleanly through the (`LlamaForCausalLM` → `LlamaModel` →
`LlamaDecoderLayer` → `LlamaAttention`) call chain in the installed
transformers version.
"""

from __future__ import annotations

from typing import Type

import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention


class RoleAwareLlamaAttention(LlamaAttention):
    """A minimal LlamaAttention subclass that accepts and *ignores* `role_ids`.

    The real role-aware rotary wiring will replace this body in the next step;
    for now we only verify that `role_ids` is reachable here at runtime.
    """

    def forward(self, *args, role_ids=None, **kwargs):  # type: ignore[override]
        # `role_ids` is accepted purely for kwarg-threading verification and
        # is intentionally unused. Everything else passes through unchanged.
        del role_ids
        return super().forward(*args, **kwargs)


def _rebuild_attention_module(
    src: LlamaAttention,
    cls: Type[LlamaAttention] = RoleAwareLlamaAttention,
) -> LlamaAttention:
    """Construct a new attention module of `cls` and copy state from `src`.

    LlamaAttention's __init__ takes `(config, layer_idx)`. We reuse the
    source module's config + layer_idx, then load_state_dict from the source
    so weights and buffers are preserved bit-for-bit.
    """
    new = cls(src.config, src.layer_idx)
    # Match dtype/device of the source before copying parameters.
    example_param = next(src.parameters(), None)
    if example_param is not None:
        new.to(device=example_param.device, dtype=example_param.dtype)
    missing, unexpected = new.load_state_dict(src.state_dict(), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return new


def patch_model_with_role_aware_attention(model: nn.Module) -> nn.Module:
    """Replace every decoder layer's `self_attn` with `RoleAwareLlamaAttention`,
    preserving weights. Returns the same model object (modified in place).
    """
    layers = model.model.layers
    for layer in layers:
        layer.self_attn = _rebuild_attention_module(layer.self_attn)
    return model
