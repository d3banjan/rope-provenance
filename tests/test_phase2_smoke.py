"""Phase 2 smoke test.

Verifies that we can subclass the attention module SmolLM2-135M actually uses
and thread a `role_ids` kwarg from `model(...)` down to `LlamaAttention.forward`
without crashing and without numerically perturbing the forward pass (since
the subclass currently ignores `role_ids`).

This test is `slow` so it is excluded from the default `uv run pytest` run.
Invoke explicitly:

    uv run pytest -q -m slow tests/test_phase2_smoke.py
"""

from __future__ import annotations

import os

import pytest
import torch

# Mark whole module as slow — downloads & loads a real model.
pytestmark = pytest.mark.slow


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
BATCH = 2
SEQLEN = 64
SEED = 0


def _captured_kwargs():
    """Build a hook that captures the kwargs passed into self_attn.forward.

    Returns (hook_fn, captured_list). The hook installs itself via the
    subclass's forward; we use a module-level list so the test can inspect
    what was actually received.
    """
    captured: list[dict] = []
    return captured


def test_role_ids_kwarg_threads_through_smollm2():
    torch.manual_seed(SEED)

    # Lazy import so that collecting tests without the model deps still works.
    from transformers import AutoModelForCausalLM

    from rope_prov.model import (
        RoleAwareLlamaAttention,
        patch_model_with_role_aware_attention,
    )

    # ---- Load the model (CPU, fp32 — no CUDA assumed). ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()

    # ---- Record the attention class hierarchy for the report. ----
    attn0 = model.model.layers[0].self_attn
    attn_cls_name = type(attn0).__name__
    mro = [f"{c.__module__}.{c.__name__}" for c in type(attn0).__mro__]
    print(f"[phase2] attention class: {attn_cls_name}")
    print(f"[phase2] attention MRO: {mro}")

    # ---- Inputs. ----
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQLEN))
    role_ids = torch.zeros(BATCH, SEQLEN, dtype=torch.long)

    # ---- Reference (unpatched) forward. ----
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids).logits.detach().clone()

    # ---- Patch in the role-aware subclass. ----
    patch_model_with_role_aware_attention(model)
    for layer in model.model.layers:
        assert isinstance(layer.self_attn, RoleAwareLlamaAttention), (
            "patch failed to install RoleAwareLlamaAttention on every layer"
        )

    # ---- Install a sentinel: capture whether role_ids actually arrives
    #      in the subclass forward. We monkey-patch one layer's forward
    #      with a wrapper that records the kwarg presence. ----
    captured: dict = {"saw_role_ids": False, "shape": None, "all_zero": None}
    target_attn = model.model.layers[0].self_attn
    original_forward = target_attn.forward

    def spy_forward(*args, role_ids=None, **kwargs):
        if role_ids is not None:
            captured["saw_role_ids"] = True
            captured["shape"] = tuple(role_ids.shape)
            captured["all_zero"] = bool(torch.all(role_ids == 0).item())
        # Delegate to the real subclass forward (which itself ignores role_ids).
        return original_forward(*args, role_ids=role_ids, **kwargs)

    target_attn.forward = spy_forward  # type: ignore[assignment]

    # ---- Patched forward with role_ids=zeros. ----
    with torch.no_grad():
        out = model(input_ids=input_ids, role_ids=role_ids)
        patched_logits = out.logits.detach().clone()

    # ---- Assertions. ----
    assert captured["saw_role_ids"], (
        "role_ids was filtered upstream before reaching LlamaAttention.forward"
    )
    assert captured["shape"] == (BATCH, SEQLEN)
    assert captured["all_zero"] is True

    max_abs_diff = (patched_logits - ref_logits).abs().max().item()
    print(f"[phase2] max_abs_logits_diff = {max_abs_diff:.3e}")

    # Subclass ignores role_ids, so outputs should match within fp32 noise.
    # Any drift here would indicate state was lost during the patch.
    assert max_abs_diff < 1e-5, (
        f"patched logits differ from unpatched by {max_abs_diff:.3e}; "
        "weights or buffers were not preserved by the patch"
    )

    # Surface the number for the user / CI logs.
    print(
        f"[phase2] OK transformers="
        f"{__import__('transformers').__version__} attn={attn_cls_name} "
        f"max_abs_diff={max_abs_diff:.3e}"
    )
