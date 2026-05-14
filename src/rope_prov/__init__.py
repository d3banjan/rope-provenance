from .parser import (
    MalformedMarkupError,
    OverlapError,
    RoleMap,
    SpanRule,
    parse_char_roles,
    parse_to_role_ids,
)
from .rotary import (
    Role,
    apply_role_aware_rotary,
    build_role_cos_sin,
    rotate_half,
)

__all__ = [
    "MalformedMarkupError",
    "OverlapError",
    "Role",
    "RoleMap",
    "SpanRule",
    "apply_role_aware_rotary",
    "build_role_cos_sin",
    "parse_char_roles",
    "parse_to_role_ids",
    "rotate_half",
]


def __getattr__(name):
    # Lazy import: model.py pulls transformers, which is heavy and not always
    # needed (parser/rotary work without it). Keep the public surface but
    # defer the import cost.
    if name in {"RoleAwareLlamaAttention", "patch_model_with_role_aware_attention"}:
        from . import model as _model
        return getattr(_model, name)
    raise AttributeError(f"module 'rope_prov' has no attribute {name!r}")
