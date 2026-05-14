"""Pytest session bootstrap.

Repairs the HF cache env var if the user's default ``~/.cache/huggingface``
symlink points at an unmounted volume. Without this, anything that imports
``transformers.AutoTokenizer`` / ``AutoModel`` fails on a fresh shell.
"""

from __future__ import annotations

import os
from pathlib import Path


def _default_hf_cache_works() -> bool:
    target = Path.home() / ".cache" / "huggingface"
    if not target.exists():
        return True  # fresh user, transformers will create it
    try:
        # Resolves through symlinks; raises if the link target is missing.
        target.resolve(strict=True)
        return True
    except (FileNotFoundError, OSError):
        return False


if "HF_HOME" not in os.environ and not _default_hf_cache_works():
    fallback = Path.home() / ".cache" / "hf_local"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(fallback)
