"""Analyze SLM LoRA adapter rank and overlap with the base->instruct delta."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


LORA_A_SUFFIX = ".lora_a.weight"
LORA_B_SUFFIX = ".lora_b.weight"
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
DEFAULT_BASE = (
    "/mnt/expansion/huggingface/hub/models--Qwen--Qwen2.5-0.5B/"
    "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)
DEFAULT_INSTRUCT = (
    "/mnt/expansion/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)


def stable_rank(matrix: torch.Tensor) -> float:
    matrix = matrix.float()
    frob_sq = float(torch.sum(matrix * matrix).item())
    if frob_sq == 0.0:
        return 0.0
    spectral = float(torch.linalg.matrix_norm(matrix, ord=2).item())
    if spectral == 0.0:
        return 0.0
    return frob_sq / (spectral * spectral)


def orth_basis(matrix: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(matrix.float(), mode="reduced")
    if r.numel() == 0:
        return q[:, :0]
    keep = torch.abs(torch.diag(r)) > 1e-7
    return q[:, keep]


def layer_index(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def module_records(
    adapter_state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
    instruct_state: dict[str, torch.Tensor],
    *,
    alpha: float,
    rank: int,
) -> list[dict]:
    prefixes = sorted(
        key[: -len(LORA_A_SUFFIX)]
        for key in adapter_state
        if key.endswith(LORA_A_SUFFIX)
    )
    records = []
    for prefix in prefixes:
        a = adapter_state[f"{prefix}{LORA_A_SUFFIX}"].float()
        b = adapter_state[f"{prefix}{LORA_B_SUFFIX}"].float()
        weight_key = f"{prefix}.weight"
        if weight_key not in base_state or weight_key not in instruct_state:
            continue
        lora_delta = (b @ a) * (alpha / rank)
        sft_delta = instruct_state[weight_key].float() - base_state[weight_key].float()
        lora_norm = float(torch.linalg.matrix_norm(lora_delta).item())
        sft_norm = float(torch.linalg.matrix_norm(sft_delta).item())
        cosine = 0.0
        if lora_norm and sft_norm:
            cosine = float(torch.sum(lora_delta * sft_delta).item() / (lora_norm * sft_norm))

        right_basis = orth_basis(a.T)
        left_basis = orth_basis(b)
        sft_frob_sq = float(torch.sum(sft_delta * sft_delta).item())
        right_capture = 0.0
        left_capture = 0.0
        both_capture = 0.0
        if sft_frob_sq:
            if right_basis.numel():
                right_capture = float(
                    torch.sum((sft_delta @ right_basis) ** 2).item() / sft_frob_sq
                )
            if left_basis.numel():
                left_capture = float(
                    torch.sum((left_basis.T @ sft_delta) ** 2).item() / sft_frob_sq
                )
            if right_basis.numel() and left_basis.numel():
                both_capture = float(
                    torch.sum((left_basis.T @ sft_delta @ right_basis) ** 2).item()
                    / sft_frob_sq
                )
        out_dim, in_dim = sft_delta.shape
        records.append(
            {
                "module": prefix,
                "layer": layer_index(prefix),
                "shape": [out_dim, in_dim],
                "rank_cap": rank,
                "lora_frob": lora_norm,
                "sft_frob": sft_norm,
                "lora_stable_rank": stable_rank(lora_delta),
                "lora_sft_cosine": cosine,
                "right_capture": right_capture,
                "left_capture": left_capture,
                "both_capture": both_capture,
                "right_random_baseline": right_basis.shape[1] / in_dim,
                "left_random_baseline": left_basis.shape[1] / out_dim,
            }
        )
    return records


def weighted_mean(records: list[dict], key: str, weight_key: str) -> float:
    total_weight = sum(float(record[weight_key]) for record in records)
    if total_weight == 0.0:
        return 0.0
    return sum(float(record[key]) * float(record[weight_key]) for record in records) / total_weight


def summarize(records: list[dict], role_emb: torch.Tensor | None) -> dict:
    by_layer: dict[str, list[dict]] = {}
    for record in records:
        by_layer.setdefault(str(record["layer"]), []).append(record)
    layer_summary = {}
    for layer, layer_records in by_layer.items():
        layer_summary[layer] = {
            "modules": len(layer_records),
            "lora_frob": sum(float(record["lora_frob"]) for record in layer_records),
            "mean_lora_stable_rank": weighted_mean(
                layer_records, "lora_stable_rank", "lora_frob"
            ),
            "mean_lora_sft_cosine": weighted_mean(
                layer_records, "lora_sft_cosine", "lora_frob"
            ),
            "mean_right_capture": weighted_mean(
                layer_records, "right_capture", "sft_frob"
            ),
            "mean_left_capture": weighted_mean(layer_records, "left_capture", "sft_frob"),
            "mean_both_capture": weighted_mean(layer_records, "both_capture", "sft_frob"),
        }
    role_summary = None
    if role_emb is not None:
        active = role_emb.float()[1:]
        role_summary = {
            "shape": list(role_emb.shape),
            "active_shape": list(active.shape),
            "stable_rank": stable_rank(active),
            "frob": float(torch.linalg.vector_norm(active).item()),
        }
    return {
        "modules": len(records),
        "mean_lora_stable_rank": weighted_mean(records, "lora_stable_rank", "lora_frob"),
        "mean_lora_sft_cosine": weighted_mean(records, "lora_sft_cosine", "lora_frob"),
        "mean_right_capture": weighted_mean(records, "right_capture", "sft_frob"),
        "mean_left_capture": weighted_mean(records, "left_capture", "sft_frob"),
        "mean_both_capture": weighted_mean(records, "both_capture", "sft_frob"),
        "layers": layer_summary,
        "role_embedding": role_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--instruct-model", default=DEFAULT_INSTRUCT)
    parser.add_argument("--cache-dir", default="/mnt/expansion/huggingface/hub")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = torch.load(args.adapter, map_location="cpu")
    adapter_state = payload["state_dict"]
    adapter_args = payload.get("args", {})
    rank = int(adapter_args.get("lora_rank", 0))
    alpha = float(adapter_args.get("lora_alpha", 1.0))
    role_emb = adapter_state.get("input_role_emb.weight")

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    instruct = AutoModelForCausalLM.from_pretrained(
        args.instruct_model,
        cache_dir=args.cache_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    records = []
    if rank > 0:
        records = module_records(
            adapter_state,
            base.state_dict(),
            instruct.state_dict(),
            alpha=alpha,
            rank=rank,
        )
    result = {
        "adapter": args.adapter,
        "base_model": args.base_model,
        "instruct_model": args.instruct_model,
        "summary": summarize(records, role_emb),
        "modules": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
