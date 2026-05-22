"""PR6 operation-detector preflight for typed policy rails.

The permission rail works when operation ids are supplied by the harness.  This
script tests the next bottleneck: can a tiny classifier recover OBEY / USE /
QUOTE from frozen Qwen hidden states on held-out policy-rail templates?
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.slm_policy_rail import (  # noqa: E402
    DEFAULT_QWEN25_05_INSTRUCT,
    OP_NAMES,
    OP_OBEY,
    OP_QUOTE,
    OP_USE,
)
from scripts.slm_policy_vector import (  # noqa: E402
    PR4_HELDOUT_SOURCE_POLICIES,
    PR4_SEEN_SOURCE_POLICIES,
    POLICY_OPS,
    apply_prompt_format,
    build_source_policy_grid_examples,
    encode_prompt,
)

LABEL_BY_OP = {OP_OBEY: 0, OP_USE: 1, OP_QUOTE: 2}
LABEL_NAMES = {0: "OBEY", 1: "USE", 2: "QUOTE"}


@dataclass
class FeatureSet:
    x: torch.Tensor
    y: torch.Tensor
    cells: list[str]
    template_splits: list[str]


def _pad_rows(rows: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(row) for row in rows)
    return torch.tensor(
        [row + [pad_value] * (max_len - len(row)) for row in rows],
        dtype=torch.long,
    )


@torch.no_grad()
def collect_features(
    model,
    tokenizer,
    examples: list[dict],
    *,
    device: torch.device,
    batch_size: int,
    hidden_layer: int,
) -> FeatureSet:
    model.eval()
    vectors: list[torch.Tensor] = []
    labels: list[int] = []
    cells: list[str] = []
    template_splits: list[str] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        ids_rows = []
        op_rows = []
        masks = []
        for ex in chunk:
            input_ids, _source_ids, operation_ids, _policy_bits = encode_prompt(
                tokenizer,
                ex["prompt"],
                ex["prompt_sources"],
                ex["prompt_operations"],
                ex["prompt_policy_bits"],
            )
            ids_rows.append(input_ids)
            op_rows.append(operation_ids)
            masks.append([1] * len(input_ids))
        input_ids = _pad_rows(ids_rows, tokenizer.pad_token_id).to(device)
        operation_ids = _pad_rows(op_rows, 0).to(device)
        attention_mask = _pad_rows(masks, 0).to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[hidden_layer]
        for row, ex in enumerate(chunk):
            op_mask = torch.zeros_like(operation_ids[row], dtype=torch.bool)
            for op_id in POLICY_OPS:
                op_mask |= operation_ids[row] == op_id
            op_mask &= attention_mask[row].bool()
            if not bool(op_mask.any()):
                raise ValueError(f"example has no operation-labeled tokens: {ex}")
            vectors.append(hidden[row, op_mask].float().mean(dim=0).cpu())
            labels.append(LABEL_BY_OP[next(k for k, v in OP_NAMES.items() if v == ex["operation"])])
            cells.append(ex.get("cell", "unbucketed"))
            template_splits.append(ex.get("template_split", "unbucketed"))
    return FeatureSet(
        x=torch.stack(vectors),
        y=torch.tensor(labels, dtype=torch.long),
        cells=cells,
        template_splits=template_splits,
    )


def train_probe(
    train: FeatureSet,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    shuffle_labels: bool,
) -> nn.Linear:
    torch.manual_seed(seed)
    random.seed(seed)
    y = train.y.clone()
    if shuffle_labels:
        y = y[torch.randperm(len(y))]
    probe = nn.Linear(train.x.shape[1], len(LABEL_NAMES))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    for _step in range(steps):
        idx = torch.randint(0, len(train.x), (batch_size,))
        logits = probe(train.x[idx])
        loss = F.cross_entropy(logits, y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return probe


@torch.no_grad()
def evaluate_probe(probe: nn.Linear, features: FeatureSet) -> dict:
    logits = probe(features.x)
    pred = logits.argmax(dim=-1)
    hits = pred.eq(features.y)

    def rate(mask: list[bool]) -> float:
        idx = torch.tensor(mask, dtype=torch.bool)
        if not bool(idx.any()):
            return 0.0
        return float(hits[idx].float().mean().item())

    cells = sorted(set(features.cells))
    template_splits = sorted(set(features.template_splits))
    by_cell = {cell: rate([item == cell for item in features.cells]) for cell in cells}
    by_template = {
        split: rate([item == split for item in features.template_splits])
        for split in template_splits
    }
    by_label = {
        name: float(hits[features.y == idx].float().mean().item())
        for idx, name in LABEL_NAMES.items()
    }
    return {
        "accuracy": float(hits.float().mean().item()),
        "by_cell": by_cell,
        "by_template_split": by_template,
        "by_label": by_label,
        "n": len(features.y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_QWEN25_05_INSTRUCT)
    parser.add_argument("--cache-dir", default="/mnt/expansion/huggingface/hub")
    parser.add_argument("--train-pairs", type=int, default=96)
    parser.add_argument("--eval-pairs", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--prompt-format", choices=("raw", "answer", "chat"), default="chat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="results/slm/qwen25_0_5b_instruct_pr6_operation_detector_s0.json",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = build_source_policy_grid_examples(
        args.train_pairs,
        heldout_values=False,
        eval_control="correct",
        source_policy_pairs=PR4_SEEN_SOURCE_POLICIES,
        template_splits=("seen",),
    )
    eval_examples = build_source_policy_grid_examples(
        args.eval_pairs,
        heldout_values=True,
        eval_control="correct",
        source_policy_pairs=PR4_SEEN_SOURCE_POLICIES + PR4_HELDOUT_SOURCE_POLICIES,
        template_splits=("seen", "heldout"),
    )
    train_examples = apply_prompt_format(train_examples, tokenizer, args.prompt_format)
    eval_examples = apply_prompt_format(eval_examples, tokenizer, args.prompt_format)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.config.use_cache = False

    start = time.monotonic()
    train_features = collect_features(
        model,
        tokenizer,
        train_examples,
        device=device,
        batch_size=args.feature_batch_size,
        hidden_layer=args.hidden_layer,
    )
    eval_features = collect_features(
        model,
        tokenizer,
        eval_examples,
        device=device,
        batch_size=args.feature_batch_size,
        hidden_layer=args.hidden_layer,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    probe = train_probe(
        train_features,
        steps=args.steps,
        batch_size=args.probe_batch_size,
        lr=args.lr,
        seed=args.seed,
        shuffle_labels=False,
    )
    trap_probe = train_probe(
        train_features,
        steps=args.steps,
        batch_size=args.probe_batch_size,
        lr=args.lr,
        seed=args.seed,
        shuffle_labels=True,
    )
    train_metrics = evaluate_probe(probe, train_features)
    eval_metrics = evaluate_probe(probe, eval_features)
    trap_metrics = evaluate_probe(trap_probe, eval_features)
    result = {
        "args": vars(args),
        "label_names": LABEL_NAMES,
        "train_n": len(train_features.y),
        "eval_n": len(eval_features.y),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "shuffle_trap_metrics": trap_metrics,
        "elapsed_sec": time.monotonic() - start,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[operation-detector] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
