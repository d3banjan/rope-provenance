"""Adapter smoke tests for gated provenance on pretrained SLMs.

This script is intentionally narrower than the SmolLM2 RoPE experiments. It
asks whether a pretrained base SLM can learn the gated selection task at all
before we spend GPU on hidden role/provenance channels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.toy_role_provenance import (  # noqa: E402
    ROLE_CONTROL_CHOICES,
    build_examples,
    build_lm_texts,
    prepare_example_roles,
)


DEFAULT_QWEN05_BASE = (
    "/mnt/expansion/huggingface/hub/models--Qwen--Qwen2.5-0.5B/"
    "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        self.lora_b = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = self.dropout(x).to(torch.float32)
        lora_out = self.lora_b(self.lora_a(lora_in)) * self.scale
        return base_out + lora_out.to(base_out.dtype)


def replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def add_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_suffixes: tuple[str, ...],
) -> list[str]:
    for param in model.parameters():
        param.requires_grad_(False)
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(target_suffixes):
            replacements.append((name, module))
    for name, module in replacements:
        replace_module(
            model,
            name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
    return [name for name, _ in replacements]


def build_prepared_examples(
    n_pairs: int,
    *,
    heldout: bool,
    template_mode: str,
    gate_kinds: tuple[str, ...],
    hide_tags: bool,
    role_control: str,
) -> list[dict]:
    examples = build_examples(
        n_pairs,
        heldout=heldout,
        template_mode=template_mode,
        gate_kinds=gate_kinds,
    )
    return prepare_example_roles(
        examples,
        hide_tags=hide_tags,
        role_control=role_control,
    )


def apply_prompt_format(examples: list[dict], tokenizer, prompt_format: str) -> list[dict]:
    if prompt_format == "raw":
        return examples
    formatted = []
    for ex in examples:
        item = dict(ex)
        prompt = ex["prompt"].rstrip()
        answer = ex["expected"]
        if prompt_format == "answer":
            item["prompt"] = f"{prompt}\nAnswer: "
            item["text"] = f"{item['prompt']}{answer}{tokenizer.eos_token or ''}"
        elif prompt_format == "chat":
            if tokenizer.chat_template is None:
                raise ValueError("tokenizer has no chat_template")
            item["prompt"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            item["text"] = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            raise ValueError(f"unknown prompt_format={prompt_format!r}")
        formatted.append(item)
    return formatted


def encode_examples(
    examples: Iterable[dict],
    tokenizer,
    *,
    max_length: int,
    fail_on_truncation: bool,
) -> list[EncodedExample]:
    encoded = []
    truncated = 0
    lost_answer = 0
    for ex in examples:
        text = ex["text"]
        prompt_len = len(ex.get("prompt", ""))
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        input_ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
        if offsets and offsets[-1][1] < len(text):
            truncated += 1
        labels = list(input_ids)
        for idx, (start, _end) in enumerate(offsets):
            if start < prompt_len:
                labels[idx] = -100
        if not any(label != -100 for label in labels):
            lost_answer += 1
        encoded.append(EncodedExample(input_ids=input_ids, labels=labels))
    if fail_on_truncation and (truncated or lost_answer):
        raise ValueError(
            "encoding lost supervision: "
            f"truncated={truncated}, lost_answer={lost_answer}, max_length={max_length}"
        )
    return encoded


def make_batch(
    encoded: list[EncodedExample],
    *,
    batch_size: int,
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = random.choices(encoded, k=batch_size)
    max_len = max(len(ex.input_ids) for ex in batch)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    labels = []
    attention_mask = []
    for ex in batch:
        pad = max_len - len(ex.input_ids)
        input_ids.append(ex.input_ids + [pad_id] * pad)
        labels.append(ex.labels + [-100] * pad)
        attention_mask.append([1] * len(ex.input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    examples: list[dict],
    *,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    correct = 0
    samples = []
    try:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            prompts = [ex["prompt"] for ex in chunk]
            enc = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            generated = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = enc["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
            )
            for ex, out in zip(chunk, decoded):
                hit = ex["expected"].lower() in out.lower()
                correct += int(hit)
                if len(samples) < 8:
                    samples.append(
                        {
                            "expected": ex["expected"],
                            "witness": ex["witness"],
                            "answer": ex["answer"],
                            "output": out[:160],
                            "hit": hit,
                        }
                    )
    finally:
        tokenizer.padding_side = old_padding_side
    exact = correct / max(len(examples), 1)
    return {
        "exact_match": exact,
        "sep": exact,
        "n": len(examples),
        "samples": samples,
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_QWEN05_BASE)
    parser.add_argument("--cache-dir", default="/mnt/expansion/huggingface/hub")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--eval-pairs", type=int, default=128)
    parser.add_argument("--lm-mix", type=int, default=0)
    parser.add_argument(
        "--template-mode",
        choices=("simple", "diverse", "gated", "gate_pretrain"),
        default="gate_pretrain",
    )
    parser.add_argument(
        "--gate-kinds",
        nargs="+",
        choices=("color_first", "no_not", "question"),
        default=["color_first", "no_not", "question"],
    )
    parser.add_argument("--hide-tags", action="store_true")
    parser.add_argument(
        "--prompt-format",
        choices=("raw", "answer", "chat"),
        default="raw",
        help="Response framing for SLM runs. Use answer for base models, chat for instruct.",
    )
    parser.add_argument(
        "--role-control",
        choices=ROLE_CONTROL_CHOICES,
        default="correct",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/slm/qwen25_0_5b_gate.json")
    parser.add_argument("--save-adapter", default=None)
    parser.add_argument("--fail-on-truncation", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="rope-provenance")
    parser.add_argument("--wandb-entity", default="d3banjan")
    parser.add_argument("--wandb-group", default="slm-gate-provenance")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.json")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples = build_prepared_examples(
        args.train_pairs,
        heldout=False,
        template_mode=args.template_mode,
        gate_kinds=tuple(args.gate_kinds),
        hide_tags=args.hide_tags,
        role_control=args.role_control,
    )
    if args.lm_mix:
        train_examples.extend(build_lm_texts(args.lm_mix))
    eval_examples = build_prepared_examples(
        args.eval_pairs,
        heldout=True,
        template_mode=args.template_mode,
        gate_kinds=tuple(args.gate_kinds),
        hide_tags=args.hide_tags,
        role_control=args.role_control,
    )
    train_examples = apply_prompt_format(train_examples, tokenizer, args.prompt_format)
    eval_examples = apply_prompt_format(eval_examples, tokenizer, args.prompt_format)
    encoded = encode_examples(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        fail_on_truncation=args.fail_on_truncation,
    )

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    patched = add_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_suffixes=tuple(args.lora_targets),
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                **vars(args),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "patched_modules": patched,
            },
        )

    print(
        f"[slm] model={args.model} total_params={total_params:,} "
        f"trainable={trainable_params:,} patched={len(patched)} "
        f"train_examples={len(encoded)} eval_examples={len(eval_examples)}",
        flush=True,
    )
    history = []
    start = time.monotonic()
    scaler_enabled = device.type == "cuda"

    def run_eval(step: int, loss_value: float | None) -> None:
        metrics = evaluate(
            model,
            tokenizer,
            eval_examples,
            device=device,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        peak_alloc_gb = 0.0
        peak_reserved_gb = 0.0
        if device.type == "cuda":
            peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)
        rec = {
            "step": step,
            "loss": loss_value,
            "elapsed_sec": time.monotonic() - start,
            "peak_alloc_gb": peak_alloc_gb,
            "peak_reserved_gb": peak_reserved_gb,
            **{k: v for k, v in metrics.items() if k != "samples"},
        }
        history.append(rec)
        if wandb_run is not None:
            wandb_run.log(rec, step=step)
        partial_path.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "patched_modules": patched,
                    "history": history,
                    "latest_samples": metrics["samples"],
                },
                indent=2,
            )
        )
        print(
            f"[slm] step={step} loss={loss_value} exact={metrics['exact_match']:.3f} "
            f"peak={peak_reserved_gb:.2f}GB elapsed={rec['elapsed_sec']:.1f}s",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.steps == 0:
        run_eval(0, None)
    for step in range(1, args.steps + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            batch = make_batch(
                encoded,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=scaler_enabled,
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                )
                loss = loss / args.grad_accum
            loss.backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            1.0,
        )
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            run_eval(step, total_loss)

    final = evaluate(
        model,
        tokenizer,
        eval_examples,
        device=device,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    result = {
        "args": vars(args),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "patched_modules": patched,
        "history": history,
        "final": final,
    }
    out_path.write_text(json.dumps(result, indent=2))
    if args.save_adapter:
        adapter_path = Path(args.save_adapter)
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": trainable_state_dict(model),
                "args": vars(args),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "patched_modules": patched,
            },
            adapter_path,
        )
    if wandb_run is not None:
        wandb_run.log(
            {
                "final/exact_match": final["exact_match"],
                "final/sep": final["sep"],
            },
            step=args.steps,
        )
        wandb_run.finish()
    print(json.dumps({k: v for k, v in final.items() if k != "samples"}, indent=2))
    print(f"[slm] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
