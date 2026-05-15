"""Phase 4 training script.

Usage:

    uv run python -m rope_prov.train --config src/rope_prov/configs/vanilla.yaml
    uv run python -m rope_prov.train --config src/rope_prov/configs/rope_prov.yaml

Paired-run guarantee (do NOT bypass): vanilla.yaml and rope_prov.yaml share
``base.yaml``. The only fields that differ are ``provenance_dims``,
``role_angles``, ``wandb_run_name``. Seed, data shuffle, LR schedule, eval
cadence, batch size — all bit-identical so the W&B group plots align without
manual fiddling.

Step-0 sanity is enforced by ``SanityWrappedCollator``: the first training
batch must have more than one unique ``role_id``, else the run aborts. This
catches the "parser silently emits all-default-role" failure mode where
rope-prov would otherwise train as vanilla and you wouldn't know until SEP
scores match.

Throughput calibration: ``--dry-run --max-steps 100`` runs N steps and
prints ``examples/sec`` to stderr. Target on a 3060: ≥2 ex/sec at
seq=1024, batch=8, grad-accum=4, bf16. If below 1.5, drop seq to 768.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .data import (
    RoleTaggingCollator,
    build_counterfactual_examples,
    filter_alpaca_examples,
)
from .model import (
    patch_model_with_pre_w_role_aware_attention,
    patch_model_with_role_aware_attention,
    patch_model_with_zeroed_prov_pairs,
)
from .parser import RoleMap


# ---------- config -----------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str) -> dict:
    p = Path(path).resolve()
    cfg = _load_yaml(p)
    if "extends" in cfg:
        base_path = (p.parent / cfg.pop("extends")).resolve()
        base = _load_yaml(base_path)
        base.update(cfg)
        cfg = base
    # Path normalization: role_map is relative to repo root.
    repo_root = _find_repo_root(p)
    cfg["role_map"] = str((repo_root / cfg["role_map"]).resolve())
    cfg["_repo_root"] = str(repo_root)
    return cfg


def _find_repo_root(start: Path) -> Path:
    cur = start.parent if start.is_file() else start
    while cur != cur.parent:
        if (cur / "pyproject.toml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def normalize_role_angles(role_angles, num_roles: int, prov_dim: int) -> list:
    """Validate role angles and reshape flat per-pair configs if needed.

    Accepted forms:

    - ``[num_roles]``: one shared phase per role.
    - ``[num_roles][prov_dim/2]``: independent phase per role and pair.
    - flat ``num_roles * prov_dim/2``: CLI-friendly per-pair form, reshaped.
    """
    if prov_dim % 2 != 0:
        raise ValueError(f"provenance_dims must be even, got {prov_dim}")
    pair_count = prov_dim // 2
    angles = list(role_angles)
    if angles and isinstance(angles[0], (list, tuple)):
        if len(angles) != num_roles:
            raise ValueError(
                f"role_angles has {len(angles)} role rows but role_map "
                f"declares {num_roles} roles."
            )
        out = [list(row) for row in angles]
        bad_rows = [i for i, row in enumerate(out) if len(row) != pair_count]
        if bad_rows:
            raise ValueError(
                f"per-pair role_angles rows must each have {pair_count} "
                f"entries for provenance_dims={prov_dim}; bad rows={bad_rows}."
            )
        return out
    if len(angles) == num_roles:
        return angles
    if len(angles) == num_roles * pair_count:
        return [
            angles[i * pair_count : (i + 1) * pair_count]
            for i in range(num_roles)
        ]
    raise ValueError(
        f"role_angles must have {num_roles} entries for shared role phases "
        f"or {num_roles * pair_count} entries for per-pair phases; "
        f"got {len(angles)}."
    )


# ---------- reproducibility ------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def snapshot_codebase(output_dir: Path) -> None:
    """Record git SHA + dirty diff into output_dir for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        diff = subprocess.check_output(["git", "diff"], text=True)
    except subprocess.CalledProcessError:
        sha = "unknown"
        dirty = ""
        diff = ""
    (output_dir / "git_sha.txt").write_text(sha + "\n")
    (output_dir / "git_status.txt").write_text(dirty)
    (output_dir / "git_diff.patch").write_text(diff)


# ---------- sanity-wrapped collator ----------------------------------------


class SanityWrappedCollator:
    """Wraps a ``RoleTaggingCollator``. On the first batch, asserts
    ``role_ids`` has more than one unique value (else rope-prov regresses to
    vanilla silently). Logs per-role token counts for the first ``log_first``
    batches so the W&B run can corroborate the parser is alive.
    """

    def __init__(self, base: RoleTaggingCollator, log_first: int = 3):
        self.base = base
        self.log_first = log_first
        self.calls = 0
        self.last_log: dict = {}

    def __call__(self, examples):
        batch = self.base(examples)
        self.calls += 1
        if self.calls <= self.log_first:
            role_ids = batch["role_ids"]
            unique = role_ids.unique().tolist()
            counts = {int(r): int((role_ids == r).sum()) for r in unique}
            mean = float(role_ids.float().mean())
            log = {
                "step": self.calls,
                "role_ids_mean": mean,
                "role_id_counts": counts,
                "role_ids_unique": unique,
            }
            self.last_log = log
            print(f"[role-sanity batch {self.calls}] {log}", flush=True)
            if self.calls == 1 and len(unique) <= 1:
                raise RuntimeError(
                    f"FIRST training batch role_ids has only one unique value "
                    f"{unique}. Aborting: rope-prov would silently train as "
                    f"vanilla. Inspect parser / role_map / collator."
                )
        return batch


class LearnableAnglesCallback(TrainerCallback):
    """Log shared learnable role-angle Parameter (θ_I, θ_D) to W&B + stderr
    every ``log_every`` steps. The angle trajectory IS the experimental
    signal for v2.5; eval loss is secondary."""

    def __init__(self, model_ref, log_every: int = 10):
        self.model_ref = model_ref
        self.log_every = log_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.log_every != 0:
            return
        param = getattr(self.model_ref, "role_angles_param", None)
        if param is None:
            return
        angles_t = param.detach().float().cpu()
        log = {}
        if angles_t.ndim == 1:
            angles = angles_t.tolist()
            log.update({f"angle/role_{i}": a for i, a in enumerate(angles)})
            if len(angles) >= 2:
                log["angle/gap_1_minus_0"] = angles[1] - angles[0]
        elif angles_t.ndim == 2:
            for role_idx in range(angles_t.shape[0]):
                for pair_idx in range(angles_t.shape[1]):
                    log[f"angle/role_{role_idx}_pair_{pair_idx}"] = float(
                        angles_t[role_idx, pair_idx]
                    )
            if angles_t.shape[0] >= 2:
                gap = angles_t[1] - angles_t[0]
                log["angle/gap_1_minus_0_mean"] = float(gap.mean())
                log["angle/gap_1_minus_0_max_abs"] = float(gap.abs().max())
                for pair_idx, value in enumerate(gap.tolist()):
                    log[f"angle/gap_1_minus_0_pair_{pair_idx}"] = value
        else:
            raise ValueError(
                f"role_angles_param must be rank 1 or 2, got {angles_t.ndim}"
            )
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(log, step=state.global_step)
        except Exception:
            pass
        print(f"[angles step={state.global_step}] {log}", flush=True)


class LearnableAngleFreezeCallback(TrainerCallback):
    """Hold learnable role angles fixed for the first N optimizer steps."""

    def __init__(self, model_ref, freeze_steps: int):
        self.model_ref = model_ref
        self.freeze_steps = freeze_steps
        self.last_trainable: bool | None = None

    def _set_trainable(self, step: int) -> None:
        param = getattr(self.model_ref, "role_angles_param", None)
        if param is None:
            return
        trainable = step >= self.freeze_steps
        if self.last_trainable is trainable:
            return
        param.requires_grad_(trainable)
        self.last_trainable = trainable
        state = "unfrozen" if trainable else "frozen"
        print(
            f"[angles-freeze step={step}] role_angles_param {state} "
            f"(freeze_steps={self.freeze_steps})",
            flush=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._set_trainable(state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        self._set_trainable(state.global_step)


class PreEvalCacheFlushCallback(TrainerCallback):
    """Flush the CUDA allocator cache right before an eval step. Cheap
    defense against fragmentation-induced OOM at eval time — the
    trainer-side allocations grow over many train steps; eval then adds
    activations whose required contiguous blocks may not fit even though
    total free memory is sufficient.

    HF's ``on_evaluate`` callback fires *after* eval, not before. We hook
    ``on_step_end`` and flush at the step preceding an eval boundary.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        if args.eval_steps and state.global_step % args.eval_steps == 0 and state.global_step > 0:
            torch.cuda.empty_cache()


class ThroughputCallback(TrainerCallback):
    """Logs examples/sec every ``log_every`` global steps. Stamps a final
    summary to stderr at train end."""

    def __init__(self, log_every: int = 50):
        self.log_every = log_every
        self.t0 = None
        self.examples_seen = 0
        self.last_log_step = 0
        self.last_t = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.last_t = self.t0
        self.examples_seen = 0
        self.last_log_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.examples_seen += eff_batch
        if state.global_step - self.last_log_step >= self.log_every:
            now = time.time()
            window_ex = (state.global_step - self.last_log_step) * eff_batch
            window_t = now - self.last_t
            rate = window_ex / window_t if window_t > 0 else float("nan")
            print(
                f"[throughput step={state.global_step}] "
                f"window_rate={rate:.2f} ex/s",
                flush=True,
            )
            self.last_log_step = state.global_step
            self.last_t = now

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.t0
        rate = self.examples_seen / elapsed if elapsed > 0 else float("nan")
        print(
            f"[throughput SUMMARY] total_ex={self.examples_seen} "
            f"elapsed={elapsed:.1f}s overall_rate={rate:.2f} ex/s",
            flush=True,
        )


# ---------- main -----------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a variant YAML.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Cap training at --max-steps and skip eval/checkpoints.",
    )
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--disable-wandb", action="store_true")
    ap.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Checkpoint path to resume from, or 'latest' to use the newest "
            "checkpoint under --output-dir. If 'latest' finds none, training "
            "starts from scratch."
        ),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    run_name = cfg["wandb_run_name"]
    out_dir = Path(
        args.output_dir or f"runs/{run_name}-seed{cfg['seed']}"
    ).resolve()
    snapshot_codebase(out_dir)

    # --- Tokenizer + role_map + collator ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    role_map = RoleMap.from_yaml(cfg["role_map"])
    # Only the rope_prov variant actually consumes role_angles; vanilla and
    # vanilla_zeroed carry placeholders that we don't want to police.
    if cfg.get("attention_variant", "rope_prov") in {"rope_prov", "rope_prov_pre_w"}:
        cfg["role_angles"] = normalize_role_angles(
            cfg["role_angles"],
            num_roles=len(role_map.roles),
            prov_dim=cfg["provenance_dims"],
        )
    base_collator = RoleTaggingCollator(
        tokenizer=tokenizer,
        role_map=role_map,
        max_length=cfg["seq_len"],
    )
    collator = SanityWrappedCollator(base_collator)

    # --- Dataset ---
    ds = load_dataset(cfg["dataset"], split=cfg["dataset_split"])
    examples = filter_alpaca_examples(ds)
    if cfg.get("train_size"):
        examples = examples[: cfg["train_size"]]
    rng = random.Random(cfg["seed"])
    rng.shuffle(examples)
    eval_n = cfg["eval_size"]
    train_examples = examples[eval_n:]
    eval_examples = examples[:eval_n]
    cf_seed = cfg.get("counterfactual_seed", cfg["seed"])
    cf_positive_fraction = cfg.get("counterfactual_positive_fraction", 0.5)
    cf_train_n = int(cfg.get("counterfactual_train_size") or 0)
    cf_eval_n = int(cfg.get("counterfactual_eval_size") or 0)
    if cf_train_n:
        train_examples.extend(
            build_counterfactual_examples(
                cf_train_n,
                seed=cf_seed,
                positive_fraction=cf_positive_fraction,
            )
        )
    if cf_eval_n:
        eval_examples.extend(
            build_counterfactual_examples(
                cf_eval_n,
                seed=cf_seed + 10_000,
                positive_fraction=cf_positive_fraction,
            )
        )
    print(
        f"[data] train={len(train_examples)} eval={len(eval_examples)} "
        f"(filtered from {len(ds)} raw; counterfactual train={cf_train_n} "
        f"eval={cf_eval_n})",
        flush=True,
    )

    # --- Model ---
    dtype = torch.bfloat16 if cfg["bf16"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype)
    variant = cfg.get("attention_variant", "rope_prov")
    if variant == "rope_prov":
        patch_model_with_role_aware_attention(
            model,
            prov_dim=cfg["provenance_dims"],
            role_angles=cfg["role_angles"],
            learnable_angles=cfg.get("learnable_angles", False),
        )
    elif variant == "rope_prov_pre_w":
        patch_model_with_pre_w_role_aware_attention(
            model,
            prov_dim=cfg["provenance_dims"],
            role_angles=cfg["role_angles"],
            learnable_angles=cfg.get("learnable_angles", False),
        )
    elif variant == "vanilla":
        # prov_dim=0 makes the subclass a no-op kwarg sink.
        patch_model_with_role_aware_attention(model, prov_dim=0)
    elif variant == "vanilla_zeroed":
        # Same positional capacity loss as rope_prov, no role signal.
        # Establishes the T2b ceiling.
        patch_model_with_zeroed_prov_pairs(
            model, prov_dim=cfg["provenance_dims"]
        )
    else:
        raise ValueError(f"unknown attention_variant: {variant!r}")

    # --- Trainer ---
    targs = TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg.get("eval_batch_size", cfg["batch_size"]),
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        num_train_epochs=cfg["epochs"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        prediction_loss_only=cfg.get("prediction_loss_only", False),
        logging_steps=cfg["logging_steps"],
        eval_strategy="no" if args.dry_run else "steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="no" if args.dry_run else "steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg.get("save_total_limit"),
        max_steps=args.max_steps if args.dry_run else int(cfg.get("max_steps") or -1),
        report_to=[] if args.disable_wandb or args.dry_run else ["wandb"],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    if not (args.disable_wandb or args.dry_run):
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])
        os.environ.setdefault("WANDB_ENTITY", cfg["wandb_entity"])
        os.environ.setdefault("WANDB_RUN_GROUP", cfg["wandb_group"])

    callbacks = [PreEvalCacheFlushCallback(), ThroughputCallback()]
    if cfg.get("learnable_angles", False):
        freeze_steps = int(cfg.get("angle_freeze_steps") or 0)
        if freeze_steps:
            callbacks.append(LearnableAngleFreezeCallback(model, freeze_steps))
        callbacks.append(LearnableAnglesCallback(model, log_every=10))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_examples,
        eval_dataset=eval_examples if not args.dry_run else None,
        data_collator=collator,
        callbacks=callbacks,
    )
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "latest":
        resume_from_checkpoint = get_last_checkpoint(str(out_dir))
        if resume_from_checkpoint:
            print(f"[resume] using latest checkpoint {resume_from_checkpoint}", flush=True)
        else:
            print(f"[resume] no checkpoint found under {out_dir}; starting fresh", flush=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if not args.dry_run:
        trainer.save_model(str(out_dir / "final"))


if __name__ == "__main__":
    main()
