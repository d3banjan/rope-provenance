"""Aggregate the three training-arm logs into a single markdown table.

Usage:

    uv run python scripts/summarize_runs.py \
        --vanilla /tmp/vanilla_full.log \
        --vanilla-zeroed /tmp/vanilla_zeroed_full.log \
        --rope-prov /tmp/rope_prov_full.log \
        --output results/three_arm_summary.generated.md

Parses HF Trainer's stdout (loss/grad_norm/eval_loss dicts emitted as ``{...}``
lines) plus our own ``[throughput SUMMARY]`` footer. Output is a markdown
file with one row per arm summarizing the training trajectory.

Robust to interrupted runs: if a log has no SUMMARY line, the script still
prints what it could parse and flags the row as ``[incomplete]``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path


# Lines emitted by HF Trainer look like:
#   {'loss': 1.8337, 'grad_norm': 2.203125, 'learning_rate': 7.4e-06, 'epoch': 0.03}
# Single quotes, Python repr — not JSON. Parse with literal_eval.
_TRAINER_DICT_RE = re.compile(r"^\{'(loss|eval_loss|train_runtime)['].*\}$")
_THROUGHPUT_SUMMARY_RE = re.compile(
    r"\[throughput SUMMARY\] total_ex=(\d+) elapsed=([\d.]+)s overall_rate=([\d.]+) ex/s"
)


@dataclass
class ArmSummary:
    name: str
    log_path: str
    train_losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    final_train_loss: float | None = None
    final_eval_loss: float | None = None
    train_runtime_s: float | None = None
    overall_rate_ex_s: float | None = None
    incomplete: bool = True


def _parse_pydict(line: str) -> dict | None:
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        import ast
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        return None


def parse_log(name: str, path: str) -> ArmSummary:
    out = ArmSummary(name=name, log_path=path)
    try:
        text = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return out

    for raw_line in text.splitlines():
        # Strip carriage-return tqdm prefix garbage if any.
        for chunk in raw_line.split("\r"):
            d = _parse_pydict(chunk)
            if d is None:
                # Also check for throughput summary
                m = _THROUGHPUT_SUMMARY_RE.search(chunk)
                if m:
                    out.overall_rate_ex_s = float(m.group(3))
                continue
            if "loss" in d and "epoch" in d and "eval_loss" not in d:
                out.train_losses.append(float(d["loss"]))
                if "grad_norm" in d:
                    out.grad_norms.append(float(d["grad_norm"]))
            if "eval_loss" in d:
                out.eval_losses.append(float(d["eval_loss"]))
            if "train_runtime" in d:
                out.final_train_loss = float(d.get("train_loss", out.final_train_loss or 0))
                out.train_runtime_s = float(d["train_runtime"])
                out.incomplete = False

    if out.eval_losses:
        out.final_eval_loss = out.eval_losses[-1]
    return out


def _range_str(xs: list[float]) -> str:
    if not xs:
        return "—"
    return f"{min(xs):.2f}–{max(xs):.2f}"


def _percentile_band(xs: list[float], lo: float, hi: float) -> str:
    if not xs:
        return "—"
    xs = sorted(xs)
    p_lo = xs[int(len(xs) * lo)]
    p_hi = xs[int(len(xs) * hi)] if int(len(xs) * hi) < len(xs) else xs[-1]
    return f"{p_lo:.2f}–{p_hi:.2f}"


def to_markdown(arms: list[ArmSummary]) -> str:
    lines = []
    lines.append("# Three-Arm Training Summary\n")
    lines.append("| Arm | Status | Final Train Loss | Final Eval Loss | Train-Loss Last 10% | Grad-Norm Early (p0-p20) | Grad-Norm Late (p80-p100) | Runtime | Throughput |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        status = "[incomplete]" if a.incomplete else "ok"
        last_decile = a.train_losses[-max(1, len(a.train_losses) // 10):] if a.train_losses else []
        late_band = f"{statistics.mean(last_decile):.3f}" if last_decile else "—"
        early_norms = a.grad_norms[: max(1, len(a.grad_norms) // 5)] if a.grad_norms else []
        late_norms = a.grad_norms[-max(1, len(a.grad_norms) // 5):] if a.grad_norms else []
        early_g = _range_str(early_norms)
        late_g = _range_str(late_norms)
        runtime = f"{a.train_runtime_s / 60:.1f} min" if a.train_runtime_s else "—"
        rate = f"{a.overall_rate_ex_s:.1f} ex/s" if a.overall_rate_ex_s else "—"
        ft = f"{a.final_train_loss:.3f}" if a.final_train_loss is not None else "—"
        fe = f"{a.final_eval_loss:.3f}" if a.final_eval_loss is not None else "—"
        lines.append(
            f"| {a.name} | {status} | {ft} | {fe} | {late_band} | {early_g} | {late_g} | {runtime} | {rate} |"
        )

    lines.append("\n## Interpretation guide\n")
    lines.append(
        "- **Compare `rope_prov` final eval loss vs `vanilla_zeroed`**, not vs `vanilla`. "
        "`vanilla_zeroed` is the architectural ceiling (T2b). If `rope_prov` ≤ `vanilla_zeroed`, "
        "the role signal is doing real work."
    )
    lines.append(
        "- **`vanilla_zeroed` ≈ `vanilla`** ⇒ the zeroed low-frequency RoPE pairs are cheap at this sequence length; "
        "rope_prov has no intrinsic capacity penalty. **`vanilla_zeroed` ≫ `vanilla`** ⇒ those pairs "
        "were structural; ceiling is constrained."
    )
    lines.append(
        "- **Late grad-norm** drifting toward 1-3 ⇒ training stabilized. Stuck at 10+ ⇒ unresolved "
        "instability; bump `max_grad_norm`."
    )

    # JSON dump for downstream automation.
    lines.append("\n## Raw\n")
    lines.append("```json")
    lines.append(json.dumps(
        [{
            "name": a.name,
            "log_path": a.log_path,
            "incomplete": a.incomplete,
            "final_train_loss": a.final_train_loss,
            "final_eval_loss": a.final_eval_loss,
            "eval_losses": a.eval_losses,
            "train_runtime_s": a.train_runtime_s,
            "overall_rate_ex_s": a.overall_rate_ex_s,
            "n_train_loss_points": len(a.train_losses),
            "n_grad_norm_points": len(a.grad_norms),
        } for a in arms],
        indent=2,
    ))
    lines.append("```")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", required=True)
    ap.add_argument("--vanilla-zeroed", required=True)
    ap.add_argument("--rope-prov", required=True)
    ap.add_argument("--output", default="results/three_arm_summary.generated.md")
    args = ap.parse_args()

    arms = [
        parse_log("vanilla", args.vanilla),
        parse_log("vanilla_zeroed", args.vanilla_zeroed),
        parse_log("rope_prov", args.rope_prov),
    ]
    md = to_markdown(arms)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md)
    print(md)
    print(f"\n[written to {args.output}]")


if __name__ == "__main__":
    main()
