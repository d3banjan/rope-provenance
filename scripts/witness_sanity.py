"""Witness-substring detector noise-floor sanity check.

Sample N SEP examples from vanilla checkpoint, dump
``(witness, output_instr, output_data, auto_match_instr, auto_match_data)``
tuples so a human can verify whether the automated substring match agrees
with a manual eyeball read.

Run on CPU to leave the GPU free for ongoing training. CPU inference on
SmolLM2-135M with 64-token outputs takes ~10-30s per generation; total
~5-10 min for 10 examples.

Output: ``results/sep/witness_sanity.json`` (machine-readable) and a
markdown table dumped to stdout for manual review.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# CPU-only — sweep needs the GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rope_prov.eval_sep import _render, generate_with_roles, load_sep_examples, _executed
from rope_prov.model import patch_model_with_role_aware_attention
from rope_prov.parser import RoleMap


def main():
    N = 10
    SEED = 0
    MAX_NEW = 64
    MODEL_PATH = "runs/vanilla-seed0/final"
    TOKENIZER = "HuggingFaceTB/SmolLM2-135M"

    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    role_map = RoleMap.from_yaml("src/rope_prov/configs/role_map.yaml")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    patch_model_with_role_aware_attention(model, prov_dim=0)
    model.eval()

    all_examples = load_sep_examples(limit=None)
    sample = random.sample(all_examples, N)

    records = []
    for i, ex in enumerate(sample):
        print(f"[{i+1}/{N}] witness={ex.witness!r}", flush=True)
        prompt_a = _render(ex.system_prompt_instructed, ex.prompt_clean)
        prompt_b = _render(ex.system_prompt_clean, ex.prompt_instructed)
        out_a = generate_with_roles(model, tokenizer, prompt_a, role_map, max_new_tokens=MAX_NEW)
        out_b = generate_with_roles(model, tokenizer, prompt_b, role_map, max_new_tokens=MAX_NEW)
        records.append({
            "idx": i,
            "witness": ex.witness,
            "output_probe_in_instr": out_a,
            "output_probe_in_data": out_b,
            "auto_match_instr": _executed(out_a, ex.witness),
            "auto_match_data": _executed(out_b, ex.witness),
        })

    Path("results/sep").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/sep/witness_sanity.json")
    out_path.write_text(json.dumps(records, indent=2))

    print("\n# Witness-substring sanity check (n=10, vanilla)\n")
    print("| # | witness | auto_instr | auto_data | output_instr (truncated) | output_data (truncated) |")
    print("|---|---|---|---|---|---|")
    for r in records:
        oi = r["output_probe_in_instr"].replace("|", "\\|").replace("\n", " ")[:80]
        od = r["output_probe_in_data"].replace("|", "\\|").replace("\n", " ")[:80]
        print(f"| {r['idx']} | `{r['witness']}` | {r['auto_match_instr']} | {r['auto_match_data']} | {oi} | {od} |")

    print(f"\n[wrote {out_path}]")
    print("Manual review: open the JSON, eyeball each output, "
          "judge whether the model 'executed' the probe (produced the witness "
          "effect) regardless of literal substring match. "
          "Tally agreement with auto_match_*. Target ≥9/10 agreement.")


if __name__ == "__main__":
    main()
