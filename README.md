# RoPE-Provenance

RoPE-Provenance tests whether an architectural, out-of-band provenance channel
can make a small RoPE-native language model distinguish executable instruction
text from inert data text.

The prototype carves out low-frequency RoPE pairs in SmolLM2-135M and uses
role-specific rotations to encode token role. The current research question is
whether the v1 negative result was caused by the post-projection rotational
placement itself, or by an Alpaca SFT curriculum that never rewarded the model
for treating directive-looking DATA as non-executable.

Start with [INDEX.md](INDEX.md). It points to the maintained research brief,
experiment tracker, results ledger, and literature synthesis.

## Quick Commands

```bash
# tests
uv run pytest -q

# train or resume the active counterfactual v2 vanilla arm
uv run env WANDB_MODE=online python -m rope_prov.train \
  --config src/rope_prov/configs/vanilla_counterfactual_v2.yaml \
  --output-dir runs/vanilla_counterfactual_v2_online-seed0 \
  --resume-from-checkpoint latest

# SEP evaluation example
uv run python -m rope_prov.eval_sep \
  --model-path runs/vanilla_counterfactual_v2_online-seed0/final \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --variant vanilla \
  --sep-json /tmp/sep_repo/SEP_dataset/SEP_dataset.json \
  --output results/sep/vanilla_counterfactual_v2.json
```

W&B project: `d3banjan/rope-provenance`.

