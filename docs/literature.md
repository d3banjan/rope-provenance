# Literature Synthesis

This is the curated related-work file. It preserves the decisions from the
earlier survey without keeping raw search notes as active docs.

## Load-Bearing Sources

Wallace et al. 2024, Instruction Hierarchy: establishes the core training
framing for privileged instructions. Modern chat models can be trained to
prefer higher-priority sources, but this is primarily coarse message/source
priority rather than arbitrary substring-level provenance.

Hines et al. 2024, Spotlighting: motivates the out-of-band channel. The key
idea is that control text and data text should be supplied to the model through
separate channels rather than through only in-band prompt strings. This project
is an architectural attempt at that premise.

Chen et al. 2024/2025, StruQ: the closest training-data predecessor for the v2
counterfactual curriculum. StruQ separates prompt and data portions of a query
and fine-tunes models to follow instructions only from the prompt portion while
ignoring instruction-like text inside the data portion.

Yi et al. 2023/2024, BIPIA: supplies the indirect-prompt-injection benchmark
setting and shows that external content can carry malicious instructions that
models execute unless the source boundary is represented and trained.

Ye, Cui, and Hadfield-Menell 2026, Role Confusion: supplies the mechanistic
motivation. Role and authority are assigned in latent space, where style can
dominate formal interface labels. This explains why weak architectural tags may
not control behavior.

Kariyappa and Suh 2025, AIR: the direct per-layer additive foil. AIR injects
instruction-hierarchy information into intermediate representations and reports
large attack-success reductions. RoPE-Provenance tests the per-layer rotational
analog and diagnoses why this placement is harder.

ASIDE, Zverev et al. 2026: input-layer rotational instruction/data separation.
This is the closest rotational lineage.

ISE, Wu et al. 2024: input-layer additive segment-style instruction hierarchy
signal.

Chiang and Yogatama 2025: RoPE dimension inefficiency. Their result concerns
the high-frequency end for long-distance retrieval. Our short-context T2b
finding concerns low-frequency pairs, so it is a complement, not a restatement.

Qwen2.5 technical report and model cards: Qwen2.5 supplies a clean small-model
bridge because the 0.5B release has both base and instruct checkpoints,
transformer/RoPE architecture, 0.49B parameters, and enough pretraining to make
semantic gates plausible. Use the base model for causal provenance experiments
and the instruct model only as a capability probe.

LoRA, Hu et al. 2021: freezes pretrained weights and inserts low-rank trainable
updates. This is the right first adaptation method for Qwen-scale tests because
it makes early kill-tests cheap and keeps the base model mostly intact.

QLoRA, Dettmers et al. 2023: supports the same adapter-first strategy under
4-bit quantization. Useful if 0.5B full/LoRA bf16 runs are too memory-heavy, but
not required before the initial Qwen 0.5B tests.

DoRA, Liu et al. 2024: improves LoRA by separating weight magnitude and
direction. Keep as a later stabilization option if LoRA underfits while full
fine-tuning is too expensive; do not add it to the first-pass harness.

## Design Space

| | Input-layer | Per-layer |
|---|---|---|
| Additive | ISE | AIR |
| Rotational | ASIDE | RoPE-Provenance |

This table is the cleanest positioning for the paper. The contribution is not
"first architectural defense." It is narrower: the first per-layer rotational
attempt in this design space, with a diagnostic negative for the tested
post-projection placement.

## Source-Guided Alternatives

Instruction/data separation: follow StruQ, ISE, ASIDE, and AIR with matched
counterfactuals. This is the current repo's natural scope.

Source attribution: follow AIS, SelfCite, and citation-pretraining style
objectives. A role bit is insufficient; generated answers need span/source ids.

Architectural provenance: combine the two by giving each source span a channel
id and adding an auxiliary loss that predicts which source ids support each
output sentence.

RWKV/SSM channel: if RoPE is the wrong substrate, inject role/provenance through
state-update gates or additive state channels. Treat high SEP on an unmodified
candidate as the gate before spending architecture work.

## Framing Guidance

Do not overclaim "first." Prior work already trains or prompts role-aware
instruction/data separation. The precise claim is "first per-layer rotational"
within the instruction/data separation design space.

Do not frame additive role embeddings as a new conceptual defense by
themselves. They are a sanity bridge and an input/additive foil: if a clean
hidden role channel works in a tiny from-scratch model, the objective is
learnable; if the per-layer RoPE variant still fails, the failure is about the
rotational carrier or adaptation path, not about the objective being impossible.

Do not treat AIR as incidental. It is the additive analog that makes the
rotational failure interpretable.

Do not collapse the v1 result into "channel dead." After the counterfactual
critique, the sharper live question is whether the negative was
curriculum-bound or placement-bound.

Do not use SEP alone below the capability floor. Negative vanilla SEP means the
base model may not have a role-discrimination substrate for an intervention to
selectively modulate.
