# Research Revision Notes

## Prompt Data as Substring Provenance

Yes: "this substring is DATA" is the simplest useful form of prompt-substring
provenance. It is binary provenance: trusted instruction channel vs untrusted
data channel. It is enough for prompt-injection separation experiments.

It is not full provenance yet. Full substring provenance would preserve finer
source identity such as source document, chunk id, tool id, retrieval rank, or
span id, and would let generated output cite or trace back to those spans. The
current RoPE-Provenance channel encodes role, not identity.

## Counterfactual Training Strategy

Plain Alpaca SFT is misaligned with the defense objective: it teaches the model
to use DATA content to answer, but rarely teaches that directive-looking DATA is
non-executable. The fix is a matched counterfactual curriculum:

- DATA-negative examples: the string "Return exactly X" appears in DATA; the
  model must ignore it and answer the original extraction task.
- INSTRUCTION-positive examples: the same string appears as INSTRUCTION; the
  model must follow it and return X.
- Benign utility examples: keep ordinary Alpaca examples so the model still
  learns to use data as evidence.

Initial mix for SmolLM2-135M:

- 70-80% ordinary Alpaca examples with non-empty input.
- 10-20% DATA-negative counterfactuals.
- 10-20% INSTRUCTION-positive matched controls.
- Use 10k-20k synthetic counterfactual examples before scaling up. If SEP does
  not move, increase style diversity before increasing count.

Diversity required:

- Directive styles: system override, developer message, JSON/XML/Markdown,
  polite request, adversarial "ignore previous".
- Placements: beginning, middle, end of DATA; before and after the relevant
  evidence.
- Task types: extraction, summarization, classification, transformation, QA.
- Surface forms: natural prose, structured records, email/tool/RAG snippets,
  escaped or tag-like markup.
- Languages can wait until the English binary result moves.

Implemented hook: `build_counterfactual_examples` in `src/rope_prov/data.py`.
Use the `*_counterfactual.yaml` configs to train matched baselines on the same
augmented data.

## Deriving the Cost Law

For a provenance pair, let role rotations be `R(a)` and `R(b)`. The cross-role
provenance logit contribution is

```text
(R(a) q)^T (R(b) k) = q^T R(b-a) k
                   = cos(delta) q^T k + sin(delta) q^T J k
```

where `delta = b-a` and `J` is the 90-degree symplectic rotation. If the
cross term `q^T J k` averages to approximately zero across examples/heads, the
mean content-match term is scaled by `cos(delta)`. The lost content-match mass
is therefore proportional to `1 - cos(delta)`.

Lean can prove the exact logit identity and the cancellation/invariance
theorems. It cannot prove the empirical loss slope without assumptions about
the learned Q/K distribution, attention mass, and cross-entropy sensitivity.
That part is a physics-style approximation:

```text
Delta loss ~= alpha * m_cross * E_prov * (1 - cos(delta))
```

where `m_cross` is cross-role attention mass and `E_prov` is useful content
energy in the provenance subspace.

## Long-Context Scaling Without Huge Contexts

The current "low-frequency pairs are cheap" result was measured at 1024 tokens.
That is not enough. A GPU-feasible scaling plan:

1. Train/evaluate at increasing `seq_len`: 512, 768, 1024, 1536, 2048 if memory
   allows.
2. Keep the task constant: labelled-key extraction with filler before/between
   evidence and directive-like distractors.
3. Measure vanilla vs vanilla_zeroed vs rope_prov at each length.
4. Plot delta eval loss and SEP-like accuracy against maximum phase movement
   of the zeroed low-frequency pairs, not just raw context length.

This tests whether the near-zero T2b cost is a short-context artifact.

## RWKV7 / SSM Gate

RWKV7 is attractive as a scale path because recurrent inference has constant
memory/time per token and released models include small sizes. But RWKV/SSM
models do not use RoPE, so this is not a RoPE-extension test. It is a broader
"out-of-band provenance channel" test.

Use high SEP as a gate:

1. Evaluate unmodified RWKV7/SSM candidates on SEP.
2. Only continue if vanilla SEP is meaningfully positive; otherwise the model
   is below the role-discrimination floor and cannot test selectivity.
3. Inject role/provenance through state update gates or additive state channels,
   not RoPE.
4. Compare against transformer baselines on the same counterfactual curriculum.

## Source-Guided Techniques for the Objective

The sources suggest three stronger interpretations of the objective:

- Instruction/data separation: follow StruQ/ISE/ASIDE/AIR and train with
  matched counterfactuals. This is the current repo's natural scope.
- Source attribution: follow AIS/SelfCite/Cite Pretrain style objectives and
  require generated answers to emit span/source ids. A role bit is not enough.
- Architectural provenance: combine the two by giving every source span a
  channel id and adding an auxiliary loss that predicts which source ids support
  each output sentence.

The most direct next experiment is not larger RoPE-Provenance. It is:

1. Counterfactual SFT on vanilla, vanilla_zeroed, and rope_prov_pi8.
2. SEP gate after training.
3. If SEP moves, add explicit citation/span-id targets.
4. If SEP does not move, try additive per-layer AIR-style channels before
   another rotational placement.

## Pre-Registration for Counterfactual v2

Timestamp: 2026-05-14T18:47:30+02:00. This note was written after seeing only
early plumbing/sanity evidence from the online `vanilla_counterfactual_v2` run:
role ids are present, step-200 eval ran, and training loss was moving normally.
Final v2 vanilla results, the other three v2 arms, SEP results, and learned
angle values were not yet known.

Primary decision metric:

```text
[SEP(rope_prov_cf) - SEP(rope_prov_alpaca)]
-
[SEP(vanilla_cf) - SEP(vanilla_alpaca)]
```

Call this positive only if the delta-of-deltas is at least +0.05 absolute SEP
points on the same SEP subset. Treat +0.02 to +0.05 as weak/marginal. Treat
anything within +/-0.02 as no architectural SEP gain unless supported by a
clear learned-angle movement.

Variance calibration before making a positive claim:

- At minimum, re-run the best-performing rope_prov counterfactual arm with a
  different training seed before calling a +0.05 delta-of-deltas real. If the
  re-run lands within +/-0.02 SEP of the original, keep +0.05 as the positive
  threshold. If it drifts more, raise the threshold to at least +0.07 or run
  more seeds.
- Separately estimate SEP measurement variance on an already-trained model.
  The current SEP evaluator uses greedy deterministic decoding, so different
  decoding seeds should be identical unless stochastic decoding is added. Use
  different SEP subsets or a future sampled-decoding mode to measure detector
  noise. If this measurement component alone exceeds +/-0.02 SEP, do not treat
  a +0.05 single-seed delta-of-deltas as strong evidence.

Expected outcomes before seeing results:

- `vanilla_counterfactual_v2`: should train stably and finish with mixed
  eval loss below 1.5. Because the synthetic eval split may be easier than
  Alpaca, lower-than-Alpaca loss is not by itself evidence of success. Expected
  SEP: improvement over the old vanilla baseline of -0.22, probably into the
  -0.12 to +0.02 range if the counterfactual curriculum transfers at all.
- `vanilla_zeroed_counterfactual_v2`: should be close to vanilla on eval loss
  and SEP. If it beats rope_prov, capacity/optimization effects explain more
  than the role rotation.
- `rope_prov_pi8_counterfactual_v2`: should preserve utility if pi/8 is small
  enough. Expected SEP is close to vanilla, with a plausible small gain. A
  large utility hit would revive the cost-law concern even under aligned data.
- `rope_prov_learnable_counterfactual_v2`: most likely learned role-angle gap
  is small, around 1-3 degrees. A gap above 5 degrees plus SEP improvement is
  strong evidence that the channel becomes useful when the loss rewards it. A
  gap near zero with stable training supports the sharper claim that this
  post-projection rotational placement is structurally inadequate.

Abort or revise the v2 curriculum before spending more GPU if vanilla shows
unstable training, final eval loss above 2.0, or no SEP movement relative to
the old vanilla baseline. In that case, the counterfactual data did not teach
role-conditioning even to the architecture-free baseline, so a rope_prov
failure would not isolate the architectural question.
