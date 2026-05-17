# Research Brief

Last updated: 2026-05-17T06:12:00+02:00.

## One-Sentence Objective

Test whether an architectural, out-of-band provenance channel can produce
measurable instruction/data separation in a small RoPE-native model, starting
with binary prompt substring provenance and later extending toward source-span
attribution.

## Three Layers

Mechanical layer: carve a subspace out of RoPE's rotation dimensions and reuse
it to encode token role. In the current SmolLM2-135M implementation,
`provenance_dims=8` takes the four lowest-frequency RoPE pairs in each
attention head and replaces positional rotation on those pairs with a
role-specific rotation. Instruction role uses angle 0; DATA can use a fixed
angle such as pi/8 or pi/2, or a learned angle.

Conceptual layer: this is an architectural answer to Hines et al.'s
out-of-band channel challenge. It occupies the per-layer rotational cell in
the input/per-layer x additive/rotational design space:

| | Input-layer | Per-layer |
|---|---|---|
| Additive | ISE | AIR |
| Rotational | ASIDE | RoPE-Provenance |

Mechanistic layer: Ye et al.'s role-confusion result motivates the test. If
authority is assigned in latent space and style dominates role perception, an
architectural role signal has to compete with that latent style channel.

## Prompt DATA as Provenance

"This substring is DATA" is the simplest useful form of prompt substring
provenance. It is binary provenance: executable instruction channel versus
untrusted data channel. It is enough for prompt-injection separation
experiments.

It is not full provenance. Full substring provenance would preserve source
identity, such as document id, chunk id, tool id, retrieval rank, or span id,
and would let generated output cite or trace back to those spans. The current
channel encodes role, not identity.

## v1 Interpretation

The original Alpaca-trained v1 result is a diagnostic negative:

- vanilla and vanilla_zeroed are effectively tied on eval loss, so the
  low-frequency RoPE pairs are cheap in the 1024-token setup.
- fixed per-layer post-projection rotation, especially at pi/2, hurts utility.
- learnable angles under Alpaca stay near zero, indicating that ordinary SFT
  did not reward use of the role channel.
- SmolLM2-135M has negative SEP, so it is below the role-discrimination floor
  under the original training setup.

This supports a narrow claim: the tested post-projection rotational placement
is not useful under the original curriculum. It does not yet prove that all
architectural provenance channels are dead.

## v2 Course Correction

The sharp critique of v1 is that Alpaca SFT rarely teaches the model to ignore
directive-looking text inside DATA. The counterfactual v2 curriculum fixes the
training signal by pairing examples:

- DATA-negative: a directive-like witness appears in DATA, but the model must
  ignore it as a command.
- INSTRUCTION-positive: the same directive appears in INSTRUCTION, and the
  model must follow it.

The split is 50/50 positive and negative to avoid "always follow" and "always
ignore" policies. The v2 generator varies directive style, task type,
placement, distractor difficulty, and surface form to reduce synthetic template
artifacts.

## Hypotheses to Kill

H0: data alone solves role separation. If vanilla counterfactual training
improves SEP as much as rope_prov, the architectural channel adds little.

H1: capacity explains the effect. If vanilla_zeroed tracks rope_prov, then the
result is explained by the zeroed/repurposed dimensions, not by role rotation.

H2: fixed pi/8 role rotation is useful. If rope_prov pi/8 beats vanilla by the
pre-registered delta-of-deltas threshold while preserving utility, the fixed
architectural channel is doing work.

Status: killed for post-projection pi/8. Under v2, fixed pi/8 reached
SEP -0.275 with instruction execution 0.020 and DATA execution 0.295. Its
delta-of-deltas versus vanilla is -0.080, so aligned counterfactual training did
not make this fixed post-projection channel useful.

More importantly, the failure is asymmetric: INSTRUCTION-slot execution falls
from 0.155 to 0.020, while DATA-slot execution is essentially unchanged
(0.290 to 0.295). This is not just "rotation unused." It is a functional
dissociation: the fixed rotation damages the harness-tag-aware compliance
pathway learned by counterfactual training, while the style-driven DATA-slot
execution pathway remains intact. Treat this as behavioral evidence for Ye et
al.'s style-dominance thesis, not yet as a localized circuit proof.

H3: the learnable channel is accepted by the model. If the learned role-angle
gap moves above 5 degrees and SEP improves, the channel becomes useful under
aligned training. If the gap stays near zero with stable training, the sharper
diagnosis is that this post-projection rotational placement is structurally
inadequate even with the right curriculum.

Because fixed pi/8 selectively destroys instruction compliance, the learnable
prediction is overdetermined. Near-zero learned angles can now mean either
equal-frequency bandwidth is too narrow or the optimizer has learned to avoid
the asymmetric compliance damage. A useful follow-up is a nonzero-initialized
learnable arm with the role gap held fixed briefly and then unfrozen. If the
gap is driven back toward zero, the optimizer can see the damaging direction. If
it stalls or oscillates, the structural-inadequacy reading is cleaner.

## Symmetry-Survey Interpretation

The equal-frequency theorem sharpens the current fixed-angle design. Applying
the same role angle to every provenance pair puts those pairs in one degenerate
angle class: a closed complex-scalar fiber, not `P/2` independent role
channels. Increasing `P` buys more role-modulated subspace size and attention
energy, but the external binary role signal is still one phase.

This is enough bandwidth for binary instruction-vs-DATA provenance in
principle. It is probably too narrow for full source/span provenance unless a
future design adds independent angle sectors, source embeddings, additive id
channels, or another higher-bandwidth carrier.

The independent-angle rotational steelman is therefore still inside the paper's
cell: break the equal-frequency collapse without moving to an additive AIR
channel. The key question is whether distinct per-pair angles pay independent
compliance damage, causing super-linear failure, or whether the compliance cost
saturates after the first role-rotated subspace, leaving room for multiple
rotational carriers.

Transport geometry gives the better diagnostic language for v2. A usable
provenance direction has to pass three filters: the observable must resolve it,
the cost must fit the available slack, and the signal must survive stepwise
transport through the block stack. Counterfactual training is the observable
refinement; vanilla_zeroed is the budget/capacity control; the pre-W smoke is a
placement test for the transport filter.

The pre-W smoke splits the transport hypothesis into two pieces. It supports
the algebraic availability claim: moving the role rotation upstream of `Wq`/`Wk`
has lower initial utility cost than post-projection pi/8. It weakly undercuts
the findability claim: between steps 400 and 600, vanilla and zeroed drop about
0.31 eval-loss nats, while pre-W drops only 0.043. That is not the trajectory of
a projection quickly learning to absorb the role rotation into a near-commutant
subspace.

The current claim should therefore avoid saying post-projection rotation is
uncompensatable in an absolute sense. The cleaner claim is that it is a
low-slack transport path through a thin commutant-compatible region. If pre-W
improves utility or SEP with a matched full run, placement was the binding
constraint but needs more budget than the smoke. If pre-W also fails under the
counterfactual curriculum, the next kill target is rotational per-layer channels
as a class, with additive channels or RWKV/SSM gates as the natural pivot.

Sector-attractor language also cautions against over-reading one SEP scalar:
two runs can share the same macro SEP while using different microscopic
allocations. Keep reporting `exec_instr`, `exec_data`, utility loss, angle
movement, and seed variance before making a mechanism claim.

The ReZero-gated pre-W smokes test a narrower optimizer hypothesis: maybe the
full-feature fine-tunes failed because the role rotation was introduced too
abruptly, so early layers never got a low-loss route into the channel. Staged
gates did open under both shared pi/8 and independent fixed-angle targets, but
SEP stayed negative and below the vanilla/pre-W band. This weakens the
"schedule/LR alone can rescue rotational provenance under SFT" interpretation,
while leaving higher-budget continued-pretraining as a separate, more expensive
future test.

## Decision Rule

Primary metric:

```text
[SEP(rope_prov_cf) - SEP(rope_prov_alpaca)]
-
[SEP(vanilla_cf) - SEP(vanilla_alpaca)]
```

Call this positive only if the delta-of-deltas is at least +0.05 absolute SEP
points on the same SEP subset. Treat +0.02 to +0.05 as marginal. Treat
anything within +/-0.02 as no architectural SEP gain unless supported by clear
learned-angle movement.

Before making a positive claim, re-run the best-performing rope_prov
counterfactual arm with a second training seed. If it drifts more than +/-0.02
SEP, raise the threshold to at least +0.07 or run more seeds. Also estimate SEP
measurement variance by using different SEP subsets or a future sampled
decoding mode.

## Formal Claim Discipline

The old broad theorem claim about general non-interference was false for a
transformer stack: DATA tokens can affect later hidden states. The current
formal targets are narrower:

- parser non-escalation: tag-like substrings inside an already-open DATA span
  remain DATA content.
- causal upstream non-interference: later DATA cannot rewrite earlier trusted
  prefix states under a causal mask.
- local attention sanity: if Q/K vectors for instruction tokens are fixed,
  changing DATA token contents cannot change instruction-to-instruction logits.

These are architecture/parser properties, not trained-model robustness claims.

## Longer Arc

If binary role provenance works, extend from role bit to source id. The natural
next target is a channel that carries document/chunk/span ids plus an auxiliary
loss tying generated statements to supporting source spans. If RoPE is the
wrong substrate, RWKV/SSM models are a plausible next architecture family:
inject provenance through state-update gates or additive state channels, gated
on whether the unmodified model has a usable SEP floor.

## Tiny-Model Sanity Track

The tiny role-provenance script is not a replacement for the SmolLM2
rotational matrix. It is a low-effort sanity track for the underlying objective:
can a small decoder learn to route identical directive-like substrings
differently when role/provenance is supplied out of band?

The current additive toy trains a fresh char-level decoder from scratch with
learned absolute position embeddings plus additive role embeddings. Hidden-tag
evals strip visible `<instr>`/`<data>` markers from text and pass role ids
separately. This makes the run a positive control for out-of-band role
conditioning, not evidence that a pretrained SLM can cheaply adopt the channel.
The current configuration has 731,656 trainable parameters.

The early hidden-tag result is suspiciously strong: near-perfect SEP appears
within a few hundred steps. Treat that as a confound warning. The synthetic
grammar may be too regular, the train/eval split is not yet a hard distribution
shift, and the model may learn field/template structure plus role ids rather
than general substring provenance. The next toy decisions must therefore come
from controls, not from the headline score.

Required controls:

- hidden tags with correct roles: positive-control target.
- hidden tags with constant roles: tests whether prompt grammar alone solves
  the task.
- hidden tags with instruction/DATA roles swapped: tests whether the out-of-band
  role channel is causal and direction-sensitive.
- no role embeddings: tests whether any remaining visible or lexical structure
  is enough.
- hard heldout split: disjoint directive surfaces, witnesses, answers, field
  order, and prompt templates.

The simple-template constant-role control is expected to overstate performance
because the model can learn two-point correlations between local substrings and
labels. The diverse-template generator is the minimum fix: vary instruction
surface forms, field labels, embedded directive prefixes, field order, and
train/eval template families so the stable feature is role/boundary assignment
rather than substring placement.

This still does not remove all shortcuts. Any finite natural-language template
set exposes non-role signals such as quotes, label names, line order, and
punctuation. The stricter toy target is an ambiguous-pair generator: keep the
visible text distribution identical or near-identical across paired examples
and change only the hidden role assignment. In that setup, a constant-role
control should be unable to choose the right source systematically, while a
correct-role model can use the out-of-band signal.

### Toy Pre-Registration

Written 2026-05-17 during the first diverse-template correct-role run, after
observing simple-template correct and constant results and after observing
diverse correct through step 1500. It is not a pre-registration for those
already-seen points; it is a decision rule for the remaining toy controls and
next generators.

Predictions:

- Simple-template constant roles will finish high (already observed), so the
  simple generator is not evidence of robust hidden-role provenance.
- Diverse-template correct roles should outperform diverse-template constant
  roles in sample efficiency and final SEP. A meaningful gap is at least +0.20
  SEP at step 500 or at least +0.10 final SEP at step 2000.
- If diverse constant roles reaches within 0.05 final SEP of diverse correct
  roles, template shortcuts are still too strong for this generator.
- Directionality should be tested by training with correct roles and evaluating
  with swapped roles. A train-and-eval-swapped run is only a reversed-codebook
  test and can succeed even when role ids are causal.
- The ambiguous-pair generator is the first toy design that can support a
  stronger scientific claim. Expected outcome: correct roles should exceed
  constant roles by at least +0.20 SEP, because text-only cues are matched by
  construction.
- Gated multi-return examples should be harder than diverse templates. The
  expected positive result is not perfect SEP, but a clear correct-vs-constant
  gap showing that role provenance composes with a learned linguistic gate.

Interpretation:

- Positive toy result: correct roles beat constant/no-role/swap controls on
  ambiguous or gated paired data. This supports out-of-band role provenance as
  a useful inductive bias in a tiny Transformer.
- Weak toy result: correct roles beat controls only on simple/diverse finite
  templates. This is a blog-post or intuition result, not a scientific claim.
- Negative toy result: controls match correct roles even on ambiguous/gated
  data. The generator still leaks the answer or the toy metric is not isolating
  provenance.

RoPE-specific toy bridge:

- RoPE positions plus additive role embeddings: checks whether the positive
  toy survives when the tiny model is RoPE-native.
- RoPE-style role rotation from scratch: checks whether rotational provenance
  fails only as a pretrained-model adaptation problem or also fails when present
  from initialization.

If RoPE-role rotation works from scratch but fails under SmolLM2 SFT/CPT, the
binding constraint is adaptation path. If it fails from scratch too, the
rotational primitive itself is weak even before scale and pretraining effects.

### Pretrained SLM Pivot

The gated toy exposed a capability floor. After fixing the silent truncation
bug with `block_size=512`, the scratch 731,656-param char model still fit the
training loss while failing heldout exact-match gate selection. This does not
show that hidden provenance is wrong; it shows that semantic and compositional
selection gates are too hard for this scratch substrate.

The next low-effort SLM experiment should start from `Qwen/Qwen2.5-0.5B` base.
This keeps the scientific target cleaner than an instruct checkpoint: the model
has language and semantic priors for colors, objects, negation, questions, and
copying, but it has fewer chat-role and instruction-hierarchy confounds. The
Qwen instruct checkpoint is still useful as a sanity probe: if instruct works
and base does not, the task is reachable but depends on post-training priors.

`HuggingFaceTB/SmolLM2-360M` base/instruct are cached as a continuity check
against the earlier SmolLM2 family. Phi mini instruct checkpoints are cached as
stronger capability probes only; they do not form the same clean base/instruct
pair, and their instruction tuning makes them less diagnostic for the paper's
main causal claim.

Pre-registered SLM queue, written before any Qwen run:

- First run the scratch syntactic-gate prerequisite with only `no_not` and
  `question` gates plus regularization. If it reaches heldout exact-match SEP
  >= 0.50, the scratch gated-role branch is unblocked. If it stays below 0.10,
  kill the scratch semantic/gated branch for the paper and use it only as a
  blog-style intuition track.
- Then run `Qwen/Qwen2.5-0.5B` base on the gate-pretrain/capability task before
  adding hidden roles. Passing threshold: heldout exact-match >= 0.50 after a
  short adapter/full smoke, with coherent copied outputs. If base cannot learn
  the gate, do not spend GPU on hidden role channels yet.
- If Qwen base passes, run the hidden-role additive channel with correct,
  constant, and eval-swapped controls. Positive threshold: correct roles exceed
  constant roles by >= +0.20 SEP and eval-swapped collapses toward zero or
  reverses.
- Qwen instruct is a confounded probe. Use it only if base fails or is marginal:
  a pass shows the task is reachable by an SLM, but not that the provenance
  channel caused the behavior.
