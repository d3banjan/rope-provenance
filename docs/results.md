# Results Ledger

This file records stable results. In-flight progress belongs in
[experiments.md](experiments.md).

## v1 Alpaca Training

SmolLM2-135M, Alpaca-cleaned, seed 0, 3 epochs, 1024 tokens.

| Arm | Final train loss | Final eval loss | Runtime | Throughput |
|---|---:|---:|---:|---:|
| vanilla | 1.673 | 1.629 | 29.9 min | 31.7 ex/s |
| vanilla_zeroed | 1.673 | 1.628 | 30.8 min | 30.8 ex/s |
| rope_prov pi/2 | 3.290 | 3.166 | 34.0 min | 27.9 ex/s |

Interpretation: `vanilla_zeroed` is effectively tied with vanilla, so the
zeroed low-frequency pairs are cheap in this short-context setup. The pi/2
post-projection rotation is highly disruptive.

## Angle Cost Law

Reference vanilla eval loss: 1.629.

| Angle | cos(theta) | 1 - cos(theta) | Delta eval loss | Slope |
|---|---:|---:|---:|---:|
| vanilla_zeroed | 1.000 | 0.000 | -0.001 | n/a |
| pi/8 | 0.924 | 0.076 | +0.103 | 1.36 |
| pi/6 | 0.866 | 0.134 | +0.168 | 1.25 |
| pi/4 | 0.707 | 0.293 | +0.372 | 1.27 |
| pi/2 | 0.000 | 1.000 | +1.537 | 1.54 |

Empirical rule for this tested family:

```text
Delta loss ~= alpha * cross-role-attention-mass * provenance-energy * (1 - cos(delta))
```

The exact logit identity is mathematical; the loss slope is empirical and
depends on learned Q/K distributions and attention mass.

## SEP Baselines

SEP subset size: 200.

| Variant | exec_instr | exec_data | SEP |
|---|---:|---:|---:|
| vanilla | 0.090 | 0.310 | -0.220 |
| vanilla_zeroed | 0.105 | 0.300 | -0.195 |
| rope_prov pi/8 | 0.005 | 0.285 | -0.280 |
| rope_prov learnable | 0.105 | 0.305 | -0.200 |

Interpretation: SmolLM2-135M under the original Alpaca setup is below the SEP
role-discrimination floor. The model executes DATA-slot probes more often than
instruction-slot probes. At pi/8, rope_prov reduces responsiveness in both
slots rather than improving selectivity.

## Learnable-Angle Diagnostic Under Alpaca

The learnable-angle arm initialized both role angles at zero. Under Alpaca SFT,
the learned role-angle gap plateaued near 0.32 degrees. That is much smaller
than pi/8 (22.5 degrees), and its SEP/eval behavior was indistinguishable from
the zeroed control.

Interpretation: ordinary Alpaca SFT did not reward the model for opening the
role channel. This motivated the counterfactual v2 curriculum rather than a
larger fixed-angle sweep.

## Counterfactual v1 Smoke

The first counterfactual generator produced 12000 synthetic training examples
and 400 synthetic eval examples. The vanilla run completed and synced to W&B as
`sskfecco`.

Use this only as a plumbing smoke run. The generator was too narrow, so the
result cannot rule out template-artifact learning.

## Counterfactual v2

The v2 curriculum adds 12000 counterfactual training examples and 400
counterfactual eval examples. It pairs directive-like substrings that should be
followed in the INSTRUCTION span with matched forms that should be treated as
data in the DATA span.

| Arm | W&B | Final train loss | Last eval loss | SEP | exec_instr | exec_data |
|---|---|---:|---:|---:|---:|---:|
| vanilla | `n3b2ajjb` | 1.6655 | 1.1782 | -0.135 | 0.155 | 0.290 |
| vanilla_zeroed | `vp7rso3y` | 1.6655 | 1.1801 | -0.125 | 0.150 | 0.275 |
| rope_prov pi/8 | `y0033rou` | n/a | 1.5224 | -0.275 | 0.020 | 0.295 |
| rope_prov pre-W pi/8 smoke | `rw38jp7x` | 1.7671 | 2.3358 | n/a | n/a | n/a |
| rope_prov learnable | `mn858blz` | 1.6656 | 1.1856 | -0.130 | 0.155 | 0.285 |
| rope_prov pre-W pi/8 full | `vmgck3dr` | 1.6654 | 1.1832 | -0.125 | 0.160 | 0.285 |
| rope_prov learnable pi/8 unfreeze | `kkrlei1m` | 1.7438 | 1.5296 | -0.290 | 0.025 | 0.315 |
| rope_prov independent angles | `i4dwm9tf` | 1.7686 | 1.5546 | -0.240 | 0.010 | 0.250 |
| rope_prov pre-W learnable | `zqfwyd7o` | 1.6653 | 1.1822 | -0.115 | 0.160 | 0.275 |
| rope_prov pre-W ReZero pi/8 smoke | `j38ih52f` | 1.7668 | 2.3369 | -0.155 | 0.085 | 0.240 |
| rope_prov pre-W ReZero independent smoke | `q8l31kt7` | 1.7668 | 2.4485 | -0.170 | 0.075 | 0.245 |
| vanilla role-contrast v3 smoke | `17yp3vws` | 1.7664 | 2.2472 | -0.160 | 0.065 | 0.225 |

Training gate: passed. The vanilla v2 run completed cleanly in 46.4 minutes at
33.35 examples/sec, and the zeroed control completed in 47.4 minutes at 32.67
examples/sec. Both have stable eval loss around 1.18 after convergence.

SEP interpretation: v2 data improved vanilla SEP by +0.085 relative to the
Alpaca vanilla baseline (-0.220 to -0.135) and improved zeroed SEP by +0.070
relative to its Alpaca baseline (-0.195 to -0.125). It still did not make either
architecture-free model actually separate roles; DATA-slot probe execution
remains higher than INSTRUCTION-slot execution. This makes the remaining
architectural arms informative: a rope_prov arm must beat this data/control gain,
not merely improve over its v1 Alpaca number.

Fixed pi/8 interpretation: the full-budget post-projection pi/8 arm failed the
pre-registered gate. Its SEP barely improved over v1 pi/8 (-0.280 to -0.275),
while vanilla improved by +0.085, giving a delta-of-deltas of -0.080. The arm
mostly suppresses INSTRUCTION-slot execution (0.020) while leaving DATA-slot
execution near vanilla/zeroed levels.

Asymmetric compliance damage: the surprising result is that fixed pi/8 does not
merely regress to vanilla-like SEP. It selectively destroys the
harness-tag-aware INSTRUCTION-slot compliance gained from counterfactual data
(0.155 to 0.020), while DATA-slot style-following is unchanged within noise
(0.290 to 0.295). This updates the cost-law framing: the aggregate loss budget
can look smooth while the capability cost lands almost entirely on one focused
role-aware pathway.

Manual output audit: a 20-example INSTRUCTION-slot audit is recorded in
`results/sep/pi8_instruction_output_audit.json`. Vanilla executed the witness
in 3/20 cases; pi/8 executed 0/20. Both models produced non-empty outputs on all
20 prompts, so the collapse is not an empty-generation artifact. The pi/8
outputs mostly drift into generic summaries or repetitive loops rather than
valid executions missed by the witness detector.

Pre-W smoke interpretation: the pre-W pi/8 arm was a 600-step placement smoke,
not a full comparison to the 2901-step vanilla/zeroed runs. It started less
disruptively than post-projection pi/8 at step 200 (2.666 vs 3.003 eval loss),
but was worse by step 400 and step 600 (2.379 vs 2.259; 2.336 vs 1.860). This
does not answer SEP, but it rules out an immediate utility-loss win for this
pre-W placement at smoke scale. The step-400 to step-600 decrease is also
slow: 0.043 for pre-W versus about 0.31 for vanilla and zeroed. The smoke
therefore supports lower initial cost but not quick optimizer absorption into a
low-loss transport path. The full-budget pre-W run supersedes this read: it
converged to the vanilla/zeroed utility band and landed at SEP -0.125. The smoke
is best treated as an operational/usability warning rather than a stable
architecture result.

Learnable post-projection interpretation: the standard learnable-angle arm
converged with vanilla-like utility and vanilla-like SEP. Its final role-angle
gap was about -0.55 degrees, far below pi/8. Under aligned counterfactual
training, the optimizer still avoids opening the post-projection rotational
channel.

Full-budget pre-W interpretation: fixed pi/8 before Wq/Wk avoids the catastrophic
instruction-compliance collapse seen in post-projection pi8, but it does not
produce positive role separation. This keeps pre-W in the comparison matrix as a
cleaner null: placement fixes utility damage but not provenance utility.

Freeze/unfreeze interpretation: initializing the learnable post-projection gap
near pi/8, freezing it for 200 steps, and then unfreezing does not recover the
zero-gap basin within budget. The final gap remains large, about 19.84 degrees,
and the SEP result is worse than fixed pi/8. This rules against the idea that
standard learnable only stayed near zero because it lacked a nonzero starting
push.

Independent-angle interpretation: assigning different fixed angles per
provenance pair was the rotational steelman against the equal-frequency
single-phase bottleneck. It failed: utility stayed in the high-loss fixed-angle
band and instruction execution collapsed to 0.010. Extra rotational bandwidth
did not create usable substring provenance at this placement and scale.

Pre-W learnable interpretation: giving the model both upstream placement and
angle freedom preserves vanilla-like utility and compliance, but the learned
role-angle gap stays near zero, about -0.46 degrees. SEP lands at -0.115, inside
the vanilla/zeroed/pre-W band rather than above it. The optimizer uses the
available freedom to close the rotational channel, not to turn it into a
substring-provenance feature.

Pre-W ReZero smoke interpretation: staged gates do not rescue the pre-W
rotational channel at smoke scale. The shared pi/8 ReZero smoke opened gates to
max_abs 0.125 and landed at SEP -0.155. The independent-angle ReZero smoke used
fp32 gates and fixed per-pair target angles, opened gates to max_abs 0.1505, and
landed at SEP -0.170. This was not expected to be cheaper than shared pi/8; it
was a route-finding test for whether extra fixed phase bandwidth plus gradual
gate opening gives SGD a usable channel. It did not.

Role-contrast v3 smoke interpretation: the stricter generator makes both roles
use DATA facts and moves the same field-referenced directive surface between
INSTRUCTION and DATA. This did not transfer to better external SEP at 600 steps:
DATA-slot execution dropped from v2 vanilla's 0.290 to 0.225, but
INSTRUCTION-slot execution also dropped from 0.155 to 0.065. The curriculum is
trainable, but it appears to teach caution or field-format behavior rather than
the desired authority separation on SEP.

Rotational-cell conclusion: the counterfactual curriculum was necessary and
worked as a data intervention, but no per-layer rotational arm produced positive
role separation. Post-projection nonzero rotations preferentially damage the
focused instruction-compliance pathway. Learnable post-projection angles avoid
damage by closing the channel. Pre-W placement restores utility and compliance
but lands in the same SEP band as zeroed, even when the role angle is learnable.
ReZero staging opens gates but still does not produce role separation. This
closes the v1/v2 rotational matrix as a negative result, with pre-W as the
strongest null and post-W nonzero rotations as the informative failure mode.

See [experiments.md](experiments.md) for the live run tracker and
pre-registered gates.

## Tiny Additive Role-Embedding Toy

These runs train a fresh char-level decoder-only Transformer from random init.
They are sanity checks for the objective, not evidence that pretrained SLMs can
cheaply adopt the same channel.

| Run | W&B | Template mode | Role control | SEP | exec_instr | exec_data | correct_data |
|---|---|---|---|---:|---:|---:|---:|
| additive hidden roles | `10oi0jn4` | simple | correct | 1.000 | 1.000 | 0.000 | 1.000 |
| additive hidden roles constant | `2l445kb2` | simple | constant | 0.953 | 0.953 | 0.000 | 1.000 |
| additive hidden roles diverse | `8itmpe7b` | diverse | correct | 0.836 | 0.836 | 0.000 | 0.859 |
| additive hidden roles diverse constant | `ub86g90v` | diverse | constant | 0.680 | 0.727 | 0.047 | 0.906 |
| gate pretrain corrected | `nl3dkyrm` | gate_pretrain | correct | 0.000 | 0.000 | 0.000 | 0.000 |
| gate pretrain syntax+regularized | `yy39ct44` | gate_pretrain, syntactic gates | correct | 0.195 | 0.195 | 0.000 | 0.000 |

Interpretation: the simple generator is too easy. A constant-role model, with no
usable instruction/DATA distinction, reaches SEP 0.953, so the simple result is
a tooling smoke and positive control only.

The diverse generator is a better calibration. Correct hidden roles beat
constant roles by +0.328 SEP at step 500 and +0.156 final SEP at step 2000,
passing the toy pre-registration thresholds. This supports the narrower claim
that software-supplied hidden role metadata gives a useful inductive bias in a
tiny Transformer. It does not yet rule out all finite-template shortcuts.

Next required toy escalation: ambiguous paired or gated multi-return examples,
where visible text is identical or near-identical across role assignments and a
small linguistic gate determines the relevant candidate. That is the first toy
setup that can support a stronger scientific claim about out-of-band substring
provenance.

Corrected gated pretraining result: the first `gate_pretrain` attempt used
`block_size=256`, while gated examples were up to 338-371 characters; that run
was invalid because answer regions could be truncated. The corrected
`block_size=512`, batch-512 run used a 731,656-parameter scratch char model and
peaked at 9.52 GB VRAM. It drove training loss to 0.027 but finished heldout
exact-match SEP 0.000, with only transient tiny hits at intermediate evals.
This is a capability-floor result for the scratch toy, not a provenance
negative. The next test should use a pretrained base SLM, starting with
`Qwen/Qwen2.5-0.5B` base.

Syntactic-gate regularized follow-up: restricting the gate to `no_not` and
`question` and adding weight decay 0.1, dropout 0.05, embedding dropout 0.05,
and label smoothing 0.02 improved heldout exact-match SEP to 0.195, with a best
intermediate SEP of 0.211 at step 1750. This shows the scratch model can learn
some syntactic selection, but it fails the pre-registered >=0.50 pass gate.
Scratch gated-role experiments remain blocked for paper evidence; the main path
moves to pretrained Qwen2.5-0.5B base.

## Qwen2.5-0.5B Gate Capability

| Run | W&B | Prompt format | Adapter | exact_match | Interpretation |
|---|---|---|---|---:|---|
| Qwen2.5-0.5B base gate pretrain raw | `yx5fus24` | raw hidden-tag-stripped prompt | LoRA r=8 | invalid | Invalidated by unshifted-label bug in the first SLM harness. |
| Qwen2.5-0.5B base gate pretrain answer cue | `g1ewvphj` | explicit `Answer:` cue | LoRA r=8 | invalid | Invalidated by unshifted-label bug in the first SLM harness. |
| Qwen2.5-0.5B base gate pretrain answer cue, shifted loss | `fd4scfjm` | explicit `Answer:` cue | LoRA r=8 | 0.188 | Failed the >=0.50 capability gate. Corrected next-token loss trains cleanly, but heldout outputs are partial copies, wrong-field copies, and repetitions rather than robust gate/copy execution. |
| Qwen2.5-0.5B-Instruct chat zero-shot | `kcawfngi` | chat | none/eval only | 0.172 | Nontrivial but not solved. The model sometimes reasons about the clue and sometimes selects a distractor. |
| Qwen2.5-0.5B-Instruct chat fine-tune, shifted loss | `8vpjaili` | chat | LoRA r=8 | 1.000 | Passed the reachability check by step 100 and stayed perfect through step 500. Samples are exact witness copies. Confounded by instruction post-training, so this is not provenance-channel evidence. |

Interpretation: the Qwen zero-shot instruct result remains valid, but the LoRA
fine-tunes before 2026-05-17T13:40+02:00 are invalid because the harness used
unshifted labels. The corrected Qwen base rerun reaches only 0.188 exact-match,
so base hidden-role additive runs remain blocked. The corrected Qwen instruct
fine-tune reaches 1.000, establishing that the finite gate/copy task is
reachable with instruction-posttraining priors. The base/instruct split is now
the key diagnostic: capability exists, but not in the clean base model under the
current short LoRA protocol.

## Qwen2.5-0.5B-Instruct Hidden Role Probe

| Run | W&B | Role control | exact_match | Interpretation |
|---|---|---|---:|---|
| Qwen2.5-0.5B-Instruct hidden-role gated correct | `hdjhcz4q` | correct out-of-band roles | 1.000 | Passed the pre-registered correct-role threshold. Paired visible prompts are identical, and samples alternate correctly between witness and answer based on the hidden role map. Constant-role control is still required before claiming role-channel causality. |
| Qwen2.5-0.5B-Instruct hidden-role gated constant | `rldphw35` | all-default roles | 0.500 | Passed the required control. With role information removed, the model selects one side of each contradictory pair, so one duplicate is right and the other is wrong. |
| Qwen2.5-0.5B-Instruct hidden-role gated eval-swap | `u9uht6a8` | train correct, eval instruction/DATA swapped | 0.000 | Passed the directionality check. The model flips to the opposite valid candidate when eval roles are swapped while labels stay fixed. |
| Strict eval reload: correct / constant / eval-swap | `qu1d8i4g`, `e54lk3vo`, `1w49y3cg` | saved adapters, strict normalized answer equality | 1.000 / 0.500 / 0.000 | Passed the eval-artifact kill test. The previous 1.000 was not an artifact of substring matching. |

Interpretation: the confounded instruct hidden-role proof-of-concept is
positive. Correct roles reach 1.000, constant roles cap at 0.500, and the
margin is +0.500 against the pre-registered +0.30 threshold. Eval-swap reaches
0.000, so the role channel carries authority direction rather than merely
adding generic capacity. This is evidence that a simple additive input channel
can carry software-supplied substring provenance when the model already has
instruction-following priors. It does not rescue the RoPE-rotation experiments
and does not yet show that a clean base SLM can learn the same behavior under
the short LoRA protocol.

Subspace audit on the saved correct-role adapter: learned LoRA deltas are low
stable-rank, with weighted mean stable-rank 1.91 under rank cap 8. The learned
role embedding's three active rows have stable-rank 2.89. A first-pass overlap
test against cached Qwen base-to-instruct weight deltas does not show a simple
global SFT-delta alignment: weighted LoRA/SFT cosine is approximately 0, and
right/left capture is near random-baseline scale. Interpretation: the adapter
is low-rank, but the naive base-to-instruct delta overlap thesis is not yet
supported by this metric.

Role-only kill test: on the harder all-gate mix, including semantic
tool/place/material gates with disjoint train/eval vocabularies, Qwen2.5-0.5B-
Instruct reaches strict exact 0.996 with only the 4 x 896 input role embedding
trainable (3,584 parameters) and no LoRA. Eval-swap of the same checkpoint is
0.000. Training the role-only model with constant prompt roles reaches only
0.438, and evaluating the correct checkpoint with roles removed reaches 0.102.
This kills the claim that LoRA/adapters are necessary for the positive hidden
role result in the instruct model. The stronger surviving claim is that the
instruction-tuned model already exposes a small input-control surface for
authority routing.

Base/instruct split check: the same role-only setup on Qwen2.5-0.5B base with
answer formatting reaches only 0.156 after 200 steps, despite near-zero train
loss. At the same point the instruct role-only run was already 0.988. This
supports the interpretation that instruction tuning creates or exposes the
authority-control surface that the software role vector can steer.

## Qwen2.5 Lazy-Rudder Geometry Cross-Check

External artifacts live in `/home/debanjan/Code/Research/lean-mining` commits
`845956c` and `91aa4c1` under experiment
`lrs1_srank_scaling_qwen25_2026-05-17`.
Per-step adapter checkpoints were verified and offloaded to
`https://huggingface.co/d3banjan/lrs1-srank-qwen25-checkpoints`; the bulky
local checkpoint directories were deleted after upload.
Historical SmolLM2 `runs/*/checkpoint-*` directories were likewise verified and
offloaded to `https://huggingface.co/d3banjan/rope-provenance-smollm2-checkpoints`;
local `final/` artifacts were kept for evaluation convenience.

| Model | d_model | attn_qkv_srank | global 7-module srank |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 896 | 3.435 | 4.600 |
| Qwen2.5-1.5B | 1536 | 3.963 | 4.426 |
| Qwen2.5-3B | 2048 | 3.939 | 4.163 |

Interpretation: Qwen2.5 DPO LoRA replicates the lazy-rudder flat floor on the
correct comparison metric: attention q/k/v modules as the analogue of Pythia's
fused `attention.query_key_value`. The constant fit is `3.779 +/- 0.243`, close
to the Pythia reference `3.653 +/- 0.289`; spread is 0.527, within the
pre-registered band. Constant srank beats falling-width laws by Delta AIC
`+7.73` versus `c/sqrt(d)` and `+6.06` versus `c/d^(1/3)`.

The 3B run required 8-bit Adam after bf16 LoRA OOM on the 12GB GPU. The
optimizer-control rerun at 0.5B passed: 32-bit Adam `3.435` versus 8-bit Adam
`3.511`, Delta `0.076 <= 0.25`, so the mixed optimizer recipe is accepted for
the primary scaling fit.

How this connects to provenance: the result supports a low-dimensional
alignment/control geometry in Qwen across scale. Combined with the role-only
probe, the strongest current thesis is that instruction tuning exposes an
authority-routing control surface, and software role vectors can steer that
surface directly. The LRS1 result does not show that the role vectors overlap
the raw DPO delta; it shows the expected alignment substrate exists.
