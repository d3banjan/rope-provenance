# Research Brief

Last updated: 2026-05-15T05:21:16+02:00.

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

H3: the learnable channel is accepted by the model. If the learned role-angle
gap moves above 5 degrees and SEP improves, the channel becomes useful under
aligned training. If the gap stays near zero with stable training, the sharper
diagnosis is that this post-projection rotational placement is structurally
inadequate even with the right curriculum.

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
