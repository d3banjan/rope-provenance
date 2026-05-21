# Parallel Policy Rail Plan

Last updated: 2026-05-22T00:09:00+02:00.

## Motivation

The current ladder suggests that text-side metadata is not enough to reliably
induce compositional permission binding. The model can learn simple gates, and
it can suppress forbidden primitives, but held-out role/permission combinations
still invite shortcuts: per-role memorization, rare-primitive collapse, and
OPEN/DECLINE class-marginal collapse.

The next architecture should stop asking the model to invent the policy
interface from prose. Instead, expose a small typed out-of-band policy IR: a
parallel rail that the software stack supplies and the model learns to use.

## Core Hypothesis

A model can learn compositional provenance control more reliably if the policy
state is represented as typed side-channel structure rather than prompt text.
The model still uses hidden layers to understand messy language, but final
behavior routes through deterministic Boolean rails.

```text
span text -> hidden semantic detector -> attempted operation P
software  -> policy IR              -> allowed(P)

decision = OPEN iff allowed(P) and source/risk constraints pass
```

This separates three jobs that text-only training entangles:

- detect what a span is trying to do.
- read the policy state.
- apply the policy state to the detected operation.

## Policy IR Shape

Do not use one giant flat list. Keep the IR typed.

```text
source:
  SYSTEM, USER, TOOL, DATA, WEB

operation:
  OBEY, USE, QUOTE, TOOL_CALL, REVEAL_SECRET, EXEC, NET

risk_domain:
  DANGEROUS, CYBER, PRIVACY, HATE, HARASSMENT, SEXUAL,
  MEDICAL, FINANCIAL, LEGAL, SELF_HARM, JAILBREAK

decision_policy:
  allow, transform, refuse, escalate
```

The source rail answers "where did this come from?" The operation rail answers
"what is being attempted?" The risk rail answers "what kind of danger surface
does this touch?" The decision policy combines them.

## Rung 0: Finish Text-Side Cleanup

Run the remaining narrow text-side control before pivoting fully:

- RPCG11d: keep the RPCG11c corpus and sampler unchanged.
- use class-weighted CE so OPEN and DECLINE contribute equal optimizer-visible
  loss mass.
- interpret failure as stronger evidence that prompt/text-side metadata is the
  wrong substrate for compositional binding.

Kill condition: if held-out cells still fail after primitive frequency and
class marginal are controlled, stop spending small-run budget on prompt-format
variants unless a new causal mechanism is identified.

## Rung 1: Source-Only Rail

Question: can an out-of-band source label alone improve instruction/data
separation?

Status: initial Qwen2.5-0.5B-Instruct smoke is positive. Training only a
5,376-parameter additive source embedding reaches strict exact 1.000 on the
synthetic paired task; constant-source eval is 0.305 and trusted/untrusted
source-swap eval is 0.000. Treat this as a source-only smoke, not evidence for
source+operation composition.

Input:

```text
tokens:    ordinary prompt text
source_id: SYSTEM / USER / TOOL / DATA / WEB per token or span
```

Training:

- trusted source spans may supply instructions.
- untrusted data/web/tool-output spans may be used as evidence but not obeyed.
- include prompt-injection-like substrings inside untrusted spans.

Metrics:

- instruction execution from trusted source.
- data/web instruction suppression.
- source-swap evaluation: same text, changed source_id.
- constant-source control.

Expected outcome: useful but incomplete. Source says who spoke, not what
operation the text is attempting.

## Rung 2: Source + Operation Rail

Question: does separating source from attempted operation create the minimum
viable compositional harness?

Input:

```text
source_id:    source of each span
operation_id: OBEY / USE / QUOTE / TOOL_CALL / REVEAL_SECRET / EXEC / NET
```

Two variants:

- oracle operation labels supplied by the generator/software stack.
- predicted operation labels from an auxiliary head over hidden states.

Training:

- for oracle labels, directly train behavior against `policy[operation_id]`.
- for predicted labels, train both operation prediction and gated behavior.

Metrics:

- per-operation OPEN coverage.
- per-operation forbidden-decline rate.
- held-out role/permission combinations.
- operation-swap evaluation with identical visible text.

Kill condition: if oracle operation labels do not compose, the issue is not
semantic detection; it is policy application or architecture.

## Rung 3: Explicit Policy Vector

Question: can the model bind arbitrary roles to reusable primitive permissions
when the role policy is a typed tensor rather than text?

Input:

```text
role_name: "auditor"
policy.operation = [OBEY=1, USE=0, QUOTE=1, TOOL_CALL=0, ...]
policy.source    = trust thresholds per source
```

Training:

- randomize role names aggressively.
- randomize policy vectors across episodes.
- hold out role names and permission combinations.
- include conflict cases where the text role name disagrees with the policy
  vector; the vector must win.

Metrics:

- held-out policy-vector combinations.
- role alias invariance.
- explicit-vector-over-name conflict accuracy.
- per-primitive coverage, not only forbidden suppression.

Success means the model has learned `role -> policy vector -> behavior`, not
`role string -> memorized behavior`.

## Rung 4: Auxiliary Rail Pretraining

Question: can the model learn the rail during continued pretraining, so SFT/DPO
only steers an existing interface?

Add auxiliary prediction heads:

```text
hidden_state_t
  -> next token
  -> source label
  -> operation label
  -> risk-domain label
  -> policy outcome label
```

Important: prediction alone is not enough. The predicted rail has to be coupled
to behavior by a gate or loss, otherwise it becomes another latent feature the
model can ignore.

Candidate objectives:

- source/operation/risk classification on structured corpora.
- policy outcome prediction on nested prompt-injection examples.
- behavior loss conditioned on supplied policy IR.
- consistency loss between predicted operation and policy-indexed decision.

Datasets:

- chat transcripts with message roles.
- tool traces and tool outputs.
- retrieval/citation documents.
- code blocks and quoted passages.
- synthetic prompt-injection spans with known attempted operation.

## Rung 5: Domain Rail

Question: can moderation-like risk categories be added without corrupting the
provenance gate?

Add `risk_domain` only after source + operation + policy vector works. Risk
domains are semantically broad and more polysemantic than operation rails.
Treat them as attributes of content, not as replacements for provenance.

Evaluation:

- same operation under different risk domains.
- same risk domain under different operations.
- threshold policy: allow / transform / refuse / escalate.
- held-out domain-operation combinations.

## Implementation Notes

Architecture options, in increasing invasiveness:

1. additive side-channel embeddings summed into token states.
2. typed rail embeddings injected at selected layers.
3. small policy MLP that maps policy tensors to per-layer gates.
4. hard or soft logit masks for tool/secret/execution primitives.
5. separate rail head whose predictions gate the LM head.

Start with additive side-channel embeddings because they are closest to the
successful hidden-role experiments. Move to gated heads only if additive rails
do not produce composition.

## Pre-Registration Discipline

For every rung:

- define exact held-out combinations before training.
- report per-primitive coverage and forbidden-decline separately.
- include constant-rail and swapped-rail controls.
- include a text-only baseline with the same visible prompt.
- mark a result VOID if the in-distribution cell does not install.
- do not call a suppression-only result compositional unless permitted
  coverage is also high.

## Expected Decision Tree

If RPCG11d succeeds, the input-side story remains viable: text metadata can
compose, but only under careful class/frequency control.

If RPCG11d fails and source-only succeeds, provenance labels help but are not a
complete binding layer.

If source + operation oracle labels succeed, the missing piece is operation
detection.

If oracle labels fail but explicit policy vectors succeed, the missing piece is
policy parsing and binding.

If explicit policy vectors fail, the model needs stronger architectural bias:
weight sharing, modular primitive heads, or hard gating outside the LM.

## Paper Framing

This plan reframes the project as a progression from text metadata to typed
control structure:

```text
RoPE role channel -> additive hidden role -> text permission gates
-> typed policy rail -> auxiliary rail pretraining
```

The core claim should stay narrow: small models can be taught behavioral gates,
but compositional policy binding appears to require a typed interface or a
training distribution that forces reusable policy parts.
