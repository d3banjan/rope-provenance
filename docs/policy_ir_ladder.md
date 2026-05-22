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

Status: initial oracle-operation Qwen2.5-0.5B-Instruct smoke is positive.
Training only 9,856 source+operation embedding parameters reaches strict exact
1.000 across trusted OBEY, untrusted OBEY suppression, DATA USE, and DATA
QUOTE. Controls remain meaningful: source-swap is 0.000; operation ablation is
0.215; OBEY/USE swap is 0.438.

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

Status: first embedding-only smoke is `VOID / installation_failed`. With source
and operation still supplied as oracle rails, a factorized OBEY/USE/QUOTE
policy bit vector trained only 15,232 additive embedding parameters. Loss fell
from 12.02 to 1.96, but exact stayed at 0.048, seen-policy exact at 0.056, and
held-out policy exact at 0.000. Because seen policies did not install, this is
not a clean held-out-composition failure. The live hypothesis is that a policy
vector needs an explicit binder or modulation site that computes
`allowed = policy[operation]`; a uniform additive prompt bias is not enough.

Diagnostic update: tiny overfit confirms the distinction. A fixed mask `001`
overfits to exact 1.000, and all six seen masks overfit to exact 1.000 on 72
training rows. The same all-seen adapter falls to 0.261 on fresh seen rows and
0.219 on fresh held-out `101` rows. This kills the "just underfit one run"
interpretation: additive policy bits can memorize a small table, but they do
not learn a reusable binding rule.

Positive pivot: the oracle permission rail succeeds. When the software stack
computes the lookup and injects a local ALLOWED/DENIED rail on candidate spans,
Qwen2.5-0.5B-Instruct reaches exact 1.000 on seen policies and held-out `101`
with only 12,544 trainable embedding parameters. The working interface is
therefore not raw `policy_bits`; it is a compiled, operation-local permission
rail.

Minimality update: the permission rail alone is sufficient on this synthetic
rung. Disabling source embeddings, operation embeddings, and raw policy-bit
embeddings leaves only a 3 x 896 permission embedding table trainable
(2,688 parameters). It still reaches exact 1.000 on seen policies and held-out
`101`, with every OPEN/DECLINE primitive cell at 1.000. This is the current
100% working path: a deterministic software compiler computes the local
permission, and the model learns to obey that rail.

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

Immediate next diagnostic: replace the oracle permission compiler with a
learned operation detector or a tiny learned binder, and test whether it
preserves the 1.000 behavior without hand-supplying the bound permission.

## PR4: 4-Cell Compiled-Permission Grid

Question: does the current 100% permission-only rail survive the same
composition pressure that killed the text-side ladders?

Status: completed on Qwen2.5-0.5B-Instruct. The 2,688-parameter
permission-only rail reaches exact 1.000 on all four cells and every
OPEN/DECLINE primitive cell. Constant-policy trap exact is 0.444; the stricter
invert-policy trap is 0.000. The original OBEY/USE `swap_policy` control is
kept as a legacy diagnostic but is weak on this grid because QUOTE is unchanged.

Current working path:

```text
software compiler: permission = policy[source, operation]
model rail:        DEFAULT / DENIED / ALLOWED at the candidate span
```

PR4 keeps the local permission rail and changes the evaluation distribution.
Every source id, every operation id, and every policy mask appears during
training. The held-out axis is the *pairing* between source and policy, crossed
with the candidate surface template.

```text
                seen template     novel template
seen source-policy      C1              C2
novel source-policy     C3              C4
```

Decision rule:

- C1 must reach exact >= 0.99 or the rung is VOID.
- C2, C3, and C4 must each reach exact >= 0.95 for `GRID_GENERALIZES`.
- constant-permission / constant-policy controls must collapse below 0.60.
- swap-policy controls must collapse below 0.60.
- per-operation OPEN and DECLINE cells must each be >= 0.95.

Kill logic:

- If C1 fails, the current rail does not install on the harder grid.
- If C2 fails, the rail is template-fragile; do not scale before fixing
  surface diversity.
- If C3 fails, source-policy recombination is still not solved.
- If C4 fails while C2/C3 pass, the rail has partial but not full
  composition; scale-up is premature.
- If all cells pass and traps collapse, PR5 becomes benchmark projection.

Scope caveat: PR4 is a systems-composition test, not proof that the model
learned the policy lookup internally. The lookup is deliberately compiled by
software. PR7 is the rung that replaces this compiler with a learned binder.

## PR5: Real Benchmark Projection

Question: does the compiled local permission rail transfer from the synthetic
grid to established prompt-injection distributions?

Status: initial eval-only SEP projection was an early kill under the >= 0.95
gate. Loading the passing PR4 adapter and evaluating 200 SEP-style denied OBEY
rows gave exact 0.900. The invert-policy trap dropped to 0.465, so the rail
still causally changed behavior, but the synthetic PR4 surface did not fully
transfer to real SEP prompt-injection surfaces without adaptation.

PR5b then added paired SEP-surface adaptation. Each SEP surface appears with
the same visible text twice: `DENIED -> ANSWER` and `ALLOWED -> witness`. The
permission rail is the only difference, and the witness is made explicit in
the untrusted span so the test measures rail use rather than world knowledge.
After 300 steps from the PR4 adapter, held-out paired SEP exact is 1.000, with
OPEN_OBEY 1.000, DECLINE_OBEY 1.000, constant-policy 0.500, and invert-policy
0.000.

Project SEP, BIPIA, and StruQ-style examples into the rail format:

- trusted instruction span gets source/operation/policy rails.
- untrusted document or tool-output span gets DATA/WEB/TOOL source rails.
- injected lower-priority instruction attempts are labeled as attempted OBEY
  but denied by the compiled permission rail.

Kill logic:

- If synthetic PR4 passes but SEP/BIPIA-style projection fails, the synthetic
  task is missing real attack-surface diversity.
- If attack suppression improves but utility collapses, the rail is too blunt
  and needs transform/refuse policy labels before risk-domain expansion.

PR5b unblocks PR6, with a caveat: benchmark-surface transfer required
adaptation. The next bottleneck is no longer "does the local permission rail
work on SEP-like surfaces?" but "can the attempted operation be detected rather
than supplied as an oracle rail?"

## PR6: Learned Operation Detector

Question: can the system stop using oracle operation labels without losing the
compiled permission rail's behavior?

Status: early-killed by detector preflight. A frozen-Qwen hidden-state linear
probe over the operation-labeled candidate span fits seen PR4 templates at
1.000, but held-out template cells fall to 0.615. The shuffled-label trap is
0.380, so the detector is learning real template signal, not random structure,
but it is not template-invariant enough to feed the permission rail.

Keep the software policy compiler and local ALLOWED/DENIED rail, but replace
oracle operation ids with a small learned detector over span hidden states.

Kill logic:

- If operation detection is below 0.95 on held-out templates, do not train the
  full rail stack; improve the detector dataset first.
- If detector accuracy is high but rail exact collapses, the handoff between
  detector and permission rail is the problem.

Next action: do not integrate predicted operation ids yet. Either add a PR6b
operation-detector diversity rung, keep operation ids oracle-supplied while
testing downstream rails, or later try PR6-RL: an independently trained detector
policy using the existing trap pairs as reward. The lookup
`permission = policy[operation]` remains definitional software; any RL belongs
only on the operation detector or a separately audited concierge module, not on
the base model and harness jointly.

## PR7: Tiny Policy Binder

Question: can a constrained module learn the lookup that raw additive policy
bits failed to learn?

Status: completed positive on the original held-out policy-mask task. Replacing
the software compiler with a 29,792-parameter MLP binder over
`[policy_bits, operation_onehot]` reaches exact 1.000 on seen masks and the
held-out `101` mask. Constant-policy is 0.429 and invert-policy is 0.000. This
rescues the raw policy-bit failure when the lookup is made architecturally
explicit.

Replace the software compiler with a small binder:

```text
(policy_bits, operation_id, optional source_id) -> permission_id
```

The binder may be an MLP, bilinear table, or tied per-primitive module. It must
be small and separately inspectable. The LM still consumes only the local
permission rail.

Kill logic:

- If the binder cannot overfit a tiny all-seen table, the architecture is
  wrong.
- If it overfits but fails C3/C4, the problem is still compositional binding.
- If it passes PR4-style C4, this is the first learned-compiler rung.

PR7b tested the same learned binder on the PR4 source-policy x template grid.
It installs on seen templates (C1/C3 = 1.000) but fails held-out templates
(C2 = 0.448, C4 = 0.438). The software compiler remains the strongest path for
PR8; the learned binder is not yet robust enough to replace it.

## PR8: Span And Long-Context Scaling

Question: does the rail survive realistic substring provenance instead of one
clean candidate span?

Status: PR8b clears the first multi-span gate. The first PR8 run was a boundary
at step 300: exact 0.965, C2 = 0.938, C4 = 0.917, constant-policy 0.444, and
invert-policy 0.003. PR8b fixed the endpoint at 200 steps, enlarged evaluation
to 2304 rows with held-out values, and added error diagnostics. It reaches exact
0.989, C2 = 0.982, C4 = 0.969, constant-policy 0.444, invert-policy 0.002, and
zero distractor errors. The residual mistakes are mostly formatting/other
outputs, not wrong-span bleed.

Add multiple candidate spans, repeated source types, long contexts, retrieved
documents, tool outputs, and irrelevant distractors. Measure both correctness
and whether the wrong span's rail bleeds into the candidate.

Use the software-compiled permission rail first, not the learned binder. PR7b
showed that the binder itself is template-fragile, so PR8 should isolate span
binding and long-context effects using the strongest available compiler.

Industry benchmark projection starts here as well. For prompt injection and
in-context security, prioritize TensorTrust, PromptInject/InjecTQA-style RAG
injections, and the local SEP projection. These test whether source/operation
rails survive creative human injections and retrieval/tool-output surfaces.

Kill logic:

- If short-context C4 passes but long-context performance decays sharply before
  the model's nominal context limit, the rail needs position/span binding or a
  different injection site.
- If multi-span examples fail while single-span examples pass, the rail is a
  local token cue rather than robust substring provenance.

Next action: move to PR9 scale/architecture replication. Keep PR8b's fixed
200-step schedule and error-type diagnostics as the baseline. Do not add
risk-domain rails yet; PR10 remains gated behind PR9.

## PR9: Scale And Architecture Replication

Question: is the rail a Qwen2.5-0.5B-Instruct artifact?

Status: first scale replication is a boundary. Qwen2.5-1.5B-Instruct fits on
the 12GB GPU and trains only `3 x 1536 = 4608` permission-rail parameters under
the PR8b fixed 200-step protocol. It reaches exact 0.948 with C1 = 0.979,
C2 = 0.917, C3 = 0.979, C4 = 0.917, constant-policy 0.444, invert-policy
0.017, and zero distractor errors. The rail is causal and span-bound, but it
does not clear the all-cell gate at the same budget.

Run the smallest passing PR4/PR5 setup on at least one larger Qwen model and
one different architecture family if local hardware permits. Prefer a 7B
replication only after PR4 and PR5 pass, because scale-up without the right
traps is low information.

Kill logic:

- If 0.5B passes and larger instruct models fail, inspect whether the rail
  injection scale or chat-template priors changed.
- If only instruct models pass, frame the rail as reusing an instruction-tuned
  authority surface, not installing provenance from scratch.

Next action: do not proceed to PR10 yet. Either run a PR9b calibration rung
that changes only the rail schedule/scale for 1.5B, or record this as the first
scale boundary and test a non-Qwen instruct model before risk-domain expansion.

## PR10: Risk-Domain Rails

Question: can broader moderation/risk concepts be added without corrupting the
source/operation/permission decomposition?

Status: gated behind PR8/PR9. The risk rail should not be added until the
source/operation/permission path survives multi-span and at least one
architecture/scale replication.

Add risk labels only after source, operation, and permission behavior survives
PR4/PR5. Risk labels should be attributes of content, not replacements for
operation labels.

Industry-grade safety evals belong on this rung:

- HarmBench / JailbreakBench: adversarial harmful-behavior and jailbreak ASR.
- XSTest: over-refusal / safety-creep capability tax.
- WildGuard / WildChat: input-output moderation and refusal classification.

These are not substitutes for prompt-injection benchmarks. They test the risk
and moderation rails after the provenance/permission rail is already stable.

Kill logic:

- If risk labels reduce prompt-injection robustness or cause over-refusal,
  keep risk handling outside the LM until the rail can represent
  `allow/transform/refuse/escalate` separately.
- If risk labels work only for synthetic labels and not benchmarked safety
  categories, treat them as classifiers, not policy rails.

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
