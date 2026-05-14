# Extended Literature Survey: Freeform exploration

Follow-on to `literature_survey.md`. Less rigid structure, more "interesting threads we found" with framing implications.

## The framing gift: Spotlighting's future-work paragraph

Hines et al. (Microsoft Research, 2024) wrote the Spotlighting paper which uses the exact term "provenance" we've been using, applied to text-level transformations. But buried in their discussion section is this:

> "Out-of-band signaling has many advantages over in-band signaling including immunity to interference, protection from fraud, and bandwidth optimization. Using this historical inspiration, it would seem that we need to devise a multi-channel analog for LLMs. In this approach, control tokens would be passed to model in a separate 'channel' from the data tokens, and the model would (somehow) only react to instructive tokens from the control layer. With current architectures of common language models, however, this is not feasible in any straightforward way. Nonetheless, this premise is compelling and future work remains to be done in this area."

This is, almost word-for-word, the motivation for our paper. They explicitly point at "control tokens in a separate channel" as the desired direction and flag it as future work. Our RoPE-subspace allocation is a concrete architectural attempt at exactly that out-of-band channel.

**Paper framing implication**: open the introduction with this quote. It establishes that a senior team at MSR has explicitly motivated this research direction and noted that "future work remains to be done in this area." Our work is that future work. Even if v1 is a negative result, we are reporting on the first architectural attempt at the out-of-band channel they describe. That's a much stronger introduction than "we tried a thing."

## The defense-architecture chronology that emerged

A clean lineage now visible:

- **2024 Mar — Spotlighting (Hines et al., MSR)**: prompt-engineering provenance signal via input transformation. *Identifies the gap: in-band-only signaling. Calls for out-of-band channel.*
- **2024 Feb / 2025 — StruQ (Chen et al., UC Berkeley)**: structured queries via fine-tuning. Two channels (prompt + data) but still in-band via delimiters. *Closes the gap partially via training; explicitly says "perhaps architectures that natively understand this separation [are needed]" as future work.*
- **2024 Oct — ISE (Wu et al.)**: BERT-style segment embedding added at input layer. First architectural attempt. *Input-layer additive.*
- **2025 — ASIDE (Zverev et al., ICLR'26)**: orthogonal rotation of data embeddings at input layer. *Input-layer rotational.*
- **2025 May — AIR (Kariyappa & Suh, NVIDIA)**: layer-specific trainable additive embeddings at intermediate representations. *Per-layer additive. Made the diagnosis we made.*
- **2025–2026 — ours**: per-layer rotational in RoPE subspace. *Per-layer rotational. Negative result; diagnosis of why.*

That's a 2×2 design that's mostly already done by the field, with our cell as the missing entry:

| | Input-layer | Per-layer |
|---|---|---|
| **Additive** | ISE | AIR |
| **Rotational** | ASIDE | ours |

This is the cleanest comparison frame for the paper. Filling in our cell with a negative result and explaining *why the rotational/per-layer combination specifically fails* is a concrete contribution that completes the design space.

## Concurrent RoPE-modification landscape (methodology peers, not competitors)

The "learnable / multi-dim / context-adaptive rotation" direction is unusually active in 2024–2026. None of these are about provenance, but they all share the "rotation as a manipulable channel" abstraction:

- **ComRoPE** (Yu et al., CVPR 2025): rotation matrices parameterized by learnable commuting skew-symmetric generators. Generalizes RoPE to arbitrary-dim with commutativity-preserving learnable angles.
- **LieRE** (Ostmeier et al., 2024): rotation via matrix exponential of weighted sum of learnable skew-symmetric matrices. Lie-group framing.
- **CARoPE** (Veisi et al., July 2025): Context-Aware RoPE; head-specific frequencies that adapt to context embeddings.
- **Circle-RoPE** (Wang et al., May 2025): cone-like decoupled RoPE for vision-language; addresses cross-modal positional bias by decoupling text and image RoPE.
- **HARoPE** (Oct 2025): head-wise adaptive RoPE for image generation.
- **Mixed RoPE** (Heo et al., 2024): learnable frequency vectors initialized from random unit-circle directions.
- **Selective RoPE** (Movahedi et al., ICLR'26): input-dependent learnable rotation primarily for linear/recurrent attention.
- **Beyond Real** (ICLR'26 under review): uses imaginary component of RoPE for long-context.
- **TAPA** (Wang et al., Meta): learnable phase function for long-context.

Why this matters: the methodological primitive (learnable / adaptive rotation in some subset of RoPE dimensions) is well-established as a 2025–2026 research direction. Our work uses this primitive for a *different purpose* (provenance, not position). That's a clean differentiation — we're not competing with these papers, we're applying their primitive to a security problem.

**One concrete idea**: cite ComRoPE and Selective RoPE specifically as "validation of learnable rotation as an architectural primitive." Then frame our contribution as "we apply this primitive to the role-encoding problem rather than the position-encoding problem." This makes our work feel like it's part of a coherent research wave rather than an isolated attempt.

## Provenance vocabulary already established

Several papers use "provenance" in nearly identical contexts:

- **Spotlighting (Hines)**: "reliable and continuous signal of its provenance"
- **InjecAgent / AgentDojo**: data-flow provenance tracking
- **StruQ**: implicit in the "structured queries" framing

We're not inventing the term. Worth using it consistently in the paper and citing these existing uses. The vocabulary alignment helps reviewers place our work — "ah, the architectural version of Spotlighting's provenance signal."

## Test-time vs training-time defense lineage

DefensiveTokens (Chen et al., ICML'25 / AISec'25) is interesting because it's a *test-time* defense via optimized special-token embeddings. Their finding: 5 optimized defensive tokens reduce ASR from 70% to 0% on Falcon3-7B-Instruct with 2.4% utility loss. This is the same Sizhe Chen / Wagner lab that did StruQ.

Why this matters for our positioning: our work is training-time and modifies the architecture. DefensiveTokens is test-time and modifies only the input. Together with AIR (training-time, layer-specific additive), there's a clean taxonomy:

| | Input modification | Architecture modification |
|---|---|---|
| **Test-time** | DefensiveTokens, Spotlighting | (open) |
| **Training-time** | StruQ | ASIDE, ISE, AIR, ours |

The "test-time architecture modification" cell is open and is a possible v3 angle: can we apply role-aware rotation only at inference, without retraining? Probably difficult given our negative result, but worth flagging as future work in the discussion.

## Recent finding that might bite us — instruction-intent analysis

**Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis** (Kang, Xiang, Kariyappa, Xiao, Li, Suh — arxiv 2512.00966, Dec 2025). This is the same Suh group as AIR. Haven't read it yet but the title suggests they're moving past architectural interventions toward intent classification. Worth checking whether they cite AIR or supersede it. If they argue intent analysis works better than architectural separation, our negative result aligns with their direction — and we should cite them.

## What I'd actually do with this for the paper

1. **Open the intro with the Spotlighting future-work quote.** This single move recasts the paper from "negative result" to "first architectural attempt at the out-of-band channel direction explicitly identified as missing."

2. **Use the 2×2 design table** (additive/rotational × input/per-layer) as the related-work organizing structure. Fills in cleanly, gives reviewers a clean map, and our cell is the missing entry that completes the design space.

3. **Cite ComRoPE, Selective RoPE, LieRE as methodology peers**, not competitors. Frame as "we apply learnable/structured rotation primitives — well-established in 2025-26 position-encoding work — to a security problem."

4. **Use "provenance" consistently** and cite its prior uses (Spotlighting primarily). Don't try to coin new terminology.

5. **In the discussion section**, frame v2 as "the next step is to apply rotation pre-Wq/Wk, restoring the learned-weight compensation we identify as the failure mode." Frame v3 as "the test-time architectural-modification cell remains open and is a possible direction once the training-time question is settled."

## Threads I'd still want to pull but haven't

- **Mechanistic interpretability of instruction-following circuits.** If there's work showing what circuits implement instruction-vs-data discrimination, that would tell us where the role signal *should* be injected. Quick search target: "instruction-following circuit mechanistic interpretability."
- **Per-head vs per-layer specialization in safety contexts.** If specific attention heads specialize for role-aware behavior, our per-head P=8 allocation might be either too uniform or too specific.
- **Length-extrapolation vs role-encoding interaction.** Several of the multi-dim RoPE papers are about length extrapolation. If allocating dims to role hurts length-extrapolation, that's a tradeoff worth quantifying. Probably future work, not v1.
- **AgentDojo as an additional eval.** PromptArmor benchmarks against it. If we can get any rope_prov variant to a reasonable AgentDojo number, that's directly comparable to PromptArmor.

These are nice-to-haves for v2 or extended journal version. Not blockers for the v1 short paper.
