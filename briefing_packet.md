# Briefing: literature survey output → Claude Code

Drop-in summary of what to use from `literature_survey.md`, `literature_survey_part2.md`, `literature_survey_part3.md` for paper artifacts. This packet is action-focused; the source surveys have the full reasoning.

## TL;DR

1. The paper's strongest framing is no longer "we tried this and it didn't work." It's **"first architectural attempt at the out-of-band channel direction called for by Hines (2024) and mechanistically explained by Ye (2026)."** Same data, better paper.

2. Three citations are now load-bearing: **Spotlighting** (motivates), **Role Confusion** (explains), **AIR** (the additive analog we're contrasting with).

3. v1 paper stands. Don't add v2.5/v2 experiments before staking. Mech interp findings go in *discussion* as motivated future work, not as completed results.

## Citations to add (with how to use each)

### Tier 1: Load-bearing (must cite, must engage with)

**Hines et al. 2024 — "Defending Against Indirect Prompt Injection Attacks With Spotlighting"** (MSR, arxiv 2403.14720)
- Use the quote in intro: "Out-of-band signaling has many advantages over in-band signaling... we need to devise a multi-channel analog for LLMs. In this approach, control tokens would be passed to model in a separate 'channel' from the data tokens... With current architectures of common language models, however, this is not feasible in any straightforward way. Nonetheless, this premise is compelling and future work remains to be done in this area."
- Frame: our work is the future work they called for.

**Ye, Cui, Hadfield-Menell 2026 — "Prompt Injection as Role Confusion"** (MIT, arxiv 2603.12277)
- Use this quote: "Security is defined at the interface but authority is assigned in latent space."
- Also: "In latent space, text that sounds like a role becomes indistinguishable from text actually tagged as that role."
- Frame: explains *why* our v1 failed mechanistically. The model's role perception is style-dominated; our architectural signal had to compete and lost at 135M scale.

**Kariyappa & Suh 2025 — "Stronger Enforcement of Instruction Hierarchy via Augmented Intermediate Representations" (AIR)** (NVIDIA, arxiv 2505.18907)
- The per-layer additive analog of our per-layer rotational. They report 1.6×-9.2× ASR reduction.
- Frame: completes the 2×2 design space. Our work is the missing rotational/per-layer cell.

### Tier 2: Required for related work

- **ASIDE** (Zverev et al., ICLR 2026) — input-layer rotational, the lineage we're in
- **ISE** (Wu et al., 2024) — input-layer additive (BERT-style segment embedding)
- **Selective RoPE** (Movahedi et al., ICLR 2026) — concurrent learnable-rotation work, primarily linear/recurrent
- **Chiang & Yogatama 2025** — RoPE dimension inefficiency theoretical motivation
- **StruQ** (Chen et al., USENIX 2025) — structured queries
- **DefensiveTokens** (Chen et al., ICML/AISec 2025) — test-time analog
- **Wallace et al. 2024** — Instruction Hierarchy (OpenAI; foundational concept)
- **Arditi et al. 2024** — refusal direction (mech interp grounding)

### Tier 3: Methodology peers (cite as related, not competitors)

- **ComRoPE** (Yu et al., CVPR 2025) — learnable commuting angle matrices
- **LieRE** (Ostmeier et al., 2024) — Lie-group rotations
- **CARoPE** (Veisi et al., 2025) — context-aware learnable frequencies
- **TAPA** (Wang et al., Meta) — learnable phase function
- **Mixed RoPE** (Heo et al., 2024) — learnable frequency vectors

## Drop-in paper structure

### Intro skeleton (3 paragraphs)

P1: Hines et al. (2024) called for an "out-of-band channel" architectural defense [quote]. The premise was compelling but the implementation path was unclear.

P2: Recent mechanistic work explains why current defenses fall short. Ye et al. (2026) show that role information is encoded in latent space dominated by style rather than architectural tags: "[security defined at interface, authority assigned in latent space]." Architectural role signals — when present — compete with style-based role inference.

P3: This work tests whether a stronger architectural signal can compete. We propose Role-Aware RoPE: allocating a subspace of RoPE's rotation dimensions to encode token role (instruction/data) rather than position. We find that per-layer rotation at the Q/K interaction site fails specifically due to post-projection placement, derive a (1-cos θ) cost law characterizing the failure, and identify the pre-projection variant as the natural next step.

### Related work organizing frame: the 2×2 table

|  | Input-layer | Per-layer |
|---|---|---|
| Additive | ISE (Wu '24) | AIR (Kariyappa & Suh '25) |
| Rotational | ASIDE (Zverev '26) | **ours** |

Each cell is one paragraph. Our cell is the missing entry; we complete the design space with a negative result + diagnostic explanation.

### Four findings (results section)

1. **(1-cos θ) cost law** — slopes 1.25-1.54 across the angle sweep. Quantitative prediction tool.
2. **T2b empirical** — vanilla ≈ vanilla_zeroed (Δ=0.0005 eval loss). Lowest-frequency RoPE pairs are dimensionally underused at 135M scale on Alpaca. Complements Chiang's theoretical work at the opposite end of the frequency spectrum.
3. **Post-projection rotation is uncompensatable** — contrasts with AIR's additive per-layer success.
4. **135M is below the SEP discrimination floor** — vanilla SEP=-0.22 means the baseline can't distinguish prompt structure regardless of mechanism.

### Discussion section (motivated future work, NOT completed)

Three v2 directions motivated by mech interp findings:

- **v2a: pre-Wq/Wk rotation** (architectural fix for compensation) — Kariyappa & Suh show additive at per-layer works; rotation analog needs the projection ordering fixed.
- **v2b: head-targeted rotation** (concentrate signal where it matters) — Zhou et al. 2024 show safety is concentrated in a small subset of heads. Uniform P=8 across all heads dilutes the signal. Target via Safety Head methodology.
- **v2c: layer-targeted rotation** (mid-early concentration) — Arditi et al. 2024 and Li et al. 2024b's "mid-early safety layer hypothesis" suggest role-relevant processing is concentrated. Modifying every layer dilutes across irrelevant ones.

State explicitly: v1 fails for at least these three independent reasons. Don't claim we've tested any of them.

## Things to NOT do

- **Don't run v2.5/v2 experiments before paper artifacts.** Priority window matters. Mech interp findings strengthen the v1 paper as-is; don't gate it on more experiments.
- **Don't over-claim "first."** ISE, ASIDE, AIR all predate us with related architectural interventions. We're "first per-layer rotational" specifically. Be precise.
- **Don't ignore AIR.** Reviewers will spot it instantly. The 2×2 table frames us as complement, not competitor.
- **Don't restate Chiang.** Our T2b is at the opposite end of the spectrum (low-freq, late-index pairs). Distinguish carefully.
- **Don't oversell the negative result.** The cost law and the diagnostic explanation are what make the negative result publishable. Lead with those, not with the SEP table.
- **Don't engineer a lenient SEP detector for v1.** Footnote the 7-8/10 manual agreement and move on.

## Concrete action items

For each artifact, what changes:

### README.md (priority 1, do today)
- Add the 2×2 design table to the "context" section.
- Add a "related work" section with the Tier 1 citations.
- Use the Hines and Ye quotes in the project description.

### Blog post on d3banjan.github.io (priority 1, do today)
- Open with the Hines quote.
- Use the 2×2 table as the organizing frame.
- Include the (1-cos θ) cost-law table with the four sweep points.
- End with "future work" pointing at v2a/v2b/v2c.

### arxiv preprint draft (priority 1, do tomorrow)
- Follow the 3-paragraph intro skeleton.
- Use the 2×2 table in related work.
- Four findings in results.
- v2a/v2b/v2c in discussion, not results.
- Cite all Tier 1 and Tier 2 papers. Tier 3 in a one-paragraph "broader context" subsection.

### PLAN.md (priority 2, after artifacts)
- Add a section noting the mech-interp-informed v2 design directions (head-targeting, layer-targeting in addition to pre-projection).
- Note that v2.5 (learnable angles) is still a valid experiment but should be done after v1 is staked.

## One-line summary for the abstract

"We test architectural rotation in RoPE subspaces as a defense against prompt injection, motivated by Hines et al.'s call for an out-of-band channel and Ye et al.'s finding that role perception in LLMs is style-dominated in latent space. We characterize a (1-cos θ) cost law for per-layer rotational interventions, complete the 2×2 design space of input/per-layer × additive/rotational defenses (with AIR, ISE, ASIDE), and identify post-Wq/Wk projection placement as the specific mechanism preventing learned-weight compensation in the rotational case."

That's the abstract. Tighten as needed.
