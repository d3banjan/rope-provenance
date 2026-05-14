# Literature Survey Part 3: Mechanistic Interpretability of Role and Safety

The mech interp thread turned up findings that materially change how we should frame the paper. Three threads matter, in decreasing order of importance.

## The paper that changes everything: "Prompt Injection as Role Confusion"

**Ye, Cui, Hadfield-Menell — arxiv 2603.12277, March 2026** (Dylan Hadfield-Menell is at MIT CSAIL).

This is the mechanistic paper our work needs to grapple with. Their findings, in their own framing:

> "Language models remain vulnerable to prompt injection attacks despite extensive safety training. We trace this failure to **role confusion**: models infer roles from how text is written, not where it comes from."

> "We design role probes which measure how models internally perceive 'who is speaking'... attacker-controllable signals (e.g. syntactic patterns, lexical choice) control role perception."

> "In the model's internal representations, text that sounds like a trusted source occupies the same space as text that actually is one."

> "The architectural role tag boundary—the primary defense against prompt injection—is overwhelmed by style in latent space."

> "Security is defined at the interface but authority is assigned in latent space."

Their headline experiments: a "CoT Forgery" attack achieves 60% success on StrongREJECT across frontier models with near-zero baselines. Position dominates tags — Systemness declines monotonically with token position, and inserting `<system>` tags has *no effect* on internal role perception.

### Why this matters for our paper

1. **It validates the *existence* of latent role representation.** The model does encode "who is speaking" — this is a real, probeable thing in activation space. Our work is attempting to manipulate that representation.

2. **It explains *why* prompt injection works at the mechanistic level.** Not "the model doesn't know about roles," but "the model's role perception is dominated by style, not by architectural tags." This is a much more specific failure mode than the field has acknowledged.

3. **It explains why our v1 negative result happened.** We added an architectural role signal via RoPE rotation. But the model's existing role-perception machinery is dominated by *style*, not by architectural signals. Our injected signal had to compete against a much stronger style-based signal that's already there. At 135M scale, the architectural signal got drowned out.

4. **It clarifies what v2 (pre-Wq/Wk rotation) would actually need to do.** Not just "deliver the role signal more effectively" but "deliver it strongly enough to overcome the existing style-based role perception." This is a much higher bar than v1's framing implied.

5. **It gives us a one-sentence framing for the paper's motivation.** "Security is defined at the interface but authority is assigned in latent space." This is the problem we're attacking. Use this exact framing in the intro — it's better than anything we'd write.

### Practical use for our paper

**For the intro**: pair this with the Spotlighting future-work quote. The story becomes:
- Hines et al. (2024) called for an out-of-band architectural channel for role signals.
- Ye et al. (2026) showed empirically why current architectural role tags fail: they're overwhelmed by style in latent space.
- Our work tests whether a stronger architectural signal — per-layer RoPE rotation — can compete. v1 finds it cannot at small scale; we identify the specific mechanism (post-projection rotation) that prevents the signal from competing effectively.

That's a tight three-paragraph intro that places our work in real dialogue with the field's current understanding.

**For the discussion**: the v2 framing also changes. Not just "fix the architecture so the signal propagates better" but "design an architectural signal that can compete with the dominant style-based role perception." This raises the bar for what success would look like, and is more intellectually honest.

## Safety is concentrated in specific attention heads

Two findings that together suggest our P=8 uniform-across-heads design might have been wrong:

**"On the Role of Attention Heads in Large Language Model Safety"** (Zhou et al., 2024)
Introduces Safety Head ImPortant Score (Ships) and Safety Attention Head AttRibution (Sahara). Finding: a small subset of attention heads dominates safety behavior.

**"Safety Alignment Should Be Made More Than Just A Few Attention Heads"** (Aug 2025)
"Ablating just a small subset of attention heads can effectively bypass the safety mechanisms of LLMs. This reveals a critical vulnerability: only a limited number of attention heads are responsible for enforcing safety constraints." Their proposed AHD method aims to distribute safety encoding across more heads.

### Implication for our design

We allocated P=8 dims uniformly across all 9 attention heads × 30 layers = 270 head-locations. The safety-relevant attention heads are a small fraction of this (probably <10 heads total at this scale, if the structure even exists at 135M).

The uniform allocation diluted the role signal across many irrelevant heads. A smarter design would identify safety-relevant heads via probing and concentrate the role rotation there. This is a v2 design improvement worth noting.

**Practical implication**: in the discussion section, flag head-specific provenance rotation as a future direction. Concretely: train vanilla, identify safety-critical heads via the Sahara/Ships methodology, then apply rope_prov rotation only at those head locations.

## Refusal is multi-dimensional, concentrated in mid-early layers

**Arditi et al. (2024)** — single refusal direction in activation space, "mid-early safety layer hypothesis" (Li et al., 2024b).

**Wollschläger et al. (ICLR 2025)** — "The Geometry of Refusal" — actually multi-dimensional concept cones, not single direction.

**Zhao et al. (July 2025)** — refusal and harmfulness are *separately* encoded directions. Steering harmfulness changes the model's "belief"; steering refusal changes surface behavior without changing belief.

### What this means for our work

The "mid-early safety layer" finding tells us where in the network role-relevant computation happens — roughly layers 5-15 in 30-layer models. Our rope_prov modification applied to every layer, including the late layers where this kind of processing isn't concentrated. We diluted the signal across irrelevant layers, just as we diluted it across irrelevant heads.

The multi-dimensionality finding (concept cones, not single directions) supports the broader idea that role/safety is encoded in a non-trivial subspace of activations. Our P=8 dims for provenance is in roughly the right ballpark of "small but non-trivial subspace" — but we put it in the wrong place (every layer, every head) rather than concentrated where it matters.

## The deeper question this opens

If style dominates architectural role tags in latent space, and if safety processing is concentrated in a small number of heads/layers, then the *naive* architectural defense (signal injected everywhere uniformly) is competing inefficiently. Effective architectural defenses need to:

1. Inject signal where the relevant computation actually happens (specific heads, mid-early layers)
2. Inject with sufficient strength to compete with style-based role inference
3. Allow the model's learned weights to actually *use* the injected signal (the pre-Wq/Wk issue we already identified)

Our v1 failed (1) by uniformity, possibly (2) by small-scale and weak-signal, and (3) by post-projection placement. v2 only fixes (3). A truly competitive architectural defense at small scale would need to address all three.

This is honestly a more interesting paper than the v1 negative-result-with-cost-law paper. But it requires more work to demonstrate. For the v1 paper, the right move is to *flag* these considerations in the discussion as "directions we'd test next" without claiming we've tested them. That keeps the v1 paper honest and scoped, while motivating v2 (and v3) more strongly.

## Updated paper framing recommendations

Given everything from parts 1, 2, and 3:

**Intro (in order):**
1. Hines et al.'s "out-of-band architectural channel" future-work quote
2. Ye et al.'s "security defined at interface, authority assigned in latent space" framing  
3. Our work tests whether architectural rotation in RoPE can compete with this latent role assignment

**Related work (in order):**
1. Defense lineage: Spotlighting (input transform) → StruQ (structured queries) → ISE (input segment embed) → ASIDE (input rotation) → AIR (per-layer additive) → ours (per-layer rotational)
2. Mechanistic grounding: role confusion (Ye et al.), safety heads (Zhou et al.), refusal directions (Arditi et al., Wollschläger et al.)
3. RoPE modification landscape (methodology peers): ComRoPE, LieRE, Selective RoPE, etc.

**Results**: the 4 findings as already stated. Add a paragraph engaging with the role-confusion finding to explain mechanistically *why* v1 negative result happened.

**Discussion**: v2 (pre-Wq/Wk) AND head-targeted variant AND layer-targeted variant as future directions, motivated by the mech interp findings.

This is a much more grounded paper than what we had before the mech interp search.
