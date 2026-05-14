# Literature Survey: ICLR 2026 and adjacent work for rope_prov paper

Searched: ICLR 2026 main + adjacent venues, with focus on (a) instruction-data separation defenses, (b) RoPE modifications, (c) instruction hierarchy / role-aware embeddings, (d) RoPE dimension inefficiency.

## The result that changes our framing

**Kariyappa & Suh (NVIDIA), "Stronger Enforcement of Instruction Hierarchy via Augmented Intermediate Representations"** — arxiv 2505.18907, May 2025. Not at ICLR 2026 main (arxiv-only as of this search), but unavoidable prior art.

They make our exact diagnostic claim: "prior works typically inject the IH signal exclusively at the initial input layer, which we hypothesize limits its ability to effectively distinguish the privilege levels of tokens as it propagates through the different layers of the model." Their solution is layer-specific trainable additive embeddings injected at intermediate representations. They report 1.6×–9.2× reduction in attack success rate over input-layer-only methods.

**Implications for our paper:**
- The "per-layer vs input-layer asymmetry" framing is *not novel* as a finding — they established it for additive embeddings.
- Our specific contribution narrows to: rotation-based (rather than additive) at intermediate, post-Wq/Wk; the (1-cos θ) cost law; the negative result that rotation breaks where addition works; and the architectural diagnosis of why.
- We should cite this prominently and position our work as "rotation analog of AIR, showing why naive rotation fails where naive addition succeeds."
- The natural v2 framing becomes: "AIR succeeds with additive embeddings; we show that the analogous rotational approach fails specifically because the rotation is downstream of learned Wq/Wk projections. Rotation applied before Wq/Wk (next paper) should recover."

## ICLR 2026 papers — directly relevant

| Paper | Authors | Relevance | Citation priority |
|---|---|---|---|
| **ASIDE: Architectural Separation of Instructions and Data** | Zverev et al. | Closest mechanical prior work. Input-layer orthogonal rotation. Their failure mode (no per-layer signal) motivates our work. | Critical |
| **Selective Rotary Position Embedding** | Movahedi, Carstensen, Afzal, Hutter, Orvieto, Cevher | Input-dependent learnable rotation. Concurrent ICLR'26. Methodologically aligned: validates "learnable rotation" as a 2026 research direction. Primarily about linear/recurrent attention, secondary on softmax. | Critical |
| **PromptArmor** | Anonymous | LLM-as-filter approach. <1% FPR/FNR on AgentDojo. Different mechanism (preprocessing) but defines the ICLR'26 defense benchmark. | High |
| **Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs** | Anonymous | Argues standard RoPE discards imaginary component. Tangential but worth citing as evidence that RoPE modifications are an active area. | Medium |

## Critical adjacent work (not necessarily ICLR 2026 but unavoidable)

| Paper | Venue | Why it matters |
|---|---|---|
| **AIR: Augmented Intermediate Representations** (Kariyappa & Suh) | arxiv 2505.18907 | Direct prior art for per-layer instruction-hierarchy injection. Our paper has to position relative to this or referees will. |
| **Instructional Segment Embedding (ISE)** | NeurIPS or ICLR'25 (Wu et al.) | BERT-style additive segment embeddings for instruction hierarchy. The non-rotation baseline. AIR cites and improves on this. |
| **DefensiveTokens** | ICML'25 / ACM AISec | Optimized special-token embeddings as test-time defense. Different mechanism (token-level, not embedding-axis) but same problem. |
| **Chiang & Yogatama, "RoPE May Cause Dimension Inefficiency"** | arxiv 2502.11276 | Theoretical motivation for our pair-allocation choice — BUT see important nuance below. |
| **TAPA: Token-Aware Phase Attention** | Wang et al., Meta | Learnable phase function for long-context. Adjacent methodology. |
| **CRoPE: Efficient Parametrization of RoPE** | Lou & Xu, Stanford / d-Matrix | RoPE reparametrization, parameter-saving. Tangential. |
| **Frayed RoPE and Long Inputs: A Geometric Perspective** | Wertheimer et al., IBM | Geometric analysis of RoPE behavior past training length. Useful for our (1-cos θ) framing. |

## Important nuance on Chiang & Yogatama

Worth getting right: Chiang argues *high-frequency* (early-index, fast-rotating) RoPE dimensions are underutilized for long-distance retrieval, because at long distances they wrap around many times and become "scrambled."

Our T2b empirical finding is about the *opposite end* of the spectrum: late-index, *low*-frequency (slow-rotating) pairs at SmolLM2-135M scale with 1024-token contexts. We found these are also empirically underutilized — but for a different reason than Chiang's mechanism. At short contexts, slow-rotating dims barely rotate at all, so they could encode position info if the model used them; the model just doesn't, possibly because pretraining didn't require it.

So our T2b finding is a complement to Chiang's, not a restatement. Two ends of the RoPE spectrum, both underutilized at this scale/task, for different reasons. Worth a careful sentence in the paper distinguishing these.

## Defense-landscape papers (cite for completeness in related work)

These are the ICLR'26 / 2025-26 prompt injection defenses to acknowledge:

- **PromptArmor** (ICLR'26) — LLM filter preprocessor
- **Polymorphic Prompt Assembling** (PPA) — structural randomization
- **OpenClaw Agent Privilege Separation** (Cheng & Tsao, TrendAI) — multi-agent isolation
- **Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis** (Kang et al., Dec 2025) — same Suh group as AIR

These are mechanism-orthogonal to architectural defenses (ASIDE, AIR, ours); cite as "defense-in-depth complements" rather than competitors.

## Recommended paper framing post-survey

Given AIR exists, the v1 paper's strongest framing is:

**Title direction**: "Provenance-Aware Rotary Position Embedding: A Negative Result on Per-Layer Rotation as Prompt-Injection Defense"

**Positioning**: "Recent work has explored injecting instruction-hierarchy signals via additive embeddings — at input-layer (ASIDE, ISE) or per-layer (AIR). We test the analogous rotation-based approach via modification of RoPE. We find that rotation fails where addition succeeds, due to a specific architectural property: per-layer rotation applied after Wq/Wk projection cannot be compensated by learned weights, while per-layer addition can. We characterize this with a quantitative (1-cos θ) cost law and identify the conditions under which rotation-based provenance would work (pre-projection application)."

**Four findings, in order of citation strength:**
1. The (1-cos θ) cost law for per-layer rotational interventions — novel, quantitative, citable.
2. Per-layer rotation post-Wq/Wk cannot be compensated by learned weights — diagnostic finding, contrasts cleanly with AIR's additive success.
3. Empirical RoPE dim inefficiency at the opposite end of the spectrum from Chiang — complementary observation.
4. 135M-scale is below the SEP discrimination floor regardless of mechanism — methodological note.

**Cited prominently in related work:**
- ASIDE (the architectural-separation lineage we're in)
- AIR (the per-layer additive analog — the foil)
- ISE (the BERT-style segment embedding lineage)
- Selective RoPE (concurrent learnable-rotation direction)
- Chiang & Yogatama (the dim-inefficiency theory)
- DefensiveTokens (the test-time analog)
- PromptArmor (the orthogonal-mechanism state-of-the-art)

## v2 framing implications

The next paper (rotation pre-Wq/Wk) now has a much cleaner positioning:
"AIR shows additive embeddings work at intermediate layers. v1 of this work showed naive rotation at intermediate layers fails specifically due to post-projection application. We resolve this by applying rotation pre-Wq/Wk, restoring the learned weights' ability to compensate, and report [v2 results]."

That's a tight one-paragraph contribution statement that's stronger than what v1 alone supports.

## Things to potentially run before paper

Given AIR exists with the same diagnostic claim, two cheap things would strengthen v1:

1. **AIR as a baseline arm** (additive layer-specific embeddings, ~1 day to implement). Direct comparison: "AIR achieves X SEP improvement; our rotation analog achieves negative result; the asymmetry is due to multiplicative-vs-additive interaction with learned projections." Strong table.

2. **ISE as a baseline arm** (BERT-style additive segment at input only, ~0.5 day). Closes the loop on the additive-vs-rotational comparison at every level (input-layer additive = ISE; per-layer additive = AIR; input-layer rotational = ASIDE; per-layer rotational = ours).

The four-cell comparison (additive/rotational × input/per-layer) would be a very clean experimental design. Worth considering if the angle sweep / v2.5 results land flat — gives the v1 paper a much stronger comparative spine even with the negative result.
