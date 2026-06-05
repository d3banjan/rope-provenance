# Power — rpcg1_capability_gate_pythia70m — 2026-05-18

**Design:** 9 cells (3 roles × 3 primitives). Per-cell observable = mean
open-token probability over 5 training frames (3 roles × 5 frames, each forward
yields all 3 primitive open-token logits).

**Statistic:** gate margin Δ = mean of 9 expected-sign-corrected per-cell shifts.
τ = 3 × SD of the same 9 shifts on the tag-only-null arm.

**Thin-DOF caveat:** the SD that sets τ is estimated from n=9 cells — a thin
estimate. This is a petri-dish screen, not a powered test. The verdict is
deliberately coarse (margin vs 3σ, plus an 8/9 sign count). A positive result
escalates to Qwen2.5-0.5B (ladder rung 2); a NO_GATE is scoped to "at 70m scale".

**Effect expectation:** if the gate exists, positive-only SFT that never
demonstrates out-of-role primitives should drop forbidden-cell open-token
probability by a clearly supra-noise margin — the forbidden cells should lose
most of the open-token probability mass the pretrain stage gave them. The 8/9
sign requirement guards against a margin driven by one or two cells.

**Trap power:** the role-swap trap shares everything but the permission map; a
cyclic shuffle makes ~half the cells disagree with the true map, so a genuine
gate's trap margin collapses toward 0. The trap is a pipeline-correctness check,
not a powered comparison — it must fail by construction.

**Null power:** the tag-only-null arm trains every primitive for every role
equally. With no role→primitive structure in its SFT data it cannot install a
gate; its 9 per-cell signed shifts are pure train+probe noise. n=9 is thin for an
SD estimate, hence the conservative 3× multiplier and the corroborating 8/9 sign
count rather than reliance on the margin alone.
