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
low-loss transport path.

See [experiments.md](experiments.md) for the live run tracker and
pre-registered gates.
