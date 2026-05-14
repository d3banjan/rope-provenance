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

No final v2 numbers are recorded yet. See [experiments.md](experiments.md) for
the live run tracker and pre-registered gates.

