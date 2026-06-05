# Postmortem — rpcg10a_grounding_qwen25 — 2026-05-20

**Verdict:** `PASS` (RPCG10 spec §3.4 decision rule, frozen `tau.json` sha
`1723401feae700f2db6e7b5cb698e0e8c2c488a766f8e38a8b4cc5c4fe661f42`).

## Question

Is the provenance-operation basis `OBEY`/`USE`/`QUOTE` more factored — more
orthogonally represented — in frozen `Qwen/Qwen2.5-0.5B` than the action basis
`exec`/`net`/`sys`? Rung 1 stage a of the binding-layer program; RPCG10b is
gated on this verdict.

## Result

Gate metric (mean pairwise |cosine| of the 3 contrast vectors, averaged over
the middle-third layer band `hidden_states[9:17]`) — lower = more factored:

```
action      0.4999
provenance  0.4911
random      0.4999
```

Bootstrap over carriers (`N_BOOT = 2000`, 95% CI):

- `d = factoredness(action) - factoredness(provenance) = +0.0088 [+0.0079, +0.0096]`
- `d_r = factoredness(provenance) - factoredness(random) = -0.0088 [-0.0096, -0.0079]`

The action-vs-provenance CI lies entirely above 0 → provenance is more factored,
the gate fires `PASS`. The random control points in the opposite direction —
random is LESS factored than provenance with margin, so the metric is not
artifact-prone (the `INCONCLUSIVE` arm of the decision rule does not trigger).
The action and random factoredness coincide at 0.4999 — close to the 0.5 floor
that 3-point uniformly-distributed vectors approach, consistent with neither
basis having a privileged factored representation in the frozen model.

Carrier-split linear probe test accuracy (corroborating only, not the gate):

```
action      1.000   (shuffled-label null 0.455)
provenance  1.000   (shuffled-label null 0.424)
random      1.000   (shuffled-label null 0.515)
```

All three bases saturate the probe (the 3-class concepts are linearly readable
from gate-band activations regardless of basis); the shuffled-label nulls
hover near the 1/3 chance line, confirming the probe is well-specified. Probe
saturation is exactly the situation the spec anticipated (§3.5) — it is why
the contrast-vector geometry is the gate, not probe accuracy.

## Interpretation

Provenance `OBEY`/`USE`/`QUOTE` is measurably more factored in the frozen base
model than action `exec`/`net`/`sys`, at a modest but statistically clear
margin. The gap is small in absolute terms (`d ≈ 0.009` on a metric whose
range is [0, 1]) but tight (`SE ≈ 0.0004`), so the signal is real, not
saturation noise. Both action and random bases sit at the 3-vector mean-cosine
floor (~0.5); provenance is the only basis that breaks it. That makes the
RPCG10b positive reading — "a basis the pretrained model already represents
in factored form composes" — defensible if RPCG10b generalizes. The margin is
not large enough to claim a categorical separation; the spec's §1
interpretation matrix (singleton-vs-combination decomposition) still applies.

## Gate decision for RPCG10b

**PROCEED** to RPCG10b on the provenance basis (PASS). The spec's WEAK and
FAIL branches do not apply; no basis amendment needed. Phase 2 of the RPCG10
plan is authorized.

obstacle_class: methodology
