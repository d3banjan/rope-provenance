# PR10 Risk-Rail Postmortem

Date: 2026-05-22.

## Question

PR10 asks whether a broader moderation-like rail can be added without corrupting
the working source/operation/permission/value-boundary stack.

The fix carried forward from PR9c is explicit value delimitation. PR9/PR9b
showed that the 1.5B model sometimes copied a whole carrier phrase instead of
the candidate value. PR9c solved that by making the answerable value boundary
explicit. PR10 keeps that interface and adds a separate hidden risk rail.

## Setup

Model: `Qwen/Qwen2.5-1.5B-Instruct`.

Trainable parameters:

```text
permission rail: 3 x 1536 = 4,608
risk rail:       4 x 1536 = 6,144
total:                       10,752
```

No LoRA, no source embeddings, no operation embeddings, and no raw policy-bit
embeddings were trained. The software compiler still supplies the local
permission rail. PR10 adds risk ids on the same candidate span:

```text
SAFE       -> allow if permission allows
SENSITIVE  -> allow if permission allows
HARMFUL    -> refuse if permission allows
DEFAULT    -> neutral / answer tokens
```

The target rule is:

```text
if permission denies:
    output ANSWER
elif risk is HARMFUL:
    output REFUSE
else:
    output candidate value
```

The prompt still contains a primary candidate plus a distractor candidate. The
distractor has its own risk label so the test can catch wrong-span bleed.

## Result

Final all-control eval used the best step-100 adapter, with 864 examples per
control.

| Control | exact | risk_allow | risk_refuse | permission_decline | Notes |
|---|---:|---:|---:|---:|---|
| correct | 0.995 | 0.988 | 1.000 | 1.000 | Main result. |
| constant_policy | 0.444 | 0.000 | 0.000 | 1.000 | Permission rail removed; fallback baseline. |
| invert_policy | 0.005 | 0.000 | 0.000 | 0.010 | Policy causality trap collapses. |
| constant_risk | 0.773 | 0.887 | 0.000 | 1.000 | Risk rail removed; harmful refusals break. |
| invert_risk | 0.444 | 0.000 | 0.000 | 1.000 | Risk causality trap collapses to fallback-like behavior. |

Main correct-control buckets:

| Metric | Value |
|---|---:|
| exact_match | 0.995 |
| seen_policy_exact | 0.997 |
| heldout_policy_exact | 0.993 |
| C1 seen source-policy x seen template | 0.997 |
| C2 seen source-policy x held-out template | 0.997 |
| C3 held-out source-policy x seen template | 0.993 |
| C4 held-out source-policy x held-out template | 0.993 |
| safe_exact | 0.986 |
| sensitive_exact | 1.000 |
| harmful_exact | 1.000 |
| distractor_error_rate | 0.000 |
| primary_value_error_rate | 0.000 |
| fallback_answer_error_rate | 0.005 |
| other_error_rate | 0.000 |

## Interpretation

PR10 passes the synthetic risk-domain rung. The risk rail did not destroy the
permission rail, source-policy recombination, held-out template transfer, or
span isolation. The useful read is not just the high correct score; it is the
trap shape:

- `invert_policy` collapses to 0.005, so the permission rail remains causal.
- `invert_risk` collapses to 0.444, so the risk rail is causal.
- `constant_risk` preserves permission-denied fallbacks but loses harmful
  refusals, which is exactly what removing the risk rail should do.
- distractor errors are zero, so the risk label on the distractor span is not
  bleeding into the primary candidate decision.

This supports the typed-rail thesis: once the deterministic harness supplies
already-bound permission and risk state at the relevant span, the model can
learn the behavioral routing with a tiny additive interface.

## Caveats

This is still a synthetic rung. It does not yet prove HarmBench,
JailbreakBench, XSTest, WildGuard, TensorTrust, or BIPIA robustness. Those
benchmarks belong after the synthetic rail stack is stable.

The result also does not solve operation detection. PR6 remains the bottleneck
for moving from oracle attempted-operation labels to language-derived labels.

Finally, `SAFE`, `SENSITIVE`, and `HARMFUL` are a minimal risk ladder, not a
complete moderation ontology. The result argues for a separate typed risk rail;
it does not yet choose the final production risk taxonomy.
