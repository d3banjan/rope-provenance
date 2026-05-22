# PR9 Postmortem: Qwen2.5-1.5B Permission Rail Scale Boundary

## Summary

PR9 tested whether the PR8b multi-span permission rail transfers from
Qwen2.5-0.5B-Instruct to Qwen2.5-1.5B-Instruct. The mechanism is unchanged:
the software stack computes `permission = policy[operation]`, and the model
trains only the three local permission vectors `DEFAULT / DENIED / ALLOWED`.

The first 1.5B run is not a clean scale pass. It is also not a dead rail. The
rail remains causal and span-local, but held-out-template OPEN behavior is
weaker than 0.5B.

## Direct Comparison

| Metric | 0.5B PR8b | 1.5B PR9 step 200 |
|---|---:|---:|
| trainable params | 2,688 | 4,608 |
| batch size | 16 | 8 |
| optimizer steps | 200 | 200 |
| sampled train examples | 3,200 | 1,600 |
| exact | 0.989 | 0.948 |
| C1 | 1.000 | 0.979 |
| C2 | 0.982 | 0.917 |
| C3 | 1.000 | 0.979 |
| C4 | 0.969 | 0.917 |
| held-out-template exact | 0.977 | 0.917 |
| open_obey | 0.974 | 0.813 |
| open_use | 0.984 | 0.938 |
| open_quote | 0.979 | 0.958 |
| decline_* | 1.000 | 1.000 |
| distractor error | 0.000 | 0.000 |
| invert-policy exact | 0.002 | 0.017 |

The initial comparison has a sample-budget confound. Because 1.5B needed
batch 8 for VRAM, 200 steps saw half the optimizer exposure of 0.5B's batch-16
run. That motivated PR9b.

## PR9b Sample-Matched Follow-Up

PR9b kept the 1.5B setup identical and changed only the planned endpoint from
200 to 400 steps, so total sampled training examples would match 0.5B:

```text
0.5B PR8b: batch 16 x 200 steps = 3,200 sampled examples
1.5B PR9b: batch  8 x 400 steps = 3,200 sampled examples
```

The deterministic prefix matched exactly through step 200, as expected. The new
evidence arrived at step 300:

| Metric | PR9/PR9b step 200 | PR9b step 300 |
|---|---:|---:|
| exact | 0.948 | 0.882 |
| C1 | 0.979 | 1.000 |
| C2 | 0.917 | 0.766 |
| C3 | 0.979 | 1.000 |
| C4 | 0.917 | 0.760 |
| held-out-template exact | 0.917 | 0.764 |
| open_obey | 0.813 | 0.813 |
| open_use | 0.938 | 0.750 |
| open_quote | 0.958 | 0.813 |
| decline_* | 1.000 | 1.000 |
| distractor error | 0.000 | 0.000 |
| fallback error | 0.024 | 0.000 |
| other/format error | 0.028 | 0.118 |
| invert-policy exact | 0.017 | 0.000 |

PR9b was early-killed at step 300. More sample exposure did not rescue the
scale run; it pushed the model into a template-format overfit.

## Failure Mode

The 1.5B rail learns the policy direction. Evidence:

- Constant-policy stays at the fallback baseline (`0.444`).
- Invert-policy nearly collapses at step 200 (`0.017`) and fully collapses by
  step 300 (`0.000`).
- Distractor error is zero throughout.
- Decline cases are perfect.

The failure is concentrated on OPEN decisions for held-out templates. By step
300, seen-template cells are perfect while held-out-template cells collapse.
The sample output shows the concrete error:

```text
expected: orange circuit
output:   retrieved note supports orange circuit
```

That is not wrong-span selection. It is not fallback refusal. It is copying the
held-out carrier phrase instead of normalizing to the bare candidate value.

## Interpretation

The current evidence points to a formatting/generalization boundary in the
1.5B residual rail, not a failure of the deterministic policy lookup.

The bigger model does not automatically make the tiny side-channel easier to
train. With the same LR and rail initialization, 1.5B learns a causal
permission direction but overfits the output format on held-out multi-span
templates. The weak axis is `OPEN` behavior under template OOD, especially
OBEY/USE/QUOTE carrier normalization; the DENIED/fallback side is stable.

## Methodological Bug

The script saved only the final adapter. That is a bad fit for these rungs
because the best checkpoint can occur before the planned endpoint. PR9b exposed
this directly: step 200 was the best checkpoint, step 300 was worse, and the
run was killed before a final adapter could be written.

Fix: save eval-step and best adapters whenever `--save-adapter` is provided.

## PR9c Fix

PR9c tested the direct fix: keep the same 1.5B model, same rail architecture,
same traps, and same multi-span task, but make candidate values explicit in the
carrier:

```text
VALUE=orange circuit
[orange circuit]
<value>orange circuit</value>
```

This removes the ambiguity between "copy the carrier phrase" and "extract the
candidate value".

Result:

| Metric | PR9c |
|---|---:|
| exact | 1.000 |
| C1 | 1.000 |
| C2 | 1.000 |
| C3 | 1.000 |
| C4 | 1.000 |
| open_obey / open_use / open_quote | 1.000 |
| decline_* | 1.000 |
| distractor / primary / fallback / other errors | 0.000 |
| constant-policy exact | 0.444 |
| invert-policy exact | 0.000 |

The postmortem diagnosis is confirmed. The 1.5B failure was output extraction
ambiguity under prose carriers, not policy lookup failure or span-binding
failure.

## Next Step

PR10 can proceed, with one constraint: the interface must make policy and value
boundaries explicit. Risk-domain benchmarks should not rely on the model to
infer the relevant answer substring from arbitrary prose while also learning a
new rail.

The next engineering fix is still useful:

- checkpoint every eval step;
- keep best adapters;
- preserve constant/invert traps and distractor-error diagnostics.
