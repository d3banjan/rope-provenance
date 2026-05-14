# Repository Index

This is the navigation file for agents and humans. Do not spread active
research state across new root-level Markdown files. Add a pointer here, then
put the substance in the pointed file.

## Maintained Docs

| File | Owns |
|---|---|
| [README.md](README.md) | Short project entrypoint and common commands. |
| [docs/research.md](docs/research.md) | Research objective, current interpretation, hypotheses, and decision tree. |
| [docs/experiments.md](docs/experiments.md) | Active run tracker, ablation plan, commands, W&B ids, and pre-registered gates. |
| [docs/results.md](docs/results.md) | Stable result snapshots and interpretation. Do not put transient run progress here. |
| [docs/literature.md](docs/literature.md) | Curated related work and source-guided alternatives. |

## Code Map

| Path | Role |
|---|---|
| `src/rope_prov/rotary.py` | Role-aware RoPE kernels. |
| `src/rope_prov/model.py` | SmolLM2/Llama attention patching for rope_prov and zeroed baselines. |
| `src/rope_prov/parser.py` | Trusted markup parser that assigns role ids. |
| `src/rope_prov/data.py` | Alpaca filtering, role-tagging collator, and counterfactual data generator. |
| `src/rope_prov/train.py` | HF Trainer entrypoint with W&B, checkpointing, and resume support. |
| `src/rope_prov/eval_sep.py` | SEP evaluation with manual role-id threaded decoding. |
| `src/rope_prov/configs/*.yaml` | Reproducible arm definitions. Paired arms should differ only in intended fields. |

## Tracker Rules

1. Update [docs/experiments.md](docs/experiments.md) for every run start,
   abort, resume, W&B id, output directory, and next action.
2. Update [docs/results.md](docs/results.md) only after a run has completed and
   the metric is stable enough to cite.
3. Keep [docs/research.md](docs/research.md) for interpretation changes. If a
   result changes the story, record the old interpretation and why it was
   superseded.
4. Keep [docs/literature.md](docs/literature.md) curated. Do not paste raw
   literature search logs into active docs.
5. Generated logs, W&B folders, checkpoints, and ad hoc summaries stay out of
   docs unless summarized into one of the maintained files above.

## Current Active Decision

The in-flight experiment family is counterfactual v2. It tests whether the v1
negative result was curriculum-bound or architecture-bound. The decisive metric
is the SEP delta-of-deltas:

```text
[SEP(rope_prov_cf) - SEP(rope_prov_alpaca)]
-
[SEP(vanilla_cf) - SEP(vanilla_alpaca)]
```

See [docs/experiments.md](docs/experiments.md) for the pre-registered threshold
and active run state.

