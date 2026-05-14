# rope-provenance

Carve a slice of RoPE rotations into a provenance-encoding subspace so that
attention can distinguish *instruction* vs *data* tokens by phase rather than
by learned content. Mechanism feasibility study on SmolLM2-135M.

See `provenance-rope.md` (parent dir) for the execution plan.

## Status

- Phase 0: scaffold — done
- Phase 1: role-aware RoPE + unit tests — done (4/4 passing)
- Phase 2+: pending

## Run tests

```
uv sync --extra dev
uv run pytest -v
```
