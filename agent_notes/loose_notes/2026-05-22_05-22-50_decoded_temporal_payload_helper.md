# Decoded Temporal Payload Helper

## Context

The multicam validation render branches had already converged on
`multicam_validation_payload_from_renders(...)`, but three paths still rebuilt
the same decoded Gaussian temporal-metric buffer inline before calling
`decoded_temporal_payload(...)`:

- base multicam external-view validation
- base multicam oracle-relative validation
- full relative-pose validation

Each path detached `xyz`, `scales`, `opacities`, and `rgbs` to CPU in the same
shape.

## Change

Added `pipeline.diagnostics.decoded_temporal_payload_from_sequence(...)`.

The helper takes a full decoded `GaussianSequence`, applies the same
detach-to-CPU policy over `DECODED_TEMPORAL_FIELDS`, and calls the existing
`decoded_temporal_payload(...)` buffer contract. The three validation render
paths now call this helper directly.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/pipeline/diagnostics.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  tests/test_pipeline_diagnostics.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_pipeline_diagnostics.py tests/test_multicam_relative_pose_trainer.py -q

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_temporal_sampling.py tests/test_mixed_same_heldout_trainer.py -q
```

Results:

- Diagnostics + relative-pose tests: `16 passed`.
- Temporal sampling + mixed same-heldout tests: `13 passed`.

## Handoff

This is a narrow cleanup, but it keeps full-sequence decoded temporal metrics on
the same field list as the streaming eval path. Future validation branches
should call `decoded_temporal_payload_from_sequence(decoded)` instead of
rebuilding field buffers locally.
