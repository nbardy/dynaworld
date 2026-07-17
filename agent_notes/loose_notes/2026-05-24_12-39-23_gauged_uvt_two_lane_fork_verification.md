# Gauged UVT two-lane fork verification

Date: 2026-05-24

## Context

The user asked to fork the Gauged UVT work into two sub-agent lanes and get both
approaches moving:

1. nonlinear/projective atlas-cell Metal evaluator
2. interval-gated q-UVT real-trainer integration

This note records the integration-level result after both workers returned.

## Lane A: projective cell evaluator

Worker A added a cell-local evaluator path:

```text
ProjectiveTraceCellTraceAtlas
projective_trace_windows_to_cell_trace_atlas(...)
eval_projective_trace_cell_torch(...)
render_projective_trace_cell_atlas_reference(...)
torch.ops.star_uvt_v0.render_projective_trace_cell_tiles(...)
render_projective_trace_cell_atlas_metal(...)
```

Accepted gauge-domain windows now lower into packed raw-time polynomial rows for:

```text
u(t), v(t), depth(t)
```

Tile-time cells index those row ids directly. This matters because the compiled
cell is now the GPU-evaluable object, not only a support/order wrapper around
the original rational coefficients.

## Lane B: interval-gated trainer backend

Worker B added a selectable trainer-harness backend:

```text
uvt.render_backend = "metal_tile_interval_gated"
render_uvt_tubes_metal_interval_gated_backward(...)
full_active_intervals(...)
validate_uvt_backend_modes(...)
```

The forward path uses native:

```text
render_gated
```

and the VJP uses native:

```text
direct_atomic_backward_gated
```

The current source-view trainer uses full-video intervals as the degenerate
single-domain case. That is intentionally boring but useful: it proves the real
trainer can select the gated backend now, while leaving the same
`active_start/active_stop` surface ready for nontrivial projective/gauge-domain
segments later.

## Verification

Combined focused gate run from the dynaworld root:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
43 passed in 7.05s
```

Worker B also ran a real `src/train/train.py` smoke from a temp config derived
from the checked-in STAR UVT RGB test-video config. It selected:

```text
render_backend = metal_tile_interval_gated
reduction_mode = index_add
sample_emission_mode = direct_atomic
```

and decreased loss:

```text
0.15689826011657715 -> 0.12637178599834442
```

## Current model

Both forked approaches are now live:

- the atlas-cell math has a GPU-evaluable local trace object
- the interval gate has a real trainer dispatch surface

The next strongest bridge is to connect them: produce nontrivial
projective/gauge-domain active intervals in a trainer-adjacent path, or add a
projective atlas-cell trainability smoke for coefficient/color updates.
