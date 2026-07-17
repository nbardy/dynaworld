# Projective Atlas Coefficient Trainability Smoke

## Context

The projective atlas had native Metal forward coverage and a direct VJP matching
Torch autograd for color, opacity, and homogeneous coefficients. The next
question was whether that derivative surface can actually move a rendered loss,
especially for geometry-like projective trace parameters.

## What Changed

Added:

```text
tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cell_one_step_coeff_training_smoke_if_available
```

The smoke:

1. Builds a degree-2 projective atlas from start homogeneous coefficients.
2. Renders a target from shifted homogeneous coefficients with the same colors
   and opacities.
3. Renders the start atlas through native Metal.
4. Computes MSE image gradients.
5. Runs `direct_backward_projective_trace_tile_time_atlas_metal(...)`.
6. Applies a small line-searched update to homogeneous coefficients only.
7. Verifies the best candidate reduces rendered MSE by at least 1%.

Color is held fixed, so the improvement must come from the projective
coefficient gradient.

## Verification

New trainability smoke:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cell_one_step_coeff_training_smoke_if_available -q
```

Result:

```text
1 passed
```

Projective atlas forward/backward/trainability cluster:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_tile_time_bins_preserve_split_window_intervals \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cells_render_in_metal_if_available \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cell_backward_matches_torch_autograd_if_available \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cell_one_step_coeff_training_smoke_if_available -q
```

Result:

```text
4 passed
```

Focused projective suite:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py -q
```

Result:

```text
42 passed
```

Focused projective plus interval-gated trainer suite:

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
44 passed
```

Existing q-UVT smoke:

```bash
PYTHONPATH=src/train uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_pair_benchmark.py \
  --scenes single_static
```

Result highlights:

```text
max_rgb_error = 5.960464477539063e-08
overflow_tile_count = 0
pair_ratio = 0.5
```

## Current Model

The projective atlas path now has a small but real optimization loop:

```text
packed projective cell forward
    -> MSE image gradient
    -> native direct VJP
    -> homogeneous coefficient update
    -> lower rendered loss
```

This is still not real trainer integration. The compiled tile membership and
order are held fixed, and the line search is a smoke-test convenience. But the
core clean-derivative condition is stronger now: the native projective VJP is
not merely numerically plausible; it moves the rendered objective in the
expected direction.

## Next Gates

1. Add a nontrivial projective/gauge-domain segment producer for the trainer so
   `active_start/active_stop` intervals are not only full-video placeholders.
2. Add a frame-count scaling microbenchmark for packed projective atlas cells.
3. Bridge WorldFoam cell-camera intersections into the same tile-time atlas
   contract.
