# High-Motion STAR UVT Trace Geometry Report

## Context

The previous artifact made the high-motion visibility row video-derived, but
only through motion centroids. The next gap was to extract actual STAR UVT trace
geometry from video/trainer artifacts.

There is no obvious full trained high-motion STAR UVT checkpoint in
`outputs/checkpoints`; the `hlaZbH` checkpoints in the current tree are mostly
PowerFoam. So the new report uses the STAR UVT trainer harness path directly.
It now includes initialization rows plus a deliberately tiny dense CPU training
smoke row, and records that this is not yet a persisted/full trained
checkpoint.

## What Changed

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_high_motion_trace_geometry_report.py
tests/test_star_uvt_projective_high_motion_trace_geometry_report.py
```

The report decodes:

```text
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
```

then builds `ScreenTimeTubeModel.from_video_samples(...)` at smoke scale:

```text
target_size = 64
frames = 16
tube_count = 64
tile_size = 8
```

It lowers the resulting STAR UVT tensors:

```text
ma, q_uvt, depth0, depth_beta, opacity, color
```

into `uvt_tubes_to_projective_trace_cell_atlas(...)`, then records complexity,
fallback, dense per-frame tile-pair denominator, and velocity statistics.

## Current Results

The saved artifact is:

```text
outputs/projective_high_motion_trace_geometry_report.json
schema_version = projective_high_motion_trace_geometry_report_v1
status = ok
```

Rows:

```text
config_faithful_zero_velocity_init:
    source = star_uvt_trainer_harness_video_samples
    velocity_init = zero
    trace_count = 64
    cell_count = 793
    interval_to_dense_tile_pair_ratio = 0.06265
    fallback_fraction = 0.0

block_match_motion_init:
    source = star_uvt_trainer_harness_video_samples
    velocity_init = block_match_gated
    trace_count = 64
    velocity_nonzero_count = 58
    velocity_max_px_per_frame = 5.65685
    cell_count = 1496
    interval_to_dense_tile_pair_ratio = 0.29278
    fallback_fraction = 0.0

block_match_motion_trained_dense_3step:
    source = star_uvt_trainer_harness_video_samples
    velocity_init = block_match_gated
    train_steps = 3
    train_lr = 0.03
    train_loss = 0.3009595 -> 0.2954695
    train_loss_ratio = 0.98176
    trained_parameter_l1_delta = 67.95
    moved_parameters = center_uv, center_t, velocity_uv, raw_precision, raw_opacity, raw_color
    depth0_l1_delta = 0.0
    trace_count = 64
    velocity_nonzero_count = 64
    velocity_max_px_per_frame = 5.78425
    cell_count = 1495
    interval_to_dense_tile_pair_ratio = 0.29417
    fallback_fraction = 0.0
```

## Interpretation

This is real STAR UVT trace geometry from the high-motion video initialization
path, not a centroid-derived proxy. The tiny trained row proves the render-active
trainable tensors can move under dense rendering loss and still compile into a
fallback-free projective atlas. `depth0` does not move in this harness because
the dense renderer sorts on `depth0.detach()`. It is still not a full saved
trained checkpoint.
The block-match rows are useful because motion increases atlas cell count and
interval/dense ratio, but the projective atlas still avoids fallback and keeps
interval entries below the dense per-frame tile-pair denominator.

The next honest step is full checkpoint trace extraction:

1. produce or locate a persisted trained high-motion STAR UVT checkpoint,
2. load its tensors,
3. run this same atlas report on the trained state,
4. compare fallback/cell-growth ratios against the initialization and tiny
   trained-smoke rows.

## Verification

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_high_motion_trace_geometry_report.py -q

2 passed in 35.36s
```

The broader focused projective suite with both trace reports also passed:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py \
  tests/test_star_uvt_projective_uv_visibility_split_report.py \
  tests/test_star_uvt_projective_high_motion_trace_geometry_report.py -q

168 passed in 231.68s
```
