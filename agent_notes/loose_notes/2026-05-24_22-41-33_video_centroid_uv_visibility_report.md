# Video-Centroid UV Visibility Report

## Context

The previous UV visibility split report still had a "high-motion" row that
only referenced the checked-in smoke video as provenance while using hand-set
synthetic line-sweep geometry. The next gate was to make that row parse the
video, while keeping the claim honest.

## What Changed

`research_experiments/star_uvt_feature_tubes/projective_uv_visibility_split_report.py`
now decodes:

```text
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
```

It reads up to 16 downsampled grayscale frames with OpenCV, computes adjacent
frame-difference energy, selects the strongest motion pairs, computes their
motion centroids, and uses those centroid `u` positions as the UV roots of the
diagnostic pairwise depth-order line.

This produces the row:

```text
high_motion_video_centroid_line_sweep
source = extracted_video_motion_centroid
```

Current extracted facts:

```text
frames_read = 16
selected_pair_indices = (7, 8, 9)
root_positions_u ~= (4.395, 4.055, 4.123)
parent_uv_event_tile_samples = 3
parent_fallback_fraction = 1.0
output_tile_size = 4
fallback_fraction = 0.0
```

The row stores decoded video metadata, selected pair indices, motion scores,
centroids, root positions, and fitted depth coefficients under `extraction`.

## Current Model

This is video-derived trace geometry, but only as a motion-centroid diagnostic.
It is not a trained STAR UVT trace and not reconstructed world geometry. It is
useful because the split/fallback mechanism now consumes real video motion
statistics instead of a purely hand-authored high-motion proxy.

The extracted roots cluster near the center of the 8-pixel diagnostic tile, so
the adaptive policy accepts child size `4` instead of the previous synthetic
row's child size `2`. That is a feature, not a regression: the report now
reflects the geometry it extracted.

## Verification

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uv_visibility_split_report.py \
  tests/test_star_uvt_projective_visibility.py::test_projective_cell_uv_visibility_adaptive_split_measures_high_motion_fallback_reduction \
  tests/test_star_uvt_projective_orbit_windows.py::test_orbit_derived_uv_visibility_split_report_reduces_fallback -q

4 passed in 4.53s
```

The report CLI also wrote a temporary JSON payload with:

```text
schema_version = projective_uv_visibility_split_report_v1
status = ok
max_parent_fallback_fraction = 1.0
max_output_fallback_fraction = 0.0
max_cell_growth = 4.0
```

After regenerating `outputs/projective_uv_visibility_split_report.json`, the
broader focused projective suite also passed:

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
  tests/test_star_uvt_projective_uv_visibility_split_report.py -q

166 passed in 93.12s
```

## Next

The real next gate is to replace this motion-centroid diagnostic row with
trainer/world trace geometry extracted from the checked-in video artifacts.
Only then should we use high-motion residual fallback and cell-growth numbers
to decide whether oblique/fiber halfspace cells are worth adding.
