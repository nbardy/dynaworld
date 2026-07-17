# Gauged UVT Extended Phase Profile

## Context

The five-source extended timing breakdown localized the strict frame-scaling
miss to:

```text
3 no-first measured/cadence pair misses
1 normalized 4-to-16-frame growth miss
```

It also showed those misses were cache/support clean. The next question was:
which timed phase explains the miss?

## New Artifact

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_phase_profile_report.py
tests/test_star_uvt_projective_real_video_multiscene_extended_phase_profile_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile/summary.json
```

The report reads saved case payloads under:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix_extended5/cases/
```

It profiles the three no-first misses plus the two growth endpoints. It uses
`step_timings_ms[1:]` to recompute no-first phase means and verifies those means
match the source matrix row ratios exactly.

## Facts

```text
phase_profile_count = 5
no_first_miss_profile_count = 3
growth_endpoint_profile_count = 2
profile_scene_count = 3
max_source_case_no_first_abs_delta = 0.0
all_profile_step_no_first_matches_source = true
max_profile_step_ratio = 1.188933546093892
max_profile_step_ratio_scene_id = Bq4rmeIvJbs_seg_000
max_profile_step_ratio_frames = 4
max_render_forward_ratio = 1.3566329017525305
max_render_forward_ratio_scene_id = Bq4rmeIvJbs_seg_000
max_render_forward_ratio_frames = 4
max_backward_ratio = 1.0839184402497806
max_backward_ratio_scene_id = Bq4rmeIvJbs_seg_000
max_backward_ratio_frames = 16
dominant_positive_phase_counts_for_no_first_misses = {'colorize_loss_ms': 1, 'render_forward_ms': 2}
max_dominant_positive_phase_delta_ms = 120.86875000143965
all_profile_pairs_cache_support_clean = true
all_profile_losses_match_cadence = true
max_profile_rebuild_ratio = 0.5
max_profile_loss_delta = 0.0
```

Profile rows:

```text
Bq4rmeIvJbs_seg_000 4f  no_first_miss
    step ratio   1.188933546093892
    render ratio 1.3566329017525305
    back ratio   0.7818155221803811
    dominant     render_forward_ms +120.86875000143965ms

Bq4rmeIvJbs_seg_000 16f no_first_miss
    step ratio   1.1381882094250788
    render ratio 1.111793076402963
    back ratio   1.0839184402497806
    dominant     render_forward_ms +75.24815266636631ms

C8kTRrtE3KU_seg_000 8f no_first_miss
    step ratio   1.0249968931082667
    render ratio 1.0065971350053353
    back ratio   0.9905494009922626
    dominant     colorize_loss_ms +9.992264333656445ms

Iagm3K8QtFw_seg_000 4f growth_endpoint
    step ratio   0.3589289819374265
    dominant     none

Iagm3K8QtFw_seg_000 16f growth_endpoint
    step ratio   0.5354614359719861
    dominant     optimizer_ms +1.6091386654200797ms
```

## Interpretation

Observed fact:
    The two meaningful no-first misses are render-forward dominated on the
    same source (`Bq4rmeIvJbs_seg_000`).

Observed fact:
    The C8k miss is small and not renderer/backward dominated.

Observed fact:
    The growth-miss source has both profiled endpoints below cadence; its
    normalized growth miss is a shape/growth-ratio issue rather than an endpoint
    pair timing failure.

Current belief:
    The immediate optimization target is Bq4 render-forward phase behavior:
    active tile density, bin traversal, first-live measured cache shape, or MPS
    variance. It is not support-rebin repair.

Confidence:
    Medium. The saved phase profiles are exact for this artifact, but still one
    timing sample per row.

## Commands

```bash
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_phase_profile_report.py
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_extended_phase_profile_report.py -q
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_phase_profile_report.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_phase_profile_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile/summary.json
```

Result:

```text
5 passed, 1 skipped in 1.19s
saved artifact verified
```

Focused phase+timing tests:

```text
12 passed in 0.22s
```

## Next Tests

1. Repeat Bq4 4f/16f cadence/measured timing several times to separate
   variance from persistent render-forward cost.
2. Add render-forward subphase counters for Bq4: active tile refs, bin scan
   count, per-frame tile distribution, and cache row shape.
3. If persistent, compare Bq4 measured versus cadence packed-bin traversal:
   same tile count does not imply same per-tile candidate distribution.
