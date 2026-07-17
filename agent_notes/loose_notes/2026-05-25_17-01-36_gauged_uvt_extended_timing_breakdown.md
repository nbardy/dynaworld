# Gauged UVT Extended Timing Breakdown

## Context

The previous five-source frame-scaling diagnostic established an important
split:

```text
correctness/cache/support:
    stable

timing:
    not yet a five-source win
```

The missing piece was localization. A single max ratio does not say whether
the timing failure comes from cache invalidation, support churn, one scene,
one frame count, or general evaluation growth.

## New Artifact

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_timing_breakdown_report.py
tests/test_star_uvt_projective_real_video_multiscene_extended_timing_breakdown_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown/summary.json
```

The report reads the failed strict five-source source:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix_extended5/summary.json
```

It builds one measured/cadence pair per scene/frame count and one growth row
per scene.

## Facts From The Breakdown

```text
source_scene_count = 5
source_row_count = 30
pair_count = 15
no_first_ratio_gt1_count = 3
no_first_ratio_gt1_fraction = 0.2
growth_ratio_gt1_count = 1
growth_ratio_gt1_fraction = 0.2
max_measured_vs_cadence_no_first_step_ms_ratio = 1.188933546093892
max_no_first_ratio_scene_id = Bq4rmeIvJbs_seg_000
max_no_first_ratio_frames = 4
max_no_first_ratio_overage = 0.1889335460938919
max_measured_no_first_growth_vs_frame_growth_ratio = 1.0009153415685994
max_growth_ratio_scene_id = Iagm3K8QtFw_seg_000
max_growth_ratio_overage = 0.0009153415685994037
distinct_any_timing_miss_scene_count = 3
all_failing_pairs_cache_clean = true
all_pair_support_clean = true
all_scene_rebuild_growth_flat = true
max_measured_vs_cadence_rebuild_ratio = 0.5
max_end_loss_abs_delta = 0.0
```

No-first pair misses:

```text
Bq4rmeIvJbs_seg_000 4f  1.188933546093892
Bq4rmeIvJbs_seg_000 16f 1.1381882094250788
C8kTRrtE3KU_seg_000 8f 1.0249968931082667
```

Frame-growth miss:

```text
Iagm3K8QtFw_seg_000 1.0009153415685994
```

## Interpretation

Observed fact:
    The failing timing rows still have measured rebuilds below cadence,
    measured rebuild growth flat, loss delta zero, and zero support rebins,
    stale refreshes, fallback marks, overflow, and visibility stratification.

Current belief:
    The five-source timing miss is not caused by cache invalidation or support
    churn. The likely branches are evaluation phase cost, run-to-run timing
    variance, or per-scene/frame-count phase shape.

Confidence:
    Medium. The breakdown is exact for the saved artifact, but it is still a
    single timing sample per source row.

## Branches

Hypothesis:
    MPS timing variance/startup phase dominates the no-first ratio.
Why it might be true:
    The worst ratio is at 4 frames, and the normalized growth miss is only
    `0.000915` over the gate.
What would make it false:
    Repeated timing keeps the same scene/frame failures with low variance.
Cheap test:
    Rerun the five-source matrix with repeat timing or use 8/16/32 frames.

Hypothesis:
    Evaluation/compositing cost dominates after projection/support rebuilds are
    amortized.
Why it might be true:
    Cache metrics are clean while no-first ratios fail.
What would make it false:
    Phase traces show compile/rebuild/support phases still dominate the miss.
Cheap test:
    Add per-row phase summaries for worst failing pairs.

Hypothesis:
    Scene/frame phase shape matters more than motion score alone.
Why it might be true:
    Worst no-first row is not the highest-motion scene; growth miss appears on
    a different source than the max no-first pair.
What would make it false:
    A larger matrix shows monotone correlation with motion/tile metrics.
Cheap test:
    Add correlation diagnostics after a repeated or larger-source matrix.

## Commands

```bash
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_timing_breakdown_report.py
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_extended_timing_breakdown_report.py -q
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_timing_breakdown_report.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_timing_breakdown_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown/summary.json
```

Result:

```text
6 passed in 0.02s
saved artifact verified
```

Combined touched pack:

```text
96 passed in 8.16s
```

## Next Tests

1. Repeat the five-source strict matrix with repeated timing and report
   mean/min/max/std for the three failing pairs.
2. Add phase-level timing breakdown for `Bq4rmeIvJbs_seg_000` at 4f/16f,
   `C8kTRrtE3KU_seg_000` at 8f, and `Iagm3K8QtFw_seg_000` growth.
3. Try an 8/16/32 frame-growth variant to reduce 4-frame startup effects.
