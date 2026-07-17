# Q2 Camera-Family Shared-Work Scaling

## Context

The active memory contract is still:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The one-parameter camera-family report proved that a local
`Q x Omega x T` chart can store projection metadata once while per-q replay
grows linearly. The next gap was the user's richer "fiber bundle / camera
gauge" concern: if the camera gauge has more than one local parameter, the
compiler should still reason over the family base instead of replaying one
path atlas per sampled camera.

## Current Model

Use a two-parameter local camera family:

```text
q = (q_phase, q_height)
B_family = Q_phase x Q_height x Omega x T
```

A world primitive induces a charted sensor-time-family trace. The report fits
projection/depth metadata with a cubic polynomial basis over
`(q_phase, q_height, tau)`:

```text
monomials with total degree <= 3
basis count = 20
```

The replay baseline compiles one `Omega x T` path chart per sampled q-pair and
uses a cubic temporal basis per path:

```text
[1, tau, tau^2, tau^3]
```

This is intentionally a metadata/work-sharing report. It is not a claim that
all high-dimensional novel-view synthesis is solved, and it is not a quality
benchmark.

## Evidence

New report:

```text
research_experiments/star_uvt_feature_tubes/projective_camera_family_2d_shared_work_scaling.py
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_shared_work_scaling/summary.json
```

Saved summary:

```text
q_axis_counts                    1,2,4,8
q_pair_counts                    1,4,16,64
family_payload_growth            1.0
per_q_replay_payload_growth      64.0
final_payload_ratio              0.0625
family_chart_growth              1.0
per_q_replay_chart_growth        64.0
final_chart_ratio                0.015625
max_family_fit_uv_error_px       0.11114689777823727
max_replay_fit_uv_error_px       0.04855698401008368
final_dense_sample_count         3072
```

Interpretation: for this smooth local Q2 family, chart payload stays tied to
primitive/chart complexity while per-q-pair replay grows with the sampled
family grid. The residual stays well below the top-level audit's `0.50px`
guard.

## Goal Audit Integration

`projective_goal_progress_audit.py` now verifies the Q2 shared-work artifact
as `camera_family_2d_shared_work` and maps it to:

```text
local_camera_family_2d_shared_metadata
```

The current saved goal-progress audit proves 31 focused requirement rows and
still keeps:

```text
full_goal_completion = open
is_goal_complete = false
```

The completion gap is now framed as broad real-scene/full-trainer acceptance
beyond focused probes, rather than a missing local Q2 family metadata story.

## Verification

Commands run:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_camera_family_2d_shared_work_scaling.py \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_camera_family_2d_shared_work_scaling.py \
  tests/test_star_uvt_projective_goal_progress_audit.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_camera_family_2d_shared_work_scaling.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
46 passed in 5.78s
```

Saved artifact checks:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_camera_family_2d_shared_work_scaling.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_shared_work_scaling/summary.json

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json \
  --verify-current-inputs
```

Both verified.

## Falsification Tests

The new verifier rejects:

- missing `Q2 x Omega x T` theory contract,
- q ranges that do not span zero,
- wrong `q_pair_count != q_axis_count^2`,
- stale summaries,
- nonconstant family payload/chart growth,
- replay growth that fails to expose 2D q-pair scaling,
- family UV residual above `0.50px`,
- per-q replay residual above `0.30px`.

The top-level audit applies stricter saved-artifact checks for the default
`1,2,4,8` q-axis grid: payload ratio below `0.15`, chart ratio below `0.05`,
replay payload growth at least `16.0`, and family residual below `0.50px`.

## Implication

This pushes the camera-family story in the direction the user wanted: gauges
and projection math carry the revolving/nearby-camera complexity first. Chart
splitting and fallback remain necessary correctness rails, but the primary
model is now a Q-family fiber-bundle atlas with explicit metadata and
derivative evidence, not "try one chart and hope."
