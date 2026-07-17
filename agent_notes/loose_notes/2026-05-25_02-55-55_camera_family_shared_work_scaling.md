# Camera-Family Shared-Work Scaling

## Context

The active thread goal is still:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

Earlier artifacts proved that the fiber-gauge value/gradient invariant extends
from one camera path to a one-parameter local camera family over
`Q x Omega x T`. The missing piece was a work-sharing artifact for that local
camera family: a mathematical guard is useful, but the goal also requires
metadata and memory growth to stay tied to chart complexity rather than the
number of rendered camera-family samples.

## Current Model

The relevant compiler object is not one atlas per nearby camera. It is a local
camera-family atlas over:

```text
Q x Omega x T
```

where `q` is a low-dimensional camera-gauge parameter. In a local trivialization,
primitive projection metadata can be represented by basis functions over
`(q, tau)` rather than replaying the whole `Omega x T` path compiler for each
q sample.

For this artifact, the family chart uses a quadratic basis:

```text
[1, q, tau, q^2, q tau, tau^2]
```

The replay baseline uses one per-q path chart with:

```text
[1, tau, tau^2]
```

This is not claiming arbitrary novel-view synthesis is free. It only proves the
local camera-family version of the shared-metadata claim.

## Evidence

New report:

```text
research_experiments/star_uvt_feature_tubes/projective_camera_family_shared_work_scaling.py
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_shared_work_scaling/summary.json
```

Saved summary:

```text
q_counts                         1,2,4,8,16
family_payload_growth            1.0
per_q_replay_payload_growth      16.0
final_payload_ratio              0.10576923076923077
family_chart_growth              1.0
per_q_replay_chart_growth        16.0
final_chart_ratio                0.0625
max_family_fit_uv_error_px       0.3059606787397726
max_replay_fit_uv_error_px       0.22764596580912055
final_dense_sample_count         768
```

This says the local `Q x Omega x T` chart stores metadata once across q samples,
while per-q replay grows linearly. The family residual is deliberately checked:
if a single chart cannot approximate the family, the verifier fails instead of
hiding the error under a vague fallback story.

## Goal-Progress Integration

`projective_goal_progress_audit.py` now imports and verifies the camera-family
shared-work report. The audit has a new proved requirement:

```text
local_camera_family_shared_metadata
```

The saved goal-progress report now proves nine sub-requirements while keeping:

```text
full_goal_completion = open
is_goal_complete = false
```

The remaining gaps are still broad real-scene quality acceptance, full
compiled-adjoint trainer replacement, and high-dimensional camera-family Metal
atlas reuse.

## Verification

Commands run:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_camera_family_shared_work_scaling.py \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_camera_family_shared_work_scaling.py \
  tests/test_star_uvt_projective_goal_progress_audit.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_camera_family_shared_work_scaling.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
23 passed in 16.00s
```

The saved camera-family shared-work artifact verifies by CLI, and the saved
goal-progress artifact verifies against current inputs:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json \
  --verify-current-inputs
```

## Falsification Tests

The new report verifier rejects:

- missing `Q x Omega x T` theory contract,
- family payload/chart growth above near-constant thresholds,
- replay growth that no longer exposes linear replay cost,
- stale summaries,
- family UV residual above `0.50px`,
- per-q replay residual above `0.30px`.

The top-level goal-progress verifier additionally rejects camera-family shared
payload ratio above `0.30`, chart ratio above `0.15`, replay payload growth
below `4.0`, and family fit residual above `0.50px`.

## Implication

This supports the user's "rich math, not just fallback" preference. For a
revolving or nearby-family camera, the right answer is a camera-gauge/fiber
bundle atlas over the relevant base, with residual/error checks deciding when
to split the chart. Fallback remains a correctness rail for pathological strata,
not the primary explanation.
