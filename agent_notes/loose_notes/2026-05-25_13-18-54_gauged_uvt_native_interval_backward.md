# Gauged UVT Native Family Interval Backward

## Context

This continuation picked up after the Q2 camera-family native interval-forward
row. The open technical gap was the matching native interval-cell backward/VJP:
the Metal interval renderer could already consume shared
`family_coeffs[N,9,B]` and `q_basis[Q,B]` for forward compositing, but the
gradient route still had to materialize per-q trace coefficient gradients and
fold them back outside the renderer.

## Current Model

The native family interval backward now mirrors the ordinary interval VJP
contract: tile membership, fallback decisions, and depth order are treated as
compiled constants. Gradients flow through opacity, time opacity, spatial
precision, color, and the u/v footprint-center coefficients, then chain into
`family_coeffs` and `q_basis` through:

```text
trace_coeff[q,n,k] = sum_b family_coeffs[n,k,b] * q_basis[q,b]
```

The VJP does not differentiate through the depth-order decision itself. That is
intentional and consistent with the existing interval renderer evidence.

## Implemented Evidence

New native report:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward/summary.json
```

Key saved metrics:

```text
q_axis_count = 5
q_pair_count = 25
trace_count = 2
family_basis_count = 6
batched_frames = 100
native_family_gradient_to_materialized_gradient_payload_ratio = 0.2926315789473684
native_family_coeff_gradient_to_materialized_gradient_payload_ratio = 0.11368421052631579
native_family_interval_backward_max_family_grad_rel_error = 2.3355269149760716e-06
native_family_interval_backward_max_q_basis_grad_rel_error = 8.51117079037067e-07
```

The report compares the native-family interval VJP against a materialized
single-launch interval VJP, then folds the materialized per-q coefficient
gradients into shared family and q-basis adjoints by the same chain rule. Base
opacity/color/time-opacity/spatial gradients are compared after summing the
materialized per-q slices.

## Goal-Progress Update

The top-level goal-progress audit now proves eighteen rows, not seventeen. The
new row is:

```text
local_camera_family_2d_metal_native_interval_backward
```

The current audit remains intentionally incomplete overall:

```text
proved_requirement_count = 18
open_requirement_count = 1
is_goal_complete = false
```

The remaining gap is no longer native forward/backward coefficient consumption.
It is broad real-scene/full-trainer acceptance and compression/reuse of
q-family tile/order metadata beyond the sampled-Q prototype.

## Verification

Focused Q2 Metal plus goal-progress matrix:

```text
70 passed in 5.25s
```

Saved-report verifiers:

```text
projective_camera_family_2d_native_interval_backward_report.py --verify-report .../summary.json
projective_goal_progress_audit.py --verify-report .../summary.json --verify-current-inputs
```

Both passed.

## Decision Implication

Future work should stop describing the Q2 Metal gap as "native interval
backward missing." The sharper next branch is q-family tile/order metadata:
can the compiler share or interpolate active-set/order records across camera
family coordinates, or does the sampled-Q tile/order index dominate memory once
the coefficient and VJP payloads are compressed?
