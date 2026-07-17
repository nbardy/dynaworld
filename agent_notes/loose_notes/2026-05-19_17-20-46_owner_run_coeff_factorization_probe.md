# Owner-Run Coeff Factorization Probe

## Context

After the packed-delta owner-run mode passed RGB-MSE moving-ray parity, the
resident tape was still dominated by `delta_coeff_f16`. At the 24-site shape,
the topology table is no longer the main storage problem; the dense
`[track, boundary, 4]` coefficient table is.

## What Changed

Added a CPU storage/math probe:

- `research_experiments/world_foam_lane2/probe_owner_run_coeff_factorization.py`
- `research_experiments/world_foam_lane2/test_probe_owner_run_coeff_factorization.py`

The probe tests the STAR-like factorization:

- store boundary planes `[boundary, nx, ny, nz, nt, b]`
- store per-track linear ray coefficients `[origin_base, origin_slope, direction_base, direction_slope]`
- reconstruct the same rational cut-depth coefficients in the shader instead
  of materializing dense `[track, boundary, 4]` coeff rows

## Evidence

Unit test:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization -v
```

Result:

```text
Ran 2 tests in 0.062s
OK
```

Real-shape probe:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/probe_owner_run_coeff_factorization.py --frame-counts 2,4,8,16 --render-size 16 --site-count 24 --out-json research_experiments/world_foam_lane2/results/2026-05-19_owner_run_coeff_factorization_probe_render16_site24_2_4_8_16.json
```

Result:

- `status`: process exit `0`
- `factorized_coefficients_match_dense_coefficients=true`
- `factorized_depth_matches_dense_coefficients=true`
- `factorized_f32_storage_below_current_coeff16_at_max_frame=true`
- `factorized_f32_storage_sublinear_vs_frames=true`
- `max_factorized_f32_vs_current_coeff16=0.02662`
- `max_depth_error_vs_dense_factorized=7.14e-5`
- `max_validity_mismatches_vs_dense_factorized=0`

At every `2/4/8/16f` row:

- current dense coeff16 bytes: `1,130,496`
- factorized f32 boundary+track bytes: `30,096`
- factorized f16 boundary+track bytes: `15,048`

The f32 factorized table is `2.66%` of the current coeff16 table. Combined with
the packed owner-run topology from the previous probe, this is the first storage
shape that looks STAR-like for WorldFoam: roughly `90KB` packed topology plus
`30KB` factorized coefficients at `16f/site24`, before accounting for the small
config/frame arrays.

## Caveats

The current Metal packed-delta path uses coeff16. The factorized f32 form
matches the dense f32 coefficient formula, not the rounded coeff16 table. That
is probably the right direction, since coeff16 introduced validity mismatches
against factorized f32 in this probe, but the next shader must compare
loss/site gradients against `owner-run-fused-mse-nomid`, not only against the
existing coeff16 packed path.

Timing is still unknown. Recomputing numerator/denominator from boundary planes
and track coefficients trades memory for extra FMAs per cut. The next fork
should be a real Metal kernel variant:

- inputs: `boundary_f32`, `track_ray_coeff_f32`, packed base/change records,
  delta offsets, `frame_t_f32`, site RGBA, target RGB
- no resident `delta_coeff_f16`
- parity gate against `owner-run-fused-mse-nomid`
- clean `2/4/8/16f` train/eval ladder with `--require-benchmark-environment-ok`
