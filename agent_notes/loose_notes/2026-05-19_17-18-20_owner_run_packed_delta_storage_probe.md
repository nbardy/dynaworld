# Owner-Run Packed-Delta Storage Probe

## Context

After the lean owner-run fused-MSE recompute keeper, compute scaled well but selected tape storage still grew almost linearly with frame count. The current `owner-run-fused-mse-nomid` selected tape is literal sample CSR:

- `offsets_i32`
- `owners_i32`
- `lengths_f32`

That is still one materialized owner-run row per rendered sample. The next hypothesis was to port the cleaner STAR-style idea: keep stable per-track structure and store only changed frame rows.

## Change

Added packed-delta storage accounting to:

`research_experiments/world_foam_lane2/probe_owner_run_boundary_tape.py`

The new helper profiles owner-run endpoint records as:

- one base row per track
- per-track change offsets
- changed frame ids
- changed-row offsets
- packed `owner,left_cut_id,right_cut_id` records as one `i32`

This is probe-only. It does not change the Metal train/eval path yet.

Added focused accounting tests:

`research_experiments/world_foam_lane2/test_probe_owner_run_boundary_tape.py`

## Verification

Focused unit test:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
```

Result: 2/2 passed.

Syntax check:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/probe_owner_run_boundary_tape.py \
  research_experiments/world_foam_lane2/test_probe_owner_run_boundary_tape.py
```

Result: passed.

Scale probe:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_owner_run_boundary_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 24 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_owner_run_boundary_packed_delta_probe_render16_site24_2_4_8_16_v2.json
```

Result artifact:

`research_experiments/world_foam_lane2/results/2026-05-19_owner_run_boundary_packed_delta_probe_render16_site24_2_4_8_16_v2.json`

## Result

The packed-delta path is worth a shader fork.

| frames | current nomid CSR bytes | packed delta bytes | packed/current |
| --- | ---: | ---: | ---: |
| 2 | 19,028 | 15,668 | 0.823 |
| 4 | 44,012 | 33,012 | 0.750 |
| 8 | 90,684 | 59,296 | 0.654 |
| 16 | 183,668 | 90,220 | 0.491 |

Scale over `2f -> 16f`:

- frame count: `8.0x`
- materialized owner-run boundary/id CSR storage: `9.78x`
- current lean nomid owner/length CSR storage: `9.65x`
- packed endpoint owner-run delta storage: `5.76x`
- owner-run count: `10.11x`

The endpoint owner/count contract still matches the current owner-run tape, and boundary ids recover run lengths exactly:

- `matches_current_owner_run_counts_and_owners=true`
- `endpoint_ids_recover_run_lengths=true`
- `max_endpoint_length_abs_error=2.22e-16`

Depth is not parity-safe without more data:

- `endpoint_continuous_density_depth_matches_current_segment_mid_depth=false`
- max depth mismatch versus current segment-mid depth: `0.1276`

For the RGB-only fused-MSE nomid path, that depth mismatch is acceptable because the hot path only needs owner and length. For RGBA/depth replay, the packed endpoint tape would need internal moments/cuts or an explicit depth semantic change.

## Decision

Do not spend on `lengths_f16` or a dense fixed-cap row first. Those reduce bytes per materialized row but keep the bad row-count scale.

The next Metal fork should implement packed endpoint owner-run delta replay for the RGB-only fused-MSE nomid path:

- use base/change row tables instead of sample CSR
- use packed `owner,left_cut_id,right_cut_id` records
- reconstruct the active frame row by applying the latest change <= frame
- recompute segment length from boundary/ray coefficients
- compare loss and site gradients against `owner-run-fused-mse-nomid`
- then run the same 2/4/8/16 timing ladder

This is the first storage result that moves WorldFoam toward the STAR UVT cleanliness story instead of merely trimming the existing CSR tape.
