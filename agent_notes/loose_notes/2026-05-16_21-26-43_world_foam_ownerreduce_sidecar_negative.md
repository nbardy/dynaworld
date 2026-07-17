# WorldFoam Owner-Reduce Sidecar Negative

## Context

The practical question was whether WorldFoam is only sublinear in theory, while
STAR UVT is actually sublinear in practice. The current promoted WorldFoam path
still remains:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse
```

The clean accepted render32/site12 artifact is still sublinear at the narrow
shader scope: from 16f to 128f, total step scales `1.464x`, backward scales
`1.536x`, and selected storage scales `1.026x` for an `8x` frame-count increase.
The STAR UVT direct-atomic smoke is cleaner at its own scope: from 2f to 32f,
step time scales `2.071x` and render time scales `2.357x` for a `16x`
frame-count increase.

The gap is not the representation math. It is the current WorldFoam hot-kernel
shape: scattered owner/site gradient reduction, i16x3 replay decode, and MPS
outlier sensitivity still make the practical speed surface fragile.

## Sidecar Implemented

Added a separately named owner-reduce op and train/eval mode instead of mutating
the promoted ABI:

```text
endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse
```

The producer builds:

```text
track_chunk_owner_offsets_i32
track_chunk_owner_i16
```

The Metal sidecar loads up to 16 unique owner ids for each track/chunk. If the
site count is <=16 it uses the existing small-site reduction. If the owner list
is valid and <=16, it reduces per-frame gradients by owner slot and flushes one
global atomic per owner. Otherwise it falls back to direct atomics.

## Verification

Build:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Syntax:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
```

Focused MPS parity passed:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_above_reduce_cap_128_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_multitrack_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk -v
```

Broader gates passed:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

```text
Ran 92 tests in 0.977s OK
```

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_world_foam_frozen_rgb_mse_objective -q
```

```text
Ran 5 tests in 0.014s OK
```

## Result

Tiny end-to-end smoke passed but was cold/slow:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_fused_mse_smoke_render16_site20_16.json
16f: total 151.446 ms, backward 25.057 ms, status ok
```

Warmed 16/32 scale failed:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_fused_mse_warm1_steps3_render16_site20_16_32.json
status failed
16f: total 4.360 ms, backward 3.651 ms, storage 859596 bytes
32f: total 341.620 ms, backward 279.870 ms, storage 862138 bytes
total scale 78.36x for a 2x frame-count increase
backward scale 76.65x for a 2x frame-count increase
storage scale 1.003x
```

Interpretation: owner-reduce is still theoretically attractive because it keeps
storage flat and reduces global atomic count on paper, but this Metal sidecar is
not practical. It reproduces the old failure mode in a safer form: correct
parity, terrible 32f timing. Do not promote it.

## Takeaway

WorldFoam is sublinear in the clean promoted fixed-geometry shader gate, but not
yet STAR-UVT-clean as a practical speed surface. STAR UVT has a simpler temporal
representation and smoother frame-count runtime evidence. WorldFoam still needs
a different kernel layout, not another owner-list slot scan inside the current
i16x3 replay loop.
