# WorldFoam Producer Owner-Reduce Negative

## Context

We tested whether the framegroup16 delta-replace fused-MSE kernel could replace
the current small-site reduction with producer-side per-track/chunk owner lists.
The topology probe made this look plausible: sampled real site64 topology had
owner-reduce/current atomic ratios near `0.09 -> 0.03` from 16f to 128f, with
fallback fraction `0.0`.

## Implementation Tried

- Added `build_delta_replace_chunk_owner_lists(...)` in
  `research_experiments/world_foam_lane2/probe_endpoint_record_delta_replay.py`.
- Temporarily threaded owner-list tensors through the promoted i16x3 framegroup16
  op ABI: Python wrapper, autograd adapter, train/eval tape, C++ binding, and
  Metal buffers.
- Metal implementation loaded up to 16 unique owner ids per track/chunk and
  reduced gradients by owner slot instead of by raw site id.

## Result

The implementation was correct on focused parity, but not viable for train/eval.

Passed while the producer-owner ABI was active:

```text
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk -v
```

Negative speed evidence:

- Tiny real train/eval with producer-owner ABI, `render16/site4/16f/steps1/warm0`,
  completed but was cold and slow: about `215 ms` total, `187 ms` backward.
- Isolated synthetic op smoke did not hang but was not a win:
  `research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_variant_timing_producer_ownerreduce_smoke_16.json`
  reported `i16x3_framegroup32_lossreduce` at `4.10 ms` on the tiny 2-track case.
- Larger real train/eval startup was slow enough that the 16/32 probe was killed
  before producing a row.

The owner-list ABI was removed from the promoted op. The current promoted
framegroup16 path is back to the small-site/loss-reduce implementation.

## Revalidation After Rollback

Build:

```text
PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace
```

Tests:

```text
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk -v
```

```text
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_world_foam_frozen_rgb_mse_objective -q
```

```text
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

All passed: focused parity, 5 objective tests, and 88 lane tests.

Fresh speed artifacts after rollback:

- `research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_revert_ownerreduce_warm3_steps5_render32_site12_16.json`
  - 16f: total mean `4.206 ms`, backward mean `3.811 ms`.
- `research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_current_revalidated_warm3_steps5_render32_site12_16_32_64_128.json`
  - status `failed` because warm3/steps5 caught MPS outliers.
  - 16f clean: total mean `3.270 ms`, backward mean `2.879 ms`.
  - 32f median clean: total median `4.517 ms`, backward median `3.912 ms`; mean was blown up by one `333 ms` backward outlier.
  - 64f still reasonable: total mean `9.009 ms`, backward mean `7.958 ms`; median total `4.959 ms`.
  - 128f row was contaminated by repeated large outliers and should not be used as the current acceptance proof.
- `research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_current_revalidated_128only_warm10_steps20_render32_site12.json`
  - 128f clean confirmation: total mean `4.628 ms`, total median `4.563 ms`, backward mean `4.096 ms`, backward median `4.051 ms`, max backward `4.584 ms`.
  - storage `1,357,262` bytes; heldout PSNR `14.686`.

## Takeaway

The math still says producer-side owner lists could reduce atomics, but the
current hot-kernel ABI/slot scan implementation is not the right way to make it
competitive. The safe current promoted path remains the existing
rowref/small-site/loss-reduce framegroup16 fused-MSE kernel. Future owner-reduce
work should be a separately named op/mode, not a mutation of the promoted op,
and should be accepted only after warm10/steps20 128-only and full scale probes
avoid MPS outlier contamination.
