# Gate4 Fast-Skip Vector Replay Cleanup

## Context

The clean benchmark preflight is still blocked by unrelated `ai_trader`
processes, so I did not run or claim a clean timing promotion. I continued the
deterministic setup/compiler cleanup after bulk CSR materialization.

The remaining per-track `sample_validation="skip"` path still bounced through
Python structures after bulk CSR:

- `row_index` was built by a nested Python `view/y/x` loop even though
  `layout="per-track"` means `row == track`.
- `fast_offsets_np` and `fast_counts_np` were converted to Python lists only to
  be converted back into tensors.
- skipped-validation `candidate_replay_iterations` still looped row/slab in
  Python even though per-track rows have one track each.

## Change

In `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`:

- For `layout="per-track"` + `sample_validation="skip"`, `row_index` now stays
  as `np.arange(track_count, dtype=np.int32)` until final tensorization.
- The fast CSR counts stay as NumPy and become the final `counts` tensor through
  `torch.from_numpy(...)`.
- `candidate_replay_iterations` is computed from
  `fast_counts_np.reshape(row_count, time_slabs)` times vectorized slab frame
  counts instead of a Python row/slab loop.

This is a small cleanup, not a shader promotion. It removes remaining host-side
list churn in the fast per-track endpoint-record path.

## Validation

Syntax:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
```

Focused high-cap parity:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v
```

Result: passed in `0.628s`.

Full focused suites:

- Gate4 compiler unit: `8/8`
- framegroup16 promotion wrapper unit: `46/46`

Integrated smoke:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 24 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_fastskip_vectorreplay_tensornative_sorted_emitted_smoke_2f_render16_site24.json
```

Result: `status=ok`, nonzero gradients, parameter update, finite outputs, and
the native sorted-delta/emitted-pack flags were effective. The train setup split
was:

- `build_endpoint_record_sequences_s`: `0.0343s`
- `build_gate4_affine_endpoint_tape_s`: `0.0231s`
- `build_gate4_endpoint_delta_replace_tape_s`: `0.0112s`

The artifact is still `benchmark_environment.status=contended`, so these timing
fields are smoke context only.

## Next

The next real shader fork should target the frame-linear candidate replay term,
not atomic count alone. Track-MSE reduced atomics but lost too much parallelism;
the more STAR-like direction is an owner-run/site-pair or boundary-pair record
that prevents rescanning and re-ownering candidates for each frame.
