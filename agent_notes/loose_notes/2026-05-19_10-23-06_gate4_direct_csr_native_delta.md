# Gate4 Direct CSR Native Delta

## Context

The prior vector-gather attempt proved that reshaping the existing Python/NumPy
sorted-depth helper was the wrong target: the gather grid was slower than the
simple row-fill loop. The next stronger fork was to bypass that helper
entirely when we already have Gate4 affine CSR coefficients and want emitted
packed endpoint records.

## Change

Added a native CPU op in the Lane2 fused-slab variant:

```text
gate4_delta_replace_packed_from_coeff_csr_cpu(...)
```

The op consumes:

- affine candidate CSR row offsets and ids
- per-candidate depth coefficients
- row-to-track mapping
- affine ray coefficients
- frame times
- site xyz/t/weight
- boundary-other-by-owner table

It emits the same 12 packed delta-replace tensors as the sorted/cut native
paths:

- base offsets/owner/left/right/packed-record
- track change offsets
- change frame/change offsets/owner/left/right/packed-record

The Python Gate4 endpoint-delta builder now takes this direct route only for
the single-slab, native sorted-delta, emitted-packed-record path. Existing
sorted/cut/Python fallbacks remain available.

## Verification

Build/visibility from this branch had already passed before this note:

```bash
rtk zsh -lc 'cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace'
```

Focused route assertion now proves the new op is actually used and matches the
packed Python oracle:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v
```

Result: `ok`.

Broader local gates:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `8 tests OK`.

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `1 test OK`.

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Result: `46 tests OK`.

Functional train/eval smoke:

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
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_directcsr_tensornative_sorted_emitted_smoke_2f_render16_site24.json
```

Result: `status=ok`; gradients nonzero, parameters updated, finite outputs.
Timing is not promotion evidence because the artifact reports a contended
environment, including a busy `ai_trader` CPU process and `MTLCompilerService`.

## Isolated Compiler Timing

In-process high-cap fixture comparison, same tape and same output parity:

```text
tracks=144 frames=16 candidates=38257 max_candidates=274
direct_csr             median 11.089 ms
sorted_native_chunk    median 26.215 ms
python_sorted_fallback median 69.733 ms
parity=ok
```

This is a real positive for the compiler/setup stage: direct-CSR avoids
materializing sorted depth/id chunks in Python and is roughly `2.4x` faster than
the previous native sorted-chunk path on this fixture, and roughly `6.3x`
faster than the Python sorted fallback.

## Interpretation

This is a keeper for setup/compiler cost, not a full WorldFoam speed fix. It
does not change the MPS replay kernel, so it does not by itself solve the
remaining frame-linear practical scaling. The next real target is still the
device replay path: consume owner/boundary/run records in a way that avoids
per-frame candidate replay, closer to STAR UVT's direct time-tube shader
contract.
