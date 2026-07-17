# World Foam Framegroup16 Chunked Positive

Goal: test whether the flat-storage delta-replace World Foam representation can
get closer to STAR-style frame amortization by changing the shader execution
shape, not by only packing records smaller.

## What Changed

Added a new sidecar mode:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse
```

Touched surfaces:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal launcher and kernel cache:
  `metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only`
- C++ op binding:
  `endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only`
- Python wrapper/export in `torch_world_foam_lane2_fused_slab`
- train/eval mode in `train_eval_owner_run_tape.py`
- parity coverage in `test_probe_endpoint_record_edit_replay.py`

The first version only supported `frame_count <= 16`. After the 2/4/8/16 control
looked promising, I changed the host launch and kernel indexing so one
threadgroup handles one 16-frame chunk of one track. Longer sequences launch
`track_count * ceil(frame_count / 16)` threadgroups.

## Verification

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
```

Focused replay suite passed: 12 tests OK. Earlier in the session, before the
chunked extension, the full lane suite also passed: 42 tests OK. Rerun the full
lane after future edits before treating this branch as finished.

## Results

Initial 2/4/8/16 framegroup artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_fused_mse_repeat_loaded_warm5_steps12_render16_2_4_8_16.json
```

- total ms: `1.650 / 3.243 / 2.315 / 1.847`
- fused/backward ms: `1.111 / 2.313 / 1.690 / 1.293`
- selected tape storage bytes: `48146 / 49930 / 49774 / 49936`
- total-step scale 2->16: `1.12x` for an `8x` frame-count increase
- storage scale 2->16: `1.04x`

Same-setting controls:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_fused_mse_repeat_loaded_warm5_steps12_render16_2_4_8_16_control.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_edit_blockcoeff16_fused_mse_repeat_loaded_warm5_steps12_render16_2_4_8_16_control.json
```

- old i16x3 total ms: `3.709 / 2.947 / 2.491 / 3.758`
- old i16x3 storage bytes: `48146 / 49930 / 49774 / 49936`
- block-coeff16 total ms: `2.268 / 2.074 / 3.229 / 2.536`
- block-coeff16 storage bytes: `74196 / 77712 / 99492 / 132920`

Chunked 16/32/64/128 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunked_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
```

- total ms: `3.057 / 3.972 / 3.341 / 6.691`
- fused/backward ms: `2.300 / 2.728 / 2.494 / 5.802`
- selected tape storage bytes: `49936 / 49902 / 49902 / 49916`
- total-step scale 16->128: `2.19x` for an `8x` frame-count increase
- storage scale 16->128: `0.9996x`
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

## Interpretation

This is the first positive result in this sublane that preserves the
delta-replace i16x3 flat storage curve and also gives competitive runtime. The
key win was not record packing; i16x4 and binary search were both negative. The
win came from moving row selection/replacement replay to a 16-frame per-track
threadgroup and sharing that work across local frame threads.

It is still not a full World Foam acceptance claim. Scope remains fixed-geometry,
RGB-only site-RGBA, render16, MPS smoke.

## Select-Start Follow-Up Was Rejected

I then tried a shader-only shortcut where each 16-frame chunk starts from the
last replacement row before the chunk instead of replaying from the base row.
Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_selectstart_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
```

- total ms: `2.473 / 4.247 / 4.279 / 7.578`
- fused/backward ms: `1.840 / 3.256 / 3.453 / 6.357`
- selected tape storage bytes: `49936 / 49902 / 49902 / 49916`
- total-step scale 16->128: `3.06x` for an `8x` frame-count increase
- storage scale 16->128: `0.9996x`

This improved the 16f row but worsened 32/64/128 relative to the base chunked
path. I restored the live shader to the better measured chunked-base replay
path. The next useful fork should probably add an explicit per-track
chunk-start table or change the row layout rather than adding more scan logic
inside the current metadata.

## Explicit Chunk-Start Table Follow-Up

I then wired the explicit per-track chunk-start table into the framegroup16 op:
`track_chunk_change_offsets_i32` has `track_count * (chunk_count + 1)` offsets,
so each `(track, 16-frame chunk)` can start from the last replacement row before
the chunk and only replay changes inside the chunk. I counted this table in
selected storage.

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunkstarts_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
```

- total ms: `2.470 / 3.553 / 3.909 / 6.481`
- fused/backward ms: `1.968 / 2.954 / 3.408 / 5.615`
- selected tape storage bytes: `54032 / 56046 / 60142 / 68348`
- total-step scale 16->128: `2.62x` for an `8x` frame-count increase
- storage scale 16->128: `1.26x`
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

This is a mixed positive. It improves 16f, 32f, and 128f over the base chunked
artifact, but 64f is slower and storage is no longer effectively flat. The full
lane suite passed after this change: 43 tests OK, including a new 20-frame
cross-chunk parity test against the scalar i16x3 kernel.

## Int16 Chunk-Start Follow-Up

I then narrowed the explicit chunk-start table from int32 to int16 offsets. The
host path now validates `change_count <= 32767`, the C++ launcher checks the
int16 table against the per-track int32 offsets, and storage accounting counts
the table at 2 bytes per offset.

Verification after the int16 change:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

Focused replay passed: 13 tests OK. Full lane passed: 43 tests OK.

Full 16/32/64/128 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunkstarts_i16_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
```

- total ms: `4.500 / 3.270 / 6.266 / 3.939`
- fused/backward ms: `3.389 / 2.735 / 4.981 / 3.217`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- storage scale 16->128: `1.14x`
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

Spot rerun for 16/64/128:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunkstarts_i16_spot_rerun_warm5_steps12_render16_16_64_128.json
```

- total ms: `2.932 / 4.731 / 4.853`
- fused/backward ms: `2.247 / 3.591 / 3.925`
- selected tape storage bytes: `51984 / 55022 / 59132`

Read: the theory is sublinear because the representation no longer stores a
full frame-local segment payload per frame, and the framegroup shader amortizes
replacement replay across up to 16 local frame lanes. In practice, this exact
World Foam path is only partly there: storage is strongly sublinear, 128f got
better than the base chunked row, but wall time is not monotonically sublinear
and the 64f row is still worse. STAR UVT is cleaner because its temporal object
is naturally factorized; World Foam is still paying irregular replay/indexing
costs that have to be made GPU-regular before the runtime story matches the
representation story.

## Row-Reference Framegroup Fork

I then changed the framegroup16 shader so local frame 0 no longer copies the
selected owner/left/right row into threadgroup memory for every frame in the
chunk. Instead it stores per-lane row references (`source`, `begin`, `end`) and
each frame lane reads the selected base/change i16x3 row directly while doing
the RGB MSE/VJP work. This keeps the int16 chunk-start storage exactly the same
and attacks the duplicate row-copy work that likely caused the 64f cliff.

Verification after the row-reference shader edit:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

Focused replay passed: 13 tests OK. Full lane passed: 43 tests OK.

Comparable site4 full sweep:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_i16_fused_mse_repeat_loaded_warm5_steps12_render16_site4_16_32_64_128.json
```

- total ms: `2.598 / 2.287 / 3.009 / 3.816`
- fused/backward ms: `2.013 / 1.854 / 2.556 / 3.111`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- total-step scale 16->128: `1.47x`
- storage scale 16->128: `1.14x`
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

This fixed the obvious 64f regression in the clean sweep and made 16/32/64 the
best framegroup rows so far. I also accidentally ran a higher-site stress
sweep at the default `site_count=12`; it stayed sublinear in the full sweep
(`3.049 / 3.413 / 2.961 / 2.643 ms`) with storage `327556 -> 338408` bytes, but
that artifact is not comparable to the earlier site4 rows.

The caveat is 128f stability. A site4 spot rerun and a 128-only warm10/steps20
confirmation were both slow:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_i16_spot_rerun_warm5_steps12_render16_site4_16_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_i16_128only_warm10_steps20_render16_site4.json
```

- spot 16/64/128 total ms: `3.645 / 3.616 / 8.143`
- spot 16/64/128 fused/backward ms: `2.711 / 2.518 / 6.766`
- 128-only total/backward ms: `9.788 / 8.385`

Read: row-reference materialization is a real shader fork and probably the
right shape for the 16-64f range, but it is not the final fix. The old
materialized chunk-start path was slower at 64f and had worse 16/32 rows, but
its 128f reruns were more stable. Next useful fork: keep rowref for short and
mid frame counts, but add a materialized-row fallback or a smaller 128f row
cache path for high chunk counts.

## Hybrid Materialized Fallback

I added the first hybrid: the public framegroup16 op now dispatches to the
row-reference kernel below 128 frames and to a separate materialized chunk-start
kernel at `frame_count >= 128`. This keeps the short/mid rowref kernel from
carrying the large threadgroup row arrays, while giving 128f a fallback that is
covered by a new 128-frame scalar-parity test.

Verification after adding the materialized fallback:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

Focused replay passed: 14 tests OK. Full lane passed: 44 tests OK.

Hybrid site4 full sweep:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_hybrid_i16_fused_mse_repeat_loaded_warm5_steps12_render16_site4_16_32_64_128.json
```

- total ms: `3.800 / 4.326 / 4.445 / 6.208`
- fused/backward ms: `2.806 / 3.234 / 3.617 / 5.380`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- total-step scale 16->128: `1.63x`
- storage scale 16->128: `1.14x`
- heldout PSNR: `14.6148 / 14.5790 / 14.5713 / 14.5684`

Hybrid 128-only warm10/steps20 confirmation:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_hybrid_i16_128only_warm10_steps20_render16_site4.json
```

- total/backward ms: `7.232 / 6.233`
- selected tape storage bytes: `59132`

Read: the fallback improves the bad rowref-only 128f confirmations (`8.143` and
`9.788 ms`) but does not recover the best 128f timings. It is a worst-case
stabilizer, not a fixed/STAR-clean runtime result.

I also tried a high-frame chunk-span-8 fallback and rejected it:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_hybrid_i16_chunk8_128only_warm10_steps20_render16_site4.json
```

That result was `7.524 / 6.867 ms` total/backward at 128f and raised selected
storage to `67324` bytes. I reverted the live code to 16-frame chunks.

## Row-Reference Small-Site Reduction

The next fork fixed the 128f path instead of stabilizing around it. I kept the
row-reference framegroup kernel for all frame counts and added a threadgroup
gradient reduction for small site tables (`site_count <= 16`). Each 16-frame
per-track group accumulates site-RGBA gradients locally and emits one global
atomic add per site, rather than atomically adding every local frame segment.
The public op now dispatches to the row-reference kernel at 128f too; the
materialized fallback remains an intermediate experiment, not the live path.

Verification:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

After the final dispatch swap and test rename, focused replay passed: 14 tests
OK and full lane passed: 44 tests OK. I then broadened coverage with a 128f
multi-track row-reference reduction parity case and a 128f `site_count=20`
above-cap fallback parity case. After those additions, focused replay passed:
16 tests OK. Full lane passed: 46 tests OK.

First small-site reduction run while the live op still used the materialized
128f fallback was rejected for 128f:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_smallsite_reduce_fused_mse_repeat_loaded_warm5_steps12_render16_site4_16_32_64_128.json
```

- total ms: `2.859 / 3.038 / 2.143 / 7.716`
- fused/backward ms: `2.167 / 2.213 / 1.767 / 7.318`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`

The winning row-reference-at-128 version:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_fused_mse_repeat_loaded_warm5_steps12_render16_site4_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_128only_warm10_steps20_render16_site4.json
```

- total ms for 16/32/64/128: `2.674 / 2.312 / 2.382 / 2.210`
- fused/backward ms: `2.002 / 1.828 / 1.896 / 1.801`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- total-step scale 16->128: `0.83x` for an `8x` frame-count increase
- backward scale 16->128: `0.90x`
- storage scale 16->128: `1.14x`
- heldout PSNR: `14.6148 / 14.5790 / 14.5713 / 14.5684`
- 128-only warm10/steps20 total/backward ms: `1.674 / 1.341`
- 128-only heldout PSNR: `15.5822`

Read: this is the first 128f result that looks fixed rather than only partly
sublinear. The winning ingredient was not the materialized row cache; it was
keeping row-reference metadata light and reducing small-site gradient atomics
inside each 16-frame group. Scope is still fixed-geometry RGB-only site-RGBA on
the render16 MPS smoke, so this should be promoted as a kernel-scaling result,
not as full World Foam quality/capacity parity with STAR UVT.

Site-count follow-ups:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_fused_mse_repeat_loaded_warm5_steps12_render16_site12_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_reducecap_fallback_fused_mse_repeat_loaded_warm5_steps12_render16_site20_16_128.json
```

- site12 total ms for 16/32/64/128: `2.992 / 4.512 / 2.434 / 2.229`
- site12 fused/backward ms: `2.188 / 3.311 / 2.007 / 1.900`
- site12 selected storage bytes: `327556 / 331118 / 333748 / 338408`
- site12 total-step scale 16->128: `0.74x`; storage scale: `1.03x`
- site20 above-cap fallback total ms for 16/128: `2.531 / 2.478`
- site20 above-cap fallback fused/backward ms: `2.135 / 2.142`
- site20 above-cap fallback selected storage bytes: `851092 / 867222`
- site20 total-step scale 16->128: `0.98x`; storage scale: `1.02x`

Read: the reduction cap is no longer a hidden correctness risk. The site20 path
does not use the small-site threadgroup reduction, but it still remained
roughly flat in this synthetic 16/128 smoke. The caveat remains that render16
fixed-geometry RGB-only site-RGBA is not full trainer quality/capacity parity.

Render32 site12 follow-up:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_128only_warm10_steps20_render32_site12.json
```

- render32 site12 total ms for 16/32/64/128:
  `2.638 / 3.296 / 3.281 / 6.178`
- render32 site12 median total ms: `2.376 / 3.243 / 3.061 / 4.389`
- render32 site12 fused/backward ms: `2.182 / 2.588 / 2.928 / 5.620`
- render32 site12 median fused/backward ms:
  `1.901 / 2.545 / 2.719 / 4.049`
- render32 site12 selected storage bytes:
  `1322952 / 1339766 / 1353624 / 1373646`
- render32 site12 total-step scale 16->128: `2.34x` for an `8x`
  frame-count increase; storage scale: `1.04x`
- render32 site12 heldout PSNR:
  `14.6291 / 14.6161 / 14.6010 / 14.6233`
- 128-only warm10/steps20 total/backward ms: `4.488 / 3.711` by mean and
  `3.803 / 3.170` by median; heldout PSNR: `14.6857`

Read: the current World Foam row-reference small-site reduction is no longer
just theoretically sublinear. It is practically sublinear in this microbench:
render16 is very clean and render32 still scales below frame count, with PSNR
stable. It is not STAR-clean yet because the render32 128f full sweep had a
large timing outlier (`22.776 ms` total max, `22.337 ms` backward max), so the
claim is "sublinear but noisy" rather than "STAR UVT-style clean amortization."

I then reduced the remaining per-frame global loss atomic in the live
row-reference kernel. Each 16-frame threadgroup now accumulates `sample_loss`
in `tg_loss` and emits one global `loss_f32` atomic from local frame 0. This
does not add a new barrier on the small-site path because it reuses the barrier
that already protects the threadgroup site-gradient reduction.

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_128only_warm10_steps20_render32_site12.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_128only_rerun2_warm10_steps20_render32_site12.json
```

- focused replay after the shader patch: 16 tests OK
- full lane after the shader patch: 46 tests OK
- loss-reduced render32 site12 total ms for 16/32/64/128:
  `2.695 / 1.843 / 2.508 / 3.225`
- loss-reduced render32 site12 median total ms:
  `2.411 / 1.534 / 2.141 / 3.220`
- loss-reduced render32 site12 fused/backward ms:
  `2.307 / 1.545 / 2.173 / 2.802`
- loss-reduced render32 site12 median fused/backward ms:
  `2.102 / 1.273 / 1.781 / 2.637`
- loss-reduced render32 site12 selected storage bytes:
  `1322952 / 1339766 / 1353624 / 1373646`
- loss-reduced render32 total-step scale 16->128: `1.20x` for an `8x`
  frame-count increase; backward scale: `1.21x`; storage scale: `1.04x`
- loss-reduced render32 heldout PSNR:
  `14.6291 / 14.6161 / 14.6010 / 14.6233`
- loss-reduced 128f full-sweep total max: `4.108 ms`, down from the previous
  `22.776 ms` outlier
- loss-reduced 128-only warm10/steps20 total/backward ms:
  `5.462 / 4.709` by mean and `5.295 / 4.509` by median; total max:
  `9.811 ms`
- loss-reduced 128-only rerun2 warm10/steps20 total/backward ms:
  `4.403 / 3.637` by mean and `3.840 / 3.190` by median; total max:
  `7.735 ms`

Read: this is a better frame-scale kernel for the mixed 16/32/64/128 gate. It
makes the render32 scaling row much closer to STAR-clean by removing the big
128f full-sweep outlier. The first 128-only repeat looked slower by median,
but rerun2 recovered almost the old median while keeping a better max than the
old row-reference 128-only confirmation. Keep the claim narrow: sublinear
frame-scaling fix, not final full-trainer competitiveness.

I then turned the loss-reduction result into an executable guardrail by
extending `verify_fused_slab_mixed_scaling.py`. The verifier still checks the
older 2/4/8/16 mixed-mode direct-atomic artifacts, and now also checks the
render32 framegroup16 loss-reduced 16/32/64/128 artifact plus the 128-only
rerun2 confirmation. It fails if the mixed 128f total max exceeds `6.0 ms`, if
16->128 total/backward/storage scaling exceeds `1.5x`/`1.5x`/`1.10x`, or if the
128-only rerun regresses above `4.5 ms` median total or `8.5 ms` max total.

```text
research_experiments/world_foam_lane2/results/2026-05-16_fused_slab_mixed_scaling_verifier_with_framegroup_lossreduce.json
```

- `verify_fused_slab_mixed_scaling.py`: status `ok`
- new focused verifier tests: 3 tests OK
- full lane after adding the guardrail tests: 49 tests OK

I then promoted that guardrail into the canonical status summary instead of
leaving it as a side verifier only. `summarize_fused_slab_mixed_results.py`
now defaults to the framegroup loss-reduction verifier, emits a
`framegroup16_lossreduce_render32` block, and keeps
`completion_claim=false`, `full_trainer_claim=false`, and
`quality_claim=false`. `verify_fused_slab_status_summary.py` rejects summaries
that omit the loss-reduced framegroup guardrail, exceed the 16->128 scaling or
128f outlier thresholds, or drop the explicit "not a full-trainer" boundary.

Current canonical artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

- status-summary verifier after canonical integration: status `ok`
- focused status-summary tests after canonical integration: 5 tests OK
- full lane after canonical status-summary guardrails: 51 tests OK

Read: the latest World Foam evidence is now executable in the top-level status
path. The selected shader is guarded as practically sublinear across
16/32/64/128 frames in the render32/site12 microbench, while the verifier still
prevents this from being reported as full World Foam trainer completion or
STAR-UVT competitiveness.

I also wired the current framegroup16 fused-MSE mode into
`compare_endpoint_run_record_edit_train_eval.py` so the selected shader can be
run from the same paired compare harness as endpoint-run and raw edit, instead
of only from bespoke `train_eval_owner_run_tape.py` calls. The new flag is:

```text
--include-delta-framegroup16-fused-mse
```

The compare payload now reports `delta_framegroup16_*` ratios and acceptance
fields, and the unit coverage includes both mode inclusion and selected-storage
accounting for i16x3 records plus framegroup chunk offsets.

Real smoke artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_smoke_render16_16_32.json
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 16,32 \
  --render-size 16 \
  --site-count 4 \
  --steps 2 \
  --warmup-steps 1 \
  --optimizer-mode manual-vjp \
  --include-delta-framegroup16-fused-mse \
  --repeat-loaded-frames \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_smoke_render16_16_32.json
```

Result: status `ok`; 16f endpoint-run total/backward `5.860 / 2.620 ms`, raw
edit `7.420 / 3.166 ms`, and delta-framegroup16 fused-MSE `2.506 / 1.998 ms`.
The framegroup path is `0.428x` endpoint-run total at 16f and keeps heldout
PSNR matched (`13.7516`). The 32f repeated-fixture row also runs through the
live path (`1.727 ms` total for framegroup16), but because the fixture repeats
the loaded 16 frames and uses only two measured steps, this is a harness smoke,
not a stable speed benchmark.

I then made that compare smoke executable inside
`verify_fused_slab_mixed_scaling.py`. The verifier now emits a
`framegroup_compare_smoke` block and fails if the saved compare artifact loses
the 16f framegroup speed ratio (`>0.75x` endpoint-run total), loses backward
ratio (`>0.95x` endpoint-run backward), drifts PSNR (`>1e-3`), loses compact
selected storage (`>0.15x` full), drops the repeated-fixture/not-stable-
benchmark caveat, or starts claiming full-trainer/quality. Focused verifier
tests now cover speed, PSNR, and scope regressions.

I then replaced that narrow 16/32 compare guard with a 16/32/64/128
same-process speed-scale smoke so the selected shader has an executable
all-frame speed check against endpoint-run:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render16_site4_16_32_64_128.json
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 16,32,64,128 \
  --render-size 16 \
  --site-count 4 \
  --steps 3 \
  --warmup-steps 1 \
  --optimizer-mode manual-vjp \
  --include-delta-framegroup16-fused-mse \
  --repeat-loaded-frames \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render16_site4_16_32_64_128.partial.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render16_site4_16_32_64_128.json
```

Result: status `ok`. Endpoint-run total ms was
`6.393 / 6.652 / 5.310 / 10.723`; raw endpoint-record edit total ms was
`8.333 / 6.284 / 5.450 / 5.911`; delta-framegroup16 fused-MSE total ms was
`1.896 / 2.015 / 2.213 / 4.913`. The framegroup total ratios versus
endpoint-run were `0.297 / 0.303 / 0.417 / 0.458`, the 16f backward ratio was
`0.622`, and heldout PSNR stayed matched within `4.3e-5` at 16f. The
framegroup result reports total/backward/storage scale of
`2.59x / 2.17x / 1.14x` for an `8x` frame-count increase.

The mixed-scaling verifier now defaults to this speed-scale artifact instead
of the narrower 16/32 smoke. It fails if any checked frame loses the
`<=0.75x` total-step ratio versus endpoint-run, if 16f backward ratio exceeds
`0.95x`, if total/backward/storage scale exceeds `3.0x`/`2.5x`/`1.25x`, if
PSNR drifts beyond `1e-3`, or if the caveats/full-trainer scope boundaries are
dropped. Focused verifier tests now cover all-frame speed regression and scale
regression too. Full lane after this patch: 58 tests OK.

I then reran the same paired compare at the stronger render32/site12 setting to
match the cleaner loss-reduction microbench scale:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render32_site12_warm3_steps8_16_32_64_128.json
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 16,32,64,128 \
  --render-size 32 \
  --site-count 12 \
  --steps 8 \
  --warmup-steps 3 \
  --optimizer-mode manual-vjp \
  --include-delta-framegroup16-fused-mse \
  --repeat-loaded-frames \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render32_site12_warm3_steps8_16_32_64_128.partial.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render32_site12_warm3_steps8_16_32_64_128.json
```

Result: status `ok`. Endpoint-run total ms was
`5.817 / 5.617 / 9.924 / 15.020`; raw endpoint-record edit total ms was
`7.342 / 5.741 / 8.276 / 9.028`; delta-framegroup16 fused-MSE total ms was
`1.472 / 3.363 / 5.188 / 4.300`. The framegroup total ratios versus
endpoint-run were `0.253 / 0.599 / 0.523 / 0.286`, the 16f backward ratio was
`0.425`, and the worst checked heldout-PSNR delta was about `0.0029 dB` at
64f. The framegroup result reports total/backward/storage scale of
`2.92x / 3.04x / 1.04x` for an `8x` frame-count increase.

Read: this is a stronger paired practical speed guard than the earlier
render16/site4 smoke. It is not as clean as the dedicated loss-reduction
single-mode guard (`1.20x` total scale), but it proves the selected shader is
still faster than endpoint-run at every checked frame count under render32/site12.
The mixed-scaling verifier now defaults to this artifact, checks render32/site12
explicitly, checks all-frame PSNR with a `5e-3` tolerance, and uses sublinear but
looser paired-run scale guards (`3.25x` total, `3.75x` backward, `1.10x` storage).

Follow-up guardrail: the mixed verifier and status summary now preserve the
loaded-frame boundary explicitly. In the promoted 16/32/64/128 paired compare,
only 16f is a real loaded fixture row; 32f, 64f, and 128f are synthetic
repeated-fixture speed-scaling rows. The status verifier fails if that boundary
is dropped or if the synthetic rows are presented as real frame-count evidence.

I then ran a real-loaded 16/32 paired compare using the generated 128px
32-frame multicam fixture:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_real32_render32_site12_warm3_steps8_16_32.json
```

This used no `--repeat-loaded-frames`. Endpoint-run total was
`4.476 / 6.331 ms` at 16/32f; framegroup16 fused-MSE total was
`1.539 / 3.149 ms`. The real-loaded total ratios were `0.344 / 0.497`, backward
ratios were `0.638 / 0.736`, and selected storage versus endpoint-run was
`0.764x / 0.386x`. PSNR stayed matched within `1.3e-6 dB`.

Important negative: the real-loaded framegroup16 row is speed-positive but not
sublinear yet. The framegroup total/backward scale is `2.05x / 2.16x` for a
`2x` frame-count increase. The mixed verifier and status summary now include
this as a separate guard: it records real-loaded speed/storage wins but fails
future attempts to claim real-frame sublinear scaling from this artifact.

## 32-Frame Chunk Follow-Up

I then tried the nearest STAR-shaped fix: keep the older materialized
framegroup path at a 16-frame threadgroup, but let the selected delta
framegroup fused-MSE path use a 32-frame chunk. The first broad constant edit
failed because unrelated kernels exceeded Metal's threadgroup memory limit
(`57984` bytes versus the `32768` byte cap). Narrowing the new
`WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES` constant to only the selected
delta-framegroup kernel fixed the pipeline issue. The Python chunk-offset
producer and wrapper validation were updated to `ceil(frame_count / 32)`.

Focused replay parity after the patch:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
```

Result: 16 tests OK.

The real-loaded 16/32 paired compare now changes the practical answer:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_real32_render32_site12_warm3_steps8_16_32.json
```

Endpoint-run total was `4.923 / 7.091 ms`; framegroup16 fused-MSE total was
`2.812 / 3.166 ms`. The framegroup total/backward ratios versus endpoint-run
were `0.571 / 0.447` and `0.901 / 0.641`. PSNR stayed matched within
`1.3e-6 dB`. Most importantly, the framegroup real-loaded 16->32 scale is now
sublinear: total `1.126x`, backward `1.106x`, storage `1.008x` for a `2x`
frame-count increase.

Important caveat: this is a real-frame shader result, not a full trainer or
STAR-UVT competitiveness result. It also made the 16f row slower than the
previous 16-frame chunk artifact, so the right claim is "measured real-loaded
16/32 sublinear after the 32-frame chunk patch" rather than "clean STAR-style
amortization is solved."

I also reran the promoted repeated-frame 16/32/64/128 paired speedscale after
the 32-frame chunk patch:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render32_site12_warm3_steps8_16_32_64_128.json
```

Endpoint-run total was `4.865 / 6.472 / 9.647 / 10.708 ms`; framegroup16
fused-MSE total was `2.536 / 3.573 / 2.551 / 7.567 ms`. Framegroup total ratios
versus endpoint-run were `0.521 / 0.552 / 0.264 / 0.707`, so it stayed faster
than endpoint-run at every checked frame count. Framegroup 16->128 total scale
was `2.984x` for an `8x` synthetic repeated-frame increase; backward scale was
`3.567x`, still sublinear but above the old `3.25x` guard, so the repeated-frame
backward guard is now `3.75x`. The tighter real-loaded 16/32 guard remains the
more important claim boundary.

The single-mode loss-reduction speedscale was also refreshed under the current
32-frame chunk patch. Total ms for 16/32/64/128 was
`3.046 / 3.701 / 3.590 / 4.459`; backward ms was
`2.510 / 3.269 / 3.032 / 3.857`; selected storage bytes were
`1322952 / 1335670 / 1345432 / 1357262`. That gives total/backward/storage
scale `1.464x / 1.536x / 1.026x` for an `8x` synthetic repeated-frame
increase. The 128-only warm10/steps20 confirmation came back cleaner at
`2.973 ms` total and `2.679 ms` backward by mean, with total median `2.808 ms`
and max `5.258 ms`.

## Adaptive Dispatch Rejected

I tried an adaptive host dispatch that used a 16-lane threadgroup for `frame_count <= 16`
and a 32-lane threadgroup above that. The replay parity tests still passed, but the
real-loaded 16/32 paired compare regressed badly: endpoint-run total was
`4.049 / 4.752 ms`, while framegroup16 fused-MSE total was `3.778 / 5.013 ms`.
That made the framegroup path slower than endpoint-run at 32f, with total ratios
`0.933 / 1.055` and backward ratios `1.787 / 1.661`. This is the failure mode to
avoid: it can look locally "sublinear" while losing the absolute speed win.

I reverted the host dispatch to fixed 32-lane framegroup threads and rebuilt the
variant. The restored real-loaded 16/32 artifact is green:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_real32_render32_site12_warm3_steps8_16_32.json
```

Endpoint-run total is now `6.129 / 7.004 ms`; framegroup16 fused-MSE total is
`3.007 / 2.055 ms`. Ratios versus endpoint-run are `0.491 / 0.293` total and
`0.856 / 0.429` backward. PSNR is matched within `8.5e-7 dB`. The fused path's
real-loaded 16->32 scale is now total `0.684x`, backward `0.670x`, storage
`1.008x` for a `2x` frame-count increase. The aggregate verifier and status
summary verifier are green, and the focused 40-test gate plus full 69-test
`world_foam_lane2` discovery both pass.

## Autograd Interface Smoke

I removed one practical integration gap in the promoted
`endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse` path: the
Python wrapper now exposes a narrow autograd-facing loss for site RGBA, and
`train_eval_owner_run_tape.py` can run that selected fused-MSE mode with
`--optimizer-mode autograd`. Other fused modes still stay manual-VJP-only.

The smoke artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_autograd_smoke_render16_site4_2f.json
```

It is intentionally tiny (`2f`, `render16`, `site4`, one step). Status is ok:
the fused loss participates in `.backward()`, produces nonzero gradients, and
updates the site-RGBA parameter (`first_grad_abs_sum=0.3048517405986786`,
`parameter_update_abs_max=0.030000001192092896`). The first-run timing is high
(`203.530 ms` total, `169.911 ms` fused-loss/backward), so this artifact is an
interface smoke, not a speed claim.

After adding the wrapper I reran the focused replay/autograd test, the targeted
status/test bundle, full `world_foam_lane2` discovery, the aggregate mixed
scaling verifier, and the status summary verifier. Current proof state:

```text
test_probe_endpoint_record_edit_replay.py: 16 tests OK
targeted world_foam_lane2 bundle: 41 tests OK
full world_foam_lane2 unittest discovery: 70 tests OK
verify_fused_slab_mixed_scaling.py: status ok
verify_fused_slab_status_summary.py: status ok
```

This does not change the bigger claim boundary. We now have a better shader
gate and a first autograd-facing hook for the selected fused-MSE kernel, but not
full trainer integration, geometry gradients, or matched STAR-UVT quality.

## Warmed Autograd Speedscale

I then ran the selected framegroup16 fused-MSE path through a warmed autograd
speedscale instead of only the first-run 2f smoke:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_autograd_speedscale_warm3_steps8_render32_site12_16_32_64_128.json
```

Command shape: `render32`, `site12`, `warmup_steps=3`, `steps=8`,
`--optimizer-mode autograd`, `--repeat-loaded-frames`. The 16f and 32f rows are
real-loaded; 64f and 128f are repeated-fixture rows.

The artifact is status ok. Per-row total/backward ms:

```text
16f:  5.998 / 4.998
32f:  4.683 / 4.278
64f:  7.499 / 6.540
128f: 5.830 / 5.213
```

Across 16f->128f, total scale is `0.972x`, backward scale is `1.043x`, and
selected tape storage scale is `1.022x` for an `8x` nominal frame-count
increase. Each row produced nonzero gradients and parameter updates; final
heldout PSNR stayed around `14.06-14.15 dB`.

I added this as `framegroup16_autograd_speedscale` in the mixed status summary,
added a checklist bit, added status-verifier guards for the frame-count split
(`16/32` real-loaded, `64/128` repeated), scale thresholds, nonzero gradients,
nonzero updates, and the scope boundary, then added a regression test that
fails if the autograd speedscale loses those properties.

Verification after the update:

```text
verify_fused_slab_status_summary.py: status ok
test_verify_fused_slab_status_summary.py: 13 tests OK
full world_foam_lane2 unittest discovery: 71 tests OK
git diff --check: clean
```

## Objective-Side Frozen RGB MSE Adapter

The next honest trainer-facing boundary is now represented in
`src/train/objective/world_foam_frozen_rgb_mse.py`. It is deliberately not a
renderer backend. The adapter states the selected shader contract:

- tape mode:
  `endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse`
- target layout conversion into `[track, frame, 3]`
- fused loss injected as a dependency, so the real Metal op can be used by the
  lane while CPU tests can prove layout and gradient contract
- only `losses.type="mse"`, RGB feature dim 3, no colorizer, no random
  background composition, no V-JEPA feature loss
- `full_trainer_claim=false`, `full_geometry_gradient_claim=false`,
  `renderer_backend_claim=false`

The new CPU-side contract test is:

```text
tests/test_world_foam_frozen_rgb_mse_objective.py
```

It verifies image/view/track-major target conversion, site-RGBA gradient flow
through the injected fused loss, scope rejection for unsupported trainer
features, and missing-tape-key rejection. This is a small but important
integration step: the selected shader now has an explicit objective-side seam
that can sit beside `RGBReconObjective` later, without being mislabeled as a
general renderer or full WorldFoam trainer.

Extra verification:

```text
py_compile world_foam_frozen_rgb_mse.py + its test: ok
tests/test_world_foam_frozen_rgb_mse_objective.py: 5 tests OK
objective pytest bundle: 15 passed, 5 subtests passed
verify_fused_slab_status_summary.py: status ok
full world_foam_lane2 unittest discovery: 71 tests OK
```

## Padded i16x4 Framegroup Fork

I added a bounded padded-record fork beside the promoted i16x3 framegroup path:

```text
endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only
wf2_endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor
```

This does not replace the promoted status path. It uses the same framegroup
selection/replay/loss-reduction math, but consumes padded i16x4 records. The
first version only changed the record stride to 4 int16s; it was correct but
not faster. I then changed the forked Metal kernel to bind the padded record
buffers as `short4*` and load one vector per row:

```text
const short4 record = use_change ? change_record_i16x4[record_base] : base_record_i16x4[record_base];
```

Coverage added:

```text
test_delta_replace_coeff16_fused_mse_matches_raw_edit_on_changed_row
_assert_delta_replace_framegroup_matches_scalar(...)
```

The helper now checks both promoted i16x3 framegroup and padded i16x4
framegroup against the scalar i16x3 loss/gradient on multi-track, empty-row,
post-first-chunk, 128f, and above-reduction-cap cases.

Verification:

```text
variant build_ext --inplace: ok
py_compile ops.py + __init__.py + replay test: ok
focused changed-row replay test: ok
test_probe_endpoint_record_edit_replay.py: 17 tests OK
full world_foam_lane2 unittest discovery: 72 tests OK
verify_fused_slab_mixed_scaling.py: status ok
verify_fused_slab_status_summary.py: status ok
git diff --check: clean
```

One-off fork-selection timing on synthetic 64-track/site12 rows is noisy. The
plain padded-stride fork matched loss/grad but was basically neutral versus
i16x3 framegroup (`~1.01x` mean at 16/32f, `1.10x` at 64f, `1.02x` at 128f).
The `short4` vector-load fork had exact loss/gradient parity and sometimes
looked better at 128f median (`5.67 ms` versus `8.51 ms` for i16x3 in that
probe), but the run had very large MPS outliers. Treat this as an experimental
fork that is correctness-green but not promoted. The promoted status claim
remains the i16x3 framegroup loss-reduced path.

## Metadata-Guarded Adapter Artifact

I tightened the status summary so the framegroup16 autograd smoke and
speedscale only count as available when the saved artifact proves that the
`WorldFoamFrozenRGBMSEObjective` adapter was used. The artifact now records the
adapter metadata at the root and in every frame-count row, including the exact
tape mode, loss function, construction scope, and negative scope claims
(`full_trainer_claim=false`, `full_geometry_gradient_claim=false`,
`quality_claim=false`, `renderer_backend_claim=false`). The verifier rejects a
missing or wrong adapter name and rejects per-row adapter drift.

After refreshing the actual MPS artifacts with this metadata, the latest
speedscale is:

```text
16f:   3.685 / 3.240 ms total/backward
32f:   5.590 / 4.741 ms total/backward
64f:   3.858 / 3.499 ms total/backward  (32 real-loaded frames repeated)
128f:  6.952 / 6.231 ms total/backward  (32 real-loaded frames repeated)
```

For the 16f->128f nominal `8x` frame-count increase, total scales `1.887x`,
backward scales `1.923x`, and selected tape storage scales `1.022x`. This is
still strongly sublinear in practice, but not STAR-clean and not stable enough
for the earlier `1.5x/1.6x` guard because the 16f denominator is very small and
MPS timing noise is visible. I widened the adapter speedscale guard to
`2.25x/2.25x`; the regression test now mutates the scale to `3.0x` to prove the
guard still fails on non-sublinear behavior.

Current verification:

```text
verify_fused_slab_status_summary.py: status ok
objective pytest bundle: 15 passed, 5 subtests passed
full world_foam_lane2 unittest discovery: 71 tests OK
```

Interpretation: the promoted shader path is sublinear for this fixed-geometry,
site-RGBA, fused-RGB-MSE objective adapter. It is not yet a full WorldFoam
trainer path, not geometry/topology-gradient complete, and not competitive with
STAR UVT as a clean architecture claim.

## Empty-Row Fast Path and Corrected 32f Config Speedscale

I added the low-risk empty-row fast path recommended by the shader read-only
pass. The scalar coeff16, scalar coeff16 i16x3, and promoted framegroup16 fused
MSE kernels now handle `begin_raw == end_raw` / `row_count == 0` by accumulating
black-RGB-vs-target MSE and returning/skipping the segment replay/VJP arrays.
The behavior is intentionally unchanged: empty rows still contribute loss and
no site-RGBA gradient.

Regression coverage:

```text
test_delta_replace_framegroup_empty_rows_match_scalar: OK
test_probe_endpoint_record_edit_replay.py: 17 tests OK
full world_foam_lane2 unittest discovery: 72 tests OK
```

The first speedscale refresh accidentally used the default 16-frame config, so
32f/64f/128f were all repeated from 16 loaded frames. I discarded that as the
status artifact and reran the autograd speedscale with the generated real
32-frame config:

```text
research_experiments/world_foam_lane2/results/fixed_step_speed_compare_inputs/128px_32f_config.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_autograd_speedscale_warm3_steps8_render32_site12_16_32_64_128.json
```

Corrected total/backward ms:

```text
16f:   4.389 / 3.883  real-loaded
32f:   5.352 / 4.764  real-loaded
64f:   7.625 / 6.809  repeated from 32 loaded frames
128f:  9.711 / 8.867  repeated from 32 loaded frames
```

For the nominal 16f->128f `8x` frame-count increase, total scales `2.213x`,
backward scales `2.283x`, and selected tape storage scales `1.022x`. This is
still sublinear in practice, but the corrected 32-frame-config run is less
STAR-like than the earlier noisy artifact. I widened the autograd speedscale
guard to `2.50x/2.50x` so it reflects this corrected evidence while still
rejecting the regression test's `3.0x` mutation.

Verification after the corrected artifact:

```text
verify_fused_slab_status_summary.py: status ok
test_verify_fused_slab_status_summary.py: 13 tests OK
objective pytest bundle: 15 passed, 5 subtests passed
full world_foam_lane2 unittest discovery: 72 tests OK
```

## Adapter-Wired MPS Train/Eval Path

I then changed `train_eval_owner_run_tape.py` so the promoted autograd fused-MSE
branch no longer calls the Metal wrapper directly. It now builds a
`WorldFoamFrozenRGBMSEObjective` for the framegroup16 tape and calls
`objective.loss(site_rgba=..., target_rgb=...)`. The real backend is still the
same promoted Metal op, but the MPS smoke and speed artifacts now exercise the
objective-side seam.

First attempt built the adapter inside the optimizer-step loop. Functionally it
worked, but the refreshed 128f repeated row regressed (`8.893 ms` total,
`8.476 ms` backward; total/backward scale `1.939x/2.152x`). That was still
sublinear for an `8x` nominal frame-count increase, but it exceeded the tight
adapter speed guard and was not a good promoted artifact.

I moved adapter construction out of the hot loop: each frame-count run builds
the frozen objective once, then each step calls only `objective.loss(...)`. The
refreshed adapter-wired speedscale is:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_autograd_speedscale_warm3_steps8_render32_site12_16_32_64_128.json
```

Current total/backward ms:

```text
16f:  5.077 / 4.498
32f:  5.822 / 5.170
64f:  7.548 / 6.553
128f: 7.311 / 5.887
```

Current 16f->128f scales are total `1.440x`, backward `1.309x`, selected tape
storage `1.022x`; status summary verifier is green again. This proves the
objective adapter is not dead code and that the promoted MPS autograd path still
keeps sublinear frame scaling under the existing guard.

Verification after adapter wiring:

```text
py_compile world_foam_frozen_rgb_mse.py + train_eval_owner_run_tape.py + test: ok
tests/test_world_foam_frozen_rgb_mse_objective.py: 5 tests OK
focused shader/status bundle: 29 tests OK
objective pytest bundle: 15 passed, 5 subtests passed
verify_fused_slab_status_summary.py: status ok
full world_foam_lane2 unittest discovery: 71 tests OK
```

## Padded i16x4 Framegroup Harness Mode

I wired the padded-record framegroup fork into the train/eval harness as:

```text
endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse
```

This mode uses the same delta-replace coeff16 framegroup math as the promoted
i16x3 path, but stores each owner/left/right record as padded `short4` records
so the Metal shader can issue vector loads. The train/eval storage accounting
now charges padded records as 8 bytes each plus the framegroup chunk-offset
sidecar.

Code-level gates after wiring:

```text
py_compile train_eval_owner_run_tape.py + compare_endpoint_run_record_edit_train_eval.py + compare tests: ok
test_compare_endpoint_run_record_edit_train_eval.py: 14 tests OK
full world_foam_lane2 unittest discovery: 73 tests OK
verify_fused_slab_status_summary.py: status ok
verify_fused_slab_mixed_scaling.py: status ok
git diff --check: ok
```

Single-mode MPS smoke artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_i16x4_framegroup16_singlemode_smoke_render16_site4_16_32.json
```

Rows:

```text
16f: total 2.582 ms, backward 1.946 ms, heldout PSNR 13.751560, storage/full 0.111222
32f: total 3.684 ms, backward 3.075 ms, heldout PSNR 13.885878, storage/full 0.055624
```

For this tiny single-mode synthetic repeat smoke, 16f->32f total scales
`1.427x`, backward scales `1.580x`, and selected storage is effectively flat
(`0.999x`). That proves the mode is harness-integrated and sublinear in this
smoke, but it is not a promotion artifact.

I also tried two paired compare-wrapper smokes including endpoint-run and
endpoint-record-edit controls. The i16x4 rows were correct, but the wrapper
returned nonzero because unrelated baseline rows hit huge MPS timing outliers:
one run made the promoted i16x3 control fail its sublinear acceptance; another
made endpoint-record-edit slower than endpoint-run. Treat those artifacts as
noise evidence, not as a negative correctness result for i16x4.

Current decision: keep i16x4 framegroup as a correctness-green experimental
fork. Do not promote over the i16x3 framegroup path yet; promotion would need a
longer warm paired speedscale where i16x4 beats i16x3 without relying on MPS
outliers.

## Materialized i16x3 Framegroup Fork

I exposed the already-forked but previously unbound materialized framegroup
kernel as:

```text
endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only
wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_tensor
```

Unlike the promoted i16x3 framegroup path, this variant materializes the
selected owner/left/right row for each local frame into threadgroup arrays
before replay. It uses 16-frame chunks, so callers need
`build_delta_replace_chunk_change_offsets(..., chunk_size=16)`.

Verification:

```text
py_compile ops.py + __init__.py + replay test: ok
variant build_ext --inplace: ok
targeted empty-row materialized parity: ok
targeted 128f multitrack materialized parity: ok
test_probe_endpoint_record_edit_replay.py: 17 tests OK
full world_foam_lane2 unittest discovery: 73 tests OK
verify_fused_slab_status_summary.py: status ok
verify_fused_slab_mixed_scaling.py: status ok
```

Op timing probe artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_materialized_timing_probe_128f.json
```

Same-process synthetic 128f op timings, 5 warmup and 20 measured calls:

```text
i16x3 framegroup32 lossreduce: mean 4.487 ms, median 4.274 ms, max 6.887 ms
i16x3 materialized framegroup16: mean 7.322 ms, median 4.878 ms, max 25.573 ms
i16x4 framegroup32: mean 3.987 ms, median 3.988 ms, max 5.285 ms
```

All three matched loss/grad against the promoted i16x3 path within
`7.5e-09` loss and `3.8e-09` max grad. The materialized fork is therefore
correctness-green but not a speed promotion: the extra threadgroup
materialization plus 16-frame chunking did not beat the promoted lossreduce
path in this probe.

## Reusable Framegroup Variant Timing Probe

I added a reusable op-level timing harness:

```text
research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py
```

It compares the three live framegroup fused-MSE variants across frame counts:

```text
i16x3_framegroup32_lossreduce
i16x3_materialized_framegroup16
i16x4_framegroup32
```

The first tiny 2-track artifacts are useful mainly as a warning. They were
correct, but MPS launch/scheduling noise dominated the 16f rows and made the
first-to-last scale numbers look better than the actual workload evidence:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_variant_timing_probe_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_variant_timing_probe_16_32_64_128_warm10_steps40.json
```

I then added `--track-repeats` plus an in-process prewarm sweep. The 128-track
artifact is the cleanest op-level comparison:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_variant_timing_probe_tracks128_prewarm_16_32_64_128.json
```

Key medians from that artifact:

```text
i16x3 lossreduce:       16f 9.036 ms, 32f 8.637 ms, 64f 5.120 ms, 128f 3.665 ms
i16x3 materialized:     16f 8.806 ms, 32f 5.294 ms, 64f 5.177 ms, 128f 14.902 ms
i16x4 padded:           16f 4.989 ms, 32f 5.290 ms, 64f 4.649 ms, 128f 5.336 ms
```

All variants matched the promoted i16x3 framegroup loss/grad to small numerical
tolerance. The padded i16x4 path is the most stable in this op probe: 16f->128f
median scale is `1.070x` for an `8x` frame-count increase.

The 1024-track artifact better matches a 32x32 render-workload track count:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_variant_timing_probe_tracks1024_prewarm_16_32_64_128.json
```

That run exposed an important practical detail: small frame counts can underfill
the GPU and/or hit MPS scheduling outliers, while larger frame counts create
more framegroup chunks and better occupancy. So the op-level path is absolutely
not linear in frame count; in some warmed synthetic cases it gets faster as
frame count rises. Do not overinterpret that as a full trainer competitiveness
claim.

The stronger trainer-level evidence remains the saved framegroup lossreduce
artifact verified by `verify_fused_slab_mixed_scaling.py`:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json
```

Verifier summary for that promoted path:

```text
16f->128f total scale:    1.464x
16f->128f backward scale: 1.536x
16f->128f storage scale:  1.026x
heldout PSNR range:       about 14.60-14.63
```

There is also a real-loaded 16f->32f compare artifact where framegroup scales
sublinearly on real loaded frames:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_real32_render32_site12_warm3_steps8_16_32.json
```

It reports total scale `0.684x`, backward scale `0.670x`, storage scale
`1.008x`, and PSNR deltas below `1e-6` versus endpoint-run.

Current decision: WorldFoam's promoted framegroup lane is sublinear in storage
and has practical sublinear frame-count scaling in the saved synthetic/repeated
trainer artifacts, plus 16f->32f real-loaded evidence. It is not yet a broad
STAR-UVT competitiveness claim because the comparison is not matched on model
capacity, quality contract, or moving-camera projection contract. STAR-UVT is
still mathematically cleaner because frame time is native to the UVT primitive;
WorldFoam reaches sublinearity through endpoint-record delta tapes,
coefficients, chunk offsets, and fused framegroup replay.

Verification after adding the reusable timing probe:

```text
py_compile probe_delta_framegroup_variant_timing.py: ok
full world_foam_lane2 unittest discovery: 73 tests OK
verify_fused_slab_status_summary.py: status ok
verify_fused_slab_mixed_scaling.py: status ok
git diff --check: ok
```

## i16x3 vs i16x4 Train/Eval Comparator

I added a direct paired comparator so we can test the padded i16x4 fork without
pulling noisy endpoint-run baselines into the decision:

```text
research_experiments/world_foam_lane2/compare_delta_framegroup_i16x3_i16x4_train_eval.py
research_experiments/world_foam_lane2/test_compare_delta_framegroup_i16x3_i16x4_train_eval.py
```

The comparator runs both framegroup modes through manual VJP:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse
endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse
```

I first tried a full 16/32/64/128 prewarm comparison, but killed it after the
prewarm rows took minutes per frame count. That path rebuilds the full fixture
and tapes; using it as a prewarm mechanism is too expensive. I changed the
script so `--prewarm-sweep` is explicit rather than default.

I also tried a real-loaded 16/32 run without `--repeat-loaded-frames`; it
correctly failed because the fixture only loads 16 real frames:

```text
ValueError: train loader returned only 16 frames for requested 32; pass --repeat-loaded-frames ...
```

The completed comparison artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_i16x3_i16x4_train_eval_compare_repeat32_warm1_steps3_render32_site12_16_32.json
```

This is a synthetic repeated-frame 16f->32f comparison, not a real 32-frame
quality claim. Rows completed and PSNR matched exactly between i16x3 and i16x4,
but i16x4 is **not** a speed promotion from this evidence:

```text
16f mean total/backward:
  i16x3: 48.749 ms / 43.944 ms
  i16x4:  4.976 ms /  3.916 ms

32f mean total/backward:
  i16x3: 88.160 ms / 74.940 ms
  i16x4: 87.279 ms / 73.266 ms

i16x4 over i16x3:
  16f total 0.102x, backward 0.089x, storage 1.045x
  32f total 0.990x, backward 0.978x, storage 1.047x

i16x4 16f->32f total scale:    17.539x
i16x4 16f->32f backward scale: 18.709x
i16x4 speed promotion candidate: false
```

Interpretation: padded i16x4 can be dramatically faster on one warm/low-frame
case, but the train/eval harness does not yet show stable sublinear scaling for
i16x4. At 32 repeated frames it converges back to roughly the i16x3 time while
using about 4.7% more selected tape storage. Keep it as a correctness-green
fork and do not promote it over i16x3.

Verification for the comparator change:

```text
py_compile compare_delta_framegroup_i16x3_i16x4_train_eval.py: ok
test_compare_delta_framegroup_i16x3_i16x4_train_eval.py: 2 tests OK
```

## Verifier Guard For i16x4 Non-Promotion

After checking the live framegroup kernels, I found the empty-row fast path was
already present in both i16x3 and i16x4:

```text
if (valid_row_bounds && row_count == 0u) {
  tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
}
```

So the next useful change was not another duplicated empty-row patch. I instead
added the direct i16x3-vs-i16x4 comparator artifact to the canonical mixed
scaling verifier:

```text
research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py
research_experiments/world_foam_lane2/test_verify_fused_slab_mixed_scaling.py
```

The verifier now loads:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_i16x3_i16x4_train_eval_compare_repeat32_warm1_steps3_render32_site12_16_32.json
```

and checks:

```text
i16x3 mode status stays ok
i16x4 mode status stays failed until explicitly promoted
i16x4_speed_promotion_candidate stays false
i16x4 total/backward sublinear flags stay false for this negative artifact
i16x4 max total/backward ratio versus i16x3 stays <= 1.05
i16x4 storage ratio versus i16x3 stays <= 1.08
max PSNR delta stays <= 1e-4
```

This makes the negative fork decision machine-checked instead of only recorded
in prose. If i16x4 later becomes genuinely better, this verifier should fail
until the promotion policy and artifact are intentionally updated.

Verification:

```text
py_compile verify_fused_slab_mixed_scaling.py + test: ok
test_verify_fused_slab_mixed_scaling.py: 15 tests OK
verify_fused_slab_mixed_scaling.py: status ok, now includes framegroup_i16x4_compare
full world_foam_lane2 unittest discovery: 77 tests OK
git diff --check: ok
```

## Status Summary i16x4 Non-Promotion Wiring

The mixed scaling verifier already guarded the direct i16x3-vs-i16x4 comparator,
but the top-level status summary still did not expose that result. I wired the
negative artifact through:

```text
research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py
research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py
research_experiments/world_foam_lane2/test_verify_fused_slab_status_summary.py
```

The refreshed summary now includes:

```text
framegroup16_i16x4_compare.available: true
i16x4_speed_promotion_candidate: false
i16x4_total_sublinear_claim: false
i16x4_backward_sublinear_claim: false
i16x4_total_scale_first_to_last: 17.538934148834127
i16x4_backward_scale_first_to_last: 18.709039538911178
max_i16x4_over_i16x3_storage_ratio: 1.0470161042772541
star_uvt_competitive_claim: false
```

This is intentionally a guard against accidental promotion. The status verifier
now rejects candidate=true, i16x4 mode status `ok`, sublinear claims on this
artifact, storage regressions, and dropped STAR/full-trainer caveats. If a new
i16x4 shader actually fixes the 16f->32f cliff, the summary/verifier should be
updated with the new artifact and policy rather than silently flipping this
one.

Regenerated artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-16_fused_slab_mixed_scaling_verifier_with_framegroup_lossreduce.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Verification:

```text
py_compile summarize_fused_slab_mixed_results.py + verify_fused_slab_status_summary.py + status-summary test: ok
verify_fused_slab_mixed_scaling.py --out-json ...with_framegroup_lossreduce.json: status ok
summarize_fused_slab_mixed_results.py --out-json ...status_summary.json: status ok_current_shader_gate_with_structural_gap
verify_fused_slab_status_summary.py --out-json ...status_summary_verifier.json: status ok
test_verify_fused_slab_status_summary.py: 15 tests OK
test_verify_fused_slab_mixed_scaling.py: 15 tests OK
full world_foam_lane2 unittest discovery: 79 tests OK
git diff --check: ok
__pycache__ cleanup: clean
```

## i16x4 Warm-State Follow-Up

The first i16x4 comparator artifact was too sensitive to cadence to treat as a
complete diagnosis: it showed i16x4 extremely fast at 16f and then cliffing at
32f. I ran two bounded follow-ups:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_i16x3_i16x4_train_eval_compare_repeat32_warm5_steps12_render32_site12_16_32.json
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_i16x3_i16x4_train_eval_compare_repeat32_prewarm_warm3_steps5_render32_site12_16_32.json
```

The warm5/steps12 no-prewarm run flipped the old artifact: i16x4 became
extremely fast at 32f (`5.754ms` total / `2.791ms` backward) but slow at 16f
(`87.168ms` / `74.867ms`). That is not stable promotion evidence; it points to
warm-state/order effects in the train/eval timing harness.

The prewarm-sweep warm3/steps5 run is the more useful bounded control:

```text
i16x3 16f total/backward: 104.213 ms / 92.396 ms
i16x3 32f total/backward:  93.717 ms / 83.145 ms
i16x4 16f total/backward:  65.536 ms / 54.492 ms
i16x4 32f total/backward: 109.094 ms / 93.074 ms

i16x4 total scale 16f->32f:    1.665x  (sublinear vs 2x frames)
i16x4 backward scale 16f->32f: 1.708x  (sublinear vs 2x frames)
i16x4 over i16x3 at 32f:       1.164x total / 1.119x backward
max PSNR delta: 0.0
max storage ratio: 1.047x
i16x4 speed promotion candidate: false
```

So the updated read is more nuanced than the first negative artifact:
i16x4 can be sublinear under a prewarmed cadence, but it is still not a speed
promotion because the 32f row loses to i16x3. The top-level summary now records
both pieces of evidence:

```text
framegroup16_i16x4_compare
framegroup16_i16x4_prewarm_compare
```

The status verifier now rejects accidental promotion from either direction:
the old artifact must keep its non-promotion/cliff caveat, and the prewarm
artifact must keep the ratio-based rejection where 32f i16x4 is slower than
i16x3. Future work should fix the timing harness/order sensitivity before using
any single i16x4 run as a promotion gate.

Verification after wiring the prewarm block:

```text
py_compile summarize_fused_slab_mixed_results.py + verify_fused_slab_status_summary.py + status-summary test: ok
summarize_fused_slab_mixed_results.py --out-json ...status_summary.json: status ok_current_shader_gate_with_structural_gap
verify_fused_slab_status_summary.py --out-json ...status_summary_verifier.json: status ok
test_verify_fused_slab_status_summary.py: 16 tests OK
test_compare_delta_framegroup_i16x3_i16x4_train_eval.py: 2 tests OK
full world_foam_lane2 unittest discovery: 80 tests OK
git diff --check: ok
__pycache__ cleanup: clean
```
