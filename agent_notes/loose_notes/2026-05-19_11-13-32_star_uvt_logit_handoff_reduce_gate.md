# STAR UVT Logit-Handoff Tile-Slot Reducer Gate

Date: 2026-05-19

## Goal

Continue the STAR UVT fast feature-shader plan after the optimizer/LR schedule
gate. The speed question was whether the image-space-prep logit handoff could
be combined with the existing stable-tile feature-gradient reducers so that the
handoff avoids the old dense F32 image-gradient path and also reduces per-channel
feature atomics.

This is a shader-side direct-kernel gate, not a first-class trainer promotion.

## Code Changes

- Added logit handoff reducer modes in
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`:
  - `logit_handoff`
  - `logit_handoff_reduce`
  - `logit_handoff_reduce_vec4`
- Updated
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
  so `direct_atomic_feature_logit_handoff_backward` can route stable tiles
  through the scalar or vec4 feature-gradient reducers, while invalid pixels
  still participate in barriers with zero contribution and unstable tiles fall
  back to direct atomics.
- Updated
  `research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py`
  and `direct_feature_mode_matrix.py` so the new modes run through the same
  tiny-parity and sequential timing harness as the older handoff modes.
- Added
  `research_experiments/star_uvt_feature_tubes/logit_handoff_reduce_report.py`
  to turn the matrix into a repeatable validation/report artifact.

## Commands

Build:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Tiny parity spot:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode logit_handoff_reduce_vec4 --feature-dims 4,32 --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_vec4_tiny_parity.json
```

Matrix:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes logit_handoff,logit_handoff_reduce,logit_handoff_reduce_vec4 \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_matrix_256_512_64f_32768t_f32
```

Report:

```bash
.venv/bin/python research_experiments/star_uvt_feature_tubes/logit_handoff_reduce_report.py
```

## Results

Report:

- `outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_report.md`
- `outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_report.json`
- Matrix:
  `outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_matrix_256_512_64f_32768t_f32/summary.md`

Direct 64f/32768t/F32 matrix:

| size | mode | total | forward | prep | backward | overflow |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | `logit_handoff` | `800.0ms` | `171.6ms` | `56.7ms` | `571.7ms` | `0` |
| 256 | `logit_handoff_reduce` | `796.1ms` | `179.2ms` | `54.5ms` | `562.3ms` | `0` |
| 256 | `logit_handoff_reduce_vec4` | `744.1ms` | `178.7ms` | `54.9ms` | `510.6ms` | `0` |
| 512 | `logit_handoff` | `2137.8ms` | `973.8ms` | `509.1ms` | `654.8ms` | `0` |
| 512 | `logit_handoff_reduce` | `1905.0ms` | `723.1ms` | `459.4ms` | `722.5ms` | `0` |
| 512 | `logit_handoff_reduce_vec4` | `1512.4ms` | `590.3ms` | `279.8ms` | `642.3ms` | `0` |

Tiny F4/F32 parity passes for all matrix rows. Max backward errors are at or
below `2.98e-08`, with forward feature error `2.98e-08` and alpha error
`1.19e-07`.

## Decision

`logit_handoff_reduce_vec4` is a real direct-kernel diagnostic candidate:

- 256px backward improves `571.7 -> 510.6ms`.
- 512px backward improves narrowly `654.8 -> 642.3ms`.
- all tested rows have zero overflow and zero unstable tiles.

Do not promote it to trainer default yet:

- scalar `logit_handoff_reduce` regresses 512px backward (`654.8 -> 722.5ms`);
- the 512px total-time win includes large forward/prep movement even though the
  edit targets backward, so MPS/session variance is still visible;
- this is direct synthetic evidence, not an end-to-end first-class overfit row.

Next speed gate: either wire a trainer-compatible native VJP/logit reducer row
and benchmark it end to end, or move to the true scalar fixedbin/tile-slot
contribution path that avoids duplicate STAR traversal.

## Current State And Workday Plan

End goal for the STAR UVT feature lane: a fast trainable F32/F64 time-tube
renderer that can use precomputed V-JEPA-style feature targets, overfit one
64-frame 512px video without falling back to projected F32 splatting, and then
scale to the prepared larger dataset only after the single-video gate has both
speed and quality evidence.

Current state:

- Best quality/visual diagnostic remains the frozen target-grid RGB-probe
  continuation chain, not the RGB-target `star-feature-512-fast` helper.
- The 1300->1400 feature1/probe40 row improves both feature loss and probe PSNR
  but is slower (`1.690-1.711s/step`); effective `lr=0.001` from the 1300
  checkpoint is faster and higher on probe PSNR but slightly worse on final
  feature/weighted loss.
- The optimizer-LR schedule gate was negative: it moved the transient spike
  instead of fixing it.
- The new logit-handoff vec4 reducer gate is the current shader sidecar:
  correct and promising in direct synthetic timing, but not first-class yet.

Next full-day order:

1. Wire a trainer-compatible native-VJP/logit-handoff path or a narrow trainer
   benchmark that exercises `logit_handoff_reduce_vec4` against the current
   frozen-probe objective without silently changing the loss.
2. Benchmark that route against the current target-grid/frozen-probe
   `feature_direct_gradcache_reduce_vec4` row at the same 64f/512px/8192t
   checkpoint state; report render, VJP/prep, backward, total step, loss, probe
   PSNR, overflow, and media when useful.
3. If the first-class route wins, run a short quality continuation from the
   1300 checkpoint; if it does not, pivot to the true scalar fixedbin/tile-slot
   contribution path rather than another dense handoff variant.
4. Only after the single-video row is both faster and quality-safe, scale to the
   prepared larger single-video dataset. Keep Gaussian/token cached conditioning
   as the dataset-scale reference until STAR feature quality closes the gap.
5. Keep WorldFoam shader investigation as a separate no-GPU-conflict lane and
   keep feature-world-tube notes separate from the STAR overfit benchmark rows.

## Validation

- `py_compile` passed for the modified Python benchmark/report files.
- `setup.py build_ext --inplace` relinked/copied the STAR UVT extension.
- Tiny parity passed for `logit_handoff_reduce_vec4`.
- The 6-row 256/512 matrix passed.
- The generated report validation passed.
