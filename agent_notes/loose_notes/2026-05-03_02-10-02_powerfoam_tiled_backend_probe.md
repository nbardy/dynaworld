# PowerFoam Metal Tiled Backend Probe

## Context

The Metal height+SV primitive was accurate and trainable, but the streaming
path still looped every pixel over every cell. Conservative projected bounds
helped but did not make 4K acceptable, especially for replay backward.

## What changed

- Added `powerfoam_tiled_stream_kernels.metal`, loaded after the existing
  streaming kernels so it can reuse the same interval, surface, height, SV
  color, and replay-gradient helpers.
- Added tiled C++/Torch ops:
  - `rasterize_tiled_count`
  - `rasterize_tiled_write`
  - `rasterize_tiled_emit_count`
  - `rasterize_tiled_emit_write`
  - `rasterize_tiled_train_forward`
  - `rasterize_tiled_train_backward`
- Added `FoamRasterConfig(use_tiled=True, tiled_builder="auto")`.
  - `auto` uses the simple sorted tile scan for `N <= 1024`.
  - Larger scenes use cell-to-tile emit plus MPS sort on `(tile, sorted_order)`
    keys, preserving front-to-back compositing order.
- Added benchmark flag `--foam-tiled` and builder selector
  `--foam-tiled-builder`.
- Added a tiled trainer config:
  `src/train_configs/local_mac_powerfoam_metal_quaternion_height_sv_texel_surface_tiled_video_1024_smoke.jsonc`.
- Added `third_party/powerfoam-metal/tests/tiled_streaming_check.py` for
  streaming-vs-tiled parity on constant features and full height+SV.

## Validation

- Rebuilt the extension with:

```bash
( cd third_party/powerfoam-metal
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

- `tiled_streaming_check.py` passed. Recent max errors:
  - constant features/alpha: `0.0` / `0.0`
  - constant grads: within `~6.6e-07`
  - height+SV features/alpha: `0.0` / `0.0`
  - height+SV grads: within `~2.4e-07`
- The 1-step trainer smoke using the tiled config also passed. Eval L1 moved
  from `0.034312` to `0.033381`; center/radius/quaternion, texel site/height,
  and SV axis/RGB deltas were all nonzero after the optimizer step.

## 4K timing

Best stable benchmark:

```bash
PYTHONPATH=src/train .venv/bin/python \
  third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py \
  --cells 1024,4096 \
  --resolutions 4096x4096 \
  --neighbors 32 \
  --warmup 1 \
  --iters 3 \
  --foam-backward \
  --foam-height-sv-texel-surface \
  --foam-tiled \
  --foam-tiled-builder auto \
  --json > outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_auto_tile8_4k_1024_4096_stable_2026-05-03.json
```

Median results:

- `1024` cells: `1198.2 ms` forward, `8166.3 ms` backward, `9364.5 ms` total.
- `4096` cells: `2332.2 ms` forward, `15567.8 ms` backward, `17899.98 ms`
  total.

This beats projected streaming but is still not fast enough to call the 4K
requirement done.

## Profiling result

For the 8x8 emit/sort path, candidate building is no longer the main cost. A
manual 4096-cell profile showed roughly:

- count/cumsum/write/sort: about `122 ms`
- tiled forward kernel: about `2086 ms`
- tiled backward kernel: about `14883 ms`

The remaining performance problem is the replay kernel shape and global
atomics, not candidate list construction.

## Failed/weak tuning

- 4x4 tiles preserved parity but made 4096-cell 4K height+SV much slower:
  `5254.6 ms` forward / `25963.5 ms` backward. Reverted to 8x8.
- Emit/sort was better for larger scenes, but worse than sorted scan at smaller
  `N`; kept an `auto` builder rather than hard-coding one path.
- A reduced-atomic constant-feature backward did work and is kept for
  `feature_mode == 0`. It reduces feature/density gradients per tile-candidate
  while leaving endpoint point/radius atomics alone. Stable 4K constant timing:
  `1024` cells `377.8 ms` forward / `733.4 ms` backward, `4096` cells
  `682.4 ms` forward / `1152.9 ms` backward.
- A full height+SV feature-gradient reduction also preserved parity, but it was
  slower than the default global-atomic replay in the actual 4K benchmark
  (`1024` cells around `10.3 s` backward; `4096` cells around `17.2 s`
  backward). The selector was reverted so mode 6 still uses the prior faster
  default. The experimental kernel remains useful as evidence that naive
  per-feature reduction is not the right next step.
- Overlap adjacency reduced average degree to about `9`, but the 4096-cell full
  height+SV 4K run still took about `17.1 s` total. The bottleneck is not just
  neighbor count.
- Reusing mode-6 SV colors and texel mixture weights inside the default tiled
  forward/backward was the first direct full-height+SV win after tiling. The
  stable 4K benchmark now reports:
  - `1024` cells: `1199.4 ms` forward, `5393.1 ms` backward, `6592.5 ms` total.
  - `4096` cells: `2066.9 ms` forward, `9025.9 ms` backward, `11092.9 ms`
    total.
  The optimized path still passed `tiled_streaming_check.py`, and the 1-step
  tiled trainer smoke still moved center/radius/quaternion/texel/SV params.
- After that optimization, `sorted_scan` beat emit/sort at `4096` cells, so
  `tiled_builder="auto"` now uses sorted scan through `N <= 4096`. The default
  auto artifact reports `1024` cells at `1280.9 ms` forward / `5936.2 ms`
  backward and `4096` cells at `2102.4 ms` forward / `9058.1 ms` backward.
  Trying int32 sort keys made MPS sort slower and was reverted.
- Added CPU `cech_aabb` adjacency as the correctness path in the local trainers.
  It uses the official Cech-style sphere-overlap predicate
  `||p_i-p_j|| <= r_i+r_j`, ignores the K cap, and logs average degree, max
  degree, required dense-overlap edges, and missing overlap edges. The focused
  test now constructs a case where `neighbor_count=1` KNN misses a true
  power-face neighbor; the `cech_aabb` render matches a dense fully connected
  one-ray reference while KNN differs.
- Switched the full height+SV tiled smoke config to `cech_aabb`. The 1-step
  smoke printed `adjacency_avg_degree=7.6113`, `adjacency_required_overlap_edges=7794`,
  and `adjacency_missing_overlap_edges=0`, with eval L1 improving from
  `0.034312` to `0.033381` and all geometry/material parameter groups moving.
- Updated older explicit-KNN PowerFoam Metal configs and the `BASELINES.md`
  PowerFoam section to label KNN as an approximate speed-ablation graph.
- Fresh full height+SV tiled 4096x4096 `cech_aabb` benchmark:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_2026-05-03.json`.
  Median `1024` cells: `1314.5 ms` forward / `5910.2 ms` backward /
  `7273.1 ms` total at average degree `8.55`. Median `4096` cells:
  `1954.5 ms` forward / `9364.2 ms` backward / `11441.3 ms` total at average
  degree `9.53`. This confirms adjacency correctness did not remove the full
  mode-6 4K backward bottleneck.
- Added a non-gradient tiled Metal auxiliary pass:
  `rasterize_power_foam_aux`, plus full height+SV aux wrappers. It emits normal
  distance, accumulated normal, fixed median-depth quantile, contribution,
  target-weighted point error, and visibility mask. The standalone
  `third_party/powerfoam-metal/tests/aux_check.py` passed for both an analytic
  one-cell constant case and a one-cell height+SV case. This gives the future
  densification/EMA code real Metal-side contribution/error/visibility inputs,
  but arbitrary official depth-quantile vectors and aux-gradient routing were
  still not implemented at the time of this entry.
- Wired the aux pass into `train_powerfoam_metal.py` validation for full
  height+SV modes. The 1-step tiled smoke now prints aux metrics; on the smoke
  fixture it reported `aux_mean_contrib` around `9.27e-4`,
  `aux_visible_fraction=1.0`, and `aux_mean_median_depth` around `2.1866`,
  while still moving all trainable geometry/material parameter groups.
- Added persistent contribution/error EMA buffers to the Metal trainer, updated
  with the official-style visible-mask decay during aux validation. The smoke
  now prints `aux_mean_contrib_ema` and `aux_mean_point_error_ema`, so future
  pruning/resampling can consume trainer state rather than recomputing raw
  stats ad hoc.
- Added fixed-capacity EMA resampling to the Metal trainer. The method keeps
  high-contribution cells, samples replacement slots from high-error valid
  cells, reindexes all per-cell tensors, preserves Adam state tensors through
  the permutation, divides carried EMAs by duplicate count, and optionally
  perturbs duplicate positions. The focused unit test verifies deterministic
  low-contribution replacement plus Adam `exp_avg` reindexing. A 2-step MPS
  smoke with `resample_every=1` exercised validation -> EMA -> resample in the
  real trainer and printed `resample_replaced=1648` across the 16-frame
  1024-cell smoke. This is still fixed-capacity replacement, not true
  capacity-changing densification/pruning.

## Remaining blocker

The next real speed step needs a different backward/replay structure, likely
less global-atomic accumulation and/or a more compact per-pixel candidate replay
contract. The current tiled path is a correct benchmarkable scaffold, not the
final fast 4K implementation. Full reproduction still also lacks static
multiview/SfM training, paper-scale densification/pruning schedules and
acceptance runs, ray tracing, and differentiable aux-gradient parts of the
official API.

## Later update: capacity resize and tile-size retune

- Extended EMA resampling from fixed-capacity replacement to true tensor
  resize. The focused unit test now covers fixed replacement, grow to 6 cells,
  and prune to 3 cells while preserving Adam state. A 2-step MPS grow smoke
  changed the live trainer from `1024` to `1050` cells, then rendered and
  backpropped the next step with `resample_cell_count=1050`.
- The naive full height+SV feature-gradient reduction remained a dead end. It
  preserved parity, but the 4K `cech_aabb` artifact
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_height_sv_reduced_2026-05-03.json`
  measured `10462.1 ms` total at `1024` cells and `20007.7 ms` at `4096`
  cells, slower than the default global-atomic mode-6 replay. The selector was
  kept on the faster default path.
- Retuned tiled geometry from 8x8/64 threads to 16x16/256 threads. This
  preserves parity and reduces tile/candidate overhead without the heavy
  reductions that hurt the mode-6 reduced kernel. The saved 4K benchmark
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_tile16_2026-05-03.json`
  reports median `1024` cells at `892.0 ms` forward / `4611.7 ms` backward /
  `5501.8 ms` total, and `4096` cells at `1614.3 ms` forward / `7138.7 ms`
  backward / `8753.0 ms` total.
- A 32x32/1024-thread probe also preserved tiled parity, but the 4K 1024-cell
  probe regressed to `7294.1 ms` total
  (`outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_tile32_probe_2026-05-03.json`),
  so the checked code stays at the measured 16x16 tile point.
- A per-pixel stop-count replay buffer was also tested because tiled backward
  otherwise replays to a tile-wide max stop. It preserved parity but got slower:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_tile16_pixelstop_2026-05-03.json`
  reported `6474.7 ms` total at `1024` cells and `10120.2 ms` at `4096`
  cells. The extra stop buffer/write traffic did not pay for itself on this
  benchmark, so the code was reverted to tile-wide stop.
- The non-gradient tiled aux API now supports arbitrary `depth_quantiles`
  instead of only the hard-coded median. `rasterize_power_foam_aux` accepts a
  quantile vector and returns `depth_quantile_depths` with shape `[B,H,W,Q]`
  (or `[H,W,Q]` for unbatched rays) while preserving `median_depth`. The aux
  check verifies `[0.25, 0.5, 0.75]` against the analytic one-cell solution.
  Differentiable aux-loss gradients are still not implemented.
- Post-retune checks run: rebuilt the extension, `tiled_streaming_check.py`,
  `linear_texture_check.py`, `aux_check.py`, and a 1-step tiled trainer smoke.
  The trainer smoke still moved learned centers, radii, quaternions, texel
  sites/heights, SV axes, and SV RGB after the optimizer step.

## Later update: posed-camera Metal trainer smoke

- Added a camera-aware training path to `src/train/train_powerfoam_metal.py`.
  The important implementation detail is that multicam samples are flattened
  from `[view, time]` to a sample batch, but the sample carries a shared
  `frame_index`. Two train cameras looking at frame `t` therefore optimize the
  same PowerFoam state for `t`; the trainer is no longer limited to one
  fixed-origin ray grid per target image.
- Reused the repo camera contract instead of inventing a PowerFoam-specific
  camera type. The path loads `multicam_val` through
  `load_multicam_video_bundle(...)`, converts `K/w2c` to `CameraSpec`, builds
  `[origin, direction]` rays with `build_camera_rays(...)`, and passes per-sample
  ray tensors into the existing Metal raster/autograd calls.
- New smoke config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_tiled_32_smoke.jsonc`.
  It uses DeepView `03_Dog`, train cameras `camera_0001` and `camera_0015`,
  heldout camera `camera_0040`, `32px`, `64` cells, `2` frames, and the
  quaternion height+SV tiled primitive.
- Focused tests now cover camera-pose ray construction and the multiview
  flattening contract. Command:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  passed with `13 passed`.
- The actual 1-step MPS multicam smoke passed. Startup printed
  `frames=2`, `samples=4`, train views `camera_0001/camera_0015`, heldout
  `camera_0040`, and `pose_source=deepview_models_relative_pinhole`.
  Metrics now include L1/MSE/PSNR/SSIM. Train eval L1/PSNR/SSIM moved from
  `0.341172` / `7.9412` / `-0.0257` to `0.340393` / `7.9490` / `-0.0237`;
  heldout L1/PSNR/SSIM after the step was `0.358152` / `7.8900` / `-0.0101`.
  The post-step drift metrics showed centers, radii, density, quaternions,
  texel sites/heights, SV axes, and SV RGB moved.
- Also regression-smoked the original explicit-video path with the new forward
  signature at `32px`, `64` cells, and `1` step. It passed with eval L1
  `0.034253 -> 0.032774` and PSNR/SSIM `25.1659` / `0.8320` to `25.5397` /
  `0.8446`, so the existing single-video path still works.
- Added `test_powerfoam_metal_synthetic_posed_views_overfit_shared_state`.
  The test is MPS-gated and uses a one-cell Metal teacher plus three posed
  camera rays to train a four-cell randomly initialized Metal student. It is a
  small constant-primitive acceptance gate, not a full height+SV benchmark. The
  assertion requires final L1 `< 0.006`, at least a 4x reduction from the first
  step, and nonzero center drift. Targeted run passed, then the focused
  PowerFoam file passed with `14 passed`.
- Added optional SSIM loss in the Metal trainer. `losses.ssim_weight` is
  default-off, but when enabled the train loop adds `1 - SSIM` using the repo
  SSIM helper and logs `ssim_loss`. A focused helper test checks identical
  images produce zero loss, raising the focused file to `15 passed`. A 1-step
  multicam MPS smoke with `losses.ssim_weight=0.05` exercised the branch in the
  real backward path and printed `ssim_loss=1.001716`.
- This closes only the tiny posed-camera smoke gate. It does not close the full
  static PowerFoam reproduction: COLMAP/SfM init, real static acceptance
  thresholds, baseline rows, official schedules/losses, ray tracing, and the
  fast 4K backward problem remain open.

## Later update: tiled memory accounting

- Added explicit memory/candidate accounting to
  `third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py`. The
  benchmark now reports the actual tiled builder, tile count, candidate count,
  candidate/offset/stop/screen-bounds bytes, saved `log_t` bytes, output/alpha
  bytes, and a conservative forbidden dense `N*H*W` float-buffer size.
- Syntax check passed:
  `rtk .venv/bin/python -m py_compile third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py`.
- Smoke artifact:
  `outputs/benchmarks/powerfoam_metal_memory_accounting_smoke_2026-05-03.json`.
  The `128x128`, `N=256`, full height+SV tiled backward row reported `2338`
  candidates over `64` tiles, `0.013 MiB` index state, `0.076 MiB` saved
  forward state including `log_t`, and `16.0 MiB` for one dense `N*H*W` float
  slab. Human output also prints these fields.
- Refreshed 4096x4096 accounting artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4096sq_1024_4096_tile16_accounting_2026-05-03.json`.
  `1024` cells used `1563158` candidates over `65536` tiles, `6.48 MiB` index
  state, `70.48 MiB` saved forward state, and would need `65536.0 MiB` for one
  dense `N*H*W` float slab. `4096` cells used `2510718` candidates,
  `10.14 MiB` index state, `74.14 MiB` saved state, and would need
  `262144.0 MiB` for one dense slab.
- The refreshed one-iteration 4096x4096 timings were noisy/slower than the
  selected median tile16 artifact: `1024` cells `1065.9 ms` forward /
  `6027.7 ms` backward / `7093.6 ms` total, and `4096` cells `2232.2 ms`
  forward / `8646.8 ms` backward / `10879.0 ms` total. Keep
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_tile16_2026-05-03.json`
  as the timing reference until a full stable rerun is needed.
- Ran the fixed P4 timing/accounting matrix:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_matrix_128_256_512_accounting_2026-05-03.json`.
  Median totals for `N=256/1024/4096` were `35.3/51.8/48.1 ms` at `128x128`,
  `36.5/47.8/65.7 ms` at `256x256`, and `73.6/126.4/227.6 ms` at `512x512`.
  The largest row (`512x512,N=4096`) saved `1.335 MiB` of tiled forward state
  versus `4096.0 MiB` for a single dense `N*H*W` float slab.
- Re-ran the full primitive Metal-vs-Torch reference check:
  `PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py`.
  The strict `quaternion_height_sv_texel_surface` case matched the Torch
  reference with feature max error `9.24e-7`, alpha max error `1.61e-6`, and
  gradient max errors no worse than `2.39e-7` across points, radii, density,
  texel sites/heights, SV axes/RGB, and quaternions. This validates the local
  Metal primitive math on tiny scenes; it does not address paper-scale static
  training or the still-slow 4K replay kernel.

## Later update: SfM point-cloud init

- Added explicit point-cloud initialization to `src/train/train_powerfoam_metal.py`.
  Supported inputs are ASCII/binary `.ply`, COLMAP `points3D.txt`, COLMAP
  `points3D.bin`, or a directory containing `input.ply`, `point_cloud.ply`,
  `points3D.*`, or `sparse/0/points3D.*`. The loader filters finite rows,
  samples deterministically to `model.cells`, optionally `fit_box` normalizes
  into the trainer bounds, and broadcasts the static centers/colors to every
  frame. Radii come from the existing KNN-radius estimator.
- Added repo-local fixture/config:
  `test_data/powerfoam_sfm_tiny_ascii.ply` and
  `src/train_configs/local_mac_powerfoam_metal_point_cloud_init_quaternion_height_sv_tiled_32_smoke.jsonc`.
  The config uses the full quaternion height+SV tiled primitive at `32px`,
  `16` cells, and one training step.
- Focused tests passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  reported `16 passed in 6.36s`. The new test verifies the PLY loader, static
  center broadcast, color initialization through SV texels, finite radii and
  densities, and quaternion-derived normals.
- Runtime single-video point-cloud smoke passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_point_cloud_init_quaternion_height_sv_tiled_32_smoke.jsonc`.
  It printed `init_point_cloud_source_count=8`, eval L1 `0.458758 -> 0.454509`,
  and nonzero post-step drift for centers, radii, density, quaternions, texel
  sites/heights, SV axes, and SV RGB.
- Runtime multicam point-cloud smoke also passed using the DeepView 3-camera
  smoke with `cells=16` and `model.init_point_cloud_path` set to the tiny PLY.
  Startup printed `frame_source=multicam_val`, train cameras
  `camera_0001/camera_0015`, heldout `camera_0040`, and
  `pose_source=deepview_models_relative_pinhole`. Eval L1 moved
  `0.378352 -> 0.377789`; heldout L1 moved `0.365270 -> 0.365193`.
- This closes the local SfM/COLMAP-style init plumbing, not the full static
  benchmark. Remaining gaps are still official schedules/losses, durable
  baseline rows, paper-scale grow/prune acceptance, ray tracing, differentiable
  aux losses if needed, and a genuinely fast 4K backward path.

## Later update: background compositing

- Added background compositing at the trainer boundary. `render.background`
  remains the fixed eval/logging RGB background; `render.background_mode` is
  `fixed` by default and can be set to `random` for per-image random training
  backgrounds. The default fixed black path preserves previous rendered values.
- The training loop now composites `rendered + (1 - alpha) * bg` before L1/MSE
  / SSIM losses, while eval/logging uses the fixed configured background so
  metrics are stable. Aux metrics still inspect the raw renderer/alpha path.
- Focused tests passed again with `17 passed in 8.00s`; the new unit test
  checks alpha compositing and random-background tensor shape/range.
- Runtime random-background smoke passed using the point-cloud init config with
  `render.background_mode=random`. Startup printed `background_mode=random`;
  train L1 was `0.255165`, eval L1 after the step was `0.455705`, and the
  same center/radius/density/quaternion/texel/SV parameter groups moved.

## Later update: official LR schedules

- Added official-style LR scheduling to `src/train/train_powerfoam_metal.py`.
  Existing multiplier-based configs keep `train.lr_schedule="constant"` by
  default. Configs can now set `train.lr_schedule="cosine"` and the upstream
  absolute LR keys (`points_lr_init/final`, `density_lr_init/final`,
  `radii_lr_init/final`, `quaternions_lr_init/final`,
  `texel_sites_lr_init/final`, `texel_sv_axis_lr_init/final`,
  `texel_sv_rgb_lr_init/final`, `texel_height_lr_init/final`). Density and
  radii use the official `1000`-step warmup; texel height uses `2000`.
- Added checked-in runtime smoke config
  `src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_tiled_32_smoke.jsonc`.
  It uses the full quaternion height+SV tiled primitive, point-cloud init, and
  upstream-style LR values from the static-scene configs.
- Validation passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  reported `18 passed in 6.21s`.
- Runtime smoke passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_tiled_32_smoke.jsonc`.
  Startup printed `lr_schedule=cosine`; eval L1 moved `0.458758 -> 0.449234`.
  The logged LR values show the intended official warmup behavior:
  step 1 after-update values had `lr_density=0.0`, `lr_radii=0.0`, and
  `lr_texel_height=0.0`; step 2 had `lr_density=0.001`,
  `lr_radii=5e-08`, and `lr_texel_height=2.5e-06`.
- This closes the trainer LR-schedule plumbing. It does not close the remaining
  loss-stack gap because the Metal aux outputs are still non-differentiable
  diagnostics unless aux-gradient routing is added.

## Later update: differentiable interpenetration loss

- Added `MetalPowerFoamVideo.interpenetration_loss(...)` and wired
  `losses.interpenetration_weight` into the Metal trainer. The edge set comes
  from the current Cech/AABB or KNN adjacency builder using detached geometry,
  matching the rest of the trainer's adjacency contract; the overlap penalty
  itself is differentiable through decoded centers and radii.
- Added official-style exponential scheduling for the interpenetration weight
  via `interpenetration_weight_final_multiplier`. This mirrors the direct
  trainer's decay shape without pretending the Metal normal/contribution aux
  diagnostics are differentiable losses.
- Focused tests passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  reported `19 passed in 4.24s`. The new unit test constructs two overlapping
  cells and verifies finite nonzero gradients on `raw_xy` and `raw_radii`.
- Re-ran the official LR schedule smoke with
  `losses.interpenetration_weight=1e-4`. It passed and logged
  `interpenetration_loss=8.9919` with weight `1e-4` at step 1, then
  `interpenetration_loss=8.9633` with scheduled weight `3.1623e-6` at step 2.
  Eval L1 moved `0.458758 -> 0.449245`.
- At this point contribution was still an open question; the follow-up below
  closes it through the differentiable alpha output. Normal-distance gradients
  remain the tiled aux-output loss gap.

## Later update: differentiable contribution loss

- Added `losses.contribution_weight` to the Metal trainer using the main
  differentiable alpha output. In this front-to-back compositor, summing
  per-cell contribution weights over cells is equal to the final rendered alpha
  per pixel, so the official contribution/sparsity regularizer can be expressed
  as `alpha.mean()` without waiting for aux-buffer gradients.
- Added `powerfoam_contribution_loss(...)` and included contribution in the
  same exponential loss-weight schedule family as the direct trainer. The pure
  unit test checks value, gradient (`1 / num_pixels`), and final scheduled
  weight (`0.1 -> 0.0001` over the full schedule).
- Focused tests passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  reported `20 passed in 6.34s`.
- Re-ran the official LR schedule smoke with both contribution and
  interpenetration enabled. It passed and logged `contribution_loss=0.38979`
  with weight `0.1` at step 1, then `contribution_loss=0.41735` with scheduled
  weight `0.0031623` at step 2. Eval L1 moved `0.458758 -> 0.449868`.
- At this point the remaining loss gap was normal-distance gradient routing.
  The follow-up below closes the official internal normal-distance loss through
  the tiled training output. External normal supervision/depth aux gradients
  remain separate.

## Later update: differentiable normal-distance loss

- Extended the tiled Metal training op so `rasterize_tiled_train_forward`
  returns `(features, alpha, normal_distance, log_t, tile_stop)`. Existing
  wrappers still return `(features, alpha)` unless callers request
  `return_normal_distance=True` on the full height+SV wrapper.
- Extended tiled replay backward with `grad_out_normal_distance`. The gradient
  contributes to the same transmittance/interval path as RGB/alpha and also
  writes direct surface-normal gradients for the learned frame. This covers the
  official internal `normal_err.mean()` style loss; it does not implement
  external Metric3D/finite-difference normal supervision.
- Added `losses.normal_weight` and `normal_weight_final_multiplier` to the
  Metal trainer schedule. The official-LR smoke config now enables normal,
  contribution, and interpenetration weights together.
- Rebuilt the extension:
  `cd third_party/powerfoam-metal && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace`.
- Validation passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  reported `21 passed in 3.06s`. The new MPS unit test sets a positive
  normal/view dot product and verifies finite nonzero gradients on
  `raw_quaternions` and `raw_densities`.
- Renderer checks still passed after the op-schema change:
  `third_party/powerfoam-metal/tests/tiled_streaming_check.py`,
  `third_party/powerfoam-metal/tests/aux_check.py`, and
  `third_party/powerfoam-metal/tests/linear_texture_check.py`.
- Re-ran the official loss smoke:
  `PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_tiled_32_smoke.jsonc`.
  It passed with `normal_weight=0.1`, `contribution_weight=0.1`, and
  `interpenetration_weight=1e-4`. The logged `normal_loss` was `0.0` for the
  default camera-facing initialization, which is expected because the internal
  loss only penalizes normals with positive dot against the ray direction.
- Refreshed a one-iteration 4K timing sample after adding the normal-distance
  output:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_normal_distance_output_1iter_2026-05-03.json`.
  It reported `1024` cells at `950.7 ms` forward / `4549.6 ms` backward /
  `5500.3 ms` total and `4096` cells at `1440.2 ms` forward /
  `6722.2 ms` backward / `8162.4 ms` total. Treat this as a quick regression
  check, not a stable new 4K baseline; the full height+SV 4K path remains much
  too slow.

## Later update: upstream source pin

- Added `research_notes/foam_papers/powerfoam_upstream_source.md` to pin the
  current official source revision used for parity checks. The scratch clone at
  `/tmp/powerfoam_official` was clean at commit
  `96392252ebd0059fe6ca98881b62e12295d9242f` (`GC to clear pytorch cache`) from
  `https://github.com/theialab/powerfoam`.
- This supersedes the older 2026-04-30 loose scan's `25d6f7b` note for current
  source-of-truth purposes. The old note is still useful chronology, but
  parity fixtures should record the new hash unless the official repo is
  intentionally refreshed again.

## Later update: baseline row

- Added a first PowerFoam Metal static-multiview smoke row to `BASELINES.md`
  under Tier 2a / DeepView 3-cam train2-test1. It points at
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_tiled_32_smoke.jsonc`
  and records the previously verified one-step numbers: train PSNR `7.9490`,
  heldout PSNR `7.8900`, heldout L1 `0.358152`, heldout SSIM `-0.0101`.
- The row is intentionally labeled a smoke/acceptance row, not a comparable
  128px/16f baseline. The proper baseline acceptance remains open until there
  is a longer run with wall time, complete metrics, and a run/artifact id.

## Later update: official geometric resample schedule

- Added `model.resample_final_cells`, `model.resample_from_step`, and
  `model.resample_until_step` to `src/train/train_powerfoam_metal.py`. When
  `model.resample_target_cells` is unset, the trainer now computes the official
  geometric growth target from the initial cell count to the final cell count.
  The helper intentionally uses Python `int(...)` truncation; for example
  `initial=4`, `final=16`, `from=2`, `until=6` yields targets
  `4, 6, 10, 15` on steps `2..5`.
- Decoupled the resample trigger from validation artifact logging. Previously a
  schedule could silently skip unless `step % image_log_every == 0`; now
  `should_resample_powerfoam_step(...)` fires from `resample_every` directly.
  If no artifact pass ran at that step, the trainer refreshes contribution and
  point-error EMAs from the current train batch before resizing.
- Validation passed:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  reported `22 passed in 2.36s`.
- Runtime smoke passed by calling `run_training(...)` with the official LR
  config overridden to `steps=3`, `image_log_every=999`, `resample_every=1`,
  `resample_final_cells=20`, `resample_from_step=1`, and
  `resample_until_step=3`. Resample printed at steps `1` and `2` despite no
  image-log pass at those steps; the live cell count grew from `16` to `20`,
  then step `3` rendered/backpropped/logged final eval with L1
  `0.458758 -> 0.438779`.

## Later update: SV color vector helper, texel-weight cache, endpoint reuse, SV-gradient reuse

- Added `stream_sv_texel_color(...)`, a vector-valued helper for mode-6
  spherical-Voronoi texel color. The active tiled forward, aux, and global
  backward paths now compute each texel RGB color with one shared SV denominator
  instead of calling `stream_sv_texel_color_component(...)` three times.
- Also cached up to 16 texel softmax weights from the texel-denominator pass
  and reused them in the immediate mode-6 color/gradient pass. The benchmarked
  configs use 4 texels, so this removes one duplicate `exp(...)` pass over the
  texels in each active forward/aux/backward cell hit.
- Added `stream_route_height_surface_endpoint_grad(...)` so tiled backward can
  reuse the base clipped interval it already computed before applying the
  height surface. Previously `stream_route_endpoint_grad(..., -4, ...)`
  recomputed the full base interval, including another Cech/AABB adjacency
  clip, for height-surface endpoint gradients.
- Added `stream_route_sv_color_grad_known_value(...)` so tiled backward can
  reuse the raw SV color and denominator computed for the texel color pass
  rather than making `stream_route_sv_color_grad(...)` recompute that first
  loop before routing SV RGB/axis gradients.
- Validation passed after rebuilding the extension:
  `third_party/powerfoam-metal/tests/tiled_streaming_check.py`,
  `third_party/powerfoam-metal/tests/aux_check.py`,
  `third_party/powerfoam-metal/tests/linear_texture_check.py`, and
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  (`22 passed in 4.89s` after the final known-value SV-gradient change).
- Saved one-iteration 4K regression artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_sv_grad_known_value_1iter_2026-05-03.json`.
  It reported `1024` cells at `566.5 ms` forward / `4180.4 ms` backward /
  `4746.9 ms` total and `4096` cells at `851.7 ms` forward /
  `6125.8 ms` backward / `6977.5 ms` total.
- Saved a short stable median:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_sv_grad_known_value_median_2026-05-03.json`.
  Median `1024` cells: `554.5 ms` forward / `4095.8 ms` backward /
  `4650.3 ms` total. Median `4096` cells: `882.1 ms` forward /
  `6137.8 ms` backward / `7013.0 ms` total.
- This is a real improvement over the prior tile16 median artifact
  (`5501.8 ms` total at `1024`, `8753.0 ms` at `4096`) and over the quick
  post-normal-output sample (`5500.3 ms`, `8162.4 ms`). It is still far too slow
  for the "fast 4K" goal; the replay/backward path remains the bottleneck.
- Rejected a follow-up per-texel view-direction cache after parity checks
  passed but timing failed to beat the known-value SV-gradient path:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_viewdir_cache_1iter_2026-05-03.json`
  reported `4778.6 ms` total at `1024` cells and `7040.4 ms` at `4096`.
  The extra temporary arrays likely added enough register pressure to erase the
  saved normalization work, so the cache was reverted. The selected code state
  was revalidated with `tests/test_powerfoam_direct.py` (`22 passed in 2.84s`).

## Later update: precomputed power-face diffs

- Added official-style `adjacency_diff` packing for the trainable tiled Metal
  path. The Python wrapper now packs one `[E,4]` float32 tensor holding
  `adjacent_center - center` plus the radical-plane power-midpoint delta, and
  the tiled forward, aux, and backward kernels use it for interval clipping
  instead of loading each neighbor point/radius in the hot edge loop.
- The extension boundary now passes `adjacency_diff` only through the tiled
  ops. The non-tiled streaming path still uses the old direct point/radius
  formula, which keeps it as an independent parity reference.
- Added an empty `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}` to
  `csrc/bindings.cpp`. Without it, direct extension import can fail with
  `dynamic module does not define module export function (PyInit__C)` and
  leave `torch.ops.powerfoam_metal` unregistered in fresh Python processes.
- Tiled-vs-streaming forward now differs at the few-ulp level because face
  constants are precomputed and then consumed in a separate Metal expression.
  The parity tolerance in `tiled_streaming_check.py` was widened to `3e-6`;
  gradient gates stayed at `1e-5`.
- Validation after rebuild:
  `third_party/powerfoam-metal/tests/tiled_streaming_check.py`,
  `third_party/powerfoam-metal/tests/aux_check.py`,
  `third_party/powerfoam-metal/tests/linear_texture_check.py`, and
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q`
  (`22 passed in 4.50s`).
- Trainer smoke still passes with the rebuilt extension:
  `PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_tiled_32_smoke.jsonc`.
  The 2-step smoke moved eval L1 from `0.458758` at step 0 to `0.449871` at
  step 2, and the state deltas confirm points/quaternions/texels/SV RGB were
  updated.
- Saved one-iteration 4K artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_adjdiff_1iter_2026-05-03.json`.
  It reported `1024` cells at `195.8 ms` forward / `1437.7 ms` backward /
  `1633.5 ms` total and `4096` cells at `333.9 ms` forward /
  `2165.1 ms` backward / `2499.0 ms` total.
- Saved a short median artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_adjdiff_median_2026-05-03.json`.
  Median `1024` cells: `207.4 ms` forward / `1443.1 ms` backward /
  `1650.5 ms` total. Median `4096` cells: `340.2 ms` forward /
  `2287.2 ms` backward / `2627.5 ms` total.
- Added a guarded selector for `powerfoam_tiled_backward_height_sv_feature_reduced`.
  It is only selected for mode 6 when `feature_dim <= 128` and the
  normal-distance output is unused by autograd; if normal-distance loss is
  active, the generic backward remains selected so those gradients are not
  dropped.
- Validation after that selector still passed:
  `tiled_streaming_check.py`, `aux_check.py`, `linear_texture_check.py`,
  `tests/test_powerfoam_direct.py` (`22 passed in 2.65s`), and the official LR
  32px trainer smoke (`0.458758 -> 0.449871` eval L1).
- Saved the now-selected reduced-mode6 median artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_adjdiff_mode6reduced_median_2026-05-03.json`.
  Median `1024` cells: `191.9 ms` forward / `1429.6 ms` backward /
  `1621.5 ms` total. Median `4096` cells: `309.4 ms` forward /
  `2149.1 ms` backward / `2456.7 ms` total.
- This materially beats the previous `sv_grad_known_value` median (`4650.3 ms`
  / `7013.0 ms` total) and the plain face-diff median (`1650.5 ms` /
  `2627.5 ms` total). It is still not the final "fast 4K" answer: full
  height+SV backward remains seconds-scale, so the next architectural step is
  still a raytrace/replay backend or another traversal that avoids replaying
  all tiled candidates.

## Later update: forward-only raytrace probe

- Added a forward-only Metal raytrace probe exposed as `raytrace_power_foam`
  and `raytrace_power_foam_flat`. It uses the same `adjacency_diff` packing,
  picks the per-camera start cell by nearest power distance, and walks outgoing
  power faces per pixel. It is intentionally not autograd yet.
- Added `third_party/powerfoam-metal/tests/raytrace_check.py`, covering a
  two-cell constant-feature parity fixture and a random 8-cell all-pairs graph.
  The two-cell case matched exactly and reported `steps range: 2 2`, so the
  probe genuinely crosses one adjacency face. The all-pairs random case matched
  raster forward within `2.98e-08` features / `0.0` alpha with mean/max steps
  `2.85` / `4`.
- Added `--foam-raytrace` to `benchmark_powerfoam_metal.py` for forward-only
  constant-feature timing. The benchmark also reports mean/max raytrace walk
  steps when this mode is used.
- The naive raytrace probe is not selected as the fast path. On random 4K
  synthetic scenes:
  `outputs/benchmarks/powerfoam_metal_constant_raytrace_cech_aabb_4k_forward_median_steps_2026-05-03.json`
  reported `231.4 ms` / `954.9 ms` forward for `1024` / `4096` cells with mean
  walk steps `11.5` / `14.1`. The KNN32 raytrace artifact
  `outputs/benchmarks/powerfoam_metal_constant_raytrace_knn32_4k_forward_median_2026-05-03.json`
  reported `275.5 ms` / `860.3 ms`, while the KNN32 tiled constant artifact
  `outputs/benchmarks/powerfoam_metal_constant_tiled_knn32_4k_forward_median_2026-05-03.json`
  reported `111.6 ms` / `178.1 ms`.
- Interpretation: the probe is useful for parity and traversal experiments, but
  the current per-pixel walk over the local adjacency graph is slower than the
  tiled candidate path. A real raytrace replacement needs a traversal-complete
  graph and a cooperative/replay design before it can plausibly solve fast 4K
  backward.

## Later update: height+SV raytrace parity and forward timing

- Added a forward-only high-level wrapper for the full height+SV primitive:
  `raytrace_power_foam_oriented_height_sv_texel_surface`. It packs the same
  flattened feature layout as `rasterize_power_foam_oriented_height_sv_texel_surface`
  and calls the Metal ray-walk kernel with `feature_mode=6`.
- Extended `third_party/powerfoam-metal/tests/raytrace_check.py` beyond the
  constant fixtures. The check now also compares a random all-pairs height+SV
  scene against the raster path. It passed with:
  - two-cell constant: exact feature/alpha match, steps `2..2`
  - random all-pairs constant: `2.98e-08` feature error, `0.0` alpha error,
    mean/max steps `2.85` / `4`
  - random all-pairs height+SV: `5.96e-08` feature error, `0.0` alpha error,
    mean/max steps `2.12` / `4`
- Extended `--foam-raytrace` in
  `third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py` to
  support forward-only height+SV timing. It still rejects backward, and result
  rows now report `backward_supported=false` for raytrace mode.
- Saved the height+SV raytrace 4K median artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_forward_median_2026-05-03.json`.
  It reported:
  - `1024` cells: `170.7 ms` median forward, mean/max steps `11.5` / `26`
  - `4096` cells: `227.2 ms` median forward, mean/max steps `14.1` / `36`
- This is now the fastest measured full height+SV 4K forward path in this local
  lane. It does not make training fast yet, because the raytrace path is still
  forward-only and has no replay/backward kernel.
- I also tried two reduced-backward shader micro-optimizations and rejected
  both after timing:
  - caching height+SV texel/SV values in local arrays preserved parity but
    regressed one-iteration 4K totals to `1990.4 ms` / `2938.0 ms`
    (`1024` / `4096`);
  - zeroing the local grad buffer up to dynamic `feature_dim` instead of fixed
    `128` preserved parity but regressed to `1858.1 ms` / `2769.4 ms`.
  Both were backed out. The selected trainable tiled artifact remains
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_adjdiff_mode6reduced_median_2026-05-03.json`.
- Validation after the wrapper/benchmark changes:
  `raytrace_check.py`, `tiled_streaming_check.py`, `aux_check.py`,
  `tests/test_powerfoam_direct.py` (`22 passed in 5.21s`), and the official LR
  32px trainer smoke. The trainer smoke again moved eval L1
  `0.458758 -> 0.449871` and state deltas confirmed centers/quaternions/
  texels/SV colors moved.

## Later update: experimental trainable raytrace replay

- Added a Metal replay/backward kernel for height+SV raytrace:
  `powerfoam_raytrace_backward_height_sv_global_atomic`. It recomputes the
  ray-walk, stores a fixed-cap event list per pixel (`FOAM_RAYTRACE_MAX_EVENTS`
  currently `64`), then replays events in reverse to route gradients for
  centers, radii, densities, texel sites, heights, SV axes/RGB, and normals.
- Wired the backward through `powerfoam_metal.raytrace_height_sv_backward`, a
  Python autograd function, and
  `raytrace_power_foam_oriented_height_sv_texel_surface`. The constant
  raytrace helper remains forward-only; the trainable path is currently
  height+SV only with `feature_dim <= 128`.
- Promoted a backward parity gate into
  `third_party/powerfoam-metal/tests/raytrace_check.py`. The small all-pairs
  height+SV fixture now checks both forward and gradients. Latest max gradient
  errors against raster were:
  `points 1.49e-08`, `radii 8.38e-09`, `densities 3.73e-09`,
  `texel_sites 2.98e-08`, `texel_heights 6.98e-10`,
  `texel_sv_axis 1.12e-08`, `texel_sv_rgb 1.86e-08`, `normals 1.40e-09`.
- The guarded 4K synthetic train median is now:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_train_capguard_median_2026-05-03.json`.
  It reported:
  - `1024` cells: `166.6 ms` forward / `826.1 ms` backward / `988.5 ms` total
  - `4096` cells: `201.8 ms` forward / `781.8 ms` backward / `983.9 ms` total
  This beats the selected tiled train median (`1621.5 ms` / `2456.7 ms` total)
  and is the first guarded sub-second full height+SV 4K forward+backward number
  in this lane. The guard checks `steps.max()` against the fixed replay event
  cap (`64`) before autograd backward can silently truncate an overlong walk.
- Wired `render.use_raytrace` into `src/train/train_powerfoam_metal.py` for
  oriented/quaternion height+SV modes and added the smoke config
  `src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_raytrace_32_smoke.jsonc`.
  At this point the config disabled `losses.normal_weight` because raytrace did
  not return differentiable `normal_distance` yet; that is superseded by the
  later normal-distance update below. The smoke passed with
  `render_backend: raytrace`, eval L1 `0.460920 -> 0.454128`, and nonzero
  center/quaternion/texel/SV state deltas.
- Regression set after the raytrace replay wiring:
  - `raytrace_check.py` passed with forward and backward parity;
  - `tiled_streaming_check.py` passed;
  - `aux_check.py` passed;
  - `tests/test_powerfoam_direct.py` passed (`22 passed in 7.08s`);
  - tiled official LR smoke still passed (`0.458758 -> 0.449871` eval L1).
- Remaining caveats before calling this full paper PowerFoam:
  raytrace replay has a fixed event cap, only supports mode-6 height+SV
  features, does not route differentiable depth-quantile / external-normal
  losses, and still uses the local Cech/AABB adjacency graph rather than a
  traversal-complete weighted-Delaunay regular triangulation as the selected
  fast path. The normal-distance caveat was closed in the later update below.

## Later update: optional regular-triangulation adjacency

- Added `third_party/powerfoam-metal/torch_powerfoam_metal/regular_triangulation.py`.
  It computes weighted-Delaunay / regular-triangulation edges by lifting points
  to `(x,y,z, ||p||^2 - r^2)`, taking SciPy/Qhull lower-hull facets, and
  extracting pairwise edges from the projected regular tetrahedra. Sites with
  hidden cells naturally get no lower-hull edges.
- Wired `regular_triangulation` into:
  - `torch_powerfoam_metal.random_scene.make_adjacency`;
  - `train_powerfoam_metal.build_csr_adjacency`;
  - the benchmark `--adjacency` choices.
  This is optional: local `.venv` does not include SciPy, so users must run with
  SciPy installed or use `uv --with scipy`. The error message points back to
  `adjacency_mode='cech_aabb'` as the fallback.
- Added a skipped-by-default pytest covering the topology contract. Running
  `uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy --with pytest python -m pytest tests/test_powerfoam_direct.py -q -k regular_triangulation`
  passed and proved the zero-weight regular graph exactly matches SciPy
  `Delaunay` edges.
- Benchmarking the regular graph showed why it is not the selected fast path
  yet. Saved artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_regular_triangulation_4k_train_capguard_median_2026-05-03.json`.
  It reported:
  - `1024` cells: avg degree `13.65`, `314.0 ms` forward /
    `1605.2 ms` backward / `1910.3 ms` total
  - `4096` cells: avg degree `13.75`, `550.4 ms` forward /
    `2482.1 ms` backward / `3040.1 ms` total
  The regular graph is available and verified, but the Cech/AABB raytrace train
  path is still much faster on this synthetic setup (`988.5 ms` / `983.9 ms`
  total).
- Regression checks after this wiring:
  - py_compile for the helper, trainer, benchmark, package, and tests passed;
  - no-SciPy `tests/test_powerfoam_direct.py` passed with the regular test
    skipped (`22 passed, 1 skipped`);
  - SciPy regular-topology test passed separately;
  - `raytrace_check.py` still passed including height+SV backward parity.

## Later update: raytrace normal-distance training output

- Finished the differentiable `normal_distance` path for height+SV raytrace.
  `powerfoam_raytrace_forward` now emits a normal-distance image, and
  `powerfoam_raytrace_backward_height_sv_global_atomic` accepts
  `grad_out_normal_distance`. The replay backward mirrors the tiled normal-loss
  route: it adds the `ndv^2` contribution to transmittance/opacity gradients and
  atomically accumulates the surface-normal gradient.
- Updated the C++/Python schema so `raytrace_forward` returns
  `(out, alpha, normal_distance, steps)` and the autograd wrapper can return
  normal distance and/or walk steps. The trainer no longer rejects
  `render.use_raytrace` with `losses.normal_weight > 0`.
- Re-enabled normal loss in
  `src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_raytrace_32_smoke.jsonc`
  (`normal_weight=0.1`, final multiplier `0.1`). The two-step MPS smoke passed
  with `render_backend=raytrace`, eval L1 `0.460920 -> 0.454128`, and nonzero
  movement in centers, density, quaternions, texel sites/heights, SV axes/RGB,
  and normals. The default camera-facing init still gives `normal_loss=0.0`,
  so the nonzero normal-distance gradient is covered by the parity fixture.
- `third_party/powerfoam-metal/tests/raytrace_check.py` now checks
  normal-distance forward parity and gradients from
  `normal_distance.square().mean()`. Latest output:
  - height+SV forward feature/alpha/normal-distance max error: `0.0` / `0.0` /
    `0.0`
  - gradient max errors: `points 3.73e-09`, `radii 8.38e-09`,
    `densities 2.79e-09`, `texel_sites 1.49e-08`,
    `texel_heights 5.82e-10`, `texel_sv_axis 9.31e-09`,
    `texel_sv_rgb 3.35e-08`, `normals 9.31e-10`.
- Regression set after the normal-distance wiring:
  - `py_compile` on the changed Python files passed;
  - `raytrace_check.py` passed;
  - `tiled_streaming_check.py` passed;
  - `aux_check.py` passed;
  - `tests/test_powerfoam_direct.py` passed (`22 passed, 1 skipped`);
  - raytrace official-LR 32px smoke passed;
  - tiled official-LR 32px smoke passed.
- Refreshed guarded 4K train artifacts after the extra output:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_train_normaldistance_median_2026-05-03.json`
  reports `1024` cells at `176.6 ms` forward / `835.8 ms` backward /
  `1016.1 ms` total and `4096` cells at `217.6 ms` forward / `798.0 ms`
  backward / `1014.4 ms` total. This is just over one second after the
  normal-distance plane/zero-gradient handling, versus the previous
  `988.5 ms` / `983.9 ms` guarded artifact without that output.
- Refreshed regular-triangulation with `uv --with scipy`:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_regular_triangulation_4k_train_normaldistance_median_2026-05-03.json`
  reports `1024` cells at `303.8 ms` forward / `1564.9 ms` backward /
  `1868.7 ms` total and `4096` cells at `529.8 ms` forward /
  `2355.7 ms` backward / `2888.3 ms` total. The proper graph is still slower
  than Cech/AABB because degree and walk steps are higher.

## Later update: regular-triangulation render parity gate

- Extended `third_party/powerfoam-metal/tests/raytrace_check.py` with an optional
  regular-triangulation block. The normal `.venv` run skips the block because
  SciPy is absent, preserving the default smoke path. The SciPy-backed run:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy python third_party/powerfoam-metal/tests/raytrace_check.py`
  passed.
- The new regular constant-fixture compares dense all-pairs raster against
  regular-triangulation raytrace. Latest max errors were `1.79e-07` features and
  `0.0` alpha; the graph had avg degree `6.6`, mean steps `3.86`, max steps `6`.
- The new regular height+SV fixture compares dense all-pairs tiled raster against
  regular-triangulation raytrace for forward and backward, including
  normal-distance. Latest max forward errors were `0.0` features / `0.0` alpha /
  `0.0` normal-distance. Gradient max errors were:
  `points 4.47e-08`, `radii 1.01e-07`, `densities 7.45e-09`,
  `texel_sites 1.30e-08`, `texel_heights 2.24e-08`,
  `texel_sv_axis 9.31e-09`, `texel_sv_rgb 5.59e-09`,
  `normals 1.49e-08`.
- This closes the tiny-scene traversal-graph parity gap for regular topology.
  It does not change the selected fast path: regular-triangulation remains
  slower than Cech/AABB on the 4K synthetic benchmark.
