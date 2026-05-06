# PowerFoam Raytrace Multicam Smoke

Date: 2026-05-05

## Context

The open PowerFoam acceptance gap was not whether the single-video Metal
raytrace path could train; that already passed. The next concrete gap was a
posed-camera smoke that exercises the raytrace height+SV backend through the
DeepView 3-cam train2/test1 loader, heldout logging, and the normal /
contribution / interpenetration loss wiring.

## Command

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke.jsonc
```

`py_compile` on `src/train/train_powerfoam_metal.py` passed immediately before
the run.

## Result

The 1-step smoke passed on MPS with:

- `render_backend: raytrace`
- `feature_mode: quaternion_height_sv_texel_surface`
- `adjacency_mode: cech_aabb`
- train views `camera_0001`, `camera_0015`
- heldout view `camera_0040`
- 64 cells, 2 frames, 32px render
- `adjacency_avg_degree: 6.78125`
- `adjacency_missing_overlap_edges: 0.0`

Initial eval:

- train/eval PSNR `8.132930`
- heldout PSNR `8.312609`
- heldout L1 `0.328643`
- heldout SSIM `0.046593`

Step 1 train:

- loss `0.379986`
- L1 `0.322378`
- MSE `0.150085`
- normal weight `0.1`
- contribution loss `0.416767`
- interpenetration loss `9.173128`
- elapsed train step `1.316392 s`

Final eval:

- train/eval L1 `0.327565`
- train/eval PSNR `8.143913`
- heldout L1 `0.328307`
- heldout PSNR `8.316828`
- heldout SSIM `0.045195`

State deltas confirmed the raytrace path is trainable on the posed-camera smoke:

- mean center delta `7.507e-04`
- mean radius delta `4.931e-05`
- mean density delta `0.336507`
- mean quaternion delta `0.099873`
- mean texel-site delta `0.002659`
- mean texel-height delta `1.400e-04`
- mean SV-axis delta `0.041286`
- mean SV-RGB delta `0.002500`

Artifacts landed under:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke/
```

including `checkpoint_final.pt`, train and heldout previews/videos, and
`resolved_config.json`.

## Interpretation

This closes the specific smoke-level gap for "raytrace PowerFoam Metal can train
on a posed multiview sample and log heldout outputs." It does not close the
paper-reproduction gap: the run is 32px, 2 frames, 64 cells, and 1 optimizer
step. It should be treated as an acceptance smoke, not as a comparable
heldout-camera baseline against the 128px/16f 2048/8192-primitive rows.

Remaining paper-level work is still: longer static multiview PowerFoam runs,
paper-scale schedules/densification acceptance, upstream parity fixtures, and
proper baseline rows with wall time plus heldout metrics.

## Follow-up 128px / 16f Probe

Added and ran:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step.jsonc
```

Config shape:

- 128px render/loss
- 16 frames
- DeepView `03_Dog`
- train cameras `camera_0001`, `camera_0015`
- heldout camera `camera_0040`
- 1024 cells
- `cech_aabb`
- raytrace height+SV backend
- normal, contribution, and interpenetration losses enabled
- 80 steps, 4 sampled train views per step

The run completed without event-cap or memory failure:

- train-loop elapsed `21.632119 s`
- initial train/eval PSNR `7.730346`
- initial heldout PSNR `8.120396`
- final train/eval PSNR `7.451306`
- final heldout PSNR `7.813736`
- final heldout L1 `0.354624`
- final heldout SSIM `0.001364`

State still moved substantially:

- mean center delta `0.008441`
- mean density delta `0.341267`
- mean quaternion delta `1.004997`
- mean texel-site delta `0.030848`
- mean SV-axis delta `0.225465`
- mean SV-RGB delta `0.039717`

Interpretation: this upgrades the evidence from tiny smoke to a real 128px/16f
posed-camera run, but it is not good. The optimizer/loss schedule appears to
over-rotate or over-move the representation: train and heldout PSNR both
degraded from initialization. The next useful run should reduce quaternion/SV
axis learning rates and probably lower or delay normal/contribution penalties,
instead of increasing cells or steps blindly.

Artifacts:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step/
```

## LR Warmup Bug

While reading the LR code, I found that the trainer initialized optimizer
groups with their initial LR and then called `update_powerfoam_learning_rates`
after `optimizer.step()`. That means warmup groups took one oversized first
update before being brought down to their warmup value. For this config, density
was initialized as an official-style group with warmup, so the first step could
use LR `1.0` before later logs showed the intended small warmup values.

Patched `src/train/train_powerfoam_metal.py` so the per-step LR update happens
before forward/backward/`optimizer.step()`.

The 32px raytrace multicam smoke still passed after the fix:

- final train/eval PSNR `8.143932`
- final heldout PSNR `8.316847`
- density/radius/height deltas stayed effectively zero on the 1-step run,
  because those groups are now genuinely in warmup
- centers/quaternions/texel sites/SV axes/RGB still moved

## Warmup-Fixed 128px Probe

Ran:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_warmupfix.jsonc
```

Result:

- train-loop elapsed `29.802491 s`
- final train/eval PSNR `7.451840`
- final heldout PSNR `7.817927`
- final heldout L1 `0.354255`
- final heldout SSIM `0.001494`
- mean density delta dropped versus the bad first run (`0.182257` vs `0.341267`),
  but mean quaternion delta was still large (`0.992997`)

Interpretation: the warmup bug was real, but fixing it alone does not save the
official-style 128px schedule.

## Low-Geometry / No-Aux Control

Ran:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_noaux.jsonc
```

Changes vs warmup-fixed probe:

- much lower point/quaternion/texel-site/SV-axis rates
- higher SV-RGB rate
- normal/contribution/interpenetration losses disabled

First result before best-checkpoint tracking:

- train-loop elapsed `36.176678 s`
- train/eval PSNR improved `7.730346 -> 8.055780`
- final heldout PSNR `8.088146`
- best heldout PSNR was step 20: `8.141685`
- final heldout L1 `0.336985`
- final heldout SSIM `0.016307`
- mean quaternion delta stayed controlled at `0.104007`
- mean SV-RGB delta was `0.116348`

Interpretation: the 128px raytrace backend can optimize the source views when
geometry is not destabilized. Heldout does not improve monotonically and final
heldout remains weak, so the next paper-reproduction work is checkpoint/early
stopping plus a schedule that introduces normal/contribution regularization
later, not more low-level Metal work.

## Best-Checkpoint Tracking

Added best-checkpoint tracking to `src/train/train_powerfoam_metal.py`.
Whenever `log_artifacts` returns validation metrics, the trainer now chooses
`heldout_eval_psnr` when heldout exists, otherwise `eval_psnr`, and writes:

```text
checkpoint_best.pt
best_metrics.json
```

It still writes `checkpoint_final.pt` at the end.

Re-ran the low-geometry/no-aux control after adding this. The rerun preserved
the same metric story and produced the best artifacts:

- train-loop elapsed `17.921399 s`
- final train/eval PSNR `8.055424`
- final heldout PSNR `8.088144`
- final heldout L1 `0.336985`
- final heldout SSIM `0.016306`
- `best_metrics.json` selected step `20`
- best heldout PSNR `8.141685`
- best heldout L1 `0.332693`
- best heldout SSIM `0.021779`

Artifacts:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_noaux/checkpoint_best.pt
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_noaux/best_metrics.json
```

## Delayed-Aux Probe

Added start-step gates for the three aux regularizers:

- `losses.normal_weight_start_step`
- `losses.contribution_weight_start_step`
- `losses.interpenetration_weight_start_step`

Then ran:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_delayed_aux.jsonc
```

This uses the low-geometry recipe, then turns on small normal/contribution/
interpenetration weights after step 40. The run completed and saved best/final
checkpoints.

Result:

- train-loop elapsed `36.806758 s`
- final train/eval PSNR `8.055440`
- final heldout PSNR `8.088228`
- final heldout L1 `0.336987`
- final heldout SSIM `0.016309`
- `best_metrics.json` selected step `20`
- best heldout PSNR `8.141685`

Interpretation: the delayed-aux plumbing works, but this schedule did not
improve heldout. The best checkpoint is still before aux losses activate, and
the final result is essentially the same as the no-aux control. Treat this as a
negative control.

## 4096-Cell Capacity / Replay-Cap Probe

Tried a 4096-cell version of the same 128px/16f low-geometry/no-aux raytrace
DeepView recipe to check whether the weak 1024-cell quality was mostly a
capacity issue. The default raytrace replay cap rejected the first training
forward:

```text
RuntimeError: raytrace height+SV backward replay cap exceeded: max_steps=75, cap=64
```

Raised the local replay cap to 128 as a temporary experiment, rebuilt
`third_party/powerfoam-metal`, and reran `raytrace_check.py`. Parity still
passed at small-scene scale. The 4096-cell DeepView run then completed, but
quality moved the wrong way:

- output dir:
  `outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_4096cells_80step_lowgeom_noaux`
- train-loop elapsed `106.455977 s`
- step-0 heldout PSNR `8.177620`
- final heldout PSNR `7.809724`
- final heldout L1 `0.354822`
- final heldout SSIM `0.001652`
- `best_metrics.json` selected step `0`

The cap-128 build also regressed the selected synthetic 4K raytrace train
benchmark badly:

```text
outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_train_cap128_median_2026-05-05.json
```

Median totals were `3103.7 ms` for 1024 cells and `3103.1 ms` for 4096 cells,
versus the selected cap-64 normal-distance artifact at roughly `1016.1 ms` /
`1014.4 ms`. The cap was reverted to 64 and the checked-in 4096 config was not
kept. This makes the conclusion sharper: simple capacity increase plus a larger
replay event budget is not the paper-quality path, and it damages the fast-4K
claim.

## Color-Only Freeze Probe

Added and ran:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_coloronly_noaux.jsonc
```

This keeps the same 1024-cell / 128px / 16f DeepView split and raytrace backend
but sets every explicit LR to zero except `texel_sv_rgb_lr_*`. The goal was to
separate geometry over-motion from pure train-view color overfit.

Result:

- train-loop elapsed `15.553846 s`
- final train/eval PSNR `8.043037`
- final train/eval L1 `0.319052`
- final heldout PSNR `8.073027`
- final heldout L1 `0.337412`
- final heldout SSIM `0.012888`
- `best_metrics.json` selected step `0`
- best heldout PSNR `8.120396`
- geometry/density/site/height/SV-axis deltas stayed zero; mean SV-RGB delta
  was `0.122881`

Interpretation: freezing geometry avoids the low-geometry run's quaternion and
site drift and makes the train-view fit faster, but heldout still gets worse.
The current failure is therefore not only "geometry LR too high"; the image-
depth geometry plus two-view color fitting is already a weak novel-view
contract.

## Train-Camera Plane-Sweep Init Probe

Added a reproducible train-camera-only plane-sweep PLY builder:

```text
research_experiments/dynamic_foam/build_multiview_plane_sweep_point_cloud.py
```

It loads a PowerFoam multicam config, uses only the configured train cameras,
samples candidate depths, projects into the opposite train view, and keeps the
best color-consistency point for each sampled source pixel. The generated
artifact for the DeepView train2/test1 lane is:

```text
research_experiments/dynamic_foam/artifacts/deepview_03_dog_train2_plane_sweep_frame0_128px_stride2_8192pts.ply
research_experiments/dynamic_foam/artifacts/deepview_03_dog_train2_plane_sweep_frame0_128px_stride2_8192pts.json
```

Builder summary:

- train cameras `camera_0001`, `camera_0015`
- heldout camera not used
- frame index `0`
- depth range `1.0..3.25`, `96` depths
- stride `2`
- valid/output point count `5799`
- median color-consistency error `0.028078`
- p90 error `0.261013`

Then ran:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_plane_sweep_init_lowgeom_noaux.jsonc
```

Result:

- train-loop elapsed `14.741095 s`
- step-0 train/eval PSNR `7.655830`
- final train/eval PSNR `7.735542`
- step-0 heldout PSNR `7.777877`
- final heldout PSNR `7.776456`
- final heldout L1 `0.357705`
- final heldout SSIM `0.001193`
- `best_metrics.json` selected step `0`

Interpretation: naive two-view plane sweep is worse than the existing
image-depth initialization on this split. The resulting cloud sits much nearer
than the image-depth init (`aux_mean_median_depth ~1.317` vs ~2.18), and the
heldout camera remains poor. This rejects "any train-view multiview point cloud
will fix it"; the next init work needs a stronger SfM/COLMAP-quality artifact,
all-train-view consistency, masks, or a different representation contract.

## Neural3D EX4DGS Point-Cloud Init Probe

The next question was whether the local Metal trainer could use a real
scene-scale point-cloud init instead of image-depth or naive two-view geometry.
I first checked the available EX4DGS `coffee_martini/input.ply` against the
Neural3D `poses_bounds.npy` loader. The EX4DGS camera centers match the
Neural3D pose bundle, but the trainer stores points in the anchor-camera frame,
so raw world coordinates need to be transformed by `inv(anchor_c2w)`.

Added:

```text
research_experiments/dynamic_foam/prepare_ex4dgs_anchor_point_cloud.py
```

Generated:

```text
research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_cam04_anchor_ex4dgs_input_128px_xy24_z4_120_trainvisible.ply
research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_cam04_anchor_ex4dgs_input_128px_xy24_z4_120_trainvisible.json
```

Artifact summary:

- sample `neural3d_coffee_martini_train_cam04_cam09_holdout_cam06`
- anchor camera `cam04`
- train cameras `cam04`, `cam09`
- heldout camera `cam06`
- source points `5498`
- kept points `5113`
- filter box `xy_extent=24`, `z=4..120`
- after box filtering, projection coverage was `81.2%` for `cam04`, `70.8%`
  for `cam09`, and `76.1%` for heldout diagnostic `cam06`

Then ran:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_ex4dgs_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

Result:

- 1024 cells, `cech_aabb`, height+SV raytrace, low-geometry/no-aux schedule
- train-loop elapsed `5.940771 s`
- initial train/eval PSNR `10.068101`
- final train/eval PSNR `10.373013`
- initial heldout PSNR `10.033043`
- final/best heldout PSNR `10.741616`
- final heldout L1 `0.229993`
- final heldout SSIM `0.160671`
- `best_metrics.json` selected step `40`
- state deltas were nonzero for center, radii, density, quaternion, texel site,
  height, SV axis, and SV RGB

Interpretation: this is the first positive real-scene SfM-style init signal for
PowerFoam Metal. The key implementation detail is coordinate-system honest
init: anchor-transform the cloud and expand the PowerFoam box to scene scale.
It is not a clean paper benchmark, because the point cloud came from an
external pretrained EX4DGS artifact and the run is only 40 steps, but it closes
the "can a real point cloud initialize and train through Metal raytrace?"
question.

## Neural3D EX4DGS Longer Probe

Added and ran a 200-step version of the same setup:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_ex4dgs_init_raytrace_128_16f_1024cells_200step_lowgeom_noaux.jsonc
```

Result:

- train-loop elapsed `55.752085 s`
- no raytrace replay-cap failure
- step-0 heldout PSNR `10.033043`
- step-50 heldout PSNR `11.321913`
- step-100 heldout PSNR `11.906818`
- step-150 heldout PSNR `11.979544`
- final step-200 heldout PSNR `11.949820`
- final heldout L1 `0.194974`
- final heldout SSIM `0.190488`
- `best_metrics.json` selected step `150`
- best checkpoint heldout L1 / SSIM were `0.194809` / `0.190030`

Interpretation: longer training did not immediately overfit the way the
DeepView image-depth lane did. It kept improving until step 150 and only dipped
slightly by step 200. The result is still not paper-clean because the geometry
comes from an external EX4DGS pretrained artifact, but it is now the strongest
real-scene heldout signal for the Metal PowerFoam path.

## First-Class World-Space Point-Cloud Init

The good EX4DGS run used a prepared anchor-frame PLY. To remove the hidden
coordinate-system footgun, I added trainer support for world-space multicam
point clouds:

- `model.init_point_cloud_coordinate_frame: "model"` keeps old behavior
- `model.init_point_cloud_coordinate_frame: "multicam_world"` applies
  `world_to_model = inv(anchor_c2w)` from the multicam loader before
  normalize/clamp/sample
- `MulticamVideoBundle` now carries `anchor_c2w`
- point-cloud init logs `init_point_cloud_coordinate_frame`

Validation:

```text
uv run --with pytest python -m pytest tests/test_powerfoam_direct.py::test_powerfoam_point_cloud_init_applies_world_to_model_transform tests/test_powerfoam_direct.py::test_powerfoam_metal_point_cloud_init_loads_ply_static_geometry
```

Result: `2 passed`.

I also ran a 1-step end-to-end smoke by overriding the 40-step Neural3D config
in memory to use:

```text
model.init_point_cloud_path = data/external/ex4dgs_pretrained/extracted/coffee_martini/input.ply
model.init_point_cloud_coordinate_frame = multicam_world
train.steps = 1
```

The smoke completed and logged `init_point_cloud_coordinate_frame:
multicam_world`, no missing Cech/AABB edges, and nonzero state deltas after the
step. Quality was poor, though: step-0 heldout PSNR was only `5.875771` versus
`10.033043` for the prepared visibility-filtered anchor-frame artifact. This
means the first-class coordinate transform is correct plumbing, but raw
unfiltered EX4DGS sampling is not the quality recipe.

I then promoted the visibility part of the prepared-Ply path into the trainer:

- `model.init_point_cloud_visibility_filter: "none" | "train_visible"`
- `model.init_point_cloud_min_visible_train_views`
- train-camera `K` / `w2c` metadata returned by `load_powerfoam_training_data`
- `PointCloudInitialization.filtered_count` logging support

Unit coverage now checks that the filter keeps a point projecting into the train
camera and drops a point outside the image. The targeted pytest slice passed
with `7 passed`.

The end-to-end raw-world smoke:

```text
model.init_point_cloud_path = data/external/ex4dgs_pretrained/extracted/coffee_martini/input.ply
model.init_point_cloud_coordinate_frame = multicam_world
model.init_point_cloud_visibility_filter = train_visible
train.steps = 1
output_dir = /tmp/powerfoam_metal_neural3d_raw_world_trainvisible_init_smoke
```

completed and the loader reports:

```text
source_count=5498
filtered_count=5113
sampled_count=1024
coordinate_frame=multicam_world
visibility_filter=train_visible
```

`best_metrics.json` selected step 0 with heldout PSNR `10.033043`, matching the
prepared filtered artifact's starting point. This proves the trainer-side
transform plus train-visibility filter reproduces the good initialization
path. The 1-step smoke is not a new quality baseline because best stayed at
initialization.

## Neural3D 4096-Cell Cap64 Smoke

I also checked whether the better filtered EX4DGS init can at least enter the
4096-cell regime without raising the raytrace replay cap. I launched a one-step
in-memory override of the 40-step Neural3D config:

```text
model.cells = 4096
train.steps = 1
output_dir = /tmp/powerfoam_metal_neural3d_filtered_init_4096cells_smoke
```

Result:

- selected cap64 raytrace path did not fail
- adjacency average degree `6.0093`
- adjacency max degree `13`
- missing overlap edges `0`
- step-0 heldout PSNR `9.784716`
- step-1 heldout PSNR `9.814147`
- step-1 heldout L1 `0.262087`
- step-1 heldout SSIM `0.138597`
- train-step elapsed `0.756979 s`

Interpretation: unlike the 4096-cell DeepView image-depth probe, this
real-scene filtered point-cloud init did not exceed the cap64 replay guard at
4096 cells. However, it started lower quality than the 1024-cell filtered run,
so this is only a capacity smoke, not a new selected quality path.

## Direct Reference Camera-Origin Fix

The P1 Torch-reference audit still had a real correctness gap: the direct
renderer assumed the camera origin was always `(0,0,0)`. That meant power-order
sorting, ray-sphere intersection, power-face clipping, surface-plane queries,
and spherical-Voronoi color view directions were only correct for the original
single-video fixed-origin setup.

I updated `src/train/powerfoam_direct.py` so `render_powerfoam_torch(...)`
accepts either old `[H,W,3]` direction grids or full `[H,W,6]` /
`[B,H,W,6]` rays. When full rays are supplied, it:

- sorts cells by `||camera_origin - p_i||^2 - r_i^2`
- computes sphere and power-face intersections from the per-ray origin
- computes surface/height query points as `origin + t * direction`
- evaluates SV color from `texel_world - ray_origin`

Added regressions in `tests/test_powerfoam_direct.py`:

- `test_powerfoam_direct_render_uses_ray_origin_for_geometry`
- `test_powerfoam_direct_sv_color_uses_ray_origin`

Validation:

```text
uv run --with pytest python -m pytest tests/test_powerfoam_direct.py
```

Result: `26 passed, 1 skipped`.

## Point-Count Metric Logging

P6 already had grow/prune/resample mechanics and optimizer-state preservation,
but the acceptance checklist still required point-count logs over time. I added
`state_cell_count` to `MetalPowerFoamVideo.parameter_drift_metrics()`, so every
validation artifact print, best-metrics JSON, checkpoint-best metric payload,
and W&B validation payload can carry the current cell count. The existing
resample test now checks that the metric reports `6` after grow and `3` after
prune.

I also made resampling explicitly reject bad cells before duplication/pruning:
finite decoded points, radii, density, features, normals, contribution EMA, and
point-error EMA are required before a cell can be selected. If a high-contrib
cell is non-finite, it is counted as `resample_invalid_pruned` and excluded
from the new parameter tensors. Added
`test_powerfoam_metal_resample_prunes_invalid_cells`; the targeted resample
tests passed with `2 passed`.

## Topology TODO Correction

I corrected the P7 TODO language around regular triangulation. Earlier notes
treated "make regular triangulation selected" as a remaining paper-fidelity
item. After checking the official source, that is the wrong blocker: official
PowerFoam builds an AABB/BVH Cech complex and accepts the Cech overlap graph as
the practical sparse topology. Our optional SciPy regular-triangulation path
stays useful for checks/ablations, but Cech/AABB should remain the selected
fast path unless a real-scene correctness gate proves otherwise.

## Canonical Tiny Parity Fixture

Added `research_experiments/dynamic_foam/make_powerfoam_parity_fixture.py` and
generated:

```text
research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json
```

The fixture records upstream PowerFoam commit
`96392252ebd0059fe6ca98881b62e12295d9242f`, a deterministic 3-cell
height+SV scene, nonzero ray origins, Cech adjacency, render options, and local
Torch-reference expected tensors for RGB, alpha, normal-distance, contribution,
and visibility. Added
`test_powerfoam_direct_loads_canonical_origin_parity_fixture`; the fixture
loads and reproduces the expected tensors. This is not official CUDA output
yet; it is the canonical local fixture needed before adding Metal/official
parity loaders.

Then added `test_powerfoam_metal_loads_canonical_origin_parity_fixture`, which
loads the same JSON, converts the padded direct adjacency into CSR, calls the
Metal quaternion height+SV wrapper on MPS, and compares RGB/alpha/
normal-distance against the fixture's Torch-reference outputs. The focused
Metal fixture test passed.

## First-Class Raw-World Train-Visible 40-Step Run

After the 1-step raw-world smoke reproduced the good filtered EX4DGS starting
point, I added and ran the checked-in config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_ex4dgs_world_trainvisible_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

This version loads the raw EX4DGS `input.ply` directly instead of the prepared
anchor-frame artifact:

```text
init_point_cloud_coordinate_frame = multicam_world
init_point_cloud_visibility_filter = train_visible
init_point_cloud_min_visible_train_views = 1
```

The trainer printed the expected intake:

- source points: `5498`
- train-visible filtered points: `5113`
- adjacency mode: `cech_aabb`
- adjacency average degree: `5.5273`
- missing overlap edges: `0`

The metrics matched the prepared-artifact 40-step run within normal sampling
noise:

- step-0 heldout PSNR: `10.033043`
- step-40 train PSNR: `10.372944`
- step-40 heldout PSNR: `10.740685`
- step-40 heldout L1 / SSIM: `0.230020` / `0.160639`
- `best_metrics.json` selected step `40`
- `state_cell_count` logged as `1024`

Important artifact caveat: the filesystem was effectively full when the run
reached the final save (`df` showed about `172 MiB` free). `checkpoint_best.pt`
and `best_metrics.json` were already written, but `checkpoint_final.pt` failed
mid-write and was truncated. I removed the truncated final checkpoint. The
training/eval result is usable as first-class transform/filter plumbing
evidence; the final checkpoint file is not.

## Direct Reference Static Posed-Camera Smoke

The remaining P1 direct-reference gap was that `render_powerfoam_torch(...)`
could accept full rays, but the direct trainer still only loaded fixed-origin
single-video data. I added `multicam_val` support to
`src/train/train_powerfoam_direct.py`:

- builds `[origin, direction]` rays from `CameraSpec`
- flattens train view/time samples while keeping shared frame-indexed foam state
- renders heldout samples with their own camera rays
- logs heldout metrics/media through the direct trainer path

Checked-in smoke config:

```text
src/train_configs/local_mac_powerfoam_direct_multicam_deepview_3cam_train2_test1_32_smoke.jsonc
```

Runtime result on CPU:

- source: `deepview_03_Dog_camera_0001_to_camera_0040`
- train views: `camera_0001`, `camera_0015`
- heldout view: `camera_0040`
- pose source: `deepview_models_relative_pinhole`
- frames: `1`
- flattened train samples: `2`
- render size / cells: `32` / `16`
- step-0 train L1 / heldout L1: `0.330641` / `0.326765`
- step-1 train L1 / heldout L1: `0.324219` / `0.326421`
- train backward completed with finite loss `0.345161`

This is not a quality baseline. Its purpose is narrower: the local Torch
reference/trainer now exercises a static posed-camera path instead of only
fixed-origin per-frame video fitting.

## Official Fixture Generator Scaffolding

The true official PowerFoam parity fixture still cannot be generated on this
Mac because the pinned upstream checkout is CUDA/Warp-bound and
`torch.cuda.is_available()` is false locally. Instead of leaving that as only a
TODO, I added a CUDA-host generator:

```text
research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py
```

It reuses the canonical tiny scene, swaps to an upstream-compatible
`TorchCamera` pinhole camera, and supports:

- `--backend local`: Mac-runnable dry run through `render_powerfoam_torch`
- `--backend official`: CUDA/Warp run through upstream `powerfoam.rasterize`,
  upstream `SphericalVoronoi`, and the pinned upstream checkout

The local dry-run artifact is checked in:

```text
research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_local_v1.json
```

It has nonzero alpha coverage (`max alpha ~= 0.605`) and is covered by
`test_powerfoam_direct_loads_official_camera_local_fixture`. The actual P0
official-output fixture remains open until the same script is run on a CUDA
host:

```text
PYTHONPATH=src/train python research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py --backend official
```

I also added
`test_powerfoam_direct_matches_official_cuda_fixture_if_present`, which skips
until
`research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json`
exists and then compares local direct output against the official CUDA/Warp
fixture.

## Atomic Checkpoint Saves

The raw-world EX4DGS 40-step run exposed a real trainability/operations bug:
when disk filled during `torch.save`, `checkpoint_final.pt` was left truncated.
I added `src/train/checkpoint_utils.py::atomic_torch_save(...)` and routed both
PowerFoam trainers through it:

- `src/train/train_powerfoam_metal.py`
- `src/train/train_powerfoam_direct.py`

The helper writes `.<checkpoint>.tmp` in the target directory and only replaces
the real checkpoint after `torch.save` succeeds. On failure it removes the temp
file and leaves any existing checkpoint untouched.

Regression:

```text
tests/test_powerfoam_direct.py::test_atomic_torch_save_preserves_existing_checkpoint_on_failure
```

I reran the direct multicam smoke after the change; it still completed and
wrote `checkpoint_final.pt` through the atomic path.

## Paper-Clean Feature-Triangulation Init Probe

The EX4DGS point-cloud path is the strongest current real-scene signal, but it
is not paper-clean because the geometry comes from an external pretrained
artifact. To make that gap concrete, I added a train-camera-only sparse
triangulation builder:

```text
research_experiments/dynamic_foam/build_multiview_feature_triangulation_point_cloud.py
```

It loads the configured `multicam_val` train cameras, extracts OpenCV SIFT/ORB
features, keeps symmetric ratio matches, triangulates with the known dataset
poses, filters by positive depth/reprojection/parallax/scene box, and writes a
model-frame ASCII PLY plus JSON audit. It never reads the heldout camera frames
for geometry.

Useful failed/weak attempts:

- Neural3D `cam04/cam09`, frame 0, 128px SIFT, 3px reprojection gate:
  `0` valid points. The JSON showed matches existed but none passed reproj.
- Neural3D `cam04/cam09`, frame 0, 256px ORB, loose 25px gate:
  `96` points, but median reprojection error was `16.32px`.
- DeepView `camera_0001/camera_0015`, frame 0, 128px SIFT, loose 25px gate:
  enough matches, but `0` points inside the trainer scene box.

The selected artifact for a clean negative control is:

```text
research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_train2_feature_triangulation_frames0_4_8_12_256px_orb_reproj8.ply
```

Its JSON audit:

- source sample: `neural3d_coffee_martini_train_cam04_cam09_holdout_cam06`
- train cameras: `cam04`, `cam09`
- heldout camera: `cam06` (not used for point generation)
- frames: `0,4,8,12`
- target size for matching: `256`
- method: ORB, symmetric ratio `0.95`
- reprojection gate: `8px`
- point count: `89`
- median / p90 reprojection error: `4.07px` / `7.40px`

I added and ran the matching Metal raytrace config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

Trainer intake:

- init source count: `89`
- train-visible filtered count: `89`
- sampled/duplicated cell count: `1024`
- adjacency: `cech_aabb`
- average / max degree: `15.8906` / `39`
- missing overlap edges: `0`

Metrics:

- step-0 heldout PSNR / L1 / SSIM: `5.6311` / `0.475119` / `0.0003`
- step-40 heldout PSNR / L1 / SSIM: `5.6311` / `0.475115` / `0.0003`
- step-40 train PSNR / L1: `5.6487` / `0.471390`
- wall-clock train loop: `7.46s`
- `checkpoint_best.pt` and `checkpoint_final.pt` both exist

Interpretation: this closes a useful clean negative control. The Metal
raytrace trainer can consume a no-pretrain train-camera-only sparse init, but a
shallow two-view ORB/SIFT triangulation artifact is far too sparse/weak for the
paper-quality setup. The next paper-clean init should be a real COLMAP or
pycolmap reconstruction, preferably with more than two train views or higher
resolution imagery, not another minor threshold tweak on this matcher.

## Train4 Feature-Triangulation Follow-Up

The data loader accepts explicit camera-list overrides, so I tried a stronger
clean reconstruction variant before closing the lane. New config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train4_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

It keeps the same Neural3D `coffee_martini` record and heldout `cam06`, but
uses four train cameras:

```text
cam04, cam09, cam13, cam20
```

The matching PLY:

```text
research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_train4_feature_triangulation_frames0_4_8_12_256px_orb_reproj8.ply
```

Artifact audit:

- train-only camera pairs: `24` pair/frame combinations
- point count: `662`
- median / p90 reprojection error: `3.29px` / `7.08px`
- coordinate frame: model/anchor frame
- no heldout or pretrained geometry used

40-step Metal result:

- samples: `64` train view/time samples
- init source / filtered count: `662` / `662`
- adjacency avg / max degree: `4.4453` / `11`
- missing overlap edges: `0`
- step-0 heldout PSNR / L1 / SSIM: `5.6727` / `0.471159` / `0.0001`
- step-40 heldout PSNR / L1 / SSIM: `5.6727` / `0.471159` / `0.0001`
- step-40 train PSNR / L1 / SSIM: `5.9217` / `0.448424` / `0.0063`
- `best_metrics.json` selected step `0`
- `checkpoint_best.pt` and `checkpoint_final.pt` both exist

Interpretation: train4 feature triangulation is a better clean artifact than
train2, but still nowhere near the EX4DGS-init row. The renderer/trainer path
is not the blocker here; the shallow pairwise feature cloud is. The remaining
paper-clean benchmark should use real COLMAP/pycolmap tracks or a substantially
better multi-view reconstruction artifact.

## Known-Pose Pycolmap/SIFT Follow-Up

I then moved from the local ORB pair matcher to actual pycolmap/COLMAP
triangulation with known dataset poses. New builder:

```text
research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py
```

Implementation notes:

- uses `pycolmap.extract_features` and `pycolmap.match_exhaustive`
- uses the known `multicam_val` train camera poses, not estimated camera poses
- writes only compact PLY/JSON by default; image/database/sparse workdirs stay
  temporary unless `--workdir` is passed
- sets `KMP_DUPLICATE_LIB_OK` before importing `torch`/`pycolmap` because this
  Mac has duplicate OpenMP runtimes between those packages

The first scripted run hit a segfault when `pycolmap` was imported before
`torch`. Reordering imports to `torch` then `pycolmap`, with
`KMP_DUPLICATE_LIB_OK=TRUE`, fixed the local run.

Generation command:

```text
PYTHONPATH=src/train uv run --with pycolmap python \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train4_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc \
  --output research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_train4_pycolmap_known_pose_frame0_1024px_sift_reproj8.ply \
  --target-size 1024 --frame-index 0 --max-features 20000 \
  --sift-ratio 0.9 --max-reproj-error 8.0 --min-tri-angle 0.1 \
  --no-ignore-two-view-tracks
```

Artifact audit:

- train cameras: `cam04`, `cam09`, `cam13`, `cam20`
- heldout camera: `cam06` (not used for geometry)
- target size: `1024`
- database keypoints: `18005`
- matched / verified image pairs: `6` / `6`
- raw points: `45`
- box-filtered PLY points: `44`
- median / p90 filtered reprojection error: `3.64px` / `6.63px`
- track length: exactly `2` for all points

Matching trainer config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train4_holdout1_pycolmap_known_pose_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

40-step Metal result:

- init source / filtered count: `44` / `44`
- sampled/duplicated cell count: `1024`
- adjacency avg / max degree: `24.5508` / `50`
- missing overlap edges: `0`
- visible fraction: `0.0308`
- step-0 heldout PSNR / L1 / SSIM: `5.6309` / `0.475125` / `0.0003`
- step-40 heldout PSNR / L1 / SSIM: `5.6309` / `0.475125` / `0.0003`
- step-40 train PSNR / L1: `5.7205` / `0.465157`
- `best_metrics.json` selected step `0`
- `checkpoint_best.pt` and `checkpoint_final.pt` both exist

Interpretation: this is the cleanest local COLMAP-style negative so far. It
shows the official-style known-pose pycolmap route can be wired locally, and
the Metal trainer can consume its PLY, but the reconstruction is too sparse and
too low-coverage to support quality. The remaining paper-clean gap is not
"write a PLY loader" or "make PowerFoam trainable"; it is getting a dense
multi-view reconstruction/tracks artifact comparable to the paper's SfM/COLMAP
initialization.

## Multiframe Known-Pose Pycolmap Follow-Up

I also tried a compact multiframe variant by generating known-pose pycolmap
snapshots for frames `0`, `4`, `8`, and `12`, then merging the resulting PLYs.
This is still train-camera-only and paper-clean in the sense that no heldout
camera or pretrained EX4DGS geometry is used, but it is not a real static
COLMAP long-track reconstruction.

Merged artifact:

```text
research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_train4_pycolmap_known_pose_frames0_4_8_12_1024px_sift_reproj8_merged.ply
```

Artifact audit:

- train cameras: `cam04`, `cam09`, `cam13`, `cam20`
- heldout camera: `cam06` (not used for geometry)
- target size: `1024`
- source frames: `0`, `4`, `8`, `12`
- per-frame point counts: `44`, `75`, `51`, `57`
- merged point count: `227`
- source frame reprojection medians: about `3.02px` to `3.80px`
- all source tracks were still two-view tracks

Matching trainer config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train4_holdout1_pycolmap_known_pose_multiframe_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

40-step Metal result:

- init source / filtered count: `227` / `227`
- sampled/duplicated cell count: `1024`
- adjacency avg / max degree: `8.994` / `37`
- missing overlap edges: `0`
- visible fraction improved to about `0.153`
- step-0 heldout PSNR / L1 / SSIM: `5.6309` / `0.475125` / `0.0003`
- step-40 heldout PSNR / L1 / SSIM: `5.6309` / `0.475125` / `0.0003`
- step-40 train PSNR / L1 / SSIM: `5.9467` / `0.448495` / `0.0042`
- `best_metrics.json` selected step `0`
- `checkpoint_best.pt` and `checkpoint_final.pt` both exist

Interpretation: the multiframe union improves coverage and source-view fitting
relative to the single-frame pycolmap artifact, but it does not improve heldout
quality at all. The clean-init failure remains reconstruction quality: we need
dense static tracks or a better local COLMAP/SfM artifact, not just more sparse
two-view pycolmap snapshots.

## Per-Image Intrinsics And DeepView Probe

The Neural3D pycolmap path was not enough, so I checked whether the local
DeepView sample could provide a denser clean init. The existing known-pose
builder assumed every train camera shared one PINHOLE K matrix. DeepView
cameras have slightly different intrinsics after target-size scaling, so I
updated:

```text
research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py
```

Changes:

- `--camera-mode auto` now uses `SINGLE` when train K matrices match and
  `PER_IMAGE` when they differ.
- reconstruction construction now adds one pycolmap camera per database
  camera id and uses the matching train-view K.
- CLI overrides can set `--train-cameras`, `--heldout-camera`,
  `--anchor-camera`, and `--condition-camera` for probe runs without creating
  a checked-in trainer config first.

Validation probes:

```text
PYTHONPATH=src/train uv run --with pycolmap python \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train4_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc \
  --output /tmp/powerfoam_neural3d_pycolmap_builder_probe.ply \
  --target-size 128 --frame-index 0 --max-features 1000 --max-points 256 \
  --max-reproj-error 8.0 --min-tri-angle 0.1 --no-ignore-two-view-tracks
```

This preserved the old Neural3D path: auto-selected `single`, verified all 6
pairs, and produced 13 filtered probe points at 128px.

```text
PYTHONPATH=src/train uv run --with pycolmap python \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_noaux.jsonc \
  --output /tmp/powerfoam_deepview_pycolmap_builder_probe.ply \
  --target-size 256 --frame-index 0 --max-features 2000 --max-points 2048 \
  --max-reproj-error 8.0 --min-tri-angle 0.1 --no-ignore-two-view-tracks
```

This exercised the new DeepView per-image path: auto-selected `per_image`,
created 2 cameras, verified the 1 image pair, triangulated 17 raw points, but
the default PowerFoam box (`xy_extent=1.25`, `z=1..3.25`) removed all points.
A wide-box rerun showed those same 17 raw points were real but too sparse to
promote.

The useful follow-up was an 8-train-camera DeepView probe:

```text
PYTHONPATH=src/train uv run --with pycolmap python \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_noaux.jsonc \
  --output /tmp/powerfoam_deepview_pycolmap_8cam_wide_probe.ply \
  --target-size 256 --frame-index 0 \
  --train-cameras camera_0001 camera_0012 camera_0002 camera_0003 camera_0015 camera_0021 camera_0013 camera_0010 \
  --heldout-camera camera_0040 --anchor-camera camera_0001 \
  --max-features 2000 --max-points 4096 --max-reproj-error 8.0 \
  --min-tri-angle 0.1 --no-ignore-two-view-tracks \
  --xy-extent 100 --z-min -100 --z-max 100
```

Result:

- train cameras: `camera_0001`, `camera_0012`, `camera_0002`,
  `camera_0003`, `camera_0015`, `camera_0021`, `camera_0013`, `camera_0010`
- heldout camera: `camera_0040` (not used for geometry)
- target size: `256`
- database keypoints: `8157`
- matched / verified image pairs: `28` / `28`
- raw points: `639`
- median / p90 reprojection error: `3.32px` / `6.22px`
- track length max / mean / median: `4` / `2.05` / `2`
- point z distribution: median `0.39`, p90 `5.78`, p99 `32.48`, max `89.33`

Interpretation at probe time: this was the first local clean pycolmap artifact
that looked dense enough to justify a PowerFoam trainer run. It needed a
deliberate trainer config with a realistic DeepView model box, likely around
`xy_extent=16`, `z_min=0.05`, and a conservative `z_max` such as `8` or `16`
to reject far outliers.

## DeepView 8cam Pycolmap Trainer Probe

Before running the trainer, I fixed an avoidable data-loader cost:
`load_multicam_video_bundle(...)` was loading full camera videos and only then
applying `max_frames` / `frame_indices`. I added early frame-count limiting so
probe configs can load only the frames they actually need. The focused test now
checks the requested camera-frame count.

One-step smoke:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  /tmp/powerfoam_deepview_8cam_pycolmap_1step_smoke.jsonc
```

Result:

- frames: `4`
- train views: 8 DeepView cameras
- heldout view: `camera_0040`
- init source / filtered count: `639` / `599`
- adjacency avg / max degree: `6.498` / `22`
- missing overlap edges: `0`
- step-0 heldout PSNR / L1 / SSIM: `7.7901` / `0.354318` / `0.00191`
- step-1 heldout PSNR / L1 / SSIM: `7.7901` / `0.354318` / `0.00191`
- state deltas after one step were nonzero for centers, features, normals,
  quaternions, texel sites, SV axes, and SV RGB

Then I promoted the artifact/config:

```text
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_256px_sift_wide.ply
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_256px_sift_wide.json
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

40-step run:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

Result:

- frames: `16`
- samples: `128`
- init source / filtered count: `639` / `599`
- train-loop elapsed: `46.53s`
- step-0 source PSNR / L1 / SSIM: `7.7475` / `0.350879` / `0.0199`
- step-40 source PSNR / L1 / SSIM: `7.8075` / `0.345982` / `0.0238`
- step-0 heldout PSNR / L1 / SSIM: `7.8250` / `0.353741` / `0.0021`
- step-40 heldout PSNR / L1 / SSIM: `7.8250` / `0.353742` / `0.0021`
- `best_metrics.json` selected step `0`
- `checkpoint_best.pt` and `checkpoint_final.pt` both exist

Interpretation: this is a better local clean init artifact than the sparse
Neural3D/2-camera pycolmap attempts, and it proves the 8-camera per-image-K
path can feed the Metal trainer. It still does not solve paper-clean heldout
quality. Training fits source views slightly, but heldout is unchanged and this
row is weaker than the earlier image-depth DeepView low-geometry control
(`8.1417` best heldout). The remaining paper gap is not just "get more pairwise
pycolmap points"; it likely needs paper-grade static COLMAP tracks / better
scene normalization / schedule work, plus the official CUDA/Warp parity fixture.

## DeepView 8cam Frame0-Only Diagnostic

I then ran the same 8-camera pycolmap artifact through a frame0-only config:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_init_raytrace_128_1f_1024cells_40step_lowgeom_noaux.jsonc
```

Result:

- frames: `1`
- samples: `8`
- init source / filtered count: `639` / `599`
- train-loop elapsed: `5.01s`
- step-0 source PSNR / L1 / SSIM: `7.5609` / `0.357772` / `0.0195`
- step-40 source PSNR / L1 / SSIM: `7.9442` / `0.333460` / `0.0364`
- step-0 heldout PSNR / L1 / SSIM: `7.7417` / `0.356101` / `0.0019`
- step-40 heldout PSNR / L1 / SSIM: `8.0377` / `0.341813` / `0.0243`
- `best_metrics.json` selected step `40`
- `checkpoint_best.pt` and `checkpoint_final.pt` both exist
- state deltas were nonzero for centers, density, features, normals,
  quaternions, texel sites, SV axes, and SV RGB

Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_init_raytrace_128_1f_1024cells_40step_lowgeom_noaux/
```

Interpretation: the same clean artifact is not hopeless. It learns a heldout
frame when the temporal problem is reduced to a single static frame. That
narrows the 16f failure toward static/dynamic temporal mismatch, schedule, or
normalization rather than pure Metal trainability or adjacency. It still does
not beat the earlier image-depth DeepView low-geometry control (`8.1417` best
heldout), so it is not paper-quality evidence.

## DeepView 8cam Frame0 200-Step Extensions

I tested whether the 40-step frame0 improvement was simply undertrained.

Default 200-step extension:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_init_raytrace_128_1f_1024cells_200step_lowgeom_noaux.jsonc
```

Result:

- train-loop elapsed: `23.54s`
- step-50 heldout PSNR / L1 / SSIM: `7.8569` / `0.348617` / `0.0086`
- step-100 heldout PSNR / L1 / SSIM: `7.8561` / `0.348442` / `0.0079`
- step-150 heldout PSNR / L1 / SSIM: `7.8694` / `0.346838` / `0.0077`
- step-200 heldout PSNR / L1 / SSIM: `7.8695` / `0.346840` / `0.0077`
- step-200 source PSNR / L1 / SSIM: `8.4144` / `0.309508` / `0.0596`
- `best_metrics.json` selected step `200`

The default extension improved source reconstruction but hurt heldout relative
to the 40-step frame0 run. The likely local mechanism is schedule drift:
density, radius, and texel-height groups use absolute official-style warmups,
so the 200-step run reaches much larger warmup LRs and larger state deltas than
the 40-step probe.

Low-motion 200-step extension:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_init_raytrace_128_1f_1024cells_200step_lowmotion_noaux.jsonc
```

Result:

- train-loop elapsed: `20.20s`
- step-50 heldout PSNR / L1 / SSIM: `7.7435` / `0.355957` / `0.0018`
- step-100 heldout PSNR / L1 / SSIM: `7.8931` / `0.346460` / `0.0065`
- step-150 heldout PSNR / L1 / SSIM: `7.9646` / `0.344208` / `0.0103`
- step-200 heldout PSNR / L1 / SSIM: `7.8706` / `0.348679` / `0.0049`
- `best_metrics.json` selected step `150`

Interpretation: lower motion prevents the worst default-200 overfit but still
does not beat the 40-step frame0 result (`8.0377`). The current selected clean
frame0 checkpoint remains the 40-step config. More steps are not the next
obvious route; the next paper-quality work should redesign the schedule or
improve the geometry/track source.

## DeepView 8cam 512px Pycolmap Artifact

I then tested whether the 256px clean geometry was just too sparse. Builder:

```text
PYTHONPATH=src/train uv run --with pycolmap python \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_noaux.jsonc \
  --output /tmp/powerfoam_deepview_pycolmap_8cam_512_probe.ply \
  --target-size 512 --frame-index 0 \
  --train-cameras camera_0001 camera_0012 camera_0002 camera_0003 camera_0015 camera_0021 camera_0013 camera_0010 \
  --heldout-camera camera_0040 --anchor-camera camera_0001 \
  --max-features 4000 --max-points 8192 --max-reproj-error 8.0 \
  --min-tri-angle 0.1 --no-ignore-two-view-tracks \
  --xy-extent 100 --z-min -100 --z-max 100
```

Promoted artifact:

```text
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_512px_sift_wide.ply
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_512px_sift_wide.json
```

Builder result:

- target size: `512`
- database keypoints: `24665`
- matched / verified image pairs: `28` / `28`
- raw / filtered points: `771` / `771`
- median / p90 reprojection error: `3.51px` / `6.75px`
- track length max / mean / median / p90: `3` / `2.01` / `2` / `2`

This fails a strict paper-grade track gate because it is still mostly two-view
tracks, but it is enough denser/better positioned to test the trainer.

Frame0 40-step run:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_512px_init_raytrace_128_1f_1024cells_40step_lowgeom_noaux.jsonc
```

Result:

- init source / filtered count: `771` / `731`
- adjacency avg / max degree: `9.385` / `61`
- missing overlap edges: `0`
- step-0 heldout PSNR / L1 / SSIM: `8.4423` / `0.313535` / `0.0478`
- step-40 heldout PSNR / L1 / SSIM: `8.4810` / `0.312108` / `0.0408`
- step-40 source PSNR / L1 / SSIM: `7.7271` / `0.337121` / `0.0517`
- `best_metrics.json` selected step `40`

16f companion:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

Result:

- frames: `16`
- samples: `128`
- init source / filtered count: `771` / `731`
- step-0 heldout PSNR / L1 / SSIM: `8.5191` / `0.311725` / `0.0476`
- step-20 heldout PSNR / L1 / SSIM: `8.5312` / `0.311007` / `0.0487`
- step-40 heldout PSNR / L1 / SSIM: `8.5355` / `0.310773` / `0.0489`
- step-40 source PSNR / L1 / SSIM: `7.8102` / `0.340064` / `0.0344`
- `best_metrics.json` selected step `40`

Interpretation: this is now the strongest clean DeepView PowerFoam Metal row.
It beats the 256px clean frame0 row (`8.0377`), the 256px clean 16f row
(`7.8250`), and the earlier image-depth DeepView low-geometry control
(`8.1417`). It still should not be called paper-quality PowerFoam: the point
cloud is only 771 points and mostly two-view tracks, and there is still no
official CUDA/Warp parity fixture.

## Official CUDA Fixture Test Gap

A read-only audit found that
`research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py`
already writes official gradients for points, radii, density, normals, texel
sites, texel height, SV axis, and SV RGB, but the skip-until-present pytest was
only checking forward outputs. I extended
`tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present`
so that, when the CUDA/Warp fixture exists, it now compares:

- rendered RGB
- alpha
- normal distance
- contribution
- normal
- visible mask
- scalar loss
- all recorded gradient tensors

This still does not generate the official fixture locally; the Mac has no CUDA
and no `warp` import. It does mean that the future CUDA-host fixture will gate
official backward parity instead of silently checking forward only.

## Fixture Gradient Schema + Metal Shared Backward Gate

Follow-up cleanup tightened the fixture parity surface and the test style.

Changes:

- `research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py`
  now writes the same backward payload for the local official-camera dry run as
  the future CUDA/Warp fixture: scalar loss plus gradients for points, radii,
  density, normals, texel sites, texel height, SV axis, and SV RGB.
- Regenerated
  `research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_local_v1.json`
  with those gradient keys.
- `tests/test_powerfoam_direct.py` now uses compact key maps and fixture helpers
  (`_fixture_param`, `_fixture_grad`, `_assert_fixture_grads`) instead of
  repeated local destructuring/assertion blocks.
- Added Metal coverage for the official-camera local fixture:
  `test_powerfoam_metal_matches_official_camera_local_fixture_shared_backward`
  checks Metal height+SV forward outputs, scalar loss, and comparable backward
  channels: density, texel height, SV axis, and SV RGB.
- Added the matching skip-until-present official CUDA/Warp Metal gate:
  `test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present`.

Why only shared gradients for Metal: the fixture/official parameterization stores
world texel sites and normals, while the Metal trainer derives world texel sites
and frames from local-site/quaternion parameters. Forward/loss should match, but
points/radii/local-site/quaternion gradients are not a one-to-one fixture
comparison. Density, height, and SV parameters are shared and did match locally.

Validation:

```text
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  tests/test_powerfoam_direct.py \
  research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py

git diff --check -- AGENTS.md tests/test_powerfoam_direct.py \
  research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py \
  research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_local_v1.json

PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
```

Result: `33 passed, 3 skipped in 3.78s`.

## 512px 16f 80-Step Dense-Eval Schedule Control

I added and ran a longer companion to the strongest clean DeepView row:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_80step_lowgeom_denseeval_noaux.jsonc
```

Command:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_80step_lowgeom_denseeval_noaux.jsonc
```

Result:

- train loop elapsed: `64.11s` with dense eval/logging every 10 steps
- best step: `20`
- best heldout PSNR / L1 / SSIM: `8.5345` / `0.310822` / `0.0487`
- final heldout PSNR / L1 / SSIM: `8.4564` / `0.315109` / `0.0444`
- final source PSNR / L1 / SSIM: `7.9138` / `0.333839` / `0.0389`

Interpretation: this does not beat the 40-step 512px 16f row (`8.5355`). The
run improves source-view PSNR while heldout degrades after the early checkpoint,
so the current clean 512px recipe is not simply under-trained. The next useful
quality work is schedule redesign or better COLMAP/track geometry, not just
more steps at the same lowgeom/no-aux settings.

## 512px 16f 120-Step Low-Motion/Low-Appearance Control

I added and ran a schedule variant that reduces both geometry/density/frame LRs
and the SV RGB appearance LR:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_120step_lowmotion_lowappearance_denseeval_noaux.jsonc
```

Command:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_120step_lowmotion_lowappearance_denseeval_noaux.jsonc
```

Result:

- train loop elapsed: `51.66s`
- best step: `120`
- best/final heldout PSNR / L1 / SSIM: `8.5270` / `0.311158` / `0.0478`
- final source PSNR / L1 / SSIM: `7.8901` / `0.336003` / `0.0371`
- state movement stayed much smaller than the 80-step default extension:
  mean center delta `0.00112`, max center delta `0.01777`, mean density delta
  `0.00239`, mean SV RGB delta `0.00385`.

Interpretation: lower motion and lower appearance LR prevent the larger heldout
collapse seen in the 80-step same-schedule extension, but they still do not beat
the selected 40-step lowgeom row (`8.5355`). This points away from "just more
careful longer training" and back toward better clean geometry/tracks or a more
substantial schedule/objective change.

## 1024px DeepView Pycolmap Geometry Probe

I also tried raising the clean DeepView known-pose pycolmap extraction from
512px to 1024px.

Builder command:

```text
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc \
  --output /tmp/powerfoam_deepview_pycolmap_8cam_1024_probe.ply \
  --target-size 1024 --frame-index 0 \
  --max-features 12000 --max-points 8192 --max-reproj-error 8.0 \
  --min-tri-angle 0.1 --no-ignore-two-view-tracks \
  --xy-extent 100 --z-min -100 --z-max 100
```

Promoted artifacts:

- `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_1024px_sift_wide.ply`
- `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_1024px_sift_wide.json`

Builder result:

- database keypoints: `78978`
- matched / verified image pairs: `28 / 28`
- raw points: `976`
- box-filtered points: `975`
- filtered reprojection mean / median / p90 / max:
  `3.8042 / 3.8247 / 6.7902 / 7.7549`
- filtered track length max / mean / median / p90:
  `3 / 2.0041 / 2 / 2`

Trainer config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_1024px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

Trainer command:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_1024px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

Result:

- train-visible source points: `948`
- adjacency avg / max degree: `16.7070 / 103`
- missing overlap edges: `0`
- train loop elapsed: `34.90s`
- best step: `0`
- best heldout PSNR / L1 / SSIM:
  `8.4415` / `0.315315` / `0.0281`
- final heldout PSNR / L1 / SSIM:
  `8.4411` / `0.315634` / `0.0281`
- final source PSNR / L1 / SSIM:
  `8.3017` / `0.317029` / `0.0524`

Interpretation: this is a negative geometry control. The 1024px extraction made
more points than 512px, but the track topology stayed mostly two-view and the
reprojection distribution got worse. It also underperformed the selected 512px
16f row (`8.5355`). The next clean-init step should be a denser/longer-track
COLMAP-style reconstruction or a better scene/source, not another simple SIFT
resolution bump.

## Long-Track Builder Controls + Duplicate-Jitter A/B

I extended the known-pose pycolmap builder so we can test the actual geometry
failure instead of guessing from aggregate point counts.

Code changes:

- `research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py`
  now exposes SIFT threshold knobs, pair verification max error, optional
  known-pose guided verification, minimum output track length, and pycolmap
  triangulation merge/transitivity/retriangulation knobs.
- It also exposes `--feature-type` / `--matcher-type` for future
  ALIKED/LightGlue probes, guarded behind `--allow-onnx-models` because the
  local pycolmap wheel aborts inside C++ for ONNX-backed matchers.
- The builder summary now records raw/filtered track-length histograms, selected
  SIFT options, verification options, and triangulation options.

First long-track probe:

```text
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc \
  --output /tmp/deepview_03_dog_8cam_pycolmap_known_pose_frame0_512px_longtrack_probe.ply \
  --target-size 512 --frame-index 0 \
  --max-features 12000 --max-points 8192 --max-reproj-error 8.0 \
  --min-tri-angle 0.1 --no-ignore-two-view-tracks --min-track-length 3 \
  --triangulation-max-transitivity 5 \
  --triangulation-complete-max-transitivity 10 \
  --triangulation-merge-max-reproj-error 8.0 \
  --triangulation-complete-max-reproj-error 8.0 \
  --triangulation-re-max-angle-error 8.0 \
  --triangulation-re-max-trials 3 \
  --xy-extent 100 --z-min -100 --z-max 100
```

Result:

- raw points: `822`
- raw track histogram: `803` length-2, `16` length-3, `3` length-4
- filtered point count with `min_track_length=3`: `19`
- filtered reprojection mean / median / p90:
  `3.5241 / 3.1785 / 5.0496`

Second long-track probe, adding `--verify-max-error 8.0` and
`--known-pose-guided-verification`:

- raw points: `599`
- raw track histogram: `598` length-2, `1` length-3
- filtered point count with `min_track_length=3`: `1`

Interpretation: the current DeepView/SIFT correspondence graph has almost no
long-track core. Looser merge/transitivity helps only slightly, and known-pose
guided verification prunes the graph further. This says the next clean-init
attempt needs a different matcher/source/undistortion path or true dense
COLMAP, not another trainer run on a track-length-3 filtered cloud.

I attempted one SIFT+LightGlue probe after adding the matcher switch:

```text
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc \
  --output /tmp/deepview_03_dog_8cam_pycolmap_known_pose_frame0_512px_sift_lightglue_longtrack_probe.ply \
  --target-size 512 --frame-index 0 \
  --feature-type sift --matcher-type sift_lightglue \
  --max-features 12000 --max-points 8192 --sift-ratio 0.9 \
  --verify-max-error 8.0 \
  --max-reproj-error 8.0 --min-tri-angle 0.1 \
  --no-ignore-two-view-tracks --min-track-length 3 \
  --triangulation-max-transitivity 5 \
  --triangulation-complete-max-transitivity 10 \
  --triangulation-merge-max-reproj-error 8.0 \
  --triangulation-complete-max-reproj-error 8.0 \
  --triangulation-re-max-angle-error 8.0 \
  --triangulation-re-max-trials 3 \
  --xy-extent 100 --z-min -100 --z-max 100
```

It aborted in pycolmap during matching:

```text
LightGlue feature matching requires ONNX support.
```

I added the ONNX opt-in guard after that failure so future local runs fail with
a normal Python error unless `--allow-onnx-models` is passed on a host with an
ONNX-enabled pycolmap build.

## Full Height+SV Raytrace Material-Overfit Test

A later audit found one local trainability gap that was still worth closing:
the synthetic posed-view overfit test used constant-feature mode, while the
real PowerFoam path we care about is full quaternion height+SV through the
raytrace backend.

I added:

```text
tests/test_powerfoam_direct.py::test_powerfoam_metal_height_sv_raytrace_overfits_tiny_material
```

The test builds a one-cell teacher and one-cell student with the same
power-cell geometry, texel sites, height, SV axis, and raytrace backend. The
teacher renders two posed cameras with a different SV RGB material. The student
then optimizes only SV RGB while all geometry/material-axis groups have zero
initial LR. The acceptance condition is direct: L1 must fall below `0.001` and
below 25% of the initial error, and the SV RGB tensor must actually move.

This is not a paper-quality scene result. It is a focused local regression that
the full height+SV raytrace path is differentiable and trainable for material
updates, instead of relying only on constant-feature overfit plus separate
raytrace/raster gradient parity.

## Saved 4K Benchmark Verifier

I added:

```text
research_experiments/dynamic_foam/verify_powerfoam_4k_benchmarks.py
```

This is a verifier for the saved benchmark artifacts, not a fresh benchmark
runner. It checks that the selected UHD full height+SV raytrace `cech_aabb`
forward+backward artifact:

- is `3840x2160`
- has backward support
- is full `oriented_height_sv_texel_surface`
- stays under `1200 ms` total median
- stays under the replay cap `64`
- beats the regular-triangulation comparison at the same cell count

Command:

```text
.venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_4k_benchmarks.py
```

Result:

```text
1024 cells: cech_aabb total 1016.073 ms vs regular 1868.673 ms, max steps 26
4096 cells: cech_aabb total 1014.387 ms vs regular 2888.311 ms, max steps 36
```

This strengthens the "4K fast enough locally" claim by tying it to exact JSON
artifacts. It still does not replace the missing paper-scale scene benchmark or
the official CUDA/Warp parity fixture.

I also added an opt-in trainer knob:

```text
model.init_point_cloud_duplicate_jitter
```

It jitters duplicated backfill cells when a point cloud has fewer points than
`model.cells`. The point-cloud init tests cover the behavior. I then ran a
direct A/B on the selected 512px 16f clean artifact:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_dupjitter_noaux.jsonc
```

Command:

```text
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
  .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_dupjitter_noaux.jsonc
```

Result:

- train-visible filtered count: `731`; duplicated backfill count: `293`
- duplicate jitter: `0.04`
- train loop elapsed: `31.53s`
- best step: `20`
- best heldout PSNR / L1 / SSIM:
  `8.3993` / `0.318282` / `0.0206`
- final heldout PSNR / L1 / SSIM:
  `8.3866` / `0.318756` / `0.0188`
- final source PSNR / L1 / SSIM:
  `8.3058` / `0.318158` / `0.0406`

Interpretation: exact duplicate radius collapse was a real implementation smell,
but naive duplicate jitter is not the missing quality lever. It hurts heldout
relative to the selected non-jitter 512px row (`8.5355`). Keep the knob for
controlled future probes, but do not promote this config.
