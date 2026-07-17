# WorldFoam real32 fixture smoke

## Context

The previous real-frame gate note correctly rejected the repeated-loaded-frame
32f result as synthetic, but its "data-blocked" framing was too broad. Checked
multicam validation manifests were only 16f, but raw DeepView `03_Dog` data was
available and long enough to build a real 32f heldout-multicam fixture.

## Added fixture/config

- `src/dataset_configs/multicam_val_deepview_03dog_128_8fps_32f.jsonc`
  builds a single 4s/8fps/32f DeepView heldout-multicam sample from
  `camera_0001` to `camera_0040`.
- Generated manifest:
  `data/multicam_val/clip_sets/multicam_val_deepview_03dog_128_8fps_32f/manifest.jsonl`
  with `sample_id=deepview_03_Dog_camera_0001_to_camera_0040`,
  `frame_count=32`, `fps=8.0`, and `target_size=128`.
- `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc`
  points WorldFoam/PowerFoam Metal at two train cameras
  (`camera_0001`, `camera_0015`) and heldout `camera_0040`.

Builder command:

```bash
PYTHONPATH=src/train uv run python src/dataset_pipeline/multicam_val.py build \
  --config src/dataset_configs/multicam_val_deepview_03dog_128_8fps_32f.jsonc
```

Loader check result:

```text
frame_count 32
train (2, 32, 3, 32, 32) ['camera_0001', 'camera_0015']
heldout (1, 32, 3, 32, 32) ['camera_0040']
pose_source deepview_models_relative_opencv_fisheye
lens ['opencv_fisheye', 'opencv_fisheye'] ['opencv_fisheye']
```

## Real32 WorldFoam smoke

Command:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc \
  --frame-counts 32 \
  --render-size 16 \
  --site-count 8 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode manual-vjp \
  --tape-mode owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid \
  --endpoint-record-source slow-owner-run \
  --experimental-selected-only-owner-run-delta-prep \
  --experimental-native-owner-run-cutwalk-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_real32_native_cutwalk_loader_smoke.json
```

Result:

- artifact status: `ok`
- row status: `ok`
- frame_count: `32`
- loaded_frame_count: `32`
- repeat_loaded_frames: `false`
- repeat scope: `real loaded frame count`
- loss decreased: `true`
- first_grad_abs_sum: `0.472179651260376`
- parameter_update_abs_max: `0.029999971389770508`
- final train PSNR: `12.413400833568957`
- final heldout PSNR: `13.872671170059306`

The smoke ended with `benchmark_environment.status=contended` because
`MTLCompilerService` was active at the end snapshot. Therefore the `727ms`
total / `677ms` backward timing is not promotable timing evidence. Treat this
as a data/loader/shader correctness gate only.

## Wrapper shape

Dry-run command:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_real32_dryrun \
  --worldfoam-config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc \
  --star-video-path data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4 \
  --frame-counts 32 \
  --require-real-loaded-frames \
  --verify-promotion \
  --dry-run \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-20_real32_dryrun.promotion_summary.json
```

The dry-run summary records the intended strict command shape and keeps
`repeat_loaded_frames=false`, `require_real_loaded_frames=true`,
`worldfoam_config=...real32_32_smoke.jsonc`, and the matched STAR video path.

## Next

The immediate 32f data-loader blocker is removed. The next real promotion run
should use the strict wrapper in a quiet benchmark environment and require
`--require-real-loaded-frames --verify-promotion`; otherwise a passing timing
row can still be contaminated by MPS/Metal compiler or live `ai_trader` export
activity.
