# Dynamic PowerFoam Implicit Camera Full-Clip Run

## Context

After wiring the learned implicit-camera path, the first 32px/3f/1-step run was only a mechanical smoke. It was visually tiny, sparse, and random-looking by construction. It should not be used as a quality artifact.

The meaningful goal is: compare the fixed-camera all-enabled 512px/56f feature PowerFoam baseline against a matched learned-camera run on the same high-motion center-cropped YouTube clip, then decide whether the camera branch helps source fit or just moves the gauge.

## Run

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train WANDB_SILENT=true .venv/bin/python \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_implicit_camera.jsonc
```

Config:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_implicit_camera.jsonc
```

Output:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_implicit_camera
```

W&B local run:

```text
wandb/run-20260506_165105-4swdbc57
```

The run was based on commit `336e96f100a0f88837e03960cb982798025d5968`.

## Metrics

Initial eval:

```text
mean PSNR 8.0476
min PSNR 6.0770
mean SNR 2.2302
min SNR -4.3788
L1 0.33038
MSE 0.16416
alpha mean 0.69206
camera rotation mean 0.0 deg
camera translation mean 0.0
```

Final eval at step 120:

```text
mean PSNR 11.2901
min PSNR 8.6798
mean SNR 5.4727
min SNR -2.5891
L1 0.22518
MSE 0.07895
alpha mean 0.99969
camera rotation mean 4.5426 deg
camera translation mean 0.30447
camera forward delta mean 0.01924
camera global residual L2 0.01111
```

Final artifacts:

```text
render_step_0120.mp4
side_by_side_step_0120.mp4
preview_step_0120.png
checkpoint_final.pt
dynamic_geometry_summary.json
per_frame_metrics_step_0120.json
train_metrics_history.jsonl
```

MP4 pixel sanity check on the first decoded final frame:

```text
render_step_0120.mp4: frames=56 fps=8 shape=(512,512,3)
mean RGB [148.782, 142.809, 147.131]
std RGB [20.613, 19.312, 57.199]
min 67 max 246 sampled unique colors 2755

side_by_side_step_0120.mp4: frames=56 fps=8 shape=(512,1024,3)
mean RGB [152.971, 148.410, 150.803]
std RGB [42.908, 42.509, 60.134]
min 0 max 250 sampled unique colors 4475
```

This confirms the final videos are not the green-video writer/player failure.

## Comparison To Fixed-Camera Baseline

Fixed-camera all-enabled baseline:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step
```

Final fixed-camera eval:

```text
mean PSNR 18.0796
min PSNR 11.1518
mean SNR 12.2622
min SNR 2.6162
L1 0.08914
MSE 0.01959
alpha mean 0.99483
state mean temporal screen delta 0.4027 px
```

The learned-camera/object-centric run is much worse on source reconstruction:

```text
mean PSNR 11.2901 vs 18.0796
L1 0.22518 vs 0.08914
MSE 0.07895 vs 0.01959
```

It did learn camera motion, but it did not use that motion to improve this source-fit benchmark. The old fixed-camera baseline is still the better-looking and better-scoring local baseline for this clip.

## Interpretation

The learned-camera path is mechanically real: camera parameters move and rendered videos are valid. But this first object-centric initialization is not a good visual baseline. It starts from a harder support/coverage problem than the fixed-camera image-plane init, saturates alpha almost completely by the end, and still trails the fixed-camera baseline by a large margin.

The next useful experiment is not another tiny smoke. It is either:

- camera branch with the stronger fixed-camera/image-plane initialization and a learned residual camera around that gauge, or
- a camera-clamp sweep where `max_rotation_degrees` and `max_translation_ratio` are tightened, so the branch can correct camera motion without destroying support/init quality.

Do not cite the 32px smoke as evidence. Cite this 512px/56f run as the first real learned-camera negative/diagnostic result.
