# Dynamic PowerFoam Residual Camera High-Res Video

## Context

The first learned-camera full-clip run used random object-centric foam and was a negative diagnostic: it learned camera motion, but quality fell far below the fixed-camera all-enabled baseline.

For a usable high-res video, we tested the residual-camera variant that preserves the strong fixed-pinhole/image-plane initialization. The camera starts exactly as the fixed-pinhole gauge (`base_position=[0,0,0]`, `look_at=[0,0,1]`) and only learns a small residual path.

## Config

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_residual_camera.jsonc
```

Output directory:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_residual_camera
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train WANDB_SILENT=true .venv/bin/python \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_residual_camera.jsonc
```

## Metrics

Initial eval matched the fixed-camera baseline, proving the gauge/init was preserved:

```text
mean PSNR 17.2777
min PSNR 10.5138
mean SNR 11.4603
min SNR 3.0375
L1 0.09850
MSE 0.02296
alpha mean 0.95313
camera rotation 0.0 deg
camera translation 0.0
```

Final eval at step 120:

```text
mean PSNR 18.0871
min PSNR 11.0965
mean SNR 12.2697
min SNR 2.6492
L1 0.08928
MSE 0.01964
alpha mean 0.99391
camera rotation mean 0.1335 deg
camera translation mean 0.01163
camera forward delta mean 0.00167
```

Comparison:

```text
fixed-camera baseline final:    mean PSNR 18.0796, L1 0.08914
residual-camera final:          mean PSNR 18.0871, L1 0.08928
object-centric camera negative: mean PSNR 11.2901, L1 0.22518
```

The residual-camera branch is effectively quality-parity with the fixed-camera high-res baseline while keeping the learned camera motion small and stable.

## Deliverables

H.264/yuv420p deliverables:

```text
outputs/dynamic_powerfoam_metal/deliverables/dynamic_powerfoam_residual_camera_512_56f_side_by_side_h264.mp4
outputs/dynamic_powerfoam_metal/deliverables/dynamic_powerfoam_residual_camera_512_56f_render_h264.mp4
```

ffprobe:

```text
side-by-side: h264 High, yuv420p, 1024x512, 56 frames, 8 fps, 7.0s
render-only:  h264 High, yuv420p, 512x512, 56 frames, 8 fps, 7.0s
```

Pixel sanity check:

```text
side-by-side: std [57.298, 57.792, 64.034], min 0, max 250, sampled unique colors 4636
render-only:  std [57.364, 58.706, 64.877], min 0, max 250, sampled unique colors 4401
```

These videos are not blank and not the previous local-player green-video failure.

## Takeaway

For this clip, learned camera should start as a residual around the strong fixed-camera/image-plane gauge. The object-centric-random initialization may be theoretically cleaner for camera factorization, but it is not currently a good high-res visual baseline.
