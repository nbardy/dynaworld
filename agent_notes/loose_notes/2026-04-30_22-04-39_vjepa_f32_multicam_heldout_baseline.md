# V-JEPA F32 Multicam Heldout Baseline

## Run

- Config: `src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha_lr3e4_camclamp.jsonc`
- Script: `PYTHONPATH=src:src/train .venv/bin/python src/train/train.py <config>`
- W&B: https://wandb.ai/nbardy/dynaworld/runs/iom0ibz8
- Steps: 1000
- Wall: 57m25s train progress wall, including final video logging
- Device: MPS
- Data: DeepView `03_Dog`, train `camera_0001` + `camera_0015`, heldout `camera_0040`

## Why This Config Exists

The original "ultimate" config launched but went NaN around step 46 in W&B run
`nlz1057l`. The failure looked like stability/camera drift rather than a data
path issue: reconstruction stayed finite while total loss/camera diagnostics
went NaN.

This stabilized config keeps the intended baseline ingredients:

- V-JEPA 2.1 ViT-B/384 precomputed features
- static/dynamic split: 96 static tokens, 32 dynamic tokens
- cross-attn-4
- F=32 feature splatting
- alpha-aware composition plus random train background
- LN + Kaiming + gain=4 colorizer
- 256px render/input
- 8192 splats
- two train cameras, one held-out camera

The stabilizing changes are:

- `train.lr`: `0.001` -> `0.0003`
- `train.camera_rig_lr`: `0.001` -> `0.0003`
- camera prediction clamps matched to the known-stable V-JEPA camera range:
  - `max_fov_delta_degrees: 3.0`
  - `max_radius_scale: 1.1`
  - `max_rotation_degrees: 1.0`
  - `max_translation_ratio: 0.03`

## Smoke Gate

Before the 1000-step W&B run, a 120-step offline gate completed without NaNs:

- Offline run: `wandb/offline-run-20260430_205610-asr0gls4`
- Final train-view metrics:
  - TrainView0 PSNR 19.0464, SSIM 0.4226
  - TrainView1 PSNR 16.9402, SSIM 0.3527
- Final heldout `camera_0040`: PSNR 10.5071, SSIM 0.1940

## 1000-Step Result

The online run completed and synced:

- W&B: https://wandb.ai/nbardy/dynaworld/runs/iom0ibz8
- Final train-view metrics:
  - TrainView0 PSNR 24.0275, SSIM 0.7150
  - TrainView1 PSNR 24.4131, SSIM 0.7389
- Final heldout `camera_0040`: PSNR 8.6923, SSIM 0.0711
- Final training loss: 0.0580

The terminal also printed an earlier validation block before the final block:

- TrainView0 PSNR 21.0358, SSIM 0.5542
- TrainView1 PSNR 20.9299, SSIM 0.5677
- Heldout `camera_0040`: PSNR 9.3172, SSIM 0.0930

## Interpretation

The run is valid and stable, but it is not a good held-out baseline yet.

It strongly overfits the two training cameras: train PSNR rises to ~24, while
heldout PSNR falls from the 120-step smoke value of 10.5071 to 8.6923 at 1000
steps. The shorter run / earlier checkpoint was better for held-out view.

This is useful evidence: the feature-splatting + V-JEPA + static/dynamic stack
can optimize the train cameras cleanly, but the current objective/camera/alpha
setup does not preserve novel-view generalization on this DeepView 3-cam probe.
Heldout PSNR remains below the recorded free dynamic gsplat baseline of 13.2940.
