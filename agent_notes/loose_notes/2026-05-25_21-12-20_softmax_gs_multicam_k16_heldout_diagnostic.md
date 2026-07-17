# Softmax-GS Multicam K16 Heldout Diagnostic

Date:
    2026-05-25 21:12 +07

Context:
    Continued the active goal after completing Softmax-GS shader work and plan
    docs. The short-term plan required a matched dynamic-GS source/heldout row
    before touching STAR UVT or WorldFoam.

Configs added:
    - `src/train_configs/local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_128splats_20step.jsonc`
    - `src/train_configs/local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_128splats_20step.jsonc`

Commands:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_128splats_20step.jsonc

PYTHONPATH=src/train WANDB_MODE=offline GSP_TAPE_CAP=16 .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_128splats_20step.jsonc
```

Results:
    No-op control: initial/final train loss `0.5910 -> 0.2261`; train
    view0/view1 PSNR `13.4197/14.3734`; heldout camera_0040 PSNR/SSIM
    `4.7369/0.0503`; step-20 total/backward/raster `291/86/58ms`; offline run
    `wandb/offline-run-20260525_210925-39a0kpp2`.

    Enabled K=16: initial/final train loss `0.5910 -> 0.2262`; train
    view0/view1 PSNR `13.4502/12.3880`; heldout camera_0040 PSNR/SSIM
    `11.7255/0.0794`; step-20 total/backward/raster `372/97/48ms`; offline run
    `wandb/offline-run-20260525_211008-vfwslw6q`.

Interpretation:
    This is the first positive heldout-style Softmax-GS signal: K=16 ties final
    train loss while substantially improving heldout PSNR on the tiny
    RGB-pyramid multicam diagnostic. It is not a promotion or baseline row
    because the setting is only 64px/4f/128splats/20step and uses cheap
    RGB-pyramid conditioning. The right next step is a repeat/scale check
    (128px/16f and/or more splats) with residual/tape-coverage diagnostics if
    cheap. Still do not port Softmax-GS into STAR UVT or WorldFoam yet.
