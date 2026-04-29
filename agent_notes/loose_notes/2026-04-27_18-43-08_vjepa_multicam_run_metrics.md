# V-JEPA multicam dynamic/static run metrics

## Context

Implemented the first V-JEPA/static-dynamic multicam trainer path for DeepView-style
three-camera data:

- train cameras: `camera_0001`, `camera_0015`
- held-out camera: `camera_0040`
- feature backend: `vjepa2_1_vitb_384`
- render/eval size: 128 px
- frames: 16
- splats: 8192
- static/dynamic split: 96 static, 32 dynamic

The trainer loads two train views and one held-out view, prebakes features from
the anchor training camera, optimizes one shared decoded world, renders through a
learnable external camera rig per view, and logs per-camera eval PSNR/SSIM.

## Commands run

Smoke compile:

```bash
rtk uv run python -m py_compile src/train/multicam_video_data.py src/train/camera_rig.py src/train/train_multicam_precomputed_feature_implicit_dynamic.py
```

RGB-pyramid smoke:

```bash
rtk env WANDB_MODE=disabled uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc
```

Online W&B V-JEPA run:

```bash
rtk ./src/train_scripts/train_multicam_static_dynamic_vjepa_features.sh
```

Cached 50-step metric run:

```bash
rtk env WANDB_MODE=disabled uv run python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src/train").resolve()))
from config_utils import load_config_file
from train_multicam_precomputed_feature_implicit_dynamic import main

cfg = load_config_file("src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc")
cfg["train"]["steps"] = 50
cfg["train"]["train_views_per_step"] = 0
cfg["logging"]["wandb_run_name"] = cfg["logging"]["wandb_run_name"] + "-cached-50step-disabled"
cfg["logging"]["log_every"] = 10
cfg["logging"]["image_log_every"] = 50
cfg["logging"]["video_log_every"] = 50
cfg["logging"]["always_log_last_step"] = True
main(cfg)
PY
```

## Results

The online W&B run created run `272ptnb5`:

```text
https://wandb.ai/nbardy/dynaworld/runs/272ptnb5
```

It reached step 86/250, then stalled badly with long per-step pauses and was
interrupted before final validation. Its step-0 eval summary was:

| View | SSIM | PSNR |
| --- | ---: | ---: |
| TrainView0 | 0.0922 | 5.1439 |
| TrainView1 | 0.0857 | 5.0774 |
| Heldout | 0.0885 | 4.5916 |

The cached W&B-disabled 50-step run completed. Step-0 eval:

| View | SSIM | PSNR |
| --- | ---: | ---: |
| TrainView0 | 0.0669 | 5.1669 |
| TrainView1 | 0.0875 | 4.9320 |
| Heldout | 0.0894 | 4.2550 |

Step-50 eval:

| View | SSIM | PSNR |
| --- | ---: | ---: |
| TrainView0 | 0.1848 | 14.4296 |
| TrainView1 | 0.2069 | 14.5911 |
| Heldout | 0.0986 | 9.8293 |

Final train loss was about `0.1968`.

## Notes

- The V-JEPA feature cache was created at
  `data/feature_cache/multicam_deepview_static_dynamic_vjepa2_1_vitb_384/5bcd6903a42607cd116ef474.pt`.
- The 50-step metrics are from a fresh model, not a resume from the interrupted
  W&B run.
- Longer runs should use the now-warm cache and may be less painful with
  `WANDB_MODE=disabled` until the training loop speed is stable.
