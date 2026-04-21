# Taichi 8192-Splat Schedule Config

## Changes

Updated the 8192-splat Taichi config:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi_8192splats.jsonc
```

- `gaussians_per_token = 64`
- `frames_per_step = 8`
- `lr = 0.001`
- `lr_schedule.type = "cosine"`
- `lr_schedule.final_lr_scale = 0.1`
- renderer metrics off

Also turned renderer metrics off in the 512-splat Taichi config:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi.jsonc
```

## Trainer Support

Added config-driven LR scheduling in `src/train/dynamicTokenGS.py`.

Supported schedule types:

- `constant` / `none` / `off`
- `cosine`

For cosine, step 1 uses the base LR and the final step reaches:

```text
base_lr * final_lr_scale
```

The current 8192-splat config therefore decays:

```text
0.001 -> 0.0001
```

`LearningRate` is now logged with scalar W&B payloads.

## Validation

- `uv run python -m py_compile src/train/dynamicTokenGS.py`
- Config parse confirmed:
  - `8192` splats
  - `frames_per_step = 8`
  - renderer metrics off
  - LR step 1 = `0.001`
  - LR final = `0.0001`
- One-step 8-frame/8192-splat Taichi smoke completed:
  - finite loss `0.1669`
  - LR schedule printed
  - renderer metrics stayed off
- `git diff --check` passed.

## Z-Range Clarification

The `z_min`/`z_max` values live in the Gaussian head, not in the renderer. They bound where the model can place predicted splat centers in its current world/canonical frame:

```python
z = sigmoid(raw_z) * (z_max - z_min) + z_min
```

They are a representation/init contract between the model and the camera coordinate scale. The wide-depth config uses `z=[0.5, 5.0]` because the 128px/4fps DUSt3R camera path can move late cameras past the old `z_max=2.5` slab.
