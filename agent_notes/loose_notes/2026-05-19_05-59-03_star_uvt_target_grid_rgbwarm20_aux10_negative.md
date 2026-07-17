# 2026-05-19 05:59:03 - STAR UVT Target-Grid RGB-Warm20 Negative Gate

## Goal

Test the next documented visual gate for the fast STAR UVT target-grid V-JEPA
route: a matched 100-step RGB warm-start schedule against the constant aux10
100-step row.

The hypothesis was that training the RGB/colorizer path first might improve the
weak target-grid media quality without changing the selected renderer or the
target-grid memory path.

## Implementation

Added config-driven staged loss weights to
`src/train/train_star_uvt_feature_overfit.py`:

```json
"feature_target": {
  "weight_schedule": [
    {
      "label": "rgb_warm20",
      "until_step": 20,
      "loss_weight": 0.0,
      "rgb_loss_weight": 20.0
    },
    {
      "label": "target_grid_aux10",
      "until_step": 100,
      "loss_weight": 1.0,
      "rgb_loss_weight": 10.0
    }
  ]
}
```

The trainer records `feature_target_weight_schedule`,
`step_feature_target_loss_weights`, and `step_rgb_loss_weights` in the output
JSON so scheduled objectives are auditable.

Focused validation before launch:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py -q
```

Result: `6 passed`.

## Run

Config:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.jsonc
```

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.jsonc
```

W&B offline run: `rih5t7h8`

Artifacts:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.json
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_sbs.mp4
```

## Result

Mechanical gate passes:

- `pass=true`
- zero tile overflow
- `colorizer_grad_seen=true`
- target grid `[32,32,16,16]`, `1.0MiB`

Metrics:

- total loss: `6.763425 -> 4.102212`
- feature loss, measured only after warmup: `0.999868 -> 0.973557`
- RGB loss: `0.338171 -> 0.312865`
- RGB PSNR: `4.709 -> 5.046`
- mean step: `1639.10ms`
- mean render forward: `548.32ms`
- mean target/loss: `27.70ms`
- mean backward: `872.48ms`
- target load/prep: `197.29ms`

Matched comparison against constant aux10 100-step:

- constant aux10: RGB PSNR `5.109`, feature loss `0.964670`,
  `1876.37ms/step`, `1032.85ms` backward
- RGB-warm20: RGB PSNR `5.046`, feature loss `0.973557`,
  `1639.10ms/step`, `872.48ms` backward

## Interpretation

RGB-warm20 is faster because the first 20 steps skip the target-grid feature
loss, but it is worse on the quality metrics we care about at the same 100-step
budget. This makes feature-loss-skipping warmup a negative visual-control gate,
not a promotion.

The next target-grid visual gate should be a trained/frozen feature-to-RGB
probe or a native-VJP/objective change, not more variants that simply delay the
feature loss.

## Docs Updated

- `BASELINES.md`
- `README.md`
- `PROJECT_INDEX.md`
- `EXPERIMENTS.md`
- `TODO/README.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.{json,md}`
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.{json,md}`
- `agent_notes/key_learnings.md`
