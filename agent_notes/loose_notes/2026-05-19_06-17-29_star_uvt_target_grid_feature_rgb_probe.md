# STAR UVT Target-Grid Feature-to-RGB Probe

## Goal

After the target-grid V-JEPA route passed speed and memory gates, the remaining
question was whether the cached target-grid features were visually decodable at
all. The RGB-aux rows were weak: aux10 improved with more steps but stayed far
below RGB STAR quality, and RGB-warm20 was a matched negative. This probe trains
only `FeatureToColor` on the actual cached V-JEPA target grid, so it is an
oracle for target-feature decodability, not a STAR rasterizer quality row.

## Implementation

Added:

- `src/train/train_star_uvt_feature_rgb_probe.py`
- `src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.jsonc`
- `tests/test_star_uvt_feature_rgb_probe.py`

The script reuses the STAR V-JEPA target loader, requires
`feature_target.materialization=target_grid`, loads the channel-adapted target
grid `[32,32,16,16]`, downsamples the 64-frame RGB video to `[32,3,16,16]`,
trains a hidden64 `FeatureToColor`, and upsamples the predicted grid RGB back to
full `[64,3,512,512]` for PSNR/media.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.jsonc
```

Rerun recorded offline W&B id `7nlur74e`.

## Result

Artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.json
outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt
outputs/media/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step_sbs.mp4
```

Measured:

- pass: `true`
- cache: hit
  `outputs/feature_cache/star_uvt_feature_targets/vjepa2_1_vitb_256crop_64f/a524619cf73c9cc18bdbe53d.pt`
- target grid: `[32,32,16,16]`, `1.0MiB`
- target RGB grid: `[32,3,16,16]`
- grid loss: `0.0470415 -> 0.0045697`
- grid PSNR: `13.275 -> 23.401`
- full upsampled loss / PSNR: `0.0098326` / `20.073`
- mean timing: `2.427ms/step`
- forward/loss: `0.757ms`
- backward: `1.003ms`
- optimizer: `0.667ms`
- target-grid load/prep: `128.842ms`

## Interpretation

The target-grid V-JEPA features are decodable. The poor STAR target-grid visual
rows are not explained by an inherently undecodable cached target representation.
The next bridge should load or freeze this trained decoder inside STAR training
or canonical probe logging, then test whether STAR-rendered feature grids move
toward the target-grid decoder image. Spending more iterations only on RGB aux
weighting or feature-loss-skipping warmup is now lower priority.

## Docs Updated

- `BASELINES.md`
- `TODO/README.md`
- `PROJECT_INDEX.md`
- `EXPERIMENTS.md`
- `README.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py`
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py`
- generated reports:
  `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  and
  `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`

## Next Gate

Wire the checkpoint as a frozen `FeatureToColor` probe in the STAR feature
trainer, with a config that logs target-grid-probe RGB and optionally adds a
small RGB-probe loss. The pass/fail condition should compare STAR-rendered
feature-grid decode against the oracle target-grid decode, not just raw RGB aux
PSNR.
