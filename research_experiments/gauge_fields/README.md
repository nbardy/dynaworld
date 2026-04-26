# Gauge Fields Material-Surfel Experiment

This is the toy renderer/trainer for the material-coordinate 4D primitive direction.

It deliberately uses the same 128px/4fps Dynaworld overfit video as the current
single-video baselines:

```text
test_data/test_video_small_128_4fps.mp4
```

The goal is not to beat FasterGS yet. The first goal is to test whether a
persistent material field can overfit the same clip at all.

## Current Baseline

The current stable baseline candidate is the 2048-element, 16-frame,
T-parameterized material-surfel run:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc
```

Reference 100-step run:

```text
W&B: https://wandb.ai/nbardy/dynaworld/runs/ajy46erb
local: outputs/gauge_fields/material_surfel_motion_128_16f_2048el_100step
eval_l1: 0.0759
eval_psnr: 18.03
alpha_mean: 0.940
alpha_coverage_050: 0.999
```

This is a valid overfit baseline for the representation. It proves the
projected material-gauge renderer and low-rank time parameterization can fit a
small 128px video window. It does not yet prove novel-view quality, real camera
geometry, or superiority over FasterGS.

## Smoke

Renderer-only smiley smoke:

```bash
uv run python research_experiments/gauge_fields/smiley_smoke.py \
  --device mps \
  --output-dir outputs/gauge_fields/smiley_smoke
```

Video-loader/training/W&B plumbing smoke:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc
```

## 128px Ablation-Budget Run

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_128_16f_512el.jsonc
```

This matches the recent 16f/128px comparison configs in video, resolution, and
250-step ablation budget. It does not match the fast-mac renderer throughput or
8192-Gaussian capacity.

## Long 128px Overfit

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_128_16f_512el_long.jsonc
```

This matches the longest recent local 128px overfit budget in the important
training axes:

```text
video: test_data/test_video_small_128_4fps.mp4
render_size: 128
max_frames: 0
train_frame_count: 16
steps: 1000
```

It still uses `frames_per_step = 1` because this renderer is pure Torch and
does not have fast-mac's fused temporal batch path.

## Stability Ladder

Run these before trusting the long config:

```bash
# Gate 1: one-frame dense coverage. If this fails, debug renderer/init first.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_static_128_1f_2048el.jsonc

# Gate 2: short-window motion capacity. If this fails after Gate 1 passes,
# debug motion basis / camera / regularization.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc
```

Expected first-order diagnosis:

```text
static 1f fails  -> coverage/init/loss issue
static 1f passes, motion 16f fails -> motion/camera issue
motion 16f passes, constrained 512el fails -> material contract too strict or too small
```

## Outputs

Each run writes:

- `config.json`
- `logs.json`
- `metrics.json`
- `checkpoint.pt`
- `preview.png`
- `side_by_side.mp4`

The preview columns are:

```text
target | render | abs_error | alpha
```

The checked-in configs also log to W&B by default:

```text
project: dynaworld
videos: Render_Video, Render_GT_Video, GT_Video
image: Render_GT_vs_Pred
scalars: Loss, Loss/RGBL1, Eval/L1, Eval/MSE, Eval/PSNR
health: Eval/AlphaCoverage005, Eval/AlphaCoverage050, Model/RadiusMeanFinal
```

Diagnostics are enabled by default in the harness and saved to `metrics.json`:

```text
projection radius p05/p50/p95
coverage budget
radius clamp fractions
motion basis / coeff / displacement stats
xmap occupancy / entropy / local smoothness
optional flow magnitude stats
```

## Cheat Probes

Run deterministic post-training probes against a saved checkpoint:

```bash
uv run python research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  --checkpoint outputs/gauge_fields/material_surfel_motion_128_16f_2048el_100step/checkpoint.pt \
  --output-dir outputs/gauge_fields/material_surfel_motion_128_16f_2048el_100step/probes \
  --device mps \
  --probe all
```

Current probes:

```text
depth_slide
radius_inflate
opacity_radius_trade
basis_scale_gauge
motion_phase_shift
```

Each probe writes `probe_summary.json`, per-probe metrics, and preview media.

## Sweep Configs

Generate the first capacity/radius/alpha sweep:

```bash
uv run python research_experiments/gauge_fields/make_sweep_configs.py \
  --base-config src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc \
  --output-dir src/train_configs/generated_gauge_fields_sweeps \
  --elements 1024,2048,4096 \
  --radii 0.05,0.07,0.09 \
  --alpha-logits=-1.2,0.0 \
  --steps 150
```

## Current Limits

This is still a pure Torch projected-disk renderer. It uses chunked pixels, but
it is still `elements x pixels` work. Keep element count and frame count small
until the representation proves it can learn.

The harness reuses Dynaworld's existing JSONC config loader, sequence loader,
and W&B media helpers. It does not reuse the full video-token trainer because
that class assumes GaussianSequence outputs and fast-mac rendering.
