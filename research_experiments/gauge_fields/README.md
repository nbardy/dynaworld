# Gauge Fields Material-Surfel Experiment

This is the toy renderer/trainer for the material-coordinate 4D primitive direction.

It deliberately uses the same 128px/4fps Dynaworld overfit video as the current
single-video baselines:

```text
test_data/test_video_small_128_4fps.mp4
```

The goal is not to beat FasterGS yet. The first goal is to test whether a
persistent material field can overfit the same clip at all.

## Support Modes

The harness now compares three support laws inside the same persistent material
field:

```text
screen_disk             current baseline: projected circular disk radius
oriented_slab           thin-initialized oriented 3D slab metric
rank_adaptive_metric    transported learned full-rank 3D PSD metric
```

All three keep persistent element identity, color, opacity, low-rank time
transport, RGB/alpha/depth/X-map outputs, and the same train loop. The support
mode changes only how each transported element becomes a projected pixel kernel.
`oriented_slab` is initialized thin but not yet hard-constrained to stay a
surface; `rank_adaptive_metric` exposes a full-rank metric and does not yet add
an eigenvalue/rank sparsity penalty.

The side-by-side 16-frame configs are:

```bash
# Baseline projected disks.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc

# Thin-initialized transported slabs.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_oriented_slab_motion_128_16f_2048el.jsonc

# Full-rank transported metric candidate.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_motion_128_16f_2048el.jsonc
```

There is also a first held-out-camera benchmark lane using the generated
multi-camera validation manifest. It trains only on the DeepView source camera
`camera_0001` and evaluates `heldout_*` metrics against `camera_0015` using the
relative pose from DeepView `models.json`:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_screen_disk_128_16f_2048el.jsonc
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_oriented_slab_128_16f_2048el.jsonc
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_128_16f_2048el.jsonc
```

This is a better representation gate than source-view PSNR, but still not the
final word: the DeepView images are fisheye and the toy gauge-field renderer
uses a pinhole projection approximation for this lane.

The matched direct 3DGS control uses the same source/held-out bundle and trains
a per-frame free Gaussian splat bank with no video encoder:

```bash
uv run python research_experiments/gauge_fields/train_splat_baseline.py \
  src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc
```

The fast plumbing smokes are:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_oriented_slab_smoke_32_2f_32el.jsonc
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc
```

## Incidence Modes

`support_mode` chooses the world/event support state. `render.incidence_mode`
chooses how a camera ray interacts with that same state.

Current incidence laws:

```text
projected_conic             existing fast projected-kernel approximation
ray_gaussian_line_peak      finite-segment ray integral with peak density
ray_gaussian_line_mass      finite-segment ray integral with calibrated mass
```

The first intended comparison keeps:

```text
support_mode = rank_adaptive_metric
x_i(t), Sigma_i(t), alpha/mass strength, c_i
```

fixed and swaps only `render.incidence_mode`.

Full-size DeepView configs:

```bash
# Fast control.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_128_16f_2048el.jsonc

# Exact line integral, peak-density interpretation.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_peak_128_16f_2048el.jsonc

# Exact line integral, mass-normalized interpretation.
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_128_16f_2048el.jsonc
```

Compact held-out-camera smoke:

```bash
for mode in projected_conic ray_gaussian_line_peak ray_gaussian_line_mass; do
  uv run python research_experiments/gauge_fields/train.py \
    src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_incidence_smoke_32_2f_64el.jsonc \
    --incidence-mode "$mode" \
    --steps 2 \
    --no-wandb \
    --output-dir "outputs/gauge_fields/multicam_deepview_incidence_smoke_32_2f_64el/$mode"
done

uv run python research_experiments/gauge_fields/summarize_runs.py \
  'outputs/gauge_fields/multicam_deepview_incidence_smoke_32_2f_64el/*' \
  --sort-by heldout_eval_psnr \
  --out-md outputs/gauge_fields/multicam_deepview_incidence_smoke_32_2f_64el/summary.md \
  --out-json outputs/gauge_fields/multicam_deepview_incidence_smoke_32_2f_64el/summary.json
```

Current smoke result, 32px / 2 frames / 64 elements / 2 steps:

| incidence_mode | eval_psnr | heldout_eval_psnr | heldout_eval_l1 | heldout_alpha_coverage_050 |
| --- | ---: | ---: | ---: | ---: |
| `projected_conic` | 6.2087 | 5.3154 | 0.5048 | 0.0234 |
| `ray_gaussian_line_mass` | 5.8856 | 5.0739 | 0.5218 | 0.0000 |
| `ray_gaussian_line_peak` | 3.9617 | 3.7377 | 0.6199 | 0.0000 |

This is a plumbing/scale sanity check, not a representation result. It says the
line-integral path trains and evaluates through the held-out-camera lane; it
also says peak density starts under-covered and mass-normalized line incidence
is the only exact candidate close enough to benchmark seriously.

Run the full 80-step incidence matrix with wall-clock logging:

```bash
uv run python research_experiments/gauge_fields/run_deepview_incidence_matrix.py \
  --steps 80 \
  --device mps \
  --no-wandb \
  --output-root outputs/gauge_fields/multicam_deepview_incidence_matrix_80step

uv run python research_experiments/gauge_fields/summarize_runs.py \
  'outputs/gauge_fields/multicam_deepview_incidence_matrix_80step/*' \
  --sort-by heldout_eval_psnr \
  --out-md outputs/gauge_fields/multicam_deepview_incidence_matrix_80step/summary.md \
  --out-json outputs/gauge_fields/multicam_deepview_incidence_matrix_80step/summary.json
```

Current 80-step held-out-camera result:

| representation | incidence | eval_psnr | heldout_eval_psnr | wall_clock_min |
| --- | --- | ---: | ---: | ---: |
| `rank_adaptive_metric` | `ray_gaussian_line_peak` | 21.6338 | 11.5705 | 16.8116 |
| `rank_adaptive_metric` | `ray_gaussian_line_mass` | 24.3293 | 9.9005 | 26.9624 |
| `free_dynamic_3dgs` | n/a | 20.5017 | 9.7392 | 2.1135 |
| `screen_disk` | `projected_conic` | 24.6535 | 9.6479 | 1.3610 |
| `rank_adaptive_metric` | `projected_conic` | 24.2230 | 9.5814 | 3.1677 |

Do not overclaim the `ray_gaussian_line_peak` row yet: it has weak source fit,
very large held-out projection coverage, and lower X-map occupancy. The cleaner
exact-incidence candidate is still `ray_gaussian_line_mass`; the implementation
problem is runtime/culling.

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
opacity_split_clone
dormant_insert
```

Each probe writes `probe_summary.json`, per-probe metrics, and preview media.
The preview columns are:

```text
target | base_render | probe_render | abs_probe_minus_base | probe_alpha
```

Each probe also writes `xmap_depth_alpha.png`; with `--include-flow`, it writes
`flow.png`.

## Sweep Configs

Generate the first capacity/radius/alpha sweep:

```bash
uv run python research_experiments/gauge_fields/make_sweep_configs.py \
  --base-config src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc \
  --output-dir src/train_configs/generated_gauge_fields_sweeps \
  --elements 1024,2048,4096 \
  --radii 0.05,0.07,0.09 \
  --alpha-logits=-1.2,0.0 \
  --support-modes screen_disk,oriented_slab,rank_adaptive_metric \
  --incidence-modes projected_conic \
  --steps 150
```

Summarize completed sweep outputs:

```bash
uv run python research_experiments/gauge_fields/summarize_runs.py \
  'outputs/gauge_fields/sweeps/gauge_fields_*_motion_128_16f_*' \
  --out-md outputs/gauge_fields/sweeps/summary.md \
  --out-json outputs/gauge_fields/sweeps/summary.json
```

The first compact 80-step sweep found:

```text
2048/r0.09: PSNR 17.78, coverage budget 4.30, xmap_occ 0.226
2048/r0.07: PSNR 17.68, coverage budget 2.85, xmap_occ 0.240
1024/r0.09: PSNR 17.60, coverage budget 2.50, xmap_occ 0.229
1024/r0.07: PSNR 17.04, coverage budget 1.74, xmap_occ 0.258
```

Early conclusion: increasing radius can substitute for element count in short
RGB overfit, but it pushes the model toward heavier coverage/smear and lower
canonical-coordinate occupancy. Keep both PSNR and xmap/coverage certificates
in the table.

## Current Limits

This is still a pure Torch projected-kernel renderer. It uses chunked pixels,
but it is still `elements x pixels` work. Keep element count and frame count
small until the representation proves it can learn.

The harness reuses Dynaworld's existing JSONC config loader, sequence loader,
and W&B media helpers. It does not reuse the full video-token trainer because
that class assumes GaussianSequence outputs and fast-mac rendering.
