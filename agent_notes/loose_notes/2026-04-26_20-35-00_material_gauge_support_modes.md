# Material Gauge Support Modes

Date: 2026-04-26

## What Changed

Extended the existing material-gauge trainer instead of forking it. The trainer
now has a `model.support_mode` switch:

```text
screen_disk
oriented_slab
rank_adaptive_metric
```

The existing `screen_disk` path remains the default and preserves the previous
projected-disk baseline.

## Math Implemented

The old support law was an image-space circular covariance:

```math
\Sigma^{screen}_i = r_{i,px}^2 I_2.
```

The new world-support paths compute a world covariance first, then project it:

```math
\Sigma^{screen}_i(t)
=
J_\pi(x_i(t))
\Sigma^{world}_i(t)
J_\pi(x_i(t))^\top.
```

`oriented_slab` uses a transported thin 3D slab:

```math
\Sigma^{world}_i(t)
=
J_i(t)
R_i
\operatorname{diag}(r_{1i}^2,r_{2i}^2,h_i^2)
R_i^\top
J_i(t)^\top
+\epsilon I.
```

`rank_adaptive_metric` uses a learned PSD metric:

```math
G_i = L_iL_i^\top
```

```math
\Sigma^{world}_i(t)
=
J_i(t)G_iJ_i(t)^\top+\epsilon I.
```

The local transport Jacobian is estimated from fixed canonical KNN neighbors:

```math
J_i(t)
\approx
I
+
\Delta_i(t)P_i^\top(P_iP_i^\top+\lambda I)^{-1},
```

where `P_i` are canonical neighbor offsets and `Delta_i(t)` are displacement
offset changes. This is an approximate Jacobian because the current low-rank
motion model is per-element, not a continuous deformation field.

## Renderer Status

Rendering still uses projected-kernel alpha compositing, not the exact Gaussian
line integral yet.

Implemented:

```text
project_points_with_jacobian
projected_support
2D covariance bounding/clamping
Mahalanobis projected ellipse kernel
optional optical_thickness transfer: alpha = 1 - exp(-mass * kernel)
projection anisotropy diagnostics
support-aware radius/mass metrics
support-aware cheat probes
```

Not implemented yet:

```text
exact ray Gaussian line integral
phase-conditioned surface/fiber/volume laws
eigenvalue/rank sparsity penalty
continuous deformation-field Jacobian
tiled fused renderer
```

## Configs Added

Smoke:

```text
src/train_configs/local_mac_gauge_fields_oriented_slab_smoke_32_2f_32el.jsonc
src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc
```

128px / 16-frame support-mode comparisons:

```text
src/train_configs/local_mac_gauge_fields_oriented_slab_motion_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_motion_128_16f_2048el.jsonc
```

Existing screen-disk configs were updated to include explicit support fields so
all side-by-side config files share the same schema.

## Verification

Parser/compile check:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  research_experiments/gauge_fields/make_sweep_configs.py \
  research_experiments/gauge_fields/summarize_runs.py
```

1-step CPU smokes all passed:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc \
  --device cpu --no-wandb --output-dir /tmp/gauge_screen_smoke_check

uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_oriented_slab_smoke_32_2f_32el.jsonc \
  --device cpu --no-wandb --output-dir /tmp/gauge_oriented_slab_smoke_check

uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc \
  --device cpu --no-wandb --output-dir /tmp/gauge_rank_metric_smoke_check
```

5-step 128px / 16-frame MPS sanity checks also passed:

| support mode | PSNR | L1 | coverage | p95 anisotropy | xmap occ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rank_adaptive_metric` | 16.2042 | 0.0972 | 2.9470 | 1.2385 | 0.2512 |
| `oriented_slab` | 16.0955 | 0.0989 | 2.6668 | 1.0179 | 0.3826 |
| `screen_disk` | 16.0704 | 0.0993 | 2.6557 | 1.0000 | 0.3892 |

These are not quality results. They prove the support modes wire through,
differentiate, save outputs, and produce comparable short-budget diagnostics.

Sweep config generation was smoke-tested with all support modes:

```bash
uv run python research_experiments/gauge_fields/make_sweep_configs.py \
  --base-config src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc \
  --output-dir /tmp/gauge_support_sweep_configs_check \
  --elements 32 \
  --radii 0.05 \
  --alpha-logits -1.2 \
  --support-modes screen_disk,oriented_slab,rank_adaptive_metric \
  --steps 1 \
  --disable-wandb
```

## Why No Fork

A fork would make fair comparison harder. Keeping the same trainer means the
support law is the controlled variable while video loading, initialization,
logging, losses, X-map diagnostics, and output formats stay shared.

## Why No Manual Gradients Yet

The first stage uses PyTorch autograd through:

```text
KNN Jacobian estimate
world covariance construction
projection Jacobian
2D covariance inverse
Mahalanobis kernel
alpha compositing
```

Manual gradients are not needed until we implement the exact ray Gaussian
line-integral renderer or a fused tiled kernel. Staying autograd-first keeps the
research surface inspectable.

## Next Step

Run the real comparison:

```text
screen_disk vs oriented_slab vs rank_adaptive_metric
2048 elements, 16 basis, 16 frames, 250 steps
same video, same seed, same loss, same camera
```

Decision rule:

```text
Do not judge only source-view PSNR.
Judge omitted-frame behavior, view-stress renders, X-map consistency, support
anisotropy, and cheat-probe deltas.
```

## Follow-Up Reproducibility Fix

The first 250-step anisotropic runs exposed a checkpoint reproducibility bug:
`support_knn_idx` was registered as a non-persistent buffer. During training the
transport Jacobian used the KNN graph from the initialization points, but after
checkpoint reload the model rebuilt KNN from the trained `x0`. That changed
anisotropic support and made probe renders fail to reproduce the training final
metrics.

Fix:

```text
support_knn_idx is now persistent in the model state_dict.
cheat_probe_material_gauge.py allows it to be missing only for older checkpoints.
```

Any support-mode probe comparison should use checkpoints produced after this
fix.
