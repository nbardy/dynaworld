# Pullback-Metric Incidence Elements

Date: 2026-04-28

## Goal

Pullback-Metric Incidence Elements are a narrow next candidate for the
gauge-fields harness. The goal is not to add a separate model family. The goal
is to keep the existing transported material/event state and derive support
from the transported canonical material neighborhood instead of learning a
screen radius or a free covariance.

The representation should answer one question:

```text
Does a support tensor derived from transported KNN neighborhoods give better
held-out-camera behavior than projected disks, learned rank-adaptive metrics,
and direct free-dynamic 3DGS, under the same data split and reporting surface?
```

Treat this as an ablation inside `research_experiments/gauge_fields`, not as a
new research lane. The existing split between `support_mode` and
`render.incidence_mode` is the right seam.

## Math

Use the current event notation:

```text
q_i                 canonical material coordinate
Phi_t(q_i)          learned transport into world at time t
x_i(t)              Phi_t(q_i)
N(i)                fixed KNN neighborhood in canonical space
w_ij                fixed neighbor weights, sum_j w_ij = 1
ell                 camera ray (o, d, s_near, s_far), ||d|| = 1
rho_i               optical strength / mass parameter
c_i                 event color/material appearance
```

The v0 support is not a learned metric. It is the second moment of the
transported neighborhood:

```text
Delta_ij(t) = x_j(t) - x_i(t)
C_i(t) = sum_{j in N(i)} w_ij Delta_ij(t) Delta_ij(t)^T
tilde C_i(t) = C_i(t) / (trace(C_i(t))/3 + eps)
Sigma_i(t) = s^2 tilde C_i(t) + sigma0^2 I
```

This is the conceptual contract: incidence is judged by how a ray cuts through
transported material neighborhoods, not by an arbitrary screen-space footprint
or a freely learned covariance. Rendering can use projected conics first, then
the existing finite-segment ray-Gaussian line integral if the fast row earns
signal.

The first admissible incidence law should be the existing mass-normalized
finite-segment ray-Gaussian line integral:

```text
tau_i(ell) = rho_i * integral_{s_near}^{s_far} exp(-0.5 r2_i(s)) ds
alpha_i(ell) = 1 - exp(-tau_i(ell))
```

`projected_conic` remains the fast approximation/control. `ray_gaussian_line_peak`
remains a diagnostic branch unless coverage-matched retuning makes it honest.

## Ablations

Run these as rows in the existing gauge matrix:

```text
free_dynamic_3dgs                                  external direct-splat control
screen_disk / projected_conic                      current fast internal control
rank_adaptive_metric / projected_conic             current transported metric fast control
rank_adaptive_metric / ray_gaussian_line_mass      current clean exact-incidence control
derived_support_metric / projected_conic           proposed fast approximation row
derived_support_metric / ray_gaussian_line_mass    proposed exact-incidence row
```

Keep the first fair run on the current stricter DeepView path:

```text
train cameras: camera_0001,camera_0015
held-out camera: camera_0040
resolution: 128 px
frames: 16
steps: 80 first, then 250 only if the 80-step signal survives
budget regimes: same rendered primitive count and same active parameter count
```

Do not tune multiple things at once. If pullback metrics lose badly at the
same initialization, only then sweep the metric eigenvalue floor, trace penalty,
condition-number penalty, and mass calibration.

## Expected Metrics

Primary selector:

```text
heldout_eval_psnr
heldout_eval_l1
heldout_eval_ssim / LPIPS if available
```

Required context:

```text
eval_psnr and eval_l1 for source fit
heldout_alpha_coverage_050
heldout_xmap_occ and xmap entropy
heldout_projection_coverage_budget
wall_clock_min, render_forward_ms, render_backward_ms when available
heldout_psnr_per_min as a cost sanity check
```

Diagnostic expectations:

```text
source-view PSNR must not select the winner
coverage explosions must be treated as suspicious even when heldout PSNR rises
mass-normalized line incidence should preserve cleaner geometry than peak mode
the pullback row should improve held-out metrics without requiring much worse
runtime than the current rank_adaptive_metric exact-incidence row
```

## Kill Criteria

Kill or pause Pullback-Metric Incidence Elements if any of these hold after the
80-step 3-camera run and one small calibration sweep:

```text
heldout_eval_psnr is not better than screen_disk / projected_conic by at least
0.3 dB in the same budget regime.

heldout_eval_l1 does not improve when heldout_eval_psnr improves, suggesting a
coverage/blur artifact rather than better geometry.

heldout_projection_coverage_budget or alpha coverage jumps like the earlier
ray_gaussian_line_peak row while xmap occupancy collapses.

runtime is more than 3x rank_adaptive_metric / ray_gaussian_line_mass without a
clear held-out-quality gain.

the representation only wins source-view metrics.

the local transport Jacobian is too ill-conditioned for stable pullback
semantics, and fixing that requires a new trainer architecture rather than a
bounded metric regularizer.
```

If killed, keep the note as negative evidence and return to the current MVP
pressure: projected controls, clean mass-normalized incidence, better culling or
fused rasterization, and held-out-camera selection.

## Implementation Pass

Implemented the first fast row as:

```text
model.support_mode = derived_support_metric
render.incidence_mode = projected_conic
```

The support tensor is derived in `MaterialSurfelField.derived_support_covariance`
from the fixed canonical KNN graph and the current transported positions. The
knobs are deliberately global/config-level:

```text
model.derived_support_scale
model.derived_support_floor
model.derived_support_weight_tau
model.derived_support_normalize_trace
```

No per-element covariance is learned in this mode. The existing
`ray_gaussian_line_mass` path can use the same world covariance through
`world_support_covariance_for_incidence`, but the first benchmark row used the
fast projected-conic approximation.

Added diagnostics:

```text
support_* spectrum ratios and phase-like scores
witness_* Plucker/angular witness confidence, no-grad
heldout_witness_* for held-out camera renders
```

Validation:

```text
uv run python -m py_compile research_experiments/gauge_fields/train.py ...
uv run --with pytest python -m pytest tests/test_gauge_incidence.py
uv run python research_experiments/gauge_fields/train.py ...derived_support...smoke... --steps 1
```

First 80-step DeepView train-2/test-1 result:

```text
free_dynamic_3dgs                         heldout PSNR 13.2940, wall 3.07 min
derived_support_metric + pair-X           heldout PSNR  8.8518, wall 2.08 min
rank_adaptive_metric + pair-X             heldout PSNR  8.1434, wall 16.19 min
rank_adaptive_metric naked                heldout PSNR  7.7890, wall 7.98 min
screen_disk naked                         heldout PSNR  7.4607, wall 3.44 min
```

Interpretation: derived support is the best gauge row so far and materially
faster than the learned rank-adaptive pair-X row, but it still does not approach
the free dynamic 3DGS baseline. This makes it worth one calibration pass, not a
full new renderer yet.
