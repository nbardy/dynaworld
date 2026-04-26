# Material Gauge Extended Probes And 250-Step Baseline

## Context

After adding first-pass material-gauge diagnostics, we continued the scientist's
diagnostics-first plan by adding the missing representation-specific probes and
running a longer measured baseline.

This stayed intentionally narrow:

```text
no new loss
no new renderer
no learned camera
no holonomy
no adversarial inner loop
```

## Source Changes

Updated:

```text
research_experiments/gauge_fields/cheat_probe_material_gauge.py
research_experiments/gauge_fields/README.md
```

New probe coverage:

```text
opacity_split_clone
dormant_insert
```

New probe visual outputs:

```text
xmap_depth_alpha.png
flow.png when --include-flow is used
```

## Verification

Syntax and whitespace checks:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  research_experiments/gauge_fields/train.py

git diff --check -- \
  research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  research_experiments/gauge_fields/README.md
```

Extended probe smoke on the 100-step baseline:

```bash
uv run python research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  --checkpoint outputs/gauge_fields/material_surfel_motion_128_16f_2048el_100step/checkpoint.pt \
  --output-dir /tmp/gauge_fields_cheat_probe_extended \
  --device mps \
  --probe all \
  --include-flow \
  --no-video
```

This passed and wrote per-probe previews plus xmap/depth/alpha strips.

## 250-Step Baseline

Command:

```bash
WANDB_SILENT=true uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc \
  --device mps \
  --steps 250 \
  --output-dir outputs/gauge_fields/material_surfel_motion_128_16f_2048el_250step_diag
```

W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/iss68j3y
```

Final metrics:

```text
eval_l1: 0.057761
eval_mse: 0.009165
eval_psnr: 20.3788
alpha_mean: 0.9232
alpha_coverage_005: 0.99995
alpha_coverage_050: 0.98959
alpha_coverage_090: 0.77868
alpha_hole_fraction: 0.0000496
projection_coverage_budget: 2.8599
projection_radius_px_p50: 2.6818
projection_radius_px_p95: 3.1599
motion_delta_mean: 0.0651
motion_delta_p95: 0.1482
motion_delta_max: 0.4586
xmap_occ: 0.2427
xmap_eff_bins: 609.4
```

Compared with the previous 100-step reference:

```text
100 step: eval_l1 0.0759, PSNR 18.03
250 step: eval_l1 0.0578, PSNR 20.38
```

The longer run clearly improves RGB fit. The preview still shows a blurred,
dark material rendering rather than a clean reconstruction, so this remains an
overfit baseline and diagnostic substrate.

## 250-Step Probe Results

Command:

```bash
uv run python research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  --checkpoint outputs/gauge_fields/material_surfel_motion_128_16f_2048el_250step_diag/checkpoint.pt \
  --output-dir outputs/gauge_fields/material_surfel_motion_128_16f_2048el_250step_diag/probes \
  --device mps \
  --probe all \
  --no-video
```

Probe deltas:

```text
basis_scale_gauge:
  delta_render_l1: 0.000000
  delta_target_l1: 0.000000
  delta_xmap_occ: 0.000000

depth_slide:
  delta_render_l1: 0.002456
  delta_target_l1: 0.000198
  delta_xmap_occ: 0.005127
  delta_coverage_budget: -0.008604

dormant_insert:
  delta_render_l1: 0.000141
  delta_target_l1: 0.00000036
  delta_xmap_occ: -0.058594
  delta_coverage_budget: 0.128327

motion_phase_shift:
  delta_render_l1: 0.035993
  delta_target_l1: 0.016106
  delta_xmap_occ: 0.000000
  delta_coverage_budget: 0.000000

opacity_radius_trade:
  delta_render_l1: 0.005335
  delta_target_l1: 0.001068
  delta_xmap_occ: -0.007080
  delta_coverage_budget: 0.316177

opacity_split_clone:
  delta_render_l1: 0.003062
  delta_target_l1: 0.000007
  delta_xmap_occ: -0.002686
  delta_coverage_budget: 1.437207

radius_inflate:
  delta_render_l1: 0.006394
  delta_target_l1: 0.000496
  delta_xmap_occ: -0.000732
  delta_coverage_budget: 0.159096
```

## Interpretation

The baseline is stronger after 250 steps, but the probes show why red-teaming
is necessary.

`basis_scale_gauge` remains exactly render-invariant. That is a harmless
parameterization gauge unless regularizers accidentally depend on the gauge.

`motion_phase_shift` is strongly detected. The learned time coefficients are
not decorative; shifting them harms RGB and raises L1.

`dormant_insert` is almost RGB-invisible but changes xmap occupancy and
coverage. This is the cleanest evidence that xmap/coverage certificates catch
something RGB barely sees.

`opacity_split_clone` is also close to RGB-null while increasing coverage budget
by 1.44. This is the dynamic material-field analogue of opacity splitting in
splat fields.

`radius_inflate` and `opacity_radius_trade` remain near-null enough to be real
cheats. They no longer improve target L1 after 250 steps, but they perturb RGB
very little relative to the structural change.

## Next Work

1. Run the generated capacity/radius/alpha sweep.
2. Add a compact sweep summarizer table from `metrics.json` files.
3. Run the same deterministic probes against a FasterGS4D checkpoint.
4. Add query/omitted-frame evaluation before training on query frames.
5. Add optional flow preview/logging to the main train output, not only probe
   output.

