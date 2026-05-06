# PowerFoam Metal Geometry-Only Motion Smoke

## Goal

Close the minimal P0.3 proof gap: show dynamic PowerFoam geometry can train on
Metal and produce motion/alpha/support changes without hiding behind feature
repainting.

This is not a nearest0040/multicam heldout result. The current
`train_dynamic_powerfoam_metal.py` path loads a plain video via
`load_video_sequence`, so the honest P0.3 row is explicit-video only until the
dynamic trainer gets a multicam/camera contract.

## What Changed

- Added a geometry-only config:

```text
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke.jsonc
```

- Updated `src/train/train_dynamic_powerfoam_metal.py` to write durable local
  metrics:
  - `train_metrics_history.jsonl`
  - `dynamic_geometry_summary.json`
  - final render and side-by-side MP4s

- Added a verifier:

```text
research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py
```

- Strengthened `tests/test_dynamic_powerfoam_metal.py`:
  - MPS alpha-change proof now uses `pytest.skip` instead of silent return.
  - Added a render-alpha loss gradient check into `raw_xy_coeff` and
    `raw_radii_coeff` with feature parameters frozen.
  - Added a CPU-only verifier contract test.

## Commands

Test gate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_dynamic_powerfoam_metal.py -q -rs
```

Result: `15 passed`.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  WANDB_MODE=disabled .venv/bin/python -u src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke.jsonc
```

Verify:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py \
  outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke \
  --require-geometry-motion \
  --require-alpha-support-motion \
  --require-appearance-freeze-control
```

Result: `ok: true`.

## Artifact

```text
outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke/dynamic_geometry_summary.json
```

Final summary:

```text
final eval_l1 0.06529680639505386
final eval_mse 0.011330553330481052
eval_alpha_mean 0.9878520965576172
state_mean_center_delta 0.08874156326055527
state_mean_radius_delta 0.0425923727452755
state_mean_temporal_screen_delta_px 0.44571685791015625
state_p95_temporal_screen_delta_px 1.0568746328353882
eval_mean_temporal_alpha_delta 0.008722909726202488
eval_mean_temporal_support_delta 0.0075358073227107525
state_mean_temporal_feature_abs_delta 0.0
```

The final feature temporal delta is exactly zero because the config disables
`dynamic_features`; the alpha/support motion therefore comes from centers/radii
and the resulting support changes, not from time-conditioned appearance.

## Remaining Gap

This proves the representation mechanics and a small Metal training artifact.
It does not prove nearest0040 heldout quality, because the dynamic trainer still
has no multicam train/heldout loader or camera model path. The next honest
quality step is a dynamic Metal multicam contract, not another explicit-video
claim.
