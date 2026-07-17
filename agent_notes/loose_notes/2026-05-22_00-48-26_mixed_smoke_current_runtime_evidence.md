# Mixed Smoke Current Runtime Evidence

## Context

After several code-organization cleanup slices, I reran the checked-in mixed
same-view plus heldout-view smoke to get current runtime evidence for the shared
interfaces. This was not a benchmark or quality run; the purpose was to prove
the current tree still executes through the unified registry, mixed scheduler,
objective/loss, W&B logging, and multicam validation media paths.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline WANDB_SILENT=true .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

## Result

Passed on MPS.

Offline W&B run:

```text
wandb/offline-run-20260522_004727-ka4lm8g5
```

The launch path was:

```text
src/train/train.py -> trainer_registry -> MixedSameHeldoutPrecomputedFeatureTrainer
```

The visible run hit cached RGB-pyramid features, alternated same-view and
heldout-view steps, and finished with multicam eval metrics:

```text
TrainView0/Eval/PSNR = 3.9744
TrainView1/Eval/PSNR = 5.4962
Heldout0_camera_0040/Eval/PSNR = 3.7891
```

## Evidence Checked

`strings` over the offline `.wandb` record found:

- `Loss/same_view_recon`
- `Loss/heldout_view_recon`
- `TrainView0/Eval/PSNR`
- `TrainView1/Eval/PSNR`
- `Heldout0_camera_0040/Eval/PSNR`
- `Heldout/Eval/PSNRMean`
- `Render_GT_vs_Pred`
- `TrainView0_Rendered_Video`
- `TrainView0_GT_Video`
- `TrainView1_Rendered_Video`
- `TrainView1_GT_Video`
- `Heldout0_camera_0040_Rendered_Video`
- `Heldout0_camera_0040_GT_Video`

Media files exist under:

```text
wandb/offline-run-20260522_004727-ka4lm8g5/files/media/
```

`file` identified the final preview as a PNG and all six train/heldout media
artifacts as MP4 files.

## Interpretation

This proves the current mixed trainer bridge still works as an interface smoke:
registry dispatch, lazy same-view manifest sampling, multicam heldout sampling,
separate mixed loss names, W&B scalar logging, final preview, and validation
videos all execute in the current tree.

It does not prove training quality, a baseline win, large-dataset stability, or
that STAR UVT/dynamic gsplat objectives are solved. Do not add this to
`BASELINES.md`.
