# V-JEPA F32 Multicam Heldout Followups

Context: run `iom0ibz8` established the first stable V-JEPA + static/dynamic
split + F32 feature-splatting baseline on the DeepView 3-cam train2/test1
probe. It trains cleanly on `camera_0001` and `camera_0015`, but held-out
`camera_0040` is weak and degrades over 1000 steps.

## Current Result

- Config: `src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha_lr3e4_camclamp.jsonc`
- W&B: https://wandb.ai/nbardy/dynaworld/runs/iom0ibz8
- Train cameras: `camera_0001`, `camera_0015`
- Heldout camera: `camera_0040`
- Final train PSNR: `24.0275`, `24.4131`
- Final heldout PSNR / SSIM: `8.6923` / `0.0711`

## Followups To Implement

1. **Separate real generalization failure from low-overlap camera choice.**
   The held-out camera appears to have limited overlap with the two train
   cameras. Before treating the run as an architecture failure, add or choose a
   3-camera sample where the heldout camera has high visual overlap with the
   training pair.

2. **Log all train and heldout views explicitly.**
   The run should make it obvious which media belongs to `TrainView0`,
   `TrainView1`, and `Heldout0_camera_0040`. If W&B already receives some of
   these videos, fix the names/panels so they are easy to find from the run
   page.

3. **Add a single comparison panel for multicam feature-splatting runs.**
   Desired layout:

   | View | GT | Splat / Pred | Alpha | Feature PCA |
   |---|---|---|---|---|
   | Train view 0 | frame/video | frame/video | frame/video | frame/video |
   | Train view 1 | frame/video | frame/video | frame/video | frame/video |
   | Heldout view | frame/video | frame/video | frame/video | frame/video |

   This is the most important diagnostic panel for the next runs: it should
   show whether the model is fitting only the source cameras, whether alpha is
   leaving holes, and whether the feature field carries coherent structure in
   the held-out view.

4. **Preserve checkpoint or early-stop evidence.**
   In this run, heldout was better at the 120-step gate and earlier validation
   than at final step 1000. Next runs should record the best heldout checkpoint,
   not only the final checkpoint.

5. **Add overlap-aware notes to `BASELINES.md`.**
   Tier 2a rows should say whether the camera split is high-overlap or
   low-overlap once that is known. Heldout PSNR is still the selector, but the
   overlap caveat matters for interpreting failures.

## Implementation Status

- Done: validation now logs explicit media for every train view and every
  heldout view (`TrainView0`, `TrainView1`, `Heldout0_camera_0040`, etc.).
- Done: validation now logs one stacked multicam diagnostic video:
  `Multicam_GT_Splat_Alpha_Feature_Grid_Video`, with rows per view and columns
  `GT | Splat/Pred | Alpha | Feature PCA`.
- Done: multicam validation now tracks `Heldout/Eval/PSNRMean`,
  `Heldout/Eval/SSIMMean`, `Heldout/BestEvalPSNR`,
  `Heldout/BestEvalSSIMAtBestPSNR`, and `Heldout/BestEvalStep`.
- Done: selected a higher-overlap DeepView `03_Dog` split. Use train
  `camera_0006` + `camera_0014`, heldout `camera_0005`, anchor/condition
  `camera_0006`. See
  `agent_notes/loose_notes/2026-05-05_14-02-06_deepview_03_dog_overlap_camera_selection.md`.
- Still open: run the higher-overlap good-set configs to completion and record
  best/final heldout metrics.
- Still open: add overlap classification and best-heldout metrics to
  `BASELINES.md` after the next real baseline run.
- Still open: checkpoint artifact preservation is not implemented yet; the
  current fix preserves best-heldout evidence as W&B scalars at validation-video
  cadence only.

## Definition Of Done

- W&B has one obvious 3-row x 4-column panel:
  `GT | Splat/Pred | Alpha | Feature PCA`.
- Rows are `TrainView0`, `TrainView1`, and `Heldout0_camera_0040`.
- The next baseline uses a higher-overlap heldout camera or explicitly logs
  the overlap limitation.
- `BASELINES.md` records both final heldout and best-heldout checkpoint if they
  differ materially.
