# Coffee Martini train2/holdout1 protocol

## Objective

Move the paper runner beyond the single-video capacity smoke and prove the
actual multicamera contract on Neural3D `coffee_martini`: train on two cameras,
never train on one heldout camera, and report the two metric surfaces
separately.

## Split and calibration

- Manifest sample: `neural3d_coffee_martini_train_cam04_cam09_holdout_cam06`.
- Train cameras: `cam04`, `cam09`.
- Heldout camera: `cam06`.
- Camera source: `poses_bounds.npy`, reported as
  `neural_3d_llff_relative_pinhole`.
- The STAR camera projection audit had been hard-coded to DeepView and crashed
  on Neural3D. It now dispatches Neural3D records through
  `neural_3d_camera_from_poses_bounds`; the focused audit/data tests pass.

## Executed evidence

- Camera audit at 64px and 128px:
  `outputs/benchmarks/2026-07-11_coffee_martini_train2_holdout1_camera_projection_audit.json`.
- World Tubes plus dynamic 3DGS Metal smoke: 64px, 4 frames, 2 optimizer
  steps, seed 17. World Tubes heldout PSNR `6.0412`; dynamic 3DGS heldout PSNR
  `4.9639`. This proves routing and metrics, not quality or ranking.
- Paper-clean WorldFoam rerun: 128px, 16 frames, 40 steps, seed 17, initialized
  from the 89-point train-camera-only ORB reconstruction. Final/best heldout
  PSNR `5.6311`, L1 `0.475115`, SSIM `0.000289`. The low support confirms the
  old clean-init negative result.
- Protocol report:
  `outputs/benchmarks/2026-07-11_coffee_martini_train2_holdout1_protocol/summary.json`.
  Its split, calibration, and separate-metric gates are green. It correctly
  records the initial route smoke as non-rankable.
- A matched seed-17 pilot then ran all three representations at
  128px/16f/40 steps/1024 primitives. World Tubes train/heldout PSNR is
  `11.1661/6.4071`; dynamic 3DGS is `7.2672/4.9550`; clean WorldFoam is
  `5.6487/5.6311`. The regenerated protocol report now records
  `matched_for_ranking=true`, but correctly leaves `paper_rankable=false`
  because only one seed exists, the comparison runner lacks W&B backing, and
  World Tubes used the non-promotable `fast_exploration` direct-atomic policy.

## Next matched run

Run all ranked lanes at 128px/16f on seeds 17, 29, and 43. Declare one matched
budget policy per table, select by heldout PSNR, retain train PSNR/SSIM/L1 and
heldout PSNR/SSIM/L1, and forbid heldout-camera frames or external pretrained
scene geometry from initialization. The EX4DGS-initialized WorldFoam row may
remain as a clearly labeled oracle/external-init upper control, not a ranked
paper-clean row.
