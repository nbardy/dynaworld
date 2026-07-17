# Coffee Martini matched three-seed table

## Closed protocol

- Dataset/scene: Neural3D `coffee_martini`.
- Train cameras: `cam04`, `cam09`.
- Heldout camera: `cam06`, never used for optimization or initialization.
- Real calibration: `neural_3d_llff_relative_pinhole` from `poses_bounds.npy`.
- Shared budget: 128px, 16 frames, 40 steps, 1024 primitives.
- Seeds: 17, 29, 43.
- Metrics: train and heldout PSNR/SSIM/L1 kept separate.
- W&B: offline because `wandb status` showed no API key; all nine runs have
  local run IDs and encoded train/heldout media.

## Aggregate result

| Representation | Train PSNR | Heldout PSNR | Heldout SSIM | Heldout L1 | Mean train loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| World Tubes | 11.2694 +/- 0.0704 | 6.3863 +/- 0.0154 | 0.033483 | 0.419681 | 562.75 s |
| WorldFoam | 5.6487 +/- 0.0000 | 5.6311 +/- 0.0000 | 0.000289 | 0.475115 | 22.46 s |
| Dynamic 3DGS | 7.2527 +/- 0.0106 | 4.9544 +/- 0.0004 | 0.141884 | 0.521717 | 2.43 s |

World Tubes wins the primary heldout-PSNR selector on this split. Dynamic 3DGS
has higher heldout SSIM but substantially worse PSNR/L1; preserve all metrics
instead of compressing that tension into one adjective. WorldFoam's almost
zero seed variance is a symptom of its 89-point train-only initialization and
near-empty support, not evidence that the representation is solved.

## Engineering findings

- `camera_projection_parity_audit.py` was DeepView-only despite accepting a
  generic multicam config. It now dispatches Neural3D through
  `neural_3d_camera_from_poses_bounds`.
- Relative output directories passed to the STAR submodule resolve under the
  submodule. The sweep runner now canonicalizes its output directory.
- Partial sweep invocations originally overwrote the top-level manifest. The
  runner now merges by seed and can rebuild from `seed_*/run_summary.json`.
- W&B media needed the optional `moviepy` runtime package. The shared W&B
  initializer now accepts `logging.wandb_run_id`, which gives WorldFoam stable
  offline IDs.
- Promotable deterministic-quality World Tubes is dramatically slower than
  direct-atomic and grows more expensive as support expands. Use it as the
  correctness/reference kernel; report direct-atomic separately for throughput.

## Evidence and next work

Canonical report:
`outputs/benchmarks/2026-07-11_coffee_martini_matched_sweep/report/summary.json`.
All eight scoped gates pass. This closes one scene and one camera split. The
next scientific work is additional `coffee_martini` camera triplets, then more
Neural3D scenes, with the same heldout-camera and three-seed contract.
