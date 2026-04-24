# Implicit-camera + 100-clip baseline pivot

## Context

We decided the known-camera overfitting script has done its job: the 65k
fast-mac v5 run can overfit the local 128px/4fps clip well enough to serve as a
mechanical target. The next baseline needs two things:

- a local 100-clip dataset path
- implicit-camera training that logs comparable convergence metrics

## Changes made

- Added `src/train/build_clip_dataset.py`.
  - Scans source videos/directories.
  - Writes loader-compatible clip folders:
    `clips/clip_XXXXXX/frames/*.png` plus `summary.json`.
  - Writes `manifest.jsonl` and `dataset.json`.
  - Default target is 100 clips, 46 frames, 4fps, 128px.
- Added `src/train_scripts/build_100_clip_dataset.sh`.
  - Usage: `./src/train_scripts/build_100_clip_dataset.sh <source-video-or-directory>`.
- Added explicit `opencv-python` to `pyproject.toml`/`uv.lock`.
  - The implicit video smoke was failing because `cv2` was not declared.
- Updated implicit-camera trainers to log comparable eval scalars:
  - `Eval/Loss`
  - `Eval/L1`
  - `Eval/MSE`
  - `Eval/SSIM`
  - `Eval/DSSIM`
  - `Eval/PSNR`
- Updated video-token implicit reconstruction to use `render_gaussian_frames` for
  temporal chunks.
  - `recon_backward_strategy=batched` now actually uses the native batch render
    path instead of looping one frame at a time.
  - This lets fast-mac v5 rasterize the 16-frame clip batch in one chunk.
- Added first 128px/4fps implicit fast-mac config:
  - `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`
  - 64 tokens x 128 gaussians/token = 8192 splats
  - `renderer=fast_mac`
  - `recon_backward_strategy=batched`
  - `losses.type=standard_gs`
- Added `src/train_scripts/train_implicit_camera_128_4fps_fast_mac_baseline.sh`.

## Verification

- `uv run python -m py_compile src/train/build_clip_dataset.py src/train/train_video_token_implicit_dynamic.py src/train/train_camera_implicit_dynamic.py`
  passed.
- Dataset builder dry run:
  - `test_data/test_video_small_128_4fps.mp4`
  - 1 clip, 16 frames, 4fps, 32px
  - planned successfully.
- Dataset builder write/load probe:
  - wrote `/tmp/dynaworld_clip_dataset_probe`
  - reloaded with `load_uncalibrated_sequence(..., frame_source="summary_sampled")`
  - got `torch.Size([16, 3, 32, 32])`, `video_fps=4.0`, `frame_source=summary_sampled`
- Existing video-token implicit smoke now runs:
  - offline W&B `offline-run-20260422_151711-6lf9gq3z`
  - 10 steps, dense 64-splat smoke
  - final sampled recon around `0.1834`
  - final full-sequence `Eval/L1 0.18659`
- New 128px/4fps fast-mac implicit 2-step probe:
  - offline W&B `offline-run-20260422_151956-1rpepa8c`
  - confirmed same 46-frame clip loads
  - confirmed fast-mac renderer with `recon_backward_strategy=batched`
  - sampled recon dropped `0.4432 -> 0.4224`
- New 128px/4fps fast-mac implicit 100-step probe:
  - W&B `z60o3ngd`
  - URL: https://wandb.ai/nbardy/dynaworld/runs/z60o3ngd
  - runtime about 6m19s
  - final sampled `Loss/Reconstruction 0.13927`
  - full-sequence eval:
    - `Eval/Loss 0.15522`
    - `Eval/L1 0.10336`
    - `Eval/MSE 0.02395`
    - `Eval/SSIM 0.27461`
    - `Eval/PSNR 16.21`
    - camera eval stayed near base: FOV `59.94`, radius `3.02`

## Interpretation

The implicit-camera fast-mac path is mechanically good and learns on the same
128px/4fps clip. It does not yet match the known-camera 65k run:

- known-camera 65k run `3piy8cww`: `Eval/L1 0.07233`, `Eval/Loss 0.10858`,
  `Eval/SSIM 0.49289`, `Eval/PSNR 18.93`
- implicit 8k run `z60o3ngd`: `Eval/L1 0.10336`, `Eval/Loss 0.15522`,
  `Eval/SSIM 0.27461`, `Eval/PSNR 16.21`

That gap is not surprising because the new implicit config is 8k splats, not
65k, and it has to learn cameras as well. The useful next step is a
capacity-matched implicit config or a better gaussian init before treating it as
the stronger baseline.
