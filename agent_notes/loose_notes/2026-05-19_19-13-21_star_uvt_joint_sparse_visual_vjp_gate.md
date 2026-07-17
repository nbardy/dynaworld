# STAR UVT Joint Sparse Visual VJP Gate

Date: 2026-05-19

## Question

The frozen sparse visual VJP gate proved native sparse RGB loss can update STAR
features, but it was quality-negative with the target-grid colorizer frozen. The
follow-up question was whether training STAR and the hidden64 colorizer together
on the same full-resolution sparse visual pixels recovers quality without
falling back to full dense autograd.

## Setup

- Config:
  `src/train_configs/star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe_lr001_50step_media.jsonc`
- Command:
  `PYTHONPATH=src/train .venv/bin/python src/train/train.py src/train_configs/star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe_lr001_50step_media.jsonc`
- Source STAR checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`
- Initial colorizer checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`
- Pixel source: deterministic `64x64` stratified full-resolution lattice on
  every frame, `262,144` pixels per step (`1.5625%` dense)
- Optimization: train STAR model and hidden64 `FeatureToColor`, `lr=0.001`,
  50 steps
- W&B: `wandb/offline-run-20260519_191321-kzhn8jjt`

The trainer uses local autograd only for the sparse RGB/colorizer part. It gets
gradients for sparse `feature_values` and `alpha_values`, then pushes those into
STAR with `direct_atomic_feature_sparse_pixels_backward_cached_bins`. In this
joint gate, local autograd also accumulates the colorizer gradients.

## Result

- Pass: `true`
- STAR gradient seen: `true`
- Colorizer gradient seen: `true`
- Colorizer init loaded: `true`
- Sparse sample loss: `0.271902 -> 0.247613`
- Sparse sample PSNR: `5.656 -> 6.062`
- Dense full-video loss: `0.249773`
- Dense full-video PSNR: `6.025`
- Mean step/render/backward: `729.45 / 135.56 / 365.95 ms`
- Mean local/native backward: `170.56 / 195.39 ms`
- Last step/render/backward: `537.98 / 105.22 / 277.15 ms`
- Dense media render: `1662.06 ms`

Artifacts:

- JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe_lr001_50step_media.json`
- Report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe.md`
- Checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe_lr001_50step.pt`
- Contact sheet:
  `outputs/media/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe_lr001_50step_contact.jpg`
- Side-by-side:
  `outputs/media/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe_lr001_50step_sbs.mp4`

## Read

Joint training is a partial recovery, not a quality bridge. It improves over the
frozen sparse visual VJP gate (`5.739 -> 6.025` full-video PSNR) and proves both
gradient paths are live. It still trails the colorizer-only stratified rendered
feature diagnostic (`6.132` full-video PSNR), costs much more
(`729.45ms/step` versus `331.52ms/step`), and the media remains sparse/streaked.

The current conclusion is that sparse RGB-only supervision is too weak or is
pulling on the wrong basis. The next gate should combine native sparse visual
VJP with the target-grid feature/probe objective instead of choosing one of
those losses alone.

## Validation

- Joint gate completed successfully and wrote JSON/checkpoint/media artifacts.
- `PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_star_uvt_rendered_feature_rgb_probe.py src/train/train.py research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py src/train/train_star_uvt_feature_overfit.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_rgb_probe.py tests/test_star_uvt_feature_target_adapter.py -q`
  - `20 passed`
- JSON/report invariants pass: joint gate has STAR+colorizer gradients, uses
  `stratified_grid`, samples `262,144` pixels/step, and the comparison report
  has `37` rows with `5.739 < 6.025 < 6.132` full-video PSNR ordering.
