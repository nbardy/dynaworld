# STAR UVT stratified rendered-feature probe

Date: 2026-05-19

## Goal

Follow the rendered-feature sparse-pixel RGB probe with a denser
full-resolution sampling gate. The prior probe trained a colorizer on actual
rendered sparse 1500 features, but it used the target-grid sparse stencil. This
gate asks whether the poor dense media was just a sampling/lattice artifact.

## Code change

Added `probe.pixel_source="stratified_grid"` to
`src/train/train_star_uvt_rendered_feature_rgb_probe.py`.

The new pixel source draws deterministic full-resolution stratified pixel ids
per chunk. For the checked-in config, it samples `64x64` pixels on every one of
the 64 frames, or `262,144` pixels/step (`1.5625%` dense). That is `4x` the
previous rendered-feature sparse-pixel gate (`65,536` pixels/step).

Added coverage in `tests/test_star_uvt_feature_rgb_probe.py` for chunk-local
stratified ids. The first runtime attempt caught a real contract bug: the
native sparse renderer requires `int32` pixel ids, while the helper initially
returned `int64`. The helper now returns `int32`.

## Command

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_rendered_feature_rgb_probe_from1500_stratified64_hidden64_lr01_100step_media.jsonc
```

W&B offline run:

```text
wandb/offline-run-20260519_185704-brrsfsdm
```

## Result

The gate passes as a sampled-loss diagnostic, but it is negative for visual
quality:

- sparse sample loss: `0.277860 -> 0.242981`
- sparse sample PSNR: `5.562 -> 6.144`
- final full-video loss: `0.243682`
- final full-video PSNR: `6.132`
- mean step/render/backward: `331.52 / 102.92 / 36.02 ms`
- last step/render/backward: `306.67 / 98.25 / 34.72 ms`
- media render: `1322.55 ms`

Artifacts:

- JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64_hidden64_lr01_100step_media.json`
- report:
  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64.md`
- checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64_hidden64_lr01_100step.pt`
- contact sheet:
  `outputs/media/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64_hidden64_lr01_100step_contact.jpg`
- side-by-side MP4:
  `outputs/media/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64_hidden64_lr01_100step_sbs.mp4`

## Read

The result rules out the easy sampling explanation. A `4x` denser
full-resolution stratified probe still lands at only `6.132` full-video PSNR,
barely above the previous target-grid sparse-pixel rendered probe (`6.096`).
The dense media remains sparse/streaked rather than reconstructing coherent
target video.

The next gate should stop training only a downstream colorizer on frozen
rendered features. Visual/probe loss needs to move back into STAR feature
optimization through a sparse/native VJP path.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py src/train/train.py
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_rgb_probe.py -q
```

Observed:

```text
6 passed
```
