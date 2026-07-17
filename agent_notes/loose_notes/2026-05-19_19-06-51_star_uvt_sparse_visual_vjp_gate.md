# STAR UVT sparse visual VJP gate

Date: 2026-05-19

## Goal

Move beyond colorizer-only rendered-feature probes. The stratified64
colorizer-only gate ruled out target-grid sampling bias, but it still froze the
STAR model. This gate tests the missing bridge: sparse RGB/probe loss should
produce gradients for rendered sparse feature/alpha values and then use the
native sparse-pixel backward path to update STAR tube parameters.

## Code change

Extended `src/train/train_star_uvt_rendered_feature_rgb_probe.py` with opt-in
native visual VJP mode:

- `probe.train_star_model=true`
- `probe.train_colorizer=false`
- `probe.colorizer_init_checkpoint=<checkpoint>`

The old colorizer-only probe behavior remains the default. In model-training
mode, each chunk:

1. renders sparse feature/alpha values with
   `render_uvt_feature_sparse_pixels_with_bins`
2. computes sparse RGB MSE through `FeatureToColor`
3. uses local autograd only to get gradients with respect to sparse
   `feature_values` and `alpha_values`
4. calls `direct_atomic_feature_sparse_pixels_backward_cached_bins`
5. applies returned gradients to STAR tensors with `torch.autograd.backward`

Added test coverage for the local sparse RGB loss/gradient helper.

## Command

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe_lr001_50step_media.jsonc
```

W&B offline run:

```text
wandb/offline-run-20260519_190651-gv6cmc2i
```

## Result

The gate passes as native sparse VJP plumbing and speed evidence:

- `model_grad_seen=true`
- `colorizer_grad_seen=false`
- sparse sample loss: `0.271902 -> 0.264276`
- sparse sample PSNR: `5.656 -> 5.779`
- final full-video loss: `0.266723`
- final full-video PSNR: `5.739`
- mean step/render/backward: `336.78 / 86.74 / 150.78 ms`
- mean local/native backward: `55.93 / 94.85 ms`
- media render: `1069.69 ms`

Artifacts:

- JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe_lr001_50step_media.json`
- report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe.md`
- checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe_lr001_50step.pt`
- contact sheet:
  `outputs/media/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe_lr001_50step_contact.jpg`
- side-by-side MP4:
  `outputs/media/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe_lr001_50step_sbs.mp4`

## Read

This proves the missing native sparse visual VJP bridge exists in the trainer
surface: sparse rendered RGB loss can update STAR parameters without dense
autograd RGB. It does not solve quality. The frozen target-grid colorizer gives
worse dense full-video PSNR (`5.739`) than the colorizer-only stratified gate
(`6.132`), and media stays sparse/streaked.

The next gate should not be another frozen-colorizer-only variant. It should
either:

- train STAR and colorizer jointly on sparse visual pixels, or
- combine native sparse visual VJP with the target-grid feature/probe objective
  so the selected V-JEPA feature basis is not destroyed while chasing RGB.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py src/train/train.py
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_rgb_probe.py -q
```

Observed:

```text
7 passed
```
