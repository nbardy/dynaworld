# STAR UVT hidden32 manual VJP gate

## Goal

Continue the STAR UVT fast feature-shader plan by testing whether a smaller
hidden decoder is a useful middle path between the expensive hidden64
full-cell8 visual VJP and the affordable but weak manual-linear route.

The original thread goal is broader: benchmark all STAR UVT and dynamic
gsplat renderers at matched frame/resolution/splat counts, identify the real
forward/backward bottlenecks, record the fast-shader lessons, port those
lessons into STAR UVT, build a fast single-video overfit route, and only then
scale to the 300-video prepared set.

## Changes

- Added generic sparse visual loss VJP aliases:
  `manual_hidden`, `manual_hidden_fastgelu`,
  `manual_hidden_star_only`, and `manual_hidden_star_only_fastgelu`.
  The previous `manual_hidden64*` names remain valid.
- Updated the sparse visual loss profile script to accept the generic
  `manual_hidden*` names.
- Updated the focused sparse visual VJP parity test to use `manual_hidden`.
- Added a hidden32 target-grid RGB-probe config.
- Added a hidden32 full-cell8 target-area sparse visual trainer config.

## Runs

Hidden32 RGB probe:

```bash
PYTHONPATH=src/train rtk .venv/bin/python src/train/train_star_uvt_feature_rgb_probe.py \
  src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden32_lr01_1000step.jsonc
```

- W&B offline run: `nub8ga5n`
- JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden32_lr01_1000step.json`
- Checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden32_lr01_1000step.pt`
- Result: pass; grid loss `0.048086 -> 0.007046`, grid PSNR
  `13.180 -> 21.520`, full upsample PSNR `19.704`,
  `2.288ms/step`, `0.919ms` backward.

Hidden32 subphase profile:

```bash
PYTHONPATH=src/train rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_hidden32_from1500_lr001_5step_media.jsonc \
  --repeat 1 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manual_hidden32_subphase_profile_fullstep
```

- JSON/MD:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manual_hidden32_subphase_profile_fullstep.{json,md}`
- Full-step extrapolated sparse render `1580.1ms`, loss VJP
  `3043.6ms`, pixel-id build `196.6ms`.
- Largest loss-side phases: GELU backward `725.6ms`, fc1 `666.5ms`,
  fc2 `386.2ms`, conv1 param grad `310.6ms`, conv1 feature grad
  `257.5ms`, target-area loss grad pred `243.1ms`.

Hidden32 full-cell8 trainer:

```bash
PYTHONPATH=src/train rtk .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_hidden32_from1500_lr001_5step_media.jsonc
```

- W&B offline run: `6urttb1z`
- JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500_lr001_5step_media.json`
- Checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500_lr001_5step.pt`
- Result: mechanics pass, not quality promotion.
- Weighted loss `1.143623 -> 1.140212`.
- Feature target loss slightly worsens `0.625418 -> 0.625438`.
- Sparse visual PSNR `5.728 -> 5.782`.
- Probe PSNR `22.028 -> 22.029`.
- Dense full RGB PSNR `5.678`.
- Mean step/backward/render `4298.4/3210.5/108.2ms`.
- Sparse visual render/loss/backward `575.3/2136.1/961.4ms`.

## Read

Hidden32 keeps most of the hidden64 target-grid RGB-probe oracle quality
(`19.704` full upsample PSNR vs hidden64 `20.073`) while being a faster
standalone probe. It is much higher capacity than the linear decoder
(`16.980`), but the full-cell8 trainer still spends seconds in Python/Torch
hidden-layer visual VJP and remains visually poor.

So the answer is not "use hidden32 as the route." It is useful evidence that
decoder capacity can be reduced somewhat, but dense Python-side hidden VJP is
still the wrong execution boundary. The next implementation should fuse the
RGB/loss/gradient handoff or visibility/prefix tape natively, while keeping the
current sparse-forward target/probe VJP as the speed comparison.

## Validation

- Focused parity/unit gate passed after the alias/config change:
  `32 passed in 1.40s`.
- Hidden32 probe, profile, and trainer all wrote JSON artifacts.
- Follow-up doc sync and comparison regeneration are tracked in the current
  active goal; this note records the run-level chronology.
