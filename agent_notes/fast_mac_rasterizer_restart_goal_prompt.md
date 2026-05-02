# Goal Prompt: Fast-mac Rasterizer High-Resolution Audit And Trainer Swap

We are restarting context. Work in:

```text
/Users/nicholasbardy/git/gsplats_browser/dynaworld
```

Read `AGENTS.md`, `BASELINES.md`, and these notes first:

- `agent_notes/loose_notes/2026-05-02_18-58-58_fast_mac_render_phase_precision_probe.md`
- `agent_notes/key_learnings.md` near the fast-mac bullets
- `src/train/renderers/fast_mac.py`
- `third_party/fast-mac-gsplat/variants/*/benchmarks/`
- `third_party/fast-mac-gsplat/variants/*/tests/`

## Objective

Audit all train-capable fast-mac rasterizer variants at high resolution for
speed and correctness, specifically to verify whether the current v5 backward
path is blowing up more than it should. Then wire the best correct variant into
the Dynaworld trainer behind a config knob and rerun the TokenGS/free-splats
baseline speed checks.

The user's remembered target is important: backward used to be roughly `2x`
forward, with about `30ms`-class backward at high point counts (`~64k`) and
`2k-4k` resolution. Do not assume this memory is wrong until the current
variant matrix proves it.

## Current Facts To Preserve

- The trainer currently uses `src/train/renderers/fast_mac.py`.
- RGB `F == 3` dispatches to
  `third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5`.
- Feature splatting `F != 3` dispatches to
  `third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features`.
- The recent free-splats and unconditioned-token high-res benchmarks were RGB
  `feature_dim=3`, so they used v5, not v5_features and not v6/v8/v9.
- Recent 4k/1f/8192 v5 free-splats phase probe:
  - projected raster forward median: `~59.5ms`
  - projected raster backward median: `~180.7ms`
  - projected raster backward mean: `~483ms` due high variance
  - full model backward median: `~836.9ms`
  - pixel reconstruction loss median: `~701.1ms`
- Full-step 4k trainer numbers include frame resize/sample, render, image loss,
  backward through projection/model, and optimizer. Do not confuse those with
  render-only or raster-only speed.

## Required Benchmark Matrix

First inventory which variants are built and train-capable. At minimum inspect
and, if needed, build/test these variant families:

- `v5`
- `v5_features`
- `v6`, `v6_upgrade`, `v6_refined`
- `v8`, `v8_hw_eval`, `v8_hw_train`, `v8_project3d`
- `v9_project3d_train`
- `v9_hw_tile_exact_probe` if it exposes a comparable train/backward path

For each train-capable variant, run a fixed synthetic/projected-input matrix:

- resolutions: `512`, `2048`, `4096`
- splat counts: `8192`, `65536`
- batch/frame counts: `1` first; add `16` only after single-frame matrix is clean
- RGB `F=3` first; feature `F=32` only after RGB is understood
- measure:
  - forward eval ms
  - forward with saved state ms
  - backward-only ms
  - forward+backward ms
  - backward/forward ratio
  - p50/p95/min/max over enough warm iterations
  - overflow fallback count or equivalent
  - max/mean image error vs reference
  - max/mean/relative gradient error vs reference for means, conics, colors, opacities

Use current variant benchmark harnesses where possible. Do not invent a new
one-off unless the existing harness cannot compare variants correctly. If you
must add a harness, save it under `research_experiments/vjepa_performance/` or
`third_party/fast-mac-gsplat/variants/.../benchmarks/` and record exact commands.

## Accuracy Gate

No variant is eligible for trainer integration unless it passes parity:

- image error is within the established tolerance for that variant family
- gradients match the reference closely enough for training
- no hidden dtype/config mismatch
- no silent CPU fallback
- no accidental use of a stale `.so`

For each variant, record whether the result is:

- `pass`
- `speed-only but accuracy failed`
- `accuracy-only but slow`
- `build/import failed`
- `not train-capable`

## Trainer Integration

After the matrix identifies the best correct train variant:

1. Add a config-controlled fast-mac variant selector, likely under
   `render.fast_mac.variant`, without breaking the default v5 path.
2. Keep `F=3` and `F!=3` dispatch explicit.
3. Run trainer smoke after wiring:
   - 1-step F=3 smoke
   - F=32/feature-splat smoke if the chosen variant supports feature channels
4. Rerun baseline throughput:
   - free_splats single-frame `128,512,2048,4096`, `8192 splats`
   - unconditioned_tokens single-frame `128,512,2048,4096`, `8192 splats`
   - 16-frame `128` and `512` for both variants
5. If quality can change due numerical differences, run a short quality smoke
   and record loss/PSNR/SSIM deltas.

## Deliverables

- A durable note under `agent_notes/loose_notes/{timestamp}_fast_mac_variant_audit.md`
  with commands, matrix tables, and interpretation.
- Update `agent_notes/key_learnings.md` only for surprising lessons.
- Update `BASELINES.md` if a trainer baseline is rerun.
- Save machine-readable benchmark artifacts under `outputs/benchmarks/`.
- Commit the scoped work. Do not bundle unrelated dirty-tree work.

## Starting Commands

Check current dispatch:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python - <<'PY'
from config_utils import load_config_file
from train_video_token_implicit_dynamic import trainer_class_for_config
from renderers.fast_mac import _ensure_fast_mac_v5_on_path

cfg = load_config_file("src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc")
cfg["model"]["train_frame_count"] = 1
cfg["train"]["steps"] = 1
cfg["logging"]["always_log_last_step"] = False
trainer = trainer_class_for_config(cfg)(cfg)
sequence_data, clip_frames, clip_times = trainer.sample_clip()
decoded = trainer.forward_clip(trainer.model_input_for_clip(sequence_data, clip_frames, clip_times), clip_times)
print("variant", trainer.model_cfg["variant"], "renderer", trainer.renderer_mode, "feature_dim", decoded.rgbs.shape[-1])
_ensure_fast_mac_v5_on_path()
import torch_gsplat_bridge_v5
print(torch_gsplat_bridge_v5.__file__)
PY
```

Build a variant from the dynaworld root without tripping uv's pyproject walk:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/<variant>
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```
