# STAR UVT Probe-Emphasis 600->800 Continuation

Date: 2026-05-19

## Goal

Continue the STAR UVT target-grid frozen RGB-probe gate past the 600-step
checkpoint, but bias the objective toward visual decodability to test whether
the same-grid oracle gap is just a matter of more probe pressure.

## Code/Config Changes

- `src/train/train_star_uvt_feature_overfit.py` now has
  `train.global_step_offset` for explicit resumed schedule/reporting semantics.
  The default is `0`, so old configs keep local-step behavior.
- Result rows now record `global_step_offset`, `start_global_step`,
  `end_global_step`, and `step_global_steps`.
- `tests/test_star_uvt_feature_target_adapter.py` covers the default and a
  schedule that starts the local segment at global step 3.
- New config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.jsonc`.

## Run

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.jsonc
```

Inputs:

- resume checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_600step_after_resume.pt`
- objective: feature loss weight `0.25`, frozen RGB-probe loss weight `40.0`
- local steps: `200`
- global steps: `600 -> 800`
- W&B offline run:
  `wandb/offline-run-20260519_074357-jde950ee`

Artifacts:

- result JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.json`
- checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_800step_after_resume.pt`
- media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_sbs.mp4`

## Results

- `pass=true`
- `resume_loaded=true`
- `resume_optimizer_loaded=true`
- `resume_checkpoint_steps=300`
- `tile_overflow_sum=0`
- probe PSNR: `19.888 -> 21.425`
- probe loss: `0.010262 -> 0.007203`
- feature loss: `0.655132 -> 0.703820`
- mean timing:
  - step `1512.4ms`
  - render `594.5ms`
  - target-grid loss prep `22.0ms`
  - RGB-probe loss `42.4ms`
  - backward `773.8ms`

## Read

This is a positive visual-pressure result and a negative objective-balance
result. It passes the standalone full-video upsample oracle number (`20.073`)
but still trails the same-grid probe oracle (`23.401`), and it gets there by
letting V-JEPA target-grid feature loss drift upward.

The next gate should not be another plain "more probe weight" run. It should
try to keep the visual probe gain while preserving feature target alignment,
for example with a staged objective, an adaptive ratio, or a native VJP that
matches the visual decoder more directly without discarding the V-JEPA target.

## Validation

- `py_compile` passed for the STAR trainer, report builders, and focused test
  file.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py -q`
  passed: `10 passed`.
- Regenerated:
  - `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.json`
  - `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  - `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.json`
  - `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
