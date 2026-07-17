# STAR UVT Feature1/Probe40 Timing Repeat

## Context

The 1300->1400 feature1/probe40 continuation kept improving feature loss and
probe PSNR, but slowed from the 1250->1300 row's `1.285s/step` to
`1.690s/step`. This run repeated the exact 1300->1400 segment from the same
1300-step checkpoint to check whether that timing was a one-off row.

## Run

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat.jsonc`
- Source checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1300step_after_resume.pt`
- Result JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat.json`
- Output checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1400step_timing_repeat.pt`
- Media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat_sbs.mp4`
- Offline W&B:
  `wandb/offline-run-20260519_093458-xs1gx1nt`
- Continuation-chain report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat.jsonc
```

## Result

- Pass: `true`
- Global steps: `1300 -> 1400`
- Resume loaded / optimizer loaded / source local steps:
  `true / true / 50`
- Objective: `feature_target.loss_weight=1.0`,
  `rgb_probe_loss_weight=40.0`
- Feature loss: `0.632124399766326 -> 0.6271196901798248`
- RGB-probe loss: `0.006360326013236772 -> 0.0063405993168998975`
- RGB-probe PSNR: `21.965205669403076 -> 21.978697776794434`
- Mean step timing:
  `1710.53ms/step`, `637.98ms` render, `25.14ms` feature target,
  `52.86ms` RGB probe, `899.07ms` backward
- Tile overflow: `0`
- Max/p95/cap tile counts: `68 / 45 / 128`

## Interpretation

The timing repeat confirms the slower 1300->1400 regime. It is not a tile
overflow problem: support stays far under cap, and quality metrics reproduce the
previous 1300->1400 row.

The next STAR UVT gate should not be another simple recovery oscillation. Either
profile the whole graph around render/probe/backward timing, or shift to native
VJP/dataset-scale work while the feature1/probe40 balance remains the current
quality-positive objective.

## Validation

- `py_compile` passed for `train_star_uvt_feature_overfit.py`, the continuation
  report script, both STAR UVT report scripts, and
  `tests/test_star_uvt_feature_target_adapter.py`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_star_uvt_feature_target_adapter.py -q` passed: `10 passed`.
- Continuation-chain report invariants passed: `8` rows, timing-repeat row is
  `pass`, global steps are `1300 -> 1400`, tile overflow is `0`, max/p95/cap is
  `68/45/128`, both feature loss and probe PSNR improve, and
  `repeat_reproduces_slow_timing=true`.
- Touched-file trailing whitespace/newline scan passed.
- `git diff --check` passed.
- `agent_notes/key_learnings.md` remains `199` lines.
- No active `train_star_uvt_feature_overfit.py` process remained.
