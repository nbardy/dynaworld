# STAR UVT Feature1/Probe40 Balance Resume100

## Context

The 1250->1300 `feature=1.0/probe=40.0` continuation was the first current
both-improving row in the 64f/512px frozen-probe target-grid chain. This run
extended the same objective from the 1300-step checkpoint for 100 local steps to
check whether the balance holds beyond the short 50-step window.

## Run

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc`
- Source checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1300step_after_resume.pt`
- Result JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json`
- Output checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1400step_after_resume.pt`
- Media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sbs.mp4`
- Offline W&B:
  `wandb/offline-run-20260519_091629-2ouws83u`
- Continuation-chain report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc
```

## Result

- Pass: `true`
- Global steps: `1300 -> 1400`
- Resume loaded / optimizer loaded / source local steps:
  `true / true / 50`
- Objective: `feature_target.loss_weight=1.0`,
  `rgb_probe_loss_weight=40.0`
- Feature loss: `0.632124399766326 -> 0.6271288748830557`
- RGB-probe loss: `0.006360326013236772 -> 0.006340551644825609`
- RGB-probe PSNR: `21.965205669403076 -> 21.978728771209717`
- Mean step timing:
  `1690.23ms/step`, `616.71ms` render, `24.12ms` feature target,
  `48.86ms` RGB probe, `909.58ms` backward
- Tile overflow: `0`
- Max/p95 tile counts: `68 / 45`, tile capacity `128`

## Interpretation

The balance objective still works at 1400 steps: both V-JEPA target-grid loss
and frozen-probe visual PSNR improved, and support stayed comfortably under the
tile cap. The improvement is now small, and the measured timing regressed from
the 1250->1300 row (`1.285s/step`) to `1.690s/step`, with backward up to
`909.6ms`.

This keeps the current quality direction alive, but it is not enough to declare
the oracle gap solved: the same-grid target-grid probe oracle remains `23.401`
PSNR. The next fork should either continue the balance row with explicit timing
scrutiny, or shift effort to native VJP/dataset-scale machinery.

## Validation

- `py_compile` passed for `train_star_uvt_feature_overfit.py`, both STAR UVT
  report scripts, and `tests/test_star_uvt_feature_target_adapter.py`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_star_uvt_feature_target_adapter.py -q` passed: `10 passed`.
- Report invariants passed: comparison JSON has `27` rows, the
  `feature1_lr005_resume100_from1300` comparison row is `pass`, bridge audit
  flag is `true`, global steps are `1300 -> 1400`, and tile overflow is `0`.
- Continuation-chain report regenerated with `7` rows and the latest
  max/p95/cap tile read `68/45/128`.
- Touched-file trailing whitespace/newline scan passed.
- `git diff --check` passed.
- No active `train_star_uvt_feature_overfit.py` process remained.
