# STAR UVT Feature Overfit Checkpoint/Resume Gate

## Context

After the 20/100/300-step frozen RGB-probe STAR target-grid gates, the next
question was whether longer probe objectives have to restart from step 0. The
trainer had no persistence hook, so extending the current best diagnostic would
mean paying the full initialization and optimization ladder again.

## Code Change

`src/train/train_star_uvt_feature_overfit.py` now accepts:

- `output.checkpoint`: optional path to save model, colorizer, optimizer,
  serialized config, final row, and loss curves.
- `train.resume_checkpoint`: optional path to load those states before the local
  training loop.
- `train.resume_optimizer`: optional bool, default `true`, to load Adam state.

Important semantic detail: this is warm-start local-step resume. `train.steps`
still means "run this many local steps after loading". It is not a global-step
resume for staged schedules. That is fine for the current constant-weight
frozen-probe gates, but scheduled objectives should treat the resumed run as a
new local segment unless we add explicit global-step offset semantics.

## Validation

Unit/compile gates:

```bash
rtk .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py tests/test_star_uvt_feature_target_adapter.py
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py -q
```

Result: `9 passed in 2.17s`.

Runtime smoke used the existing real STAR feature-overfit route:

```text
src/train_configs/star_uvt_feature_testvideo_8f_64_rgbpyramid_target_gradcache_reduce_vec4_10step.jsonc
```

The script mutated it in memory to run 2 local steps, save
`/tmp/star_uvt_checkpoint_resume_smoke/first.pt`, then load that checkpoint with
optimizer state and run 2 more local steps, saving
`/tmp/star_uvt_checkpoint_resume_smoke/resume.pt`.

Smoke summary:

```json
{
  "first_checkpoint_exists": true,
  "first_step_ms": 159.2177915154025,
  "first_steps": 2,
  "first_tile_overflow_sum": 0,
  "resume_checkpoint_exists": true,
  "resume_checkpoint_steps": 2,
  "resume_loaded": true,
  "resume_optimizer_loaded": true,
  "resume_step_ms": 42.785812460351735,
  "resume_steps": 2,
  "resume_tile_overflow_sum": 0
}
```

The first tiny run passed with feature loss `0.340055 -> 0.330431`; the resumed
run passed with feature loss `0.320850 -> 0.311150`. Both had zero tile overflow.

## Handoff

The current keeper quality diagnostic remains the 300-step frozen RGB-probe row:
feature loss `0.999935 -> 0.811652`, probe PSNR `13.985 -> 16.560`,
`1.355s/step`, offline W&B `jhv2lgdj`. Checkpoint/resume does not improve that
result by itself; it just makes the next longer or scheduled probe cheaper to
stage.

Next useful run: add `output.checkpoint` to the selected frozen-probe config,
rerun or extend from a saved checkpoint, and compare whether longer local
segments close the remaining gap to the standalone feature-to-RGB oracle
(`20.073` full-video PSNR, `23.401` grid PSNR). If the next run uses a staged
weight schedule, add explicit global-step offset semantics first or keep each
segment's schedule intentionally local.
