# Microlib: Gaussian 512px Promotion Guard

## Problem

The 300-clip V-JEPA/static-dynamic Gaussian lane is cache-hot and reasonable at
256px, but the 256-to-512 multires run hit NaNs near promotion. This microlib
should evolve guardrails and diagnostics around the render-size promotion.

## Why Now

The lane is blocked at 512px promotion. Throughput evidence at 256px does not
prove the 512px stage is stable, and the current multires config should not be
treated as a completed baseline.

## Allowed Edits

Likely surface:

- `src/train/train_video_token_implicit_dynamic.py`
- `src/train/config_utils.py` only if config normalization is needed
- `src/train_configs/*multires*.jsonc` only for explicit smoke configs
- `tests/` for promotion-schedule validation

Keep guard knobs in JSONC. Do not add environment-variable fanout.

## Baseline

Current config:

- `local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc`
- `render_size_schedule`: 256 at step 0, 512 at step 2400
- recorded failure: NaNs after the 512px switch

## Evaluator Cascade

Stage 0:

- render-size schedule still validates
- no silent downgrade to 256px only
- config remains JSONC-driven

Stage 1:

- unit test for promotion schedule and checkpoint-before-promotion behavior
- finite-check helper catches nonfinite decoded tensors, render outputs, and
  loss before optimizer step

Stage 2:

- short smoke with promotion moved early, e.g. 1-2 steps at 256 then 1-2 steps
  at 512
- writes diagnostic JSON with the first nonfinite source or `finite=true`

Stage 3:

- resumed pre-promotion checkpoint run over a larger but still local budget
- W&B enabled if the result is going to support a lane claim

## Primary Metrics

- `promotion_reached == true`
- `finite == true`
- checkpoint exists before promotion
- diagnostic fields present
- no target/render-size downgrade

## Hard Rejects

- Masking NaNs after optimizer corruption.
- Lowering target resolution or frame count and calling it fixed.
- Moving guard config to env vars.
- Continuing after nonfinite state without a clear abort/resume path.

## Promotion Gate

This microlib can be promoted when the early-promotion smoke proves the guard
works. It does not unblock the 300-clip baseline until the real multires config
is rerun or resumed successfully through the 512px stage.
