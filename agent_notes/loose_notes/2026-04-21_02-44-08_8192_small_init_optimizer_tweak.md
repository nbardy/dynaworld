# 8192 Small Init And Optimizer Tweak

## Context

The 128px/4fps Taichi 8192-splat run was stable and fast, but the visual fit was still soft. Increasing from 512 to 8192 splats only modestly improved sampled train loss, which suggests the next simple baseline tweak should not be more splats. The user asked to keep the token/splat count fixed and apply the cleaner training/init changes.

## Changes

- Kept the 8192 capacity fixed: `128 tokens x 64 gaussians/token`.
- Added config-driven Gaussian head initialization knobs:
  - `token_init_std`
  - `head_output_init_std`
  - `position_init_raw_jitter`
  - `scale_init_log_jitter`
  - `opacity_init`
  - `rotation_init`
- Updated the 8192 Taichi config to use:
  - `scale_init=0.02`
  - `scale_init_log_jitter=0.7`
  - `opacity_init=0.02`
  - `token_init_std=0.01`
  - `head_output_init_std=0.002`
  - `position_init_raw_jitter=0.9`
  - `rotation_init="identity"`
- Added config-driven optimizer creation:
  - `optimizer.type`: `adam` or `adamw`
  - `optimizer.weight_decay`
  - `optimizer.exclude_bias_norm`
  - optional `optimizer.lr_multipliers` by parameter-name prefix
- Updated the 8192 config to use `AdamW(weight_decay=0.05)` with bias/norm excluded.
- Added `clip_grad_norm` to the train config and applied it after finite optimizer diagnostics and before `optimizer.step()`.
- Left optimizer metrics enabled for this config so W&B can show gradient norms while we compare this run to the previous 8192 run.

## Scale Initialization Clarification

Before this change, scale was not literally fixed. The model used:

```text
scales = exp(raw_scale) * scale_init
```

where `raw_scale` came from the scale MLP, so splats had random sizes around `scale_init`. However, adding a TokenGS-like tiny final head init would have made `raw_scale` very close to zero and therefore made initial sizes almost fixed. The new `scale_init_log_jitter` intentionally gives each scale output a random trainable bias in log-scale space, so initial scales are roughly:

```text
scale_init * exp(U[-jitter, jitter])
```

For `scale_init=0.02` and `jitter=0.7`, that is about `0.01..0.04`.

## Validation

- Syntax check passed:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/gs_models/blocks.py \
  src/train/gs_models/dynamic_token_gs.py \
  src/train/dynamicTokenGS.py
```

- Two-step disabled-W&B smoke passed on MPS with Taichi and AdamW.
- Thirty-step disabled-W&B probe was finite. Initial loss is much higher than the old fat/opaque init because this config starts nearly transparent and smaller, but it moved down from roughly `0.43..0.51` to roughly `0.23` within 30 short-schedule steps. The 30-step probe is not a final quality test because cosine decay compressed the LR schedule into only 30 steps.
- `git diff --check` passed for the touched files.

## Open Question

This small-init config is deliberately less eager to paint with broad opaque splats. If it underfits after a real 1000-step run, the first cheap adjustment is likely `opacity_init=0.05` rather than increasing splat count again.

## Follow-Up Probe

Running the full default with `opacity_init=0.02` showed that it was stable but not a good working default: around step 100 it was still roughly `0.12..0.15` sampled loss and had slowed to about `1.1..1.3 it/s`. The likely issue is renderer work, not just optimizer quality: very low opacity prevents early alpha saturation, so Taichi evaluates many more transparent splats per pixel.

An override probe with `opacity_init=0.1`, optimizer metrics disabled, and a 120-step short cosine schedule was more usable:

```text
steps=120
opacity_init=0.1
final sampled loss ~= 0.109
average speed ~= 1.72 it/s
```

This is still not the final quality setting, but it is a better default for "make the 128px baseline work first." The config now uses `opacity_init=0.1` and leaves optimizer metrics off by default.
