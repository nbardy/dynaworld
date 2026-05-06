# Pre-PowerFoam Direct Splat Speed Probe

## Context

User corrected that the remembered `15-30 it/s` run was pre-PowerFoam. I checked
the current direct-splat paths and Adam dtype behavior.

There are two relevant direct baselines:

- `src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
  - same-source trainer path
  - no video encoder
  - no token-to-splat decoder
  - direct per-source-frame Gaussian parameters plus the same implicit camera
  - fast-mac renderer
- `src/train_configs/local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_free_dynamic_3dgs_128_16f_2048splats.jsonc`
  - calibrated multicam heldout-camera direct 3DGS baseline
  - direct per-frame Gaussian parameters
  - dense renderer in the checked-in config

The old `17 it/s` note points at
`src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_8192splats.jsonc`,
but that exact config is no longer present in `src/train_configs/`.

## Step-Only Same-Source Free-Splats Timing

I timed `trainer.step(keep_preview=False)` directly, with W&B disabled and no
validation/logging inside the timed region.

Command shape:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python - <<'PY'
# load local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc
# instantiate Trainer
# warm up 5 step() calls
# time 50 step() calls
PY
```

Results on local MPS:

```text
free_splats fast_mac, train_frame_count=16
warmup=5 timed_steps=50 elapsed_sec=23.282 avg_it_s=2.148
last_loss=0.346125 last_recon=0.339459

free_splats fast_mac, train_frame_count=4
warmup=5 timed_steps=50 elapsed_sec=8.705 avg_it_s=5.744
last_loss=0.340116 last_recon=0.333871
```

This does not reproduce `15-30 it/s` in the current code/config state. The
closest current direct fast-mac path is around `5.7 it/s` for 4 frames/step and
around `2.1 it/s` for 16 frames/step. The older `~17 it/s` note was tied to a
pre-fix fast-mac wide-depth config that also collapsed to white renders before
the near-plane/LR stability fix.

## Full Trainer Wall-Clock Gotcha

A naive `trainer.run()` timing is misleading for speed because step 0 always
satisfies `step % video_log_every == 0`, so it still renders/encodes validation
video even when `video_log_every` is set very high.

Observed naive 20-step run:

```text
SPEED_PROBE config=free_splats_fast_mac steps=20 elapsed_sec=13.802 avg_it_s=1.449
```

This included step-0 validation video encoding and should not be used as the
renderer/optimizer loop throughput.

## Multicam Direct 3DGS Smoke

I also ran the calibrated heldout-camera direct-splat baseline for 10 steps:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/gauge_fields/train_splat_baseline.py \
  src/train_configs/local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_free_dynamic_3dgs_128_16f_2048splats.jsonc \
  --steps 10 \
  --no-wandb \
  --output-dir /tmp/dynaworld_free_dynamic_3dgs_speed_probe
```

It completed and produced:

```text
train cameras: camera_0001,camera_0015
heldout: camera_0040
steps=10 device=mps renderer=dense splats=2048 mode=per_frame
final eval_psnr=18.4116
heldout_eval_psnr=13.4614
```

This path is useful as the true calibrated "optimize splat params then raster"
heldout-camera baseline, but it is not the old fast-loop reproduction because
the checked-in config uses the dense renderer and does final eval/artifact work.

## Adam / BF16 Finding

Current `train_video_token_implicit_dynamic.py` constructs:

```python
torch.optim.Adam(trainable_parameters, lr=..., fused=device.type in {"cuda", "mps"})
```

It does not cast model parameters to bf16 before optimizer construction.
`train.amp=true` only controls autocast for forward compute. It does not make
Adam state bf16.

Probe on `free_splats` after one optimizer step:

```text
DTYPE_PROBE before_step first_param=raw_xyz param_dtype=torch.float32 amp_available=False amp_dtype=torch.float16 optimizer=Adam fused=True
DTYPE_PROBE after_step first_param=raw_xyz param_dtype=torch.float32 state_dtypes={'step': 'torch.float32', 'exp_avg': 'torch.float32', 'exp_avg_sq': 'torch.float32'}
```

So the answer is: no, current direct-splat/video-token training is not using
bf16 Adam. It is fused Adam on MPS/CUDA when available, with fp32 parameters and
fp32 Adam moments unless we explicitly add a different optimizer/state strategy.
