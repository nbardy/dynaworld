# Softmax-GS 5-Step Diagnostic

## Context

After the Metal forward shader and guarded Torch fallback passed one-step
smokes, I ran a tiny matched 5-step diagnostic pair. The purpose was only to
answer: can enabled Softmax-GS survive more than one optimizer step after the
shader work?

## Configs

- `src/train_configs/local_mac_softmax_gs_noop_diagnostic_32_2f_64splats_5step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_diagnostic_32_2f_64splats_5step.jsonc`

Both use:

```text
render_size = 32
train_frame_count = 2
tokens = 4
gaussians_per_token = 16
total splats = 64
steps = 5
W&B disabled
```

The enabled row used `softmax_gs_enabled=true` through the then-current Torch
fallback. The no-op row uses the same `v5_softmax_gs` fork with
`softmax_gs_enabled=false`. A later pass changed enabled training to Metal
forward plus Torch recompute backward; see the next loose note.

## Commands

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_noop_diagnostic_32_2f_64splats_5step.jsonc

PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_diagnostic_32_2f_64splats_5step.jsonc
```

## Results

No-op:

```text
initial loss = 0.4376
step 1 = 0.4479
step 2 = 0.4258
step 3 = 0.4743
step 4 = 0.4307
step 5 = 0.4891
tqdm mean = 3.62s/it
```

Enabled:

```text
initial loss = 0.4375
step 1 = 0.4331
step 2 = 0.4381
step 3 = 0.4483
step 4 = 0.4335
step 5 = 0.4885
tqdm mean = 4.85s/it
```

## Interpretation

This is a stability diagnostic, not a quality result. Both rows finish, losses
stay finite, and the final losses are nearly tied. The enabled path does not
show an obvious short-run win, but it also does not explode.

Do not update `BASELINES.md` from this. The runs are tiny, W&B-disabled,
same-view, and fallback-backed.

## Next Gate

Native/tape backward is still the right next serious implementation gate. A
larger slow-fallback run would be possible, but it will spend time on a path we
already know is not the final renderer. If we do another fallback ablation, it
should be short, explicitly diagnostic, and chosen only to decide whether native
backward is worth implementing.
