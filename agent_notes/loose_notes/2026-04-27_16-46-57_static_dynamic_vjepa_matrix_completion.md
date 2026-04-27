# Static/Dynamic V-JEPA Matrix Completion

Context: the previous best static/dynamic V-JEPA run was described as `~525`
steps. Local W&B shows the precise status as run `mybv736f`, `_step=520`,
with step-500 media and output around `499/1000`; it used the 1000-step config
but did not finish the 1000-step schedule.

## Runs Added

Launcher:

```bash
rtk ./src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh all
```

New wiring:

- Added static/dynamic split support to `UnconditionedTokenGSImplicitCamera`.
- Added unconditioned static/dynamic 96/32 strong-init configs for 250 and 1000 steps.
- Added local static/dynamic 96/32 strong-init 1000-step config.
- Extended `train_static_dynamic_vjepa_features_ablation.sh` with `matrix-250`, `matrix-1000`, and `all` modes.

One accidental model-construction smoke initialized W&B run `hccnqdvz`; it has
no training/eval metrics and should be ignored.

## Source-View Matrix

All rows are same source-video, 128px render/loss, 8192 splats, static/dynamic
96/32 split where marked.

| variant | run | steps | Eval/Loss | L1 | SSIM | PSNR | temporal adj ratio | decoded XYZ adj | cam adj rot | cam adj trans | cam rot mean | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local static/dynamic old | `sc25ek8t` | 250 | 0.1195 | 0.0779 | 0.4287 | 18.42 | 0.3408 | 0.0455 | 0.0159 | 0.00280 | 7.710 | 488s |
| V-JEPA static/dynamic old | `oaor6um2` | 250 | 0.0881 | 0.0615 | 0.6109 | 20.29 | 0.6322 | 0.0945 | 0.1309 | 0.0171 | 4.193 | 810s |
| V-JEPA static/dynamic interrupted | `mybv736f` | 520 | 0.0547 | 0.0413 | 0.7836 | 23.69 | 0.8009 | 0.1305 | 0.1827 | 0.0190 | 3.564 | 2424s |
| unconditioned static/dynamic | `twh5to1q` | 250 | 0.0762 | 0.0555 | 0.6815 | 21.36 | 0.6619 | 0.0208 | 0.0244 | 0.00155 | 2.458 | 216s |
| unconditioned static/dynamic | `qstqjup2` | 1000 | 0.0588 | 0.0448 | 0.7706 | 23.03 | 0.7837 | 0.0271 | 0.0160 | 0.00167 | 1.601 | 912s |
| local static/dynamic | `x803a6ra` | 1000 | 0.0781 | 0.0551 | 0.6599 | 21.49 | 0.7012 | 0.0644 | 0.0334 | 0.00498 | 5.124 | 1587s |
| V-JEPA static/dynamic | `x4uc6va3` | 1000 | 0.0455 | 0.0360 | 0.8336 | 24.93 | 0.8426 | 0.1546 | 0.2909 | 0.0355 | 6.798 | 1486s |

## Interpretation

The matched V-JEPA static/dynamic 1000-step run is the best same-source result
so far on full-frame eval: it improves over unconditioned static/dynamic 1000
by `0.0133` loss and `+0.0630` SSIM, and improves over the interrupted V-JEPA
checkpoint by `0.0092` loss and `+0.0501` SSIM.

The unconditioned static/dynamic control is much stronger than expected:
`twh5to1q` at 250 steps already beats the old V-JEPA 250 row, and `qstqjup2`
nearly matches the old interrupted V-JEPA run on SSIM/PSNR without using video
features. This means a lot of the previous win came from static/dynamic split,
strong init, and the token/head decoder optimization path, not just JEPA.

The red flag remains camera compensation. The best new V-JEPA run also has the
largest adjacent camera deltas: `0.2909 deg` adjacent rotation and `0.0355`
translation, versus unconditioned 1000 at `0.0160 deg` and `0.00167`.
That does not invalidate the source-view improvement, but it means the next
selector cannot be source-view loss alone.

Next control:

- Run V-JEPA static/dynamic 1000 with camera clamp or camera regularization.
- Run the same unconditioned/static-dynamic control under the camera clamp.
- Prefer held-out-camera or scene-distinct eval for any claim about video
conditioning helping generalization.
