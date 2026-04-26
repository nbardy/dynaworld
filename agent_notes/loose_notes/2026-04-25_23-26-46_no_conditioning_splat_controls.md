# No-Conditioning Splat Controls

## Context

The current 16-frame dog overfit line needed controls that remove video conditioning entirely:

- a modelless/direct-splat baseline: fit free Gaussian parameters, not tokens or a decoder
- a token+decoder baseline: learned tokens decoded to Gaussians, but no video encoder or feature memory

These are not replacements for the V-JEPA/LTX/Wan feature-prior work. They are controls for checking whether the renderer, initialization, camera head, and loss can fit the clip without any image/video conditioning.

## What Changed

Added two implicit-camera model variants in `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`:

- `FreeGaussianBankImplicitCamera`
  - no video encoder
  - no token-to-Gaussian decoder
  - direct learnable Gaussian parameters
  - configured as per-source-frame free banks for the current 46-frame clip
  - still uses the same implicit camera head so camera learning stays comparable
- `UnconditionedTokenGSImplicitCamera`
  - no video encoder
  - learned query tokens plus the normal Gaussian decoder heads
  - time-conditioned, so it can memorize a sequence from normalized frame time
  - still uses the same implicit camera head

Added configs:

- `src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc`

Both match the current RGB/uniform strong-init 16f/128px overfit setup:

- `test_data/test_video_small_128_4fps.mp4`
- 16-frame sampled windows from 46 source frames
- 128px render/loss size
- 128 splat slots x 64 Gaussians = 8192 Gaussians per frame
- fast-mac renderer
- 250 steps
- same losses and camera regularization

Extended `src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh`:

- `free`
- `tokens`
- `controls`
- `all`

## Verification

No training was started.

Smokes run:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/gs_models/dynamic_video_token_gs_implicit_camera.py \
  src/train/gs_models/__init__.py \
  src/train/train_video_token_implicit_dynamic.py

uv run ruff check --select F \
  src/train/gs_models/dynamic_video_token_gs_implicit_camera.py \
  src/train/gs_models/__init__.py \
  src/train/train_video_token_implicit_dynamic.py

bash -n src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh
```

Forward smoke output:

```text
src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc
  FreeGaussianBankImplicitCamera xyz (16, 8192, 3) cameras 16 camera_state True
src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc
  UnconditionedTokenGSImplicitCamera xyz (16, 8192, 3) cameras 16 camera_state True
```

`git diff --check` passed for the touched files.

## Interpretation

These two runs separate different failure modes:

- If free per-frame splats cannot fit the dog, the blocker is renderer/loss/init/camera/optimization rather than representation conditioning.
- If free splats fit but unconditioned tokens do not, the token decoder is a bottleneck.
- If unconditioned tokens fit similarly to the video-conditioned model, then the current video encoder path may not be contributing meaningful per-frame information on this overfit task.
- If V-JEPA/local conditioned models beat both no-conditioning controls, then video features are doing useful work.

## Full Suite Results: 2026-04-26

Command:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh all
```

I waited for an already-running trainer process to exit, then ran the full comparison suite:

- local video encoder implicit camera
- frozen V-JEPA fpc16/256 implicit camera
- known-camera DUSt3R control
- direct free-splat implicit camera
- unconditioned token+decoder implicit camera

All five runs completed. Step-0 initialization diagnostics printed before the first optimizer step.

| Variant | W&B run | Final train loss | Eval loss | Eval L1 | Eval SSIM | Eval PSNR | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| local video encoder implicit camera | `bbk7maml` | 0.13265 | 0.14906 | 0.09864 | 0.29852 | 16.63 | 6:10 |
| frozen V-JEPA fpc16/256 implicit camera | `pwvybmao` | 0.13155 | 0.14510 | 0.09466 | 0.30623 | 16.71 | 19:04 |
| known-camera DUSt3R control | `ut58yk3z` | 0.14758 | 0.15595 | 0.10320 | 0.26607 | 16.55 | 2:46 |
| direct free-splat implicit camera | `kttrbewl` | 0.25769 | 0.26334 | 0.28706 | 0.66310 | 9.66 | 1:49 |
| unconditioned token+decoder implicit camera | `xenc4w06` | 0.13574 | 0.14388 | 0.09569 | 0.32673 | 16.88 | 1:51 |

W&B links:

- local: https://wandb.ai/nbardy/dynaworld/runs/bbk7maml
- V-JEPA: https://wandb.ai/nbardy/dynaworld/runs/pwvybmao
- known-camera: https://wandb.ai/nbardy/dynaworld/runs/ut58yk3z
- free-splats: https://wandb.ai/nbardy/dynaworld/runs/kttrbewl
- unconditioned tokens: https://wandb.ai/nbardy/dynaworld/runs/xenc4w06

Interpretation changed after the run:

- V-JEPA is slightly better than the local video encoder on full-video eval, but about 3x slower on this Mac/MPS setup.
- The DUSt3R known-camera control is not winning here, so camera precomputation alone does not explain the missing dog/detail issue.
- Direct free splats failed badly despite having free parameters. This points at optimization/parameterization/camera coupling, not just raw degrees of freedom.
- The unconditioned token+decoder control matched or beat the video-conditioned paths. That means this single-video 16f/128px overfit is too weak to prove that video conditioning is contributing useful information.
- The token decoder/head parameterization appears to be doing important optimization work; it is not merely a compact way to store free splats.

## Linear-Time Free-Splat Follow-Up

After visual inspection, the direct free-splat run looked sharper than its L1/PSNR suggested: washed out, but more edge-aware and shape-like. I added a new control:

- `src/train_configs/local_mac_compare_free_linear_time_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `model.variant = "free_linear_time_splats"`
- one shared free Gaussian bank
- learned linear xyz velocity with respect to normalized time
- learned linear opacity-logit slope with respect to normalized time
- fixed RGB, scale, and rotation over time
- same implicit camera head

The comparison script now supports:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh free-linear
```

Forward smoke:

```text
LinearTimeFreeGaussianBankImplicitCamera xyz (16, 8192, 3) opacity (16, 8192, 1) cameras 16
velocity_extent 1.0 velocity_abs_mean 0.0 opacity_slope_abs_mean 0.0
```

Init diagnostics now support direct free models too. Probe command:

```bash
PYTHONPATH=src/train uv run python src/train/probe_init_diagnostics.py CONFIG --seed 0
```

Key init probe summaries:

| Variant | Scale mean | Scale P99 | Opacity mean | RGB mean | RGB std | RGB entropy | XYZ cross/within |
|---|---:|---:|---:|---:|---:|---:|---:|
| free per-frame splats | 0.02167 | 0.03972 | 0.1000 | 0.5003 | 0.2888 | 1.0000 | 1.004 |
| free linear-time splats | 0.02163 | 0.03969 | 0.1000 | 0.5001 | 0.2890 | 0.9999 | 1.004 |
| unconditioned tokens | 0.02380 | 0.05739 | 0.1023 | 0.4560 | 0.2924 | 0.9910 | 0.2435 |
| local video encoder | 0.02144 | 0.03986 | 0.1001 | 0.5000 | 0.0113 | 0.2317 | 0.0462 |

This does not support the simple "token models start with larger splats" hypothesis. At initialization, scale is similar, and unconditioned tokens actually have the largest P99 scale. The stronger measured difference is that the local video-encoder path starts almost gray/low-color-entropy and has very low cross-token xyz spread, while free splats start as a full-entropy colored cloud with independent spatial support.

## Crash-Recovery Completion: Free Linear-Time Run

After the user's computer crash, I checked for surviving DynaWorld train
processes. None were running. The only active Python/W&B process was unrelated
`font_maker` work.

The earlier comparison suite had added the free linear-time control but had not
run it yet, so I completed it:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh free-linear
```

W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/zj2kis2e
```

Result:

| Variant | W&B run | Eval loss | Eval L1 | Eval SSIM | Eval PSNR | Temporal adjacent ratio | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| free linear-time splats | `zj2kis2e` | 0.23910 | 0.21842 | 0.35638 | 11.58 | 0.03691 | 4:42 |

Updated no-conditioning comparison:

| Variant | W&B run | Eval loss | Eval L1 | Eval SSIM | Eval PSNR | Temporal adjacent ratio | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| local video encoder implicit camera | `bbk7maml` | 0.14906 | 0.09864 | 0.29852 | 16.63 | 0.02191 | 6:10 |
| frozen V-JEPA fpc16/256 implicit camera | `pwvybmao` | 0.14510 | 0.09466 | 0.30623 | 16.71 | 0.03287 | 19:04 |
| known-camera DUSt3R control | `ut58yk3z` | 0.15595 | 0.10320 | 0.26607 | 16.55 | 0.14403 | 2:46 |
| direct free-splat implicit camera | `kttrbewl` | 0.26334 | 0.28706 | 0.66310 | 9.66 | 0.67861 | 1:49 |
| free linear-time splats | `zj2kis2e` | 0.23910 | 0.21842 | 0.35638 | 11.58 | 0.03691 | 4:42 |
| unconditioned token+decoder implicit camera | `xenc4w06` | 0.14388 | 0.09569 | 0.32673 | 16.88 | 0.06708 | 1:51 |

Interpretation:

- Linear-time free splats improved over independent free per-frame splats on
  L1/PSNR, but still lost badly to token-decoder models.
- Free splats are not a strong oracle yet. Their parameterization/optimization
  is weaker than the token decoder despite having more direct degrees of freedom.
- The unconditioned token+decoder remains the best no-conditioning control in
  this tiny single-video setup.
- The strongest overall result in the temporal suite remains the structured
  static/dynamic token split (`sc25ek8t`), not direct splat fitting.
