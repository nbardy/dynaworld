# Residual Free-Bank Architecture Matrix

## Context

The 16f/128px single-video comparison exposed a mismatch between scalar metrics and visual quality:

- direct free splats had poor L1/PSNR but sharper visible shapes and much higher SSIM
- unconditioned tokens were visually second-best and competitive in eval loss
- local/V-JEPA conditioned token decoders often looked smooth or blurry

Init diagnostics weakened the simple "token models start with larger splats" hypothesis. The stronger issue is that plain `GaussianParameterHeads` use a per-split output bias shared across all tokens, so token diversity depends on token/head residuals. The local video-token config had low cross-token spread and near-gray RGB init.

## Current Model

The residual-free-bank idea keeps the good part of direct free splats:

```text
per-token/per-split free Gaussian base bank
```

and adds the useful part of token/video models:

```text
bounded residuals predicted from time/video-conditioned tokens
```

The head parameterization is raw-space residual, not decoded-space residual:

```text
raw_xyz      = base_raw_xyz      + tanh(delta_xyz)      * residual_xyz_raw_scale
raw_scale    = base_raw_scale    + tanh(delta_scale)    * residual_scale_log_scale
raw_quat     = base_raw_quat     + tanh(delta_quat)     * residual_rot_raw_scale
raw_opacity  = base_raw_opacity  + tanh(delta_opacity)  * residual_opacity_logit_scale
raw_rgb      = base_raw_rgb      + tanh(delta_rgb)      * residual_rgb_logit_scale
```

Then the usual Gaussian squashing applies:

```text
xy      = tanh(raw_xy) * xy_extent
z       = sigmoid(raw_z) * (z_max - z_min) + z_min
scale   = exp(raw_scale) * scale_init
opacity = sigmoid(raw_opacity)
rgb     = sigmoid(raw_rgb)
quat    = normalize(raw_quat)
```

## Variants Added

### Existing Controls Kept

- `free_splats`: direct independent per-frame free bank
- `free_linear_time_splats`: shared free bank plus linear xyz velocity and opacity slope
- `unconditioned_tokens`: learned tokens plus plain Gaussian decoder, no video conditioning
- `learned_time_orbit_path + local`: repo-native local video encoder
- `learned_time_orbit_path + vjepa_hf`: frozen V-JEPA fpc16/256 encoder

### New Residual Free-Bank Variants

- `unconditioned_residual_free_bank`
  - no video encoder
  - learned time-conditioned tokens
  - residuals on a per-token free Gaussian base bank
  - isolates whether the free-bank prior explains the sharpness

- `residual_free_bank + video_encoder_backend=local`
  - local video-token memory
  - same query/cross-attn/camera path as the local baseline
  - residual free-bank Gaussian head

- `residual_free_bank + video_encoder_backend=vjepa_hf`
  - frozen V-JEPA fpc16/256 memory
  - same residual free-bank head
  - tests whether V-JEPA adds useful conditioning once the head has a better base prior

## Configs Added

- `src/train_configs/local_mac_compare_unconditioned_residual_free_bank_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `src/train_configs/local_mac_compare_residual_free_bank_local_video_encoder_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `src/train_configs/local_mac_compare_residual_free_bank_vjepa2_vitl_fpc16_256_frozen_16f_implicit_camera_128_fast_mac_8192splats.jsonc`

All keep the current comparison invariants:

- `test_data/test_video_small_128_4fps.mp4`
- `train_frame_count=16`
- `size=128`
- `render_size=128`
- `tokens=128`
- `gaussians_per_token=64`
- 8192 Gaussians/frame
- 250 steps
- same camera config, renderer config, LR, and loss weights

## Script Modes

Updated:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh residual-tokens
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh residual-local
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh residual-vjepa
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh residual
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh matrix
```

`matrix` excludes the known-camera DUSt3R control because it changes camera/data source and is not a fair head/conditioning comparison.

## Initial Diagnostics

Forward/config smoke:

```text
UnconditionedResidualFreeBankImplicitCamera xyz (16, 8192, 3) rgb_std 0.2894
ResidualFreeBankVideoTokenGSImplicitCamera xyz (16, 8192, 3) rgb_std 0.2880
V-JEPA residual config resolve-only; skipped HF checkpoint construction.
```

Init probe:

| Variant | Scale mean | Scale P99 | Opacity mean | RGB std | RGB entropy | XYZ cross/within |
|---|---:|---:|---:|---:|---:|---:|
| unconditioned residual free-bank | 0.02165 | 0.03968 | 0.1000 | 0.2882 | 0.9998 | 1.006 |
| local residual free-bank | 0.02174 | 0.03973 | 0.1000 | 0.2871 | 0.9999 | 1.004 |

This means the residual-free heads start with free-splat-like diversity while retaining a conditioned residual path.

## Agent Proposals Incorporated

Three subagents were used:

- architecture explorer: recommended unconditioned residual-free bank, video-conditioned residual-free bank, and later linear-time residual-free bank
- config explorer: recommended retaining current baselines, adding residual local/V-JEPA/unconditioned configs, and adding `matrix` script mode without known-camera
- diagnostics explorer: recommended sharpness, washed-out appearance, background leakage, coverage, and decoded-drift metrics

The first two proposal tracks were implemented here. The diagnostics proposal is not fully implemented yet beyond init/decoded probes.

## Open Next Step

Run order for the next actual training suite:

```text
residual-tokens
residual-local
residual-vjepa
```

If residual-local beats local visually and in SSIM/sharpness, the repeated-bias Gaussian head was a real bottleneck. If residual-tokens already matches residual-local, the single-video task is still too easy to solve from time alone and needs 256px or held-out/multi-clip eval before making conditioning claims.

## Matrix Run Results: 2026-04-27

Command:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh matrix
```

All eight cells completed without process failures. This was still the 128px/16f/single-overfit contract, so it is a systems/architecture check and a weak conditioning-quality verdict.

| Variant | Run | Eval/Loss | Eval/L1 | Eval/SSIM | Eval/PSNR | Runtime |
|---|---|---:|---:|---:|---:|---:|
| local video encoder | `leunxckm` | 0.1485 | 0.0975 | 0.2946 | 16.56 | 0:59 |
| V-JEPA fpc16/256 | `vph578vo` | 0.1601 | 0.1088 | 0.2690 | 16.19 | 11:32 |
| unconditioned tokens | `fja3e512` | 0.1289 | 0.0863 | 0.4009 | 17.41 | 1:12 |
| free linear-time splats | `9pzjrm9a` | 0.2419 | 0.2224 | 0.3603 | 11.43 | 1:03 |
| free splats | `0cwcdva7` | 0.2650 | 0.2888 | 0.6611 | 9.62 | 0:57 |
| unconditioned residual free-bank | `a3d5yojf` | 0.1469 | 0.1018 | 0.3454 | 16.48 | 1:51 |
| residual free-bank local | `2a0vmenl` | 0.1590 | 0.1125 | 0.3106 | 15.85 | 2:30 |
| residual free-bank V-JEPA | `1shque5e` | 0.1497 | 0.1045 | 0.3397 | 16.38 | 15:00 |

W&B links:

- local: https://wandb.ai/nbardy/dynaworld/runs/leunxckm
- V-JEPA: https://wandb.ai/nbardy/dynaworld/runs/vph578vo
- unconditioned tokens: https://wandb.ai/nbardy/dynaworld/runs/fja3e512
- free linear-time splats: https://wandb.ai/nbardy/dynaworld/runs/9pzjrm9a
- free splats: https://wandb.ai/nbardy/dynaworld/runs/0cwcdva7
- unconditioned residual free-bank: https://wandb.ai/nbardy/dynaworld/runs/a3d5yojf
- residual local: https://wandb.ai/nbardy/dynaworld/runs/2a0vmenl
- residual V-JEPA: https://wandb.ai/nbardy/dynaworld/runs/1shque5e

Interpretation:

- The winner by full-video reconstruction metrics is unconditioned tokens, not local conditioning and not V-JEPA. This reinforces that the current single-video 16f/128px task can be solved from learned time/query structure alone.
- V-JEPA did not help this setup. It was slower and worse than local/unconditioned in this 250-step short run, so V-JEPA should not be judged again until the comparison is 256px source/load/render/loss and uses a harder held-out or multi-clip contract.
- Free splats still show the metric/visual split: very high SSIM but bad L1/PSNR. This supports the user's visual observation that edge/shape quality and full-frame photometric loss are not aligned here.
- Linear-time free splats improved PSNR/L1 over static free splats but did not close the gap to token decoder models. Time-parameterized free splats are not a drop-in replacement for the decoder in this contract.
- Residual free-bank heads did not beat the simpler unconditioned token decoder. They may still be useful as a head-design ablation, but this matrix does not justify making them the default.
- The residual-V-JEPA run slowed badly near the end on MPS, likely memory/attention pressure plus media/logging overhead. Its quality did not justify that cost here.

Next implication:

The next fair run should not be more 128px single-clip cells. Move the main comparison to 256px end-to-end with step-0 media, foreground/high-pass/motion metrics, and either held-out windows or scene-distinct clips. Keep unconditioned tokens as a required baseline in every future V-JEPA/local comparison.

Static/dynamic caveat:

This matrix did not include the known strongest static/dynamic recipe. See `agent_notes/best_tweaks.md`: the 96/32 static/dynamic split plus precomputed V-JEPA 2.1 ViT-B/384 tokens and 4 cross-attention layers reached `Eval/Loss 0.0881`, `SSIM 0.6109` at 250 steps, and the interrupted ~525-step run reached `Eval/Loss 0.0547`, `SSIM 0.7836`. That means the recent matrix only says "plain V-JEPA and residual-free-bank heads are weak here"; it does not falsify the static/dynamic + V-JEPA recipe.
