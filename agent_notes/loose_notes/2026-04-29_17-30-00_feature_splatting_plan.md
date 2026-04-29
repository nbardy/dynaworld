# Feature Splatting Plan

Date: 2026-04-29

Goal: replace fixed-RGB-per-splat with **F-channel features per splat**,
rasterize into an `F`-channel feature map, then run a per-pixel `1x1` MLP that
maps features to RGB. Default `F = 32`. The colorize MLP is fixed (no time
conditioning) and shared across pixels.

This layers on top of the unconditioned token-GS baseline
(`local_mac_unconditioned_tokens_fast.jsonc`, run `s6xnvoch`). All other
architecture pieces stay the same.

## Current vs new pipeline

```
current:  tokens -> (xyz, scale, quat, opacity, rgb[3])  -> raster[B,T,3,H,W] -> RGB image -> L1+DSSIM loss
new:      tokens -> (xyz, scale, quat, opacity, feat[F]) -> raster[B,T,F,H,W] -> 1x1 MLP -> RGB image -> L1+DSSIM loss
```

The only points where `3` becomes `F`:
- the splat-feature head output and the `GaussianSequence.rgbs` field
- the rasterizer's color tensor + output
- the new colorize-MLP input

The loss, the camera path, the projection math, and the splat geometry
(`xyz/scale/quat/opacity`) are unchanged.

## Feature-dim contract

- `F` is configurable per run via `model.feature_dim` (default 32).
- `F = 3` MUST stay valid as an identity / parity case. With `F=3` and a
  colorize MLP that's a no-op (identity init, learnable), the run should
  reproduce the current RGB baseline within optimization noise. This is the
  cleanest "did anything break" test.
- Features are unconstrained reals in the splat output. The colorize MLP
  produces RGB in `[0, 1]` via a sigmoid (or learnable bias + sigmoid).

## File-by-file changes

| # | File | What changes |
|---|---|---|
| 1 | `third_party/fast-mac-gsplat/variants/v5/` (FORK to `v5_features`) | Replace 3-channel hardcoding with runtime `F`. See companion handoff doc `2026-04-29_17-30-00_codex_handoff_v5_features_rasterizer.md`. |
| 2 | `src/train/renderers/fast_mac.py` | Point at the new bridge. Drop the `clamp(0.0, 1.0)` and `permute(...,3,...)` assumptions for `F!=3`; output is `[B,T,F,H,W]` unclamped. |
| 3 | `src/train/renderers/projection.py`, `src/train/renderers/common.py` | Verify `colors` is passed through channel-agnostically. The current functions only reorder/filter splats; should already work for any `C`. Confirm by grep + tests. |
| 4 | `src/train/gs_models/blocks.py` (`GaussianParameterHeads`) | Replace `rgb_head` (out=`G*3`) with `feature_head` (out=`G*F`). Drop `rgb_init`/`rgb_init_min`/`rgb_init_max` since features aren't RGB-bounded. Optional small-init for stability (head_output_init_std stays). Forward returns `features` of shape `[B, N, F]` instead of `rgbs [B, N, 3]`. |
| 5 | `src/train/gs_models/blocks.py` (`DynamicResidualGaussianBankHead`, `ResidualFreeGaussianParameterHeads`) | Same change. Their `A_*` time-basis tensors and dynamic_alpha bits stay. |
| 6 | `src/train/runtime_types.py` (`GaussianSequence`) | Rename `rgbs` -> `features` (or add `features` and keep `rgbs` only for the RGB-3 legacy path; pick one). Update `GaussianFrame` similarly. |
| 7 | `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py` | Thread `feature_dim` through the model constructors. Pass to head kwargs. The `_decode_static_dynamic_split` and `_decode_single_time` paths just need to rename the local `rgbs` -> `features`. |
| 8 | `src/train/colorize.py` (NEW) | Tiny module: `FeatureToColor(in_dim=F, hidden=None or small, out=3, activation='sigmoid')`. For v0 it's a `nn.Conv2d(F, 3, kernel_size=1)` + `sigmoid`, applied after raster. Document at top: "applied per-pixel after rasterization, no time conditioning, weights shared across all pixels and frames." |
| 9 | `src/train/train_video_token_implicit_dynamic.py` (`recon_backward`, `build_model_from_config`) | After raster: `rgb = colorize(features)`. Construct the colorize module from `model.feature_dim`. Add it to the optimizer's parameter list. |
| 10 | `src/train/train_video_token_implicit_dynamic.py` (`MODEL_OPTION_DEFAULTS`) | Add `"feature_dim": 32` (or 3 if we want the parity-case default). |
| 11 | Train config | Add `"feature_dim": 32` and a colorize section. Recommend `local_mac_unconditioned_tokens_fast.jsonc` keeps `feature_dim: 3` (parity) and a sibling `..._F32.jsonc` enables 32. |

## Build order (recommend)

1. **Codex**: fork v5 -> v5_features, expose runtime `F` in the bridge + Metal kernels. Land before any model-side changes — this is the only blocker.
2. **Identity Python harness** (no model): a tiny test that constructs random splats with random F-channel features, calls the new raster, and checks: (a) shape is `[B,H,W,F]`, (b) backward returns `dL/dfeatures` with the right shape, (c) at `F=3` outputs match the v5 raster bit-for-bit.
3. **Plumb `GaussianParameterHeads` -> `feature_head`** behind a `feature_dim` constructor arg, while keeping the legacy `rgbs` field name. Wire `feature_dim=3` end-to-end first; rerun `unconditioned_tokens_fast` and confirm PSNR matches `s6xnvoch` within 1 dB.
4. **Add colorize MLP** with identity init at `F=3`. Rerun parity. Then bump `F=32`.
5. **Loss is unchanged**. The colorize MLP is just another layer in front of the L1+DSSIM. Backward already flows through.

## Tests / acceptance

- **Parity at F=3 + identity colorize**: same fast baseline (200 steps, 64 px, 2048 splats) lands within ~1 dB of `s6xnvoch`. If it doesn't, the problem is in the head/colorize plumbing, not the rasterizer (assuming the rasterizer parity test in step 2 passed).
- **Shape + grad check at F=32**: forward produces `[B,T,32,H,W]`, backward produces non-zero `dL/dfeatures` and `dL/dW_colorize`. No NaNs over 50 warmup steps.
- **Wall clock at F=32**: should not exceed `F=3` baseline by more than `~F/3` factor on raster (Metal kernel bandwidth) plus a small constant for the colorize MLP. If it's much worse, the kernel isn't using the new runtime `F` cleanly.
- **No regressions in existing trainers**: the residual_free and known_camera paths still depend on `rgbs`. Either preserve the field for legacy callers or migrate them too. Listed callers: `runtime_types.py:618` (`DECODED_TEMPORAL_FIELDS`), and every `decoded.rgbs` use site found by grep.

## Out-of-scope for this session

- Touching v8 / v9 / v6 forks. Stay on v5 -> v5_features.
- Feature distillation losses (matching V-JEPA/DINO/CLIP features). The first cut is **just RGB reconstruction** through a feature bottleneck. Distillation is an obvious next step but separable.
- Any model-architecture change beyond swapping `rgb_head` -> `feature_head`. The token bank, the static/dynamic split, the camera heads, and the time projector all stay.

## Risk notes

- The biggest risk is the rasterizer fork being slower at `F=32` than expected. The v5 Metal hot path uses threadgroup memory sized for 3 channels (`sh_colors[b3 + 0u..2u]`) and `float3` atomics. Generalizing this is non-trivial — Codex should pay attention to threadgroup memory budget and atomic accumulator layout.
- The second risk is silent gradient corruption from the rasterizer fork. The parity test at F=3 (kernel output bit-for-bit identical to v5) is the only durable safety net here. Don't merge the model changes until that test is green.
