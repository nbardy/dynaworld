# Camera-Clamped Static/Dynamic V-JEPA Control

## Question

The best completed V-JEPA static/dynamic run (`x4uc6va3`) improved reconstruction
but also used much larger implicit camera motion than the matched unconditioned
control. The open question was whether V-JEPA was actually helping the
static/dynamic token decoder, or whether the apparent win was mostly camera
compensation.

## Test

Added explicit camera-clamp configs for the 1000-step matched pair:

- `src/train_configs/local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_camera_clamp_video_implicit_128_fast_mac_8192splats_1000step.jsonc`
- `src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_camera_clamp_video_implicit_128_fast_mac_8192splats_1000step.jsonc`

The clamp changed only camera bounds and run labels relative to the prior 1000
step pair:

- `max_fov_delta_degrees: 3.0` instead of `15.0`
- `max_radius_scale: 1.1` instead of `1.5`
- `max_rotation_degrees: 1.0` instead of `5.0`
- `max_translation_ratio: 0.03` instead of `0.2`

Loss weights, renderer, strong RGB/uniform/token/head init, token split, splat
count, and step count stayed matched. The V-JEPA config kept browser export
enabled; unconditioned export stayed disabled because the browser bundle expects
the token/head export path used by the feature-conditioned model.

Launcher mode added:

```bash
rtk bash src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh camera-clamp
```

## Runs

| lane | run | Eval/Loss | Eval/L1 | SSIM | PSNR | Adj camera rot deg | Adj camera trans | Mean camera rot deg | Eval FOV | Eval radius |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unconditioned static/dynamic clamp | `mqaz9ohc` | 0.0805315 | 0.0571003 | 0.651487 | 21.1044 | 0.019156 | 0.001812 | 1.14476 | 59.2816 | 2.91622 |
| V-JEPA static/dynamic clamp | `yhezacn8` | 0.0424945 | 0.0341260 | 0.848063 | 25.2567 | 0.136823 | 0.016429 | 1.11681 | 59.8276 | 2.98035 |
| unconditioned static/dynamic unclamped | `qstqjup2` | 0.0587532 | 0.0447682 | 0.770613 | 23.0296 | 0.015992 | 0.001669 | 1.60104 | 54.9819 | 2.65645 |
| V-JEPA static/dynamic unclamped | `x4uc6va3` | 0.0454669 | 0.0360370 | 0.833627 | 24.9261 | 0.290897 | 0.035472 | 6.79779 | 57.9133 | 2.76715 |

Additional temporal / decoded-state metrics:

| lane | run | TemporalAdjacentL1Ratio | TemporalPredAdjacentL1 | DecodedXYZAdjacentL2 |
| --- | --- | ---: | ---: | ---: |
| unconditioned static/dynamic clamp | `mqaz9ohc` | 0.656533 | 0.0565847 | 0.0365736 |
| V-JEPA static/dynamic clamp | `yhezacn8` | 0.883486 | 0.0760652 | 0.139775 |
| unconditioned static/dynamic unclamped | `qstqjup2` | 0.783669 | 0.0675422 | 0.0271374 |
| V-JEPA static/dynamic unclamped | `x4uc6va3` | 0.842614 | 0.0725462 | 0.154602 |

Browser export from the clamped V-JEPA run:

```text
outputs/browser_exports/20260427T114103Z_ablate-time-static-dynamic-96-32-crossattn4-precomputed-vjepa2-1-vitb-384-rgb-uniform-strong-cam-yhezacn8/manifest.json
```

## Interpretation

This does not show that V-JEPA is useless. It shows the opposite for this
same-source overfit lane: V-JEPA plus static/dynamic split survives the camera
clamp and is the best source-view result in this matrix.

The previous red flag was still valid because unconstrained camera motion can
fake scene dynamics by moving the camera path instead of the splats. The
camera-clamped result answers that specific concern:

- V-JEPA clamp beats unconditioned clamp by `0.0380` Eval/Loss, `+0.1966` SSIM,
  and `+4.15 dB` PSNR.
- V-JEPA clamp slightly beats V-JEPA unclamped (`0.0425` vs `0.0455` Eval/Loss)
  while cutting adjacent camera rotation from `0.291 deg` to `0.137 deg`,
  adjacent translation from `0.0355` to `0.0164`, and mean camera rotation from
  `6.80 deg` to `1.12 deg`.
- The unconditioned clamp got worse versus unconditioned unclamped
  (`0.0805` vs `0.0588` Eval/Loss), which says the clamp is a real constraint,
  not a no-op.
- V-JEPA still has more adjacent camera motion than unconditioned clamp
  (`0.137 deg` vs `0.019 deg`, `0.0164` vs `0.0018`), but this motion is far
  below the old V-JEPA red flag and the mean camera rotation is no longer high.

The result is still source-view overfit evidence. It supports the local claim
"precomputed V-JEPA features help the strong static/dynamic token decoder under
tighter camera bounds." It does not prove scene-generalization, novel-view
truth, or true 3D dynamics. The next selector for that is held-out camera /
multiview or scene-distinct evaluation.

## Follow-Up

Useful next ablations, if we continue instead of wrapping:

- Keep this clamp as the default camera-control setting for future V-JEPA
  static/dynamic same-source probes.
- Run the same clamp on a 256px end-to-end matrix before comparing to
  non-V-JEPA feature paths.
- Promote the V-JEPA clamp run to the current best tiny same-source recipe if
  the visual media agrees with the scalar metrics.
- Do not remove the unconditioned baseline; it remains the guardrail showing
  how much same-source overfit can be solved from time plus decoder capacity.
