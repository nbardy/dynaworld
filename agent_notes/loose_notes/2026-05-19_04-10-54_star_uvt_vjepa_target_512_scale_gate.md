# STAR UVT Real V-JEPA Target 512px Scale Gate

Date: 2026-05-19 04:10:54

## Goal

Continue the STAR UVT fast feature-shader plan by taking the real V-JEPA target
bridge beyond the 8f/64px smoke and testing it at the selected 512px
no-pre-norm reduce-vec4 renderer scale.

## Work Done

- Added the scale config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_2step.jsonc`.
- Ran the exact 64f/512px/8192t/F32/chunk2 V-JEPA target gate.
- Hit a pre-rasterizer target-prep failure:
  `RuntimeError: Invalid buffer size: 48.00 GiB`.
- Patched `src/train/train_star_uvt_feature_overfit.py` so feature channel
  adaptation runs before temporal/spatial upsampling.
- Moved benchmark JSON persistence before requirement assertions so failed
  gates leave evidence.
- Reran the same config and got a passing cache-hit row.
- Refreshed the generated bridge audit:
  `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`.

## Result

Passing JSON:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_2step.json
```

Key numbers:

```text
pass: true
loss: 1.000014 -> 0.999547
mean step: 3010.2 ms
mean render forward: 1084.9 ms
mean feature-target loss prep: 344.1 ms
mean backward: 1373.3 ms
tile overflow: 0
max tile load: 33 / 128
```

Feature target path:

```text
vjepa_tokens [1,8192,768]
token_grid_shape [32,16,16]
adapted target [64,32,512,512]
channel_adapter_applied_before_grid true
```

## Interpretation

The real V-JEPA STAR target bridge is no longer just a smoke. It reaches
64f/512px scale and exercises the selected reduce-vec4 feature shader without
tile overflow.

The selected `star-feature-512-fast` helper is still not a V-JEPA route; it
remains RGB-target `FeatureToColor` training. The new V-JEPA target config is a
separate scale regression to compare against the Gaussian/token precomputed
V-JEPA family.

The channel-before-grid fix is a general shader-pipeline lesson: when adapting
high-dimensional cached features to a lower-dimensional renderer target, reduce
channels before dense grid expansion. Doing the reverse created a 48 GiB
temporary for a 2 GiB final target.

## Remaining Work

- Replace full dense target materialization with chunked/lazy target adaptation
  before longer 512px or 300-set runs.
- Run a matched comparison against existing Gaussian/token precomputed V-JEPA
  rows.
- Decide whether the next speed win comes from scalar fixedbin/tile-slot
  feature-gradient accumulation or a faster image-space VJP/handoff.
- Do not promote quality from this two-step row; it is a bridge and timing gate.
