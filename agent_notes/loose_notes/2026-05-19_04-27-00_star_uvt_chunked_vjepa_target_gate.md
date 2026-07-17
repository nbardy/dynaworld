# STAR UVT Chunked V-JEPA Target Gate

Date: 2026-05-19 04:27:00

## Goal

Continue the STAR UVT fast feature-shader plan by replacing full dense V-JEPA
target materialization with a chunked target path that can be used before longer
512px or 300-set runs.

## Work Done

- Added `FeatureTargetTensor` to `src/train/train_star_uvt_feature_overfit.py`.
- Added `feature_target.materialization` with values `dense` and `chunked`.
- Added exact chunk-grid adaptation for `trilinear` and `nearest`; CPU
  equivalence checks pass against dense `F.interpolate`, including
  channel-standardization stats.
- Added streaming channel-standardization stats for chunked targets.
- Added `feature_target_ms` to timing rows, synchronized on MPS so the target
  bucket reflects device work.
- Replaced the brittle checked 2-step chunked config with a 5-step LR `0.005`
  gate:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_chunkedtarget_lr005_5step.jsonc`.
- Refreshed the V-JEPA bridge audit so it points at the chunked gate.

## Result

Passing JSON:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_chunkedtarget_lr005_5step.json
```

Key numbers:

```text
pass: true
loss: 1.000014 -> 0.999545
mean step: 3743.3 ms
mean render forward: 815.6 ms
mean feature-target chunk/loss: 1734.2 ms
mean backward: 1077.4 ms
tile overflow: 0
max tile load: 33 / 128
```

Feature target path:

```text
vjepa_tokens [1,8192,768]
channel-adapted source [32,32,16,16]
token_grid_shape [32,16,16]
adapted logical target [64,32,512,512]
materialization chunked
materialization_chunk_size 2
```

## Interpretation

The target-memory blocker is now addressed for training state: the trainer no
longer needs to keep the full dense 2 GiB target tensor resident. The cost moved
into per-step target chunk generation, which is now a visible timing bucket and
is larger than renderer backward on this row.

The earlier 2-step gates are too noisy for hard loss-decrease assertions at
512px V-JEPA target scale. Keep the 5-step LR `0.005` chunked gate as the
current regression.

The selected `star-feature-512-fast` helper is still RGB-target
`FeatureToColor`, not cached V-JEPA. The chunked V-JEPA config is a separate
cached-feature scale regression.

## Remaining Work

- Compare this STAR V-JEPA target path against matched Gaussian/token
  precomputed V-JEPA rows.
- Optimize target chunk interpolation or move V-JEPA target evaluation closer to
  the renderer/loss before longer runs.
- Keep Gate 4 quality claims separate; this is not a source-view quality win.
