# STAR UVT Selected-Shader 128/256/512 Scale Gate

## Goal

After selecting no-pre-norm `feature_direct_gradcache_reduce_vec4` as the
current 512px feature-tube speed diagnostic, run a first-class scale gate at
128/256/512 to check whether that selection is consistent across resolution.

## Setup

All rows use:

- `arch=star_uvt_feature_overfit`
- `test_data/test_video_384_128_6fps.mp4`
- 64 frames
- 8192 F32 feature tubes
- `frame_chunk_size=2`
- `colorize.pre_norm=false`
- 20 optimizer steps

Modes compared:

- `feature_direct_gradcache`
- `feature_direct_gradcache_reduce_vec4`

## Support Validity Finding

128px with the same 8192 tubes is crowded because the tile grid is much smaller.
The attempted 128px validity ladder failed before producing JSON at:

- cap128, default alpha: `2053` overflow tiles
- cap256, default alpha: `1487` overflow tiles
- cap256, `alpha>=1/72`: `160` overflow tiles

The first valid 128px row used cap256 plus `alpha>=1/32`, ending with max tile
`232/256` and p95 tile `205`.

## Results

| size | mode | cap | alpha | step s | backward s | forward s | color/loss s | end PSNR | overflow | max tile | p95 tile |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | `feature_direct_gradcache` | 256 | 0.03125 | 0.668 | 0.375 | 0.146 | 0.085 | 5.300 | 0 | 232 | 205 |
| 128 | `feature_direct_gradcache_reduce_vec4` | 256 | 0.03125 | 0.658 | 0.378 | 0.138 | 0.082 | 5.300 | 0 | 232 | 205 |
| 256 | `feature_direct_gradcache` | 128 | 0.003922 | 1.112 | 0.564 | 0.338 | 0.134 | 5.250 | 0 | 97 | 81 |
| 256 | `feature_direct_gradcache_reduce_vec4` | 128 | 0.003922 | 1.069 | 0.561 | 0.302 | 0.132 | 5.250 | 0 | 97 | 81 |
| 512 | `feature_direct_gradcache` | 128 | 0.003922 | 2.858 | 1.327 | 1.053 | 0.344 | 4.941 | 0 | 36 | 22 |
| 512 | `feature_direct_gradcache_reduce_vec4` | 128 | 0.003922 | 2.491 | 1.184 | 0.911 | 0.287 | 4.941 | 0 | 36 | 22 |

## Decision

Keep `feature_direct_gradcache_reduce_vec4` as the selected 512px fast
diagnostic. Do not generalize it blindly:

- at 128px, vec4 is a step-time tie and a slight backward loss
- at 256px, vec4 is only a small win
- at 512px, vec4 is a meaningful first-class win

The low-resolution support result matters: 128px needs much stronger pruning
than 256/512 for the same tube count, so future scale tables must include tile
validity columns rather than only timing.
