# STAR UVT First-Class Backward Split

Date: 2026-05-18 23:52 +0700

## Goal

The previous STAR UVT feature notes identified backward as the dominant cost,
but the first-class trainer timing lumped the image-space `FeatureToColor` /
loss backward together with the Metal renderer backward. This session added a
diagnostic gate that separates those parts on the actual checked-in configs.

## New Script

Added:

```text
research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py
```

The script loads a normal `arch=star_uvt_feature_overfit` config, target video,
model initialization, and colorizer. It then times:

- STAR UVT feature render forward
- `FeatureToColor` plus loss forward
- image-space backward to `grad_feature_image` and `grad_alpha`
- manual Metal STAR UVT feature backward with those image gradients

It writes JSON and markdown tables. This is a diagnostic split, not a full
trainer replacement: it excludes optimizer time and media/logging.

## Commands

Smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  --config src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc \
  --warmup 0 --repeat 1 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_smoke_8f64.json \
  --out-md outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_smoke_8f64.md
```

256px cap256 mode comparison:

```bash
STAR_UVT_TILE_CAPACITY=256 PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_256_gradcache_chunk4_32768t_alpha1_72_cap256_20step.jsonc \
  --modes feature_direct_gradcache,feature_direct_gradcache_reduce,feature_direct_gradcache_reduce_vec4 \
  --warmup 1 --repeat 2 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.json \
  --out-md outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.md
```

512px gradcache split:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_4096t_2step.jsonc \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step.jsonc \
  --warmup 1 --repeat 2 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.json \
  --out-md outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.md
```

512px no-pre-norm A/B:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_4096t_2step.jsonc \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step.jsonc \
  --colorize-pre-norm false --warmup 1 --repeat 2 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.json \
  --out-md outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.md

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 WANDB_MODE=offline .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_2step.jsonc
```

## Results

Artifacts:

```text
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.md
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.md
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.md
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_2step.json
```

256px / 64f / 32768t / alpha>=1/72 / cap256:

| mode | manual total | color/loss backward | renderer backward | renderer share of backward |
| --- | ---: | ---: | ---: | ---: |
| `feature_direct_gradcache` | `1994.7ms` | `987.8ms` | `553.0ms` | `35.9%` |
| `feature_direct_gradcache_reduce` | `1764.8ms` | `887.8ms` | `526.6ms` | `37.2%` |
| `feature_direct_gradcache_reduce_vec4` | `1727.7ms` | `867.4ms` | `494.7ms` | `36.3%` |

512px / 64f / gradcache:

| tubes | manual total | color/loss backward | renderer backward | renderer share of backward |
| ---: | ---: | ---: | ---: | ---: |
| `4096` | `6566.8ms` | `3775.0ms` | `1071.7ms` | `22.1%` |
| `8192` | `5372.8ms` | `3430.1ms` | `700.0ms` | `16.9%` |

512px / 64f / gradcache, no pre-norm:

| tubes | manual total | color/loss backward | renderer backward | renderer share of backward |
| ---: | ---: | ---: | ---: | ---: |
| `4096` | `2018.9ms` | `317.1ms` | `674.8ms` | `68.0%` |
| `8192` | `2403.5ms` | `400.6ms` | `751.5ms` | `65.2%` |

512px / 64f / 8192t / no-pre-norm actual trainer:

| row | value |
| --- | ---: |
| pass | `true` |
| loss | `0.33817 -> 0.33764` |
| mean step | `3715.4ms` |
| mean backward | `1585.6ms` |
| mean forward | `1268.3ms` |
| mean colorize/loss | `440.9ms` |
| overflow | `0` |

All rows passed finite/no-overflow checks. The 512px timing remains noisy, but
the split is not ambiguous: the image-space colorizer/loss backward dominates
the first-class graph at 512px.

## Interpretation

This changes the next-step plan. The STAR UVT feature renderer still needs a
real optimized fixedbin/two-pass feature-gradient path, but renderer-only work
cannot recover most of the 512px step time because renderer backward is only
about `17-22%` of backward there. The dense `FeatureToColor`/loss VJP is the
main 512px cost.

The no-pre-norm A/B shows that the current LayerNorm pre-norm is a large part
of the image-space VJP cost. It is the first practical 512px speed candidate:
the actual 8192t first-class row drops from the previous pre-norm `7.94s/step`
and `4.88s` backward to `3.72s/step` and `1.59s` backward. This should not be
promoted on speed alone; it needs a longer media/quality overfit comparison.

Do not read the synthetic direct-kernel benchmark as the whole training
bottleneck. It is still useful for shader parity and renderer micro-optimizing,
but first-class speed claims now need the backward split gate too.

Practical next options:

- optimize fixedbin/two-pass feature-gradient accumulation for the renderer
- add a native image-space VJP for the current simple colorizer/loss path
- try a better handoff that avoids both dense F32 image-gradient backprop and
  per-pixel renderer-side `W^T` over all feature channels
- run the no-pre-norm/simple colorizer variant for 20-50 steps with media
  before building another complex handoff

## Docs Updated

- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `PROJECT_INDEX.md`
- `README.md`
- `agent_notes/key_learnings.md`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_2step.jsonc`

2026-05-19 addendum: the 20-step media A/B requested here was run in
`agent_notes/loose_notes/2026-05-19_00-24-59_star_uvt_no_prenorm_quality_ab.md`.
It confirmed no-pre-norm is faster but did not promote it on quality.

## Validation

- `py_compile` passed for:
  - `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
  - `research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py`
  - `src/train/train_star_uvt_feature_overfit.py`
- `trainer_entry_for_config` loaded all `26`
  `src/train_configs/star_uvt_feature*.jsonc` configs through
  `train_star_uvt_feature_overfit`.
- Artifact sanity passed for the backward-breakdown JSONs, no-pre-norm trainer
  JSON, and regenerated scale report. The scale report now carries the current
  no-pre-norm row (`3715.4ms` step, `1585.6ms` backward), not the overwritten
  first-run timing.
- `agent_notes/key_learnings.md` remains under the cap at `195` lines.
- `git diff --check` passed.
