# STAR UVT Feature1/Probe40 Whole-Graph Profile

## Context

The feature1/probe40 1300->1400 continuation kept improving quality but slowed
from the 1250->1300 row. A matched repeat reproduced the slower trainer row at
`1.711s/step` with zero tile overflow and `68/45/128` max/p95/cap tile count.

This profile gate asks whether the slowdown is intrinsic to the current
target-grid plus frozen RGB-probe objective, and where the backward time sits
when we split image-space loss VJP from the STAR UVT Metal renderer backward.

## Implementation

Added:

```text
research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py
```

The script loads one or more existing first-class trainer configs, restores the
configured checkpoint, renders each frame chunk with the forward-only STAR UVT
feature Metal path, detaches the rendered feature image, runs the target-grid
feature loss and frozen RGB-probe loss to get `grad_feature_image`, and then
calls `direct_atomic_feature_backward` manually with the selected renderer
backward mode.

It records separate timings for:

- render forward
- target-grid prep
- feature-loss forward
- frozen RGB-probe forward/loss
- image-space loss backward
- STAR UVT Metal renderer backward

It intentionally excludes optimizer time and media/checkpoint work, so it is a
split diagnostic, not an end-to-end trainer replacement.

## Command

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.jsonc \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat.jsonc \
  --warmup 1 \
  --repeat 3 \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.json \
  --out-md outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md
```

## Results

Output:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.json
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md
```

The gate passed for both rows.

| source checkpoint | trainer step ms | manual total ms | render ms | image-loss backward ms | renderer backward ms | renderer share of backward | overflow | max/p95/cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global step 1250 | `1285.0` | `1565.9` | `605.5` | `159.8` | `700.9` | `81.4%` | `0` | `63/41/128` |
| global step 1300 | `1710.5` | `1504.0` | `581.2` | `154.3` | `669.1` | `81.3%` | `0` | `63/42/128` |

## Interpretation

The current target-grid/frozen-probe objective is no longer dominated by the
same image-space `FeatureToColor`/loss VJP seen in earlier RGB-target rows.
Under this objective, renderer backward is about `81%` of manual backward, and
the frozen-probe/image-loss backward is about `19%`.

The isolated split does **not** reproduce the trainer slowdown. The 1300-source
checkpoint is slightly faster than the 1250-source checkpoint in this manual
profile, while the saved trainer references are `1285.0ms/step` for 1250->1300
and `1710.5ms/step` for the matched 1300->1400 timing repeat.

That means the remaining speed question is not tile overflow and not obviously
geometry becoming more expensive at step 1300. The next gate should either:

- add an end-to-end trainer timing trace with per-step samples, or
- move directly to native VJP / scalar fixedbin work, since the current
  objective remains quality-positive and renderer backward is the manual
  majority bucket.

## Docs Updated

- `README.md`
- `PROJECT_INDEX.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`

## Validation

- `py_compile` passed for
  `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py`.
- Whole-graph profile invariants passed: two rows, both `pass=true`, zero tile
  overflow, renderer backward share above `80%`, and the isolated manual 1300
  row did not reproduce the trainer slowdown.
- `agent_notes/key_learnings.md` remains `199` lines.
- No active `star_uvt_feature1_wholegraph_profile.py` process remained.
