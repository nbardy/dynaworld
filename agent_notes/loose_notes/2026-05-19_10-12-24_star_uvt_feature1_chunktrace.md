# STAR UVT Feature1/Probe40 Chunk Trace

## Context

The end-to-end trainer timing trace reproduced the 1300-source slowdown and
showed a late quality spike at global step `1318`. Step-level timing did not
make `1318` look like the slowest timing outlier, so this gate traces chunk
losses and chunk timings around global steps `1317`, `1318`, and `1319`.

## Implementation

`src/train/train_star_uvt_feature_overfit.py` now supports:

```json
"train": {
  "trace_global_steps": [1317, 1318, 1319]
}
```

The trainer writes:

- `chunk_trace_global_steps`
- `chunk_traces`

Each traced step records per-frame-chunk weighted loss, feature loss, probe
loss, render time, target/probe time, and backward time. The trace is opt-in;
normal configs default to an empty trace list.

Added config:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_chunktrace20_from1300.jsonc
```

Added report script:

```text
research_experiments/star_uvt_feature_tubes/star_uvt_feature1_chunktrace_report.py
```

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_chunktrace20_from1300.jsonc

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_chunktrace_report.py
```

Offline W&B:

```text
wandb/offline-run-20260519_100920-sudhs7r2
```

## Artifacts

```text
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace20_from1300.json
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.json
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md
```

## Results

The traced run exits cleanly and preserves `pass=false`.

| global step | loss | feature loss | probe loss | step ms | render ms | backward ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1317` | `0.881625` | `0.630693` | `0.006273` | `2072.0` | `715.3` | `1144.6` |
| `1318` | `0.896184` | `0.632314` | `0.006597` | `1827.0` | `650.7` | `992.0` |
| `1319` | `0.895442` | `0.632250` | `0.006580` | `1714.4` | `612.5` | `933.8` |

Spike delta from `1317 -> 1318`:

- weighted loss: `+0.014559`
- feature loss: `+0.001620`
- probe loss: `+0.000323`
- positive/negative chunks: `27/5` out of `32`

Weighted-loss delta by frame range:

- frames `0-15`: `+0.006475` (`44.5%` of spike)
- frames `16-31`: `+0.003956` (`27.2%`)
- frames `32-47`: `+0.002432` (`16.7%`)
- frames `48-63`: `+0.001696` (`11.6%`)

Largest chunk deltas:

- frame `0`: `+0.001703`
- frame `18`: `+0.001327`
- frame `2`: `+0.001123`
- frame `6`: `+0.000899`
- frame `22`: `+0.000855`

## Interpretation

The `1318` spike is distributed across most chunks, not a single bad frame
chunk or timing outlier. The elevated loss persists at `1319`, so this looks
like an optimizer/objective-state jump rather than a transient render bug.

The next useful gate is no longer tile capacity or more trace plumbing. Either
checkpoint/lower LR before this region if continuing the same objective, or move
to the native VJP/scalar fixedbin shader path because renderer backward remains
the manual majority bucket.

## Docs Updated

- `README.md`
- `PROJECT_INDEX.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`

## Validation

- `py_compile` passed for `src/train/train_star_uvt_feature_overfit.py`,
  `star_uvt_feature1_chunktrace_report.py`, and
  `star_uvt_feature1_trainer_trace_report.py`.
- Chunk-trace config resolves with `steps=20`, trace steps
  `[1317, 1318, 1319]`, and `require_loss_decrease=false`.
- Chunk-trace invariants passed: raw trace row keeps `pass=false`, traced steps
  are `[1317, 1318, 1319]`, each traced step has `32` chunks, report says the
  spike is distributed rather than localized, positive/negative chunks are
  `27/5`, weighted-loss delta is above `0.014`, and the first-quarter share is
  above `40%`.
- `agent_notes/key_learnings.md` remains `199` lines.
