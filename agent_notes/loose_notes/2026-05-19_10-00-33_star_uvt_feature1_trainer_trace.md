# STAR UVT Feature1/Probe40 Trainer Timing Trace

## Context

The matched 1300->1400 timing repeat reproduced the slower trainer row at
`1.711s/step`, but the manual whole-graph split did not reproduce that slowdown:
it reported `1565.9ms` manual total at step 1250 and `1504.0ms` at step 1300.
That meant the next gate needed an end-to-end trainer trace with per-step
samples.

## Implementation

`src/train/train_star_uvt_feature_overfit.py` now writes two timing trace fields
to its result JSON:

- `step_timings_ms`: the per-step timing dictionaries already accumulated by
  the training loop
- `timing_trace_summary_ms`: min/max/first/last per timing bucket

Added two no-media timing configs:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_trace20_from1250.jsonc
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_trace20_from1300.jsonc
```

The trace configs set `require_loss_decrease=false` so timing probes can exit
cleanly while still preserving the row's own `pass` flag.

Added report script:

```text
research_experiments/star_uvt_feature_tubes/star_uvt_feature1_trainer_trace_report.py
```

## Commands

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_trace20_from1250.jsonc

PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_trace20_from1300.jsonc

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_trainer_trace_report.py
```

Offline W&B:

```text
wandb/offline-run-20260519_095606-ylo7rlho
wandb/offline-run-20260519_095815-0es8wfdo
```

## Artifacts

```text
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace20_from1250.json
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace20_from1300.json
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.json
outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md
```

## Results

| trace | pass | global | feature loss | probe PSNR | step mean | no-first mean | render no-first | loss-region no-first | backward no-first | max step | overflow | max/p95/cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1250->1270 | `true` | `1250->1270` | `0.638803->0.634913` | `21.933->21.947` | `1722.0ms` | `1705.3ms` | `614.5ms` | `127.6ms` | `919.8ms` | `2039.7ms` | `0` | `63/42/128` |
| 1300->1320 | `false` | `1300->1320` | `0.632124->0.632249` | `21.965->21.818` | `1859.9ms` | `1850.7ms` | `656.5ms` | `152.5ms` | `992.5ms` | `2472.8ms` | `0` | `63/43/128` |

The no-first deltas for 1300-source minus 1250-source:

- step: `+145.4ms`
- render forward: `+42.0ms`
- combined loss region: `+24.9ms`
- backward: `+72.7ms`
- feature-target sub-bucket: `+8.2ms`
- RGB-probe sub-bucket: `+4.2ms`

The 1300-source trace has a late objective spike at global step `1318`:
feature loss jumps from `0.630693` to `0.632314`, and probe loss jumps from
`0.006273` to `0.006597`. The run exits cleanly because the trace config does
not assert loss decrease, but the row's own `pass=false` status is preserved.

## Interpretation

The end-to-end trainer trace does reproduce a real 1300-source slowdown, unlike
the isolated manual whole-graph split. The slowdown is not a first-step-only
optimizer warmup artifact and not tile overflow. It is spread across render
forward, the combined target/probe loss region, and backward.

The late spike makes this more than a pure speed issue. Continuing the exact
feature1/probe40 schedule should either trace the autograd/MPS state around
global step `1318`, or move to the shader path the evidence already points to:
native VJP / scalar fixedbin feature backward.

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
  `star_uvt_feature1_trainer_trace_report.py`, and
  `star_uvt_feature1_wholegraph_profile.py`.
- Both trace configs resolve with `steps=20` and
  `train.require_loss_decrease=false`.
- Trainer trace report invariants passed: two rows, first row `pass=true`,
  second row `pass=false`, zero tile overflow in both rows, no-first step
  delta above `100ms`, global step `1318` recorded as the quality spike, and
  both raw trace JSONs contain `20` per-step timing samples plus
  `timing_trace_summary_ms`.
- `agent_notes/key_learnings.md` remains `199` lines.
