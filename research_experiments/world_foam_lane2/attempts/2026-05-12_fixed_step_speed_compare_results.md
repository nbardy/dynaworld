# Fixed-Step Speed Compare Results

Command for the main run:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --steps 8 \
  --warmup-steps 2 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/fixed_step_speed_compare_default.json
```

The main run produced all 128px rows and failed on STAR-UVT at `256x32` with
`RuntimeError('Invalid buffer size: 12.00 GiB')`. The missing dynamic and World
Foam `256x32` rows were run separately with STAR skipped.

| shape | renderer | loaded f | mean step s | mean render s | measured 8-step s |
|---|---:|---:|---:|---:|---:|
| 128x128 8f | star_uvt | 8 | 0.144 | 0.005 | 1.155 |
| 128x128 8f | free_dynamic_splats | 8 | 0.301 | 0.144 | 2.404 |
| 128x128 8f | world_foam | 8 | 0.106 | 0.044 | 0.849 |
| 128x128 16f | star_uvt | 16 | 0.342 | 0.008 | 2.740 |
| 128x128 16f | free_dynamic_splats | 16 | 0.624 | 0.298 | 4.990 |
| 128x128 16f | world_foam | 16 | 0.201 | 0.096 | 1.607 |
| 128x128 32f | star_uvt | 32 | 0.525 | 0.007 | 4.199 |
| 128x128 32f | free_dynamic_splats | 32 | 1.141 | 0.582 | 9.125 |
| 128x128 32f | world_foam | 32 | 0.322 | 0.157 | 2.579 |
| 256x256 32f | star_uvt | 32 | failed | failed | failed |
| 256x256 32f | free_dynamic_splats | 32 | 0.545 | 0.223 | 4.360 |
| 256x256 32f | world_foam | 32 | 1.234 | 0.606 | 9.873 |

Interpretation:

- The 12 GiB failure is STAR-UVT full-sequence backward workspace, not the raw
  video tensor and not World Foam.
- Dynamic GSplats did not hit the memory blow-up in its isolated `256x32` run.
- World Foam also completed `256x32`, but it is not yet a full trainer and its
  step renders all train-camera rays, while the STAR/dynamic rows render one
  train camera per step.
- At 128px, World Foam step time grows `0.106 -> 0.201 -> 0.322` for `8 -> 16
  -> 32` frames, which is sublinear with frame count in this fixed-geometry
  accounting. That does not yet prove a full train loop win.

Follow-up STAR direct-backward check:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --cases 128x8,128x16,128x32,256x32 \
  --steps 8 \
  --warmup-steps 2 \
  --skip-dynamic \
  --skip-world-foam \
  --uvt-sample-emission-mode direct_atomic \
  --out-json /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/world_foam_lane2/results/fixed_step_speed_compare_star_directatomic.json \
  --input-dir /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/world_foam_lane2/results/fixed_step_speed_compare_inputs_directatomic
```

| shape | renderer | loaded f | mean step s | mean render s | measured 8-step s |
|---|---:|---:|---:|---:|---:|
| 128x128 8f | star_uvt direct_atomic | 8 | 0.034 | 0.003 | 0.268 |
| 128x128 16f | star_uvt direct_atomic | 16 | 0.039 | 0.004 | 0.314 |
| 128x128 32f | star_uvt direct_atomic | 32 | 0.038 | 0.003 | 0.304 |
| 256x256 32f | star_uvt direct_atomic | 32 | 0.123 | 0.005 | 0.984 |

Interpretation update:

- The STAR 12 GiB failure is not inherent to spacetime tubes. It is the
  historical sample-emission backward path allocating
  `tile_count * tile_pixels * tile_capacity` gradient rows before reduction.
- The existing `direct_atomic` path uses one direct per-tube gradient row and
  clears `256x32`.
- Chunked backward should be treated as a fallback, not the main design. The
  better target is a FasterGS-style deterministic per-Gaussian/per-tube backward
  that keeps the direct path's memory shape without nondeterministic float
  atomics.

Default switch and isolated scaling reruns:

- `fixed_step_speed_compare.py` and `multicam_train_step_timing_probe.py` now
  default STAR timing to `uvt_sample_emission_mode=direct_atomic`.
- Reproduce the historical failure path with
  `--uvt-sample-emission-mode atomic_append`.
- The no-flag STAR rerun wrote
  `results/fixed_step_speed_compare_star_directatomic_default_no_flag.json`.
- The dynamic-only rerun wrote
  `results/fixed_step_speed_compare_dynamic_only_matrix.json`.
- A broad all-algorithm direct-atomic run was stopped after it stayed CPU-bound
  past nine minutes; isolated per-renderer rows are the cleaner comparison for
  now.
- A World Foam isolated full matrix was also stopped after seven minutes with no
  output file; use the earlier 128px rows plus the separate 256px/32f row until
  the World Foam runner gets progress logging or a narrower one-camera mode.

No-flag STAR default/direct-atomic rows:

| shape | renderer | loaded f | mean step s | mean render s | measured 8-step s |
|---|---:|---:|---:|---:|---:|
| 128x128 8f | star_uvt direct_atomic default | 8 | 0.024 | 0.002 | 0.193 |
| 128x128 16f | star_uvt direct_atomic default | 16 | 0.034 | 0.003 | 0.274 |
| 128x128 32f | star_uvt direct_atomic default | 32 | 0.033 | 0.003 | 0.265 |
| 256x256 32f | star_uvt direct_atomic default | 32 | 0.067 | 0.005 | 0.533 |

Dynamic-only isolated rows:

| shape | renderer | loaded f | mean step s | mean render s | measured 8-step s |
|---|---:|---:|---:|---:|---:|
| 128x128 8f | free_dynamic_splats | 8 | 0.131 | 0.054 | 1.047 |
| 128x128 16f | free_dynamic_splats | 16 | 0.252 | 0.104 | 2.018 |
| 128x128 32f | free_dynamic_splats | 32 | 0.444 | 0.168 | 3.549 |
| 256x256 32f | free_dynamic_splats | 32 | 0.500 | 0.189 | 4.002 |

128px frame-count growth:

```text
STAR direct/default: 8f -> 32f = 1.37x for 4x frames
dynamic isolated:    8f -> 32f = 3.39x for 4x frames
World Foam prior:    8f -> 32f = 3.04x for 4x frames
STAR old sample:     8f -> 32f = 3.63x for 4x frames
```

Sublinear gain table:

```text
Renderer                  8f step  16f step  16f growth  16f vs linear  32f step  32f growth  32f vs linear
------------------------  -------  --------  ----------  -------------  --------  ----------  -------------
STAR direct/default        0.024s    0.034s        +42%           +29%    0.033s        +37%           +66%
Dynamic GSplats isolated   0.131s    0.252s        +93%            +4%    0.444s       +239%           +15%
World Foam prior           0.106s    0.201s        +89%            +5%    0.322s       +204%           +24%
STAR old sample            0.144s    0.342s       +137%           -19%    0.525s       +263%            +9%
```

Speedup table for STAR direct/default:

```text
Shape     STAR step  Baseline                  Base step  Speedup  Time reduction
--------  ---------  ------------------------  ---------  -------  --------------
128x8f       0.024s  Dynamic GSplats              0.131s     5.4x             82%
128x8f       0.024s  World Foam                   0.106s     4.4x             77%
128x8f       0.024s  STAR old sample              0.144s     6.0x             83%
128x16f      0.034s  Dynamic GSplats              0.252s     7.4x             86%
128x16f      0.034s  World Foam                   0.201s     5.9x             83%
128x16f      0.034s  STAR old sample              0.342s    10.0x             90%
128x32f      0.033s  Dynamic GSplats              0.444s    13.4x             93%
128x32f      0.033s  World Foam                   0.322s     9.7x             90%
128x32f      0.033s  STAR old sample              0.525s    15.9x             94%
256x32f      0.067s  Dynamic GSplats              0.500s     7.5x             87%
256x32f      0.067s  World Foam                   1.234s    18.5x             95%
```

Camera scope:

- this data is multicam and the saved STAR run reports train cameras
  `camera_0006` and `camera_0014`
- the STAR timing loop picks one camera per step with `view = step % view_count`
- in `view_sequence` mode, that one selected camera renders all loaded frames
  for the step
- therefore this is not a fixed single camera position, but it is also not a
  simultaneous all-camera batch timing
- current STAR sequence projection selects `w2c[view, 0]` and `K[view, 0]`, so
  it does not yet handle per-frame camera motion inside one rendered sequence
- heldout-camera quality and direct-atomic determinism are still separate gates

Naive per-frame-camera stress test:

```text
Mode                    8f step  16f step  32f step  8f->32f growth
----------------------  -------  --------  --------  --------------
STAR static-view seq     0.027s    0.029s    0.033s           1.25x
STAR per-frame loop      0.122s    0.219s    0.399s           3.26x
```

The per-frame loop uses the same direct STAR backend but calls projection/render
once per frame with `select_K_for_view_time(...)` and
`select_w2c_for_view_time(...)`. It supports the per-frame-camera contract as a
stress test, but it discards sequence-level amortization and is not the target
variable-camera implementation.

Same-step quality smoke:

```text
Shape/steps       Method                  Train PSNR  Heldout PSNR  Train loop
----------------  ----------------------  ----------  ------------  ----------
128px 8f / 100    STAR direct_atomic          15.408        13.061      3.623s
128px 8f / 100    Dynamic GSplats             7.257         7.123      2.562s
```

This is a smoke only: same step count, but STAR uses temporal-window sequence
training while the existing dynamic comparator trains sampled frames. The fixed
step speed table remains timing evidence, not a PSNR-parity proof.

Interpretation update: for this fixed-step timing harness, STAR direct/default
is now the clearest sublinear frame-count result. The remaining blocker is not
memory; it is whether a direct/reduced backward can be made deterministic and
quality-stable enough to become the reporting/training default outside speed
probes.
