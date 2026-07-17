# Trained High-Motion Trace Scaling

## Context

The previous high-motion projective reports had two useful but incomplete
layers:

- video-derived motion-centroid UV visibility splits, which were real video
  diagnostics but not trained STAR UVT geometry;
- trainer-frame cache reuse, which used the actual trainer route but with a
  synthetic tensor loader.

This pass added a third layer: train a tiny STAR UVT feature model on the
checked-in high-motion smoke video, save the checkpoint, reload the learned
tensors, then compile them into projective interval trace-cell atlases over
growing frame prefixes.

## Artifact

Script:

```text
research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py
```

Outputs:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/trained_high_motion_checkpoint.pt
```

Command:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --frame-counts 4,8,16 \
  --steps 4 \
  --size 32 \
  --tube-count 64 \
  --run-metal-timing \
  --include-per-frame-baseline \
  --timing-iterations 2 \
  --timing-warmup 1 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling
```

## Evidence

The trainer route uses the checked-in high-motion smoke video:

```text
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
```

Tiny smoke settings:

```text
size = 32
tube_count = 64
steps = 4
feature_dim = 3
projective_interval.enabled = true
```

The trainer checkpoint row:

```text
loss: 0.298236 -> 0.296121
pass: true
tile_overflow_sum: 0
cache rebuilds/live updates/staleness checks: 1 / 3 / 3
```

Compiled trained checkpoint geometry:

```text
frames                     4      8       16
trace_count                64     64      64
dense_per_frame_tile_pairs 1542   3061    6016
interval_trace_entries     392    477     573
interval/dense tile ratio  0.254  0.156   0.095
fallback_fraction          0      0       0
```

The regenerated artifact also runs native interval Metal forward/backward on
each compiled atlas and compares against replaying one single-frame interval
atlas per frame:

```text
frames                         4       8       16
shared_interval_entries        392     477     573
per_frame_replay_entries       392     1956    3862
shared_forward_ms              127.2    25.3    57.0
per_frame_forward_ms           147.0   179.4   355.2
shared_backward_ms             111.1    43.9    60.3
per_frame_backward_ms          126.4   183.3   367.9
```

The important ratio:

```text
dense per-frame tile pairs grow 3.90x
interval trace entries grow       1.46x
16f shared/per-frame forward      0.160x
16f shared/per-frame backward     0.164x
```

## Interpretation

This replaces the old "motion centroid only" high-motion evidence with a real
saved trainer-checkpoint geometry gate. It is still tiny and should not be
used as a final wall-clock claim, especially because the backward timing is
noisy at this scale. It still makes the core objective more true: learned STAR
UVT tensors from a real video can be pulled into a reusable sensor-time trace
atlas whose world-side interval work grows slower than dense per-frame tile
work, and that atlas beats a same-checkpoint framewise replay route in the
tiny native interval forward/backward diagnostic.

## Verification

```text
py_compile projective_trained_high_motion_trace_scaling_benchmark.py: passed
benchmark run: status ok
focused high-motion report + trainer cache tests: 3 passed in 94.63s
```

## Next

Scale this beyond the tiny smoke: larger tube count, larger image size, and
warm-timed wall-clock rows. Then use the fallback/cell-growth numbers to decide
whether oblique/fiber halfspace cells are necessary or whether the current
trace-cell atlas is enough for this class of high-motion source-view video.

## Expansion: 64px / 128-Tube Rerun

I reran the same saved-checkpoint interval-vs-per-frame gate at a slightly
larger smoke scale:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --frame-counts 4,8,16 \
  --steps 4 \
  --size 64 \
  --tube-count 128 \
  --run-metal-timing \
  --include-per-frame-baseline \
  --timing-iterations 1 \
  --timing-warmup 1 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t
```

Artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/trained_high_motion_checkpoint.pt
```

Training still behaved as a smoke gate rather than a quality run:

```text
loss: 0.317323 -> 0.316218
pass: true
tile_overflow_sum: 0
cache rebuilds/live updates/staleness checks: 1 / 3 / 3
```

Compiled trained checkpoint geometry:

```text
frames                     4      8       16
trace_count                128    128     128
dense_per_frame_tile_pairs 3578   7158    14363
interval_trace_entries     956    1158    1371
interval/dense tile ratio  0.267  0.162   0.095
fallback_fraction          0      0       0
```

Same-checkpoint per-frame replay:

```text
frames                         4       8       16
shared_interval_entries        956     1158    1371
per_frame_replay_entries       956     4811    9605
shared_forward_ms              295.0   183.2   469.7
per_frame_forward_ms           301.3   770.8   802.0
shared_backward_ms             113.6    57.5   303.3
per_frame_backward_ms          189.8   657.5   1779.1
```

Interpretation:

The forward timing is noisier than the 32px artifact, but the entry-count and
backward rows strengthen the working model. At 16 frames, the shared interval
atlas uses `0.143x` as many interval entries as framewise replay, the native
forward is `0.586x` the replay time, and native backward is `0.170x` the replay
time. The important durable fact is zero fallback and zero overflow while the
same learned tensors scale through the reusable sensor-time atlas.

## Expansion: 96px / 256-Tube Cap256 Rerun

The first attempt at this scale failed before training:

```text
RuntimeError: meta tile_capacity must match STAR_UVT_TILE_CAPACITY
```

That was not a theory failure. The benchmark configured
`feature_uvt.tile_capacity = 256`, but only pushed the corresponding Metal
environment before standalone timing. The trainer render still inherited the
old/default cap. I patched the benchmark to call the tile-env synchronization
before `run_training`, and added a focused test for the non-default cap256
contract.

Command after the fix:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --frame-counts 4,8,16 \
  --steps 4 \
  --size 96 \
  --tube-count 256 \
  --tile-capacity 256 \
  --run-metal-timing \
  --include-per-frame-baseline \
  --timing-iterations 1 \
  --timing-warmup 1 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256
```

Artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/trained_high_motion_checkpoint.pt
```

Training:

```text
loss: 0.317038 -> 0.315874
pass: true
tile_overflow_sum: 0
cache rebuilds/live updates/staleness checks: 1 / 3 / 3
```

Compiled trained checkpoint geometry:

```text
frames                     4      8       16
trace_count                256    256     256
dense_per_frame_tile_pairs 7820   15628   31255
interval_trace_entries     2045   2349    2831
interval/dense tile ratio  0.262  0.150   0.091
fallback_fraction          0      0       0
```

Same-checkpoint per-frame replay:

```text
frames                         4       8       16
shared_interval_entries        2045    2349    2831
per_frame_replay_entries       2045    10279   20547
shared_forward_ms                76.8     83.6   250.6
per_frame_forward_ms            968.3   3820.0  2512.6
shared_backward_ms               59.3    202.5   247.2
per_frame_backward_ms           430.5   2168.2  2018.8
```

Interpretation:

This strengthens the scale story. At 16 frames, shared interval uses `0.138x`
the per-frame replay entries, forward timing is `0.100x`, and backward timing
is `0.122x`. It is still only a smoke-scale native MPS diagnostic, but it now
passes at 96px/256 tubes with cap256 and no fallback/overflow.
