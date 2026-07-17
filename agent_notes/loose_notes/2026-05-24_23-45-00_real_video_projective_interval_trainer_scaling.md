# Real-Video Projective Interval Trainer Scaling

## Context

The earlier trainer frame-scaling artifact used the real `run_training` route
but monkeypatched a synthetic tensor loader. That proved the compatible
projective interval trainer path could reuse cache metadata, but it did not
exercise the real high-motion video loader and target frames.

This pass added a sibling benchmark that runs the actual high-motion source
video through the trainer for `4/8/16` frame prefixes and compares cadence
cache rebuilds against measured live-cache reuse.

## Artifact

Script:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_trainer_frame_scaling_benchmark.py
```

Outputs:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling/cases/*.json
```

Command:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_trainer_frame_scaling_benchmark.py \
  --frame-counts 4,8,16 \
  --steps 4 \
  --size 64 \
  --tube-count 128 \
  --tile-capacity 128 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling
```

## Evidence

The benchmark uses:

```text
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
size = 64
tube_count = 128
steps = 4
refresh_every = 2
```

Rows:

```text
frames  policy    end_loss  no_first_ms  rebuilds  live_updates  support_rebins  overflow
4       cadence   0.302624  2953.1       2         2             2               0
4       measured  0.302624   900.4       1         3             3               0
8       cadence   0.280947  4775.2       2         2             2               0
8       measured  0.280947  1880.8       1         3             3               0
16      cadence   0.311539  3350.2       2         2             2               0
16      measured  0.311539  2490.6       1         3             3               0
```

Summary:

```text
measured rebuilds: 1/1/1
cadence rebuilds:  2/2/2
max end-loss delta: 0.0
measured/cadence no-first-step ratios: 0.305, 0.394, 0.743
max_tile_count: 18
tile_overflow_sum: 0 for all rows
```

## Interpretation

This is a stronger end-to-end trainer-level link than the synthetic loader
artifact. It shows that, on the real high-motion clip, the measured
projective-interval policy halves full cache rebuilds, preserves identical
training loss versus cadence, and remains within tile capacity.

It is not yet the final support-lifecycle result. The measured row still does
support rebins on every live update (`3/3/3`), so the next cache work should
attack support churn instead of only reducing full rebuilds.

## Expansion: Guarded Support-Churn Reruns

I extended the benchmark to expose the existing support guard and
tail-alpha certificate knobs:

```text
--support-guard-padding
--support-guard-policy
--support-guard-bisect-steps
--support-stale-overshoot-epsilon
--support-stale-tail-alpha-epsilon
```

The first guarded rerun:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_trainer_frame_scaling_benchmark.py \
  --frame-counts 4,8,16 \
  --steps 4 \
  --size 64 \
  --tube-count 128 \
  --tile-capacity 128 \
  --support-guard-padding 1.0 \
  --support-guard-policy slack_budgeted \
  --support-stale-tail-alpha-epsilon 0.001 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard1_tail001
```

Result:

```text
frames  policy    no_first_ms  rebuilds  support_rebins  stale_refreshes  overflow
4       cadence     705.4      2         0               0                0
4       measured    663.3      1         0               0                0
8       cadence    1871.7      2         0               0                0
8       measured    752.8      1         0               0                0
16      cadence    1659.3      2         0               0                0
16      measured   1083.6      1         0               0                0
```

The guard1 artifact keeps exact cadence loss, zero overflow, measured rebuilds
at `1/1/1`, and removes support rebins/stale refreshes entirely. Its
measured/cadence no-first-step ratios are:

```text
0.940, 0.402, 0.653
```

I also tried guard2:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard2_tail001/summary.md
```

Guard2 also removes support rebins, but its measured/cadence no-first-step
ratios are:

```text
0.322, 0.807, 1.753
```

The 16-frame measured row is slower than cadence there. So the update to our
model is precise: support guards do solve the real-video support churn for
this clip, but guard size is a performance knob. The next production policy
should prefer the smallest certified guard that covers live motion and leaves
tile pressure bounded, rather than treating larger guard padding as safer by
default.

Follow-up smaller guards:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard05_tail001/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.md
```

Both use the same `slack_budgeted` policy and `tail001` certificate. Both
also remove support rebins/stale refreshes entirely with exact cadence loss
and zero overflow. Guard `0.25px` is therefore the smallest certified no-churn
guard in this ladder, while guard `0.5px` has the best timing ratios:

```text
guard0.25 measured/cadence no-first-step ratios: 0.893, 0.218, 0.747
guard0.5  measured/cadence no-first-step ratios: 0.536, 0.324, 0.279
```

## Verification

```text
py_compile projective_real_video_trainer_frame_scaling_benchmark.py: passed
focused real-video verifier tests after guarded artifacts: 20 passed in 13.33s
focused real-video + trained high-motion verifier tests: 28 passed in 22.45s
saved artifact verifiers: base, guard025_tail001, guard05_tail001, guard1_tail001, and guard2_tail001 passed
```
