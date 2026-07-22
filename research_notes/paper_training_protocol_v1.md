# Unified Paper Training Protocol v1

## Scope

This protocol compares World Tubes, WorldFoam, and free dynamic 3DGS without
pretending that their representations or kernels are identical. The shared
surface owns data, sample order, progressive stages, cost accounting, and
evaluation. Representation adapters retain their own model state and Metal
entry points.

World Tubes are structured 4D splats: each tube persists through time and has
a constrained trajectory/temporal support. They are not arbitrary Gaussians in
joint `(x, y, z, t)` with a free 4D covariance. The structural restriction is
part of the hypothesis: a shared tube should cover more time with less repeated
state than independent per-frame Gaussians.

## Dataset Contract

- Primary scene: Neural3D `coffee_martini`.
- Primary temporal set: all 300 synchronized frames at 30 fps.
- Primary task: train-camera reconstruction and heldout-camera novel-view
  evaluation at every timestamp.
- The first full-sequence runner keeps the established `cam04`/`cam09` train
  and `cam06` heldout split. Camera-triplet and scene breadth are separate
  experiment axes.
- Images preserve source aspect ratio. A `4:3` source must not be square
  resized.

## Space-Time Batch Contract

Each shuffled epoch contains every train `(camera, time)` pair exactly once.
No pair is sampled with replacement inside an epoch. Batch construction is
best-effort structured:

1. choose a shuffled anchor pair;
2. add other train cameras at the same time;
3. add nearby times from the anchor camera;
4. fill the remaining slots from the shuffled global pool.

This combines cross-camera geometry constraints, local motion constraints, and
global temporal coverage while retaining an auditable uniform epoch measure.

## Progressive Contract

Stages explicitly declare:

- `[start_step, end_step)`;
- aspect-preserving `(height, width)`;
- active primitive count;
- frames per optimizer step;
- learning-rate multiplier.

Resolution and active capacity are non-decreasing. Stage transitions must keep
the model finite and preserve optimizer state. World Tubes and dynamic 3DGS
preallocate the final capacity and activate a prefix. WorldFoam uses its
optimizer-state-preserving cell resampler when capacity changes.

The coarse-to-fine row must be compared with a fixed-resolution row at matched
target-pixel budget. Otherwise progressive training is a compute change hidden
inside an optimization claim.

## Kernel Adapters

The shared runner records, but does not erase, the implementation boundary:

| Representation | Forward | Backward |
| --- | --- | --- |
| World Tubes | STAR UVT Metal tile/projective interval | selected STAR UVT Metal policy |
| WorldFoam | PowerFoam Metal raster or raytrace | matching custom Metal autograd path |
| dynamic 3DGS | fast-mac Metal | fast-mac Metal autograd |

Deterministic correctness policies are reported separately from practical
throughput policies. A reference kernel is never substituted for the normal
training kernel without changing the row label.

## Cost Contract

Every lane reports:

- optimizer steps;
- target frames and target pixels;
- rasterized frames and rasterized pixels;
- total/trainable parameters and parameter bytes;
- optimizer-state bytes;
- elapsed training time;
- peak device memory when the backend exposes it;
- active primitives per stage.

Nominal primitive count is not a capacity match. In particular, `N` free
per-frame Gaussians over `T` frames carry approximately `T` times the
frame-local state of `N` shared tubes. Tables must expose both active render
count and total stored state.

## Required Ablations

1. shared full-sequence sampler versus the old 16-frame smoke;
2. structured space-time batches versus global shuffled batches;
3. coarse-to-fine versus fixed resolution at matched target pixels;
4. capacity curves for each representation;
5. normal fast kernel versus deterministic correctness audit timing;
6. heldout camera quality, train quality, wall time, and memory;
7. additional camera triplets, seeds, and Neural3D scenes.

The existing 128px/16-frame/40-step result remains a correctness and runner
smoke. It is not relabeled as the full-sequence paper experiment.

## Implementation Status (2026-07-19)

The shared protocol and all three adapters are implemented. World Tubes uses
the existing temporal-window STAR Metal kernel at each selected global time,
so K-frame paper updates rasterize K frames rather than whole sequences. A
two-stage 4-frame MPS smoke and a low-resolution all-300-frame MPS smoke pass
with exact cost validation.

The configured primary row ends at 384x512 and uses the current eager
multicamera bundle. Native 2028x2704 targets are deliberately not claimed:
they require on-demand K-frame decode/ray generation and streamed evaluation.
The executable remaining-work contract is
`TODO/unified_paper_ablation_pipeline.md`.
