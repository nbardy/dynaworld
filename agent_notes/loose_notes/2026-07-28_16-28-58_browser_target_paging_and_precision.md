# Browser Target Paging And Precision

## Why This Follow-up Existed

The first July 28 scaling pass stopped at 192x144 because the 384x288 benchmark
attempted to bind every target image as one RGBA32F storage buffer:

```text
384 * 288 * 18 cameras * 16 times * 4 channels * 4 bytes = 509,607,936 bytes
```

The active tiled step selects only one camera/time pair, so that allocation was
an ownership mistake rather than useful residency.

## Changes

The base WebGPU trainer now owns overridable target-allocation and
target-initialization hooks. The full-frame tiled trainer overrides them with a
single RGBA32F frame page. Before each step it uploads the selected frame,
writes configuration with `targetOffset=0`, and submits the matching command
buffer. All operations use the same `GPUQueue`, so deep submission remains
ordered without `onSubmittedWorkDone()` or a per-step map.

The target fix exposed two global-capacity reservations and two latent
correctness ceilings:

- checkpoints were `pixels * (4096 / 16) * vec4<f32>`;
- pair gradients were `tiles * 4096 * 96 bytes`;
- the first bounded version kept only 2,048 splats per tile;
- backward encoded every active pair in indirect-dispatch X, which is limited
  to 65,535 workgroups per dimension.

The final layout uses `nextPow2(splatCount)` slots per tile, so every splat can
contribute once without overflow. Tile IDs and compact active-pair references
share one buffer. Backward maps compact pair index
`wid.y * 65535 + wid.x`, guards the padded final row, reduces the 16x16 pixel
workgroup, and atomically adds 24 FP32 gradient values into one record per
splat. This removes both the per-pair gradient slab and the `splat x tile`
inverse map.

Forward's per-pixel stop rank shares the otherwise unused fourth pixel-gradient
lane. It crosses that boundary as the exact numeric conversion
`u32 -> f32 -> u32`; a raw bitcast would encode small ranks as subnormal floats
that implementations may flush before backward reads them.

The checkpoint planner increases replay stride only when needed to remain under
the device's storage-binding limit. Every major tiled buffer receives a
pre-allocation size gate.

## 384x288 Memory Plan

Apple M4 WebGPU reports a 128 MiB storage-binding limit.

| Buffer | Before | After |
| --- | ---: | ---: |
| Target | 509,607,936 B | 1,769,472 B |
| Checkpoints | 452,984,832 B | 113,246,208 B |
| Pair gradients | 169,869,312 B | 0 B |
| Pair IDs and references | 14,155,776 B | 14,155,776 B |
| FP32 gradient accumulator | n/a | 393,216 B |

The exact 4,096-splat plan uses 4,096 pair slots per tile. FP32 checkpoint
stride is 64; packed-FP16 checkpoint stride is 32. Both tapes are 108 MiB at
384x288 because FP16 spends its per-record saving on twice as many checkpoints.

## Execution Result

The real in-app WebGPU benchmark completes at 384x288 with 4,096 active splats,
finite loss, and zero tile overflow.

FP32, 32 warmup plus 128 measured steps:

```text
steps/s: 118.67, 117.62, 117.96
median: 117.96
loss after 160 submissions: 0.471943
```

The final FP32 scaling matrix was:

| Raster | 768 | 1,536 | 4,096 |
| --- | ---: | ---: | ---: |
| 96x72 | 1,233 | 863 | 359 |
| 192x144 | 675 | 462 | 240 |
| 384x288 | 266 | 182 | 118 |

These are current ownership-path numbers, not directly comparable to older
sampled-ray or per-pair-gradient measurements.

## Precision Decision

The trainer retains FP32 for:

- trainable splat parameters;
- first and second Adam moments;
- projection, covariance, conic, depth, and alpha math;
- compositing arithmetic, rendered pixels, and SSIM statistics;
- image gradients and atomic gradient reductions;
- target pages.

Packed-FP16 forward checkpoints use core `pack2x16float` and
`unpack2x16float`, not FP16 arithmetic. The Apple adapter also advertises native
`shader-f16`.

Matched packed-checkpoint repeats at 384x288/4,096:

```text
steps/s: 139.24, 131.07, 131.65
median: 131.65
loss after 160 submissions: 0.471949
absolute loss delta from FP32: about 6e-6
```

At 1,024 submissions, FP32 loss was 0.284515 and packed-FP16 loss was 0.284516.
The corresponding 512-step timed intervals were 88.10 and 99.97 steps/s.
Packed checkpoints were therefore promoted to the SPA default, while FP32
remains selectable.

The strongest next compact-storage target is canonical atlas bytes:

1. retain RGBA8 frames from decode instead of immediately expanding the whole
   dataset to Float32;
2. share that canonical storage between main, training, and validation workers;
3. page one packed target and decode with `unpack4x8unorm` or `textureLoad`;
4. preserve FP32 loss and optimizer arithmetic.

That reduces both the per-step target upload and the much larger host-memory
problem. At present, the 384x288 Float32 dataset is still about 486 MiB and can
be cloned main to training worker and again to validation worker. The isolated
GPU benchmark is fixed; a normal high-resolution SPA mode still needs this host
storage pass.

## Scope Boundary

The sampled-ray control dynamically chooses many camera/time targets inside one
dispatch and still binds the full tensor. It continues to reject 384x288. It
also has a fixed 2,048-entry workgroup order cache and now rejects larger splat
counts before shader execution. The SPA default is the tiled backend; changing
the sampled control requires a separate sample-target staging design or packed
all-dataset representation.
