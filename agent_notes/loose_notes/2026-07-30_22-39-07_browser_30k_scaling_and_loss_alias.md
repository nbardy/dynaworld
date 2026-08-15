# Browser 30K Scaling And Loss Alias

## Why the objective chart oscillated

Coffee Martini has 17 train cameras and 16 times, so the full-frame schedule
contains 272 pairs. The pair stride is 169 and visits every pair exactly once
per cycle. The worker previously requested a metric every 256 submitted steps.
Because `gcd(256, 272) = 16`, those readbacks observed only 17 pair phases and
repeated after 4,352 optimizer steps. The visible ripple was camera/time
difficulty aliasing, not an LR reset, shuffle reset, or topology reset.

Changing the requested interval to 257 was not sufficient: the worker submits
eight-step bursts, so the usual realized interval became 264. The durable fix
is a 272-entry GPU-resident objective ring. Every optimizer step writes its
objective/L1/DSSIM and absolute step stamp. Asynchronous UI readback copies the
ring and reports the mean of the most recent dynamic-fit cycle. The raw current
pair remains visible in the metric tooltip and the camera/frame diagnostic.

Topology growth is separate. For the default 4,096-to-8,192 run, 16 hidden
slots activate every 100 steps from step 600 through step 26,100. Once capacity
is full there is no recycling. The metric stream now reports topology operations
since the prior readback so a real split event is distinguishable from pair
difficulty.

## Large-capacity memory changes

- Raised the tiled model/render ID path to 15-bit IDs and an internal 32,768
  stress ceiling.
- Kept the default reserve at 8,192.
- Removed the inherited sampled view/time/splat depth-order cache from tiled
  allocation. At 30K this avoids 32,640,000 bytes.
- Removed the unused tiled sampled-gradient slab, leaving only the small base
  binding placeholder.
- Kept one paged RGBA32F target, packed-FP16 forward checkpoints, and FP32
  parameters/projection/loss/gradients/Adam.
- Added exact per-category buffer accounting to the UI.

The current capacity-scaled state is about 808 bytes per power-of-two splat,
plus roughly 15.7 MiB of fixed raster/checkpoint workspace. The avoidable next
memory cuts are the persistent parameter readback and repeated camera rotation
inside each projected packet; neither is the current 30K blocker.

## Full-cycle Apple M4 stress

Each count used a fresh device, 96x72 full-image training, packed-FP16
checkpoints, 11x11 Gaussian SSIM, 8 warmup steps, and 272 measured GPU-drained
steps. Preview, validation, metric readback, compilation, and initialization
were excluded.

| Active splats | Steps/s | Allocated buffers | Max tile | Cumulative overflow |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 99.54 | 31.38 MiB | 2,680 | 0 |
| 30,000 | 63.93 | 38.86 MiB | 3,993 | 0 |
| 32,768 | 58.80 | 40.92 MiB | 4,096 | 9,125 |

The short 32K test had misleadingly reported zero overflow on its final pair.
The cumulative counter caught violations elsewhere in the 272-pair cycle. The
SPA therefore exposes 30K only as a stress option; 32K remains an internal
limit for ID and failure-path tests.

## Relation to the remembered Metal numbers

The memory of large fast forward/backward cuts is valid, but the count needs a
contract:

- fast-mac v5 processed 262K projected instances in roughly 256-274 ms F+B;
- a dynamic v8 note records 2.097M frame-expanded projected instances at
  256x256 in 252.7 ms and at 512x512 in 693 ms;
- WorldFoam compacted about 1.3M segment records to about 63K owners in a
  roughly 6 ms microstep at a tiny 64-cell model;
- no surviving artifact proves one million distinct trainable splats in one
  scene through projection, image objective, backward, Adam, and topology.

The browser already uses tiled source-over compositing, saved order/stop state,
fixed GPU capacity, indirect compact-pair backward, target paging, and no
hot-loop CPU wait. Its main missing Metal pattern is staged VJP: the current
pair kernel repeats the expensive 3D mean/covariance/quaternion derivative for
each tile/splat pair. Project3D Metal first accumulates compact projected-space
gradients, then runs one projection VJP per splat. That is the highest-value
20K-30K speed patch, ahead of lower-level memory packing.

Other scaling work remains: replace the three-panel global preview bitonic sort,
parallelize the serial densification selector, add pass timestamp queries, and
add an exact tile-overflow route before treating 32K as valid.
