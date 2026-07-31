# Browser Progressive 30K Defaults

Date: 2026-07-31

The browser SPA now defaults to the measured fast tiled lane with a progressive
4,096 -> 30,000 splat topology. Initialization still uses each checked-in SfM
seed once. Creating 8K active splats immediately would duplicate the 4,096 seed
points at identical positions, scales, colors, and opacity, which is not a
better initialization.

The 30K reserve is affordable at the checked-in 96x72 raster: the live SPA
reports 38.2 MiB of WebGPU buffers with the conservative FP32 projection VJP.
Dormant slots allocate state
but skip raster projection, gradient clear, and Adam update until residual-led
GPU splitting activates them. Early-step compute therefore remains close to
the 4K-active path and grows with useful topology rather than reserved capacity.

Non-training defaults were relaxed without disabling observability:

- live three-view preview: 20 -> 15 FPS;
- asynchronous metric readback: every 256 -> 512 submitted steps;
- full validation request: every 8,192 -> 16,384 completed steps;
- live preview and full-image validation remain enabled.

At the current 96x72 SPA rate, a 512-step loss update is still roughly
sub-second. The change does not promote the diagnostic 384x288 timing result or
packed projection-VJP FP16 as a speed default. It uses the already validated
packed transmittance checkpoints, FP32 projection VJP, compact shared targets,
and sparse active-prefix optimizer.

A worker-backed smoke reached step 5,696 in about six seconds and reported
972.5 steps/s, finite objective `0.19342`, zero current/cumulative tile
overflow, nonblank three-camera results, and no console errors. This is a
same-session health observation, not a controlled before/after benchmark.
