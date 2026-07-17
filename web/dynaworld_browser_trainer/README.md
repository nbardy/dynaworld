# Dynaworld Browser Trainer

Standalone WebGPU SPA for in-browser dynamic splat training.

It preloads a tiny source-view dynamic NeRF-style target, trains a compact
dynamic Gaussian/tube field with WGSL compute, and renders the current result
live in the page.

## Run

Serve from the repo root so the Neural3D coffee preview can be fetched:

```bash
python3 -m http.server 8080
```

Open:

```text
http://localhost:8080/web/dynaworld_browser_trainer/
```

If the local Neural3D preview clip is unavailable, the app falls back to a
deterministic synthetic D-NeRF mini fixture.

The local Neural3D preview videos are side-by-side camera previews. The browser
trainer crops wide preview videos to the left source-view pane by default,
because the current objective is a single source-view image-space overfit, not a
two-camera or heldout-view renderer.
`converge48` keeps that training boundary but shows both source and target
preview crops in a small camera strip and loops preview time by default. This is
for visual context only: the WebGPU trainer still optimizes the left/source crop
and does not train on, or render into, the target camera.

## Current Boundary

This is a browser-first training prototype, not parity with the Metal trainer.
It ports the practical shape of the STAR UVT / WorldTubes and dynamic-splat
training loops to a small WGSL testbed: dynamic splat/tube parameters,
source-view reconstruction loss, compute-shader SGD, and live splat rendering.

The current train kernel uses a mean static background plus fixed-order
source-over dynamic splats so dark moving objects can occlude the background.
The training objective now matches the browser render blend order, but it does
not yet implement the full Metal tile/depth/alpha compositor, native VJP,
camera-family trace atlas, shared-backward/tape accumulation, or heldout camera
training contract.

The Motion Model menu currently switches between two browser approximation
branches inside the WGSL trainer:

- `World Tubes-style shared motion`: linear velocity plus a small harmonic
  time basis.
- `Dynamic splats-style velocity`: per-splat linear screen-space velocity.

Initialization is deterministic target-grid/color initialization from the
preloaded frames plus motion-aware seeding. Most splats still cover an
aspect-aware grid, but a later-drawn fraction is seeded directly from
high-motion frame/pixel samples with a local time center and slightly higher
opacity. The current default seeds 48% of splats from those motion samples with
slightly broader, more opaque initial support, after diagnostics showed the
older init was under-covering moving pixels. The browser still does not run COLMAP, pycolmap, VGGT, or any
point-cloud geometry initializer. This is much better than random noise for a
source-view overfit, but it is not a true 3D geometry seed.
`converge46` adds a lightweight source-view motion prior: frame-level motion
centroid velocities are estimated from target-vs-mean-background residuals, and
motion-seeded splats initialize their linear velocity from that estimate while
back-solving the base center so the splat still lands on its chosen frame.
This is still an image-space initializer, not SfM.
`converge47` makes that prior local: each motion-seeded splat searches nearby
residual pixels in adjacent frames, blends the local match velocity with the
frame-centroid fallback, then uses the same base-center back-solve. This is the
current browser init default.

The default is 768 splats, with a 96-768 splat-count slider. The first
384/512/768 sweep made 512 the best default under the old 75% motion sampler,
but the post-`converge25` retest reversed that: at 95% motion sampling, 768
splats reached lower true motion loss than a matched 512-splat rerun at similar
observed step rates. Higher counts are still untested because this prototype
currently uses a simple all-splats-per-sample objective.
The default temporal support is `0.30`, with a 0.14-0.32 slider. Lower values
make splats more frame-local but can starve background/view-change gradients;
higher values are more stable but blurrier.
The default learning rate is `0.90`, raised from `0.45` after a short browser
sweep showed much faster motion-loss reduction without producing console errors.
The default requested `Motion Mix` is `95%`, raised from the original hardcoded
75% after a live Dynamic splats-style sweep showed better true motion loss at
the same 512-splat / temporal `0.30` setting. The default `Static Mix` reserve
is `8%`, so the effective motion readout is `92%`; set `Static Mix` to `0%` to
recover the old v42-style sampler for A/B checks.
The default `Support Guard` is `52%`. It exposes the simplified WGSL motion
coverage hinge target so the browser is no longer locked to the older hidden
44% support floor.
`FPS` is the browser render-loop rate. `Steps/s` is the actual train-step
throughput and is the better number for comparing splat-count settings.

The default Neural3D preview file is 512x256: two 256x256 camera panes packed
side by side. The browser crops one pane and decodes the training target to
128x128x8. The visible WebGPU canvas can be larger because it is just the render
target. For this prototype, training cost is mostly splat count and
samples-per-step: the current compute shader dispatches one worker per splat and
re-evaluates all splats for each sampled pixel. Increasing target/canvas
resolution is therefore cheaper than increasing splats for training, although
render fragments and CPU validation still scale with displayed/output pixels.

## Current Math

The browser model stores each primitive as:

- `posRadius.xy`: normalized image-space center
- `posRadius.z`: temporal center
- `posRadius.w`: image-space radius, measured in target-height units
- `motion.xy`: linear screen-space velocity
- `motion.zw`: harmonic screen-space offsets for the World Tubes-style mode
- `colorOpacity.xyz`: primitive RGB used by the source-over residual
- `colorOpacity.w`: opacity logit

The trainer now preserves the source video aspect ratio, uses aspect-aware
Gaussian distances, and renders the target aspect into the WebGPU canvas instead
of stretching the fit across the wide preview pane. Each primitive also has a
soft temporal gate around `posRadius.z`; this keeps the source-view overfit from
collapsing every frame into one static blur.
Training now ignores splat contributions beyond the same 3-sigma support that
the render billboard can actually draw, so the train objective no longer chases
tiny invisible far-field Gaussian tails.

The displayed sample loss is an EMA-smoothed stochastic loss from the training
mini-batch, not a full-frame validation loss. The sidebar also reports a
deterministic sparse `Grid Loss`, `Val MAE`, `Val PSNR`, `Val SSIM`,
`Motion Loss`, `Motion Cov`, `Static Cov`, `Peak Alpha`, `Active`,
`Mean Opac`, and `Mean Radius` diagnostics. `Grid Loss` is sparse validation
MSE, `Val MAE` is mean absolute RGB error, `Val PSNR` is derived from sparse
MSE, and `Val SSIM` is a global luma SSIM approximation over the same sparse
validation grid. `Motion Loss` evaluates the packed high-motion frame/pixel set
directly, so it is the better quick signal for whether moving objects are
actually improving instead of being hidden by the mean static background.
`Motion Cov` is the average dynamic alpha coverage on those motion samples,
`Static Cov` is dynamic alpha leaking onto a thinned low-motion grid,
`Peak Alpha` is the mean strongest per-sample splat alpha, `Active` counts
splats with opacity above 5%, and the mean opacity/radius readbacks make it
easier to tell whether the model is shrinking or fading out. The target pane
also has a throttled `Validation error` view that draws a heat map from the
current source-view prediction error for the selected frame.

Training samples are intentionally biased toward moving pixels: the dataset
loader compares each frame against the mean background, keeps high-energy
frame/pixel samples, and the WGSL train shader draws a configurable fraction
from that motion set. The current default is 95% motion samples and 5% uniform
samples before the v43 static reserve is applied. The loader also now keeps a
low-motion frame/pixel set, and the train shader reserves 8% of samples for
that static set so the low-motion alpha penalty is not starved by the
motion-heavy sampler. The `Motion Px` and `Static Px` stats show how many
packed frame/pixel samples the current target produced.

Training still uses RGB reconstruction plus lightweight browser-specific
regularization, not a full standard perceptual loss stack. Current
regularizers/guards include temporal gating, radius/opacity clamps, a low-motion
alpha penalty, global opacity decay, the static-sample reserve, and the
motion-sample support guard. SSIM is currently validation-only; adding DSSIM,
multi-scale SSIM, LPIPS, total-variation, acceleration/velocity smoothness, or
alpha-budget regularizers should be tested behind explicit controls because
they can easily improve a scalar while worsening support or motion fit in this
simplified source-view prototype.

`converge31` added a small motion-coverage hinge inside the train shader, but
the first `target=0.50` / `weight=0.20` setting over-preserved support and
slowed motion-loss improvement. `converge32` weakens this to a late guard:
on motion-sampled pixels only, if total dynamic alpha coverage falls below
`0.44`, the alpha/radius/center/time gradients receive a support-preserving term
with weight `0.08`. This is not a new renderer; it is a source-view guard
against the simplified browser objective winning MSE by letting dynamic support
fade or specialize away.
The extended `converge32` browser trace reached step `861`, true `Motion Loss`
`0.005914`, and `Motion Cov` `47.0%`. That is a small MSE tradeoff versus the
best `converge28` trace (`0.005459` at step `854`), but it keeps much healthier
motion support than `converge28`'s `38.2%` coverage.
`converge33` leaves the math unchanged, renames the UI control from
`Shader Mode` to `Motion Model`, and records the current mode comparison:
World Tubes-style reached step `629`, true `Motion Loss` `0.005938`, and
`Motion Cov` `47.4%`, close to Dynamic splats-style at step `861` /
`0.005914` / `47.0%`. The current default remains World Tubes-style because
the two branches are effectively tied under the weakened support guard.
`converge34` leaves the math unchanged again and fixes the visual debugging
surface: target and render panes now use equal desktop columns, and the target
pane has an RGB/motion-residual selector. The residual view displays amplified
`abs(frame - mean_background)` so sparse moving-object support is visible. A
live post-edit trace reached step `268`, `Grid Loss` `0.000193`, true
`Motion Loss` `0.006505`, `Motion Cov` `50.2%`, mean radius `0.0141`, and no
browser warnings/errors after pausing the run.
`converge38`/`39` add the matching result-side diagnostics. `Result View` can
show RGB, a bright dynamic layer, or alpha support on a black background. An
intermediate fragment-storage-buffer attempt blacked out the render pane and was
backed out; the final path keeps the splat fragment independent of the
background storage buffer. The alpha-support view showed that the model does
find the moving person, but also keeps a broad active layer across the
background. `converge39` responds by lowering the temporal gate floor from
`sigma * 0.70` (`21%` at the default temporal support) to `sigma * 0.30`
(`9%` at the default). Boot motion loss improved from `0.011522` to `0.011099`
and boot coverage dropped from `63.1%` to `59.7%`; the short trace reached
step `72`, true `Motion Loss` `0.007788`, and `Motion Cov` `53.9%`. This is a
small improvement, not a complete sparsity fix.
`converge40` adds the first explicit sparsity objective: a `Static Cov`
readback, a low-motion alpha penalty, and global opacity decay. The initial
decay weight `0.055` was too blunt: by step `239` it reached true `Motion Loss`
`0.007012` but drove `Motion Cov` down to `42.4%`, below the support guard.
`converge41` keeps the low-motion alpha penalty but lowers opacity decay to
`0.025`; by step `294` it reached `Motion Loss 0.006751`, `Motion Cov 44.6%`,
`Static Cov 2.6%`, `Active 406/768`, and `Mean Opac 7.3%`. That is still broad
visually, but it is a better trade than v40 because it reduces the background
layer without collapsing motion support. `converge42` keeps the v41 train
constants and thins the `Static Cov` validation pass to one quarter of
low-motion grid samples to reduce readback overhead; boot and one-step browser
smokes loaded the new assets and exercised the 96-byte train uniform path.
`converge43` adds the dedicated low-motion sample buffer and static sample
reserve. The first v43 browser trace loaded `Motion Px 2018`, `Static Px
16384`, reached step `259`, true `Motion Loss 0.006803`, `Motion Cov 45.5%`,
`Static Cov 2.6%`, `Active 420/768`, and `Mean Opac 7.4%` with no browser
warnings/errors. This makes the static-cleanup objective less accidental; it is
not yet a proof of visual convergence.
`converge44` keeps the v43 math and exposes the static reserve as a `Static Mix`
slider. Default labels now show effective `Motion Mix 92%` and `Static Mix 8%`;
setting `Static Mix` to `0%` restores effective `Motion Mix 95%` for direct
v42-style sampler comparisons. The in-app smoke loaded v44 assets, stepped the
trainer once, and returned no browser warnings/errors.
The matched v44 control found that static reserve is not the convergence
culprit: `Static Mix 0%` reached step `274`, `Motion Loss 0.006794`, and
`Motion Cov 45.0%`, while default `8%` reached step `271`,
`Motion Loss 0.006822`, and `Motion Cov 45.3%`. Both arms converged to the
old hidden 44% support neighborhood. `converge45` responds by exposing that
target as `Support Guard` and defaulting it to `52%` with the same weak `0.08`
hinge weight. The v45 trace loaded `app.js?v=20260707-converge45` and reached
step `297`, `Motion Loss 0.007060`, `Motion Cov 48.2%`, `Static Cov 2.7%`,
and `Active 406/768`. This preserves more support at a modest MSE cost; it is
still not a renderer/init parity fix.
`converge46` keeps the same support guard and adds frame-motion centroid
initialization. The first v46 trace loaded `app.js?v=20260707-converge46` and
reached step `290`, `Motion Loss 0.007036`, `Motion Cov 47.0%`,
`Static Cov 2.8%`, and `Active 407/768`. This is a small fit win over v45 but
a support loss, so it should be treated as a useful init diagnostic rather than
a convergence fix.
`converge47` replaces the global velocity with a local residual-match velocity.
The first v47 trace loaded `app.js?v=20260707-converge47` and reached step
`279`, `Motion Loss 0.006885`, `Motion Cov 48.1%`, `Static Cov 2.7%`, and
`Active 414/768`. This is the first clean positive result in this lane: better
motion fit than v45/v46 at similar support. It is still source-view image-space
training, not a full depth/tile/heldout renderer.

Important simplifications remain: color is static per primitive, visibility is
fixed instance order rather than depth sorted, and the backward pass evaluates
all splats for every sampled pixel.

`converge50`-`59` replace the SGD-style update with per-parameter Adam moments,
add persistent absolute-gradient/contribution statistics, and add fixed-capacity
density maintenance. Every 256 steps, up to eight weak slots are recycled into
small splats at the highest sampled motion residuals; both parameter ping-pong
buffers, optimizer moments, and statistics are reset for those slots without
reallocating WebGPU resources. The UI reports cumulative recycled splats and
mean absolute parameter delta. Validation readbacks are serialized, training
pauses during the CPU validation pass, and slider changes cannot reset the
trainer before the dataset finishes loading.

This is infrastructure completion, not a new quality claim. The first Adam
rates inherited from SGD were rejected because validation worsened; the final
rates are an order of magnitude smaller. Short probes show stable finite
updates, but they do not yet beat `converge47` on true motion loss. Windowed
D-SSIM remains validation-only because adding a 3x3 or 5x5 neighborhood inside
the current all-pairs train kernel would multiply its dominant work. The next
quality step is a tiled image-space backward or exported geometry/init, then a
matched Adam+density-control ablation.
