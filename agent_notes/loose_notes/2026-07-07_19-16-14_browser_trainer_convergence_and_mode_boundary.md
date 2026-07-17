# Browser Trainer Convergence And Mode Boundary

## Context

The first `web/dynaworld_browser_trainer/` prototype loaded and ran, but the
user correctly noticed that it did not visibly feel like it was converging and
asked whether the browser app contained the real ultra-fast Metal shader
patterns, both WorldTubes and dynamic Gaussian-splat shader families, and what
initialization it used.

## Truth boundary

- The browser app is a standalone WebGPU/WGSL source-view training prototype.
- It is not a port of the full Metal STAR UVT/WorldTubes shared-backward path,
  dynamic 3DGS fast-mac renderer, or PowerFoam/WorldFoam kernels.
- The repo has those Metal lanes separately, including STAR UVT direct-atomic /
  feature direct-atomic work, dynamic 3DGS fast-mac variants, and WorldFoam /
  PowerFoam Metal variants.
- The browser app currently implements a small screen-space splat/tube
  objective over a mean static background so it can train directly inside the
  browser.

## Changes

- Fixed a real visual/objective mismatch: the WebGPU render pass now uses
  additive blending, matching the additive WGSL training objective.
- Then superseded the positive-only additive residual with a mean static
  background plus alpha-over-style dynamic residual splats. This is still not a
  sorted transmittance compositor, but it lets dark moving objects occlude the
  background instead of only adding light.
- Replaced the orderless alpha training approximation with fixed-order
  source-over compositing to match the browser render blend order. Each splat's
  opacity/geometry gradient now uses the color under that splat and the
  suffix-transmittance of later splats in the fixed draw order.
- Fixed a target distortion bug: the Neural3D preview source is 512x256, but
  the first browser loader decoded it into 96x96. The loader now preserves
  aspect and decodes the preview to 128x64 by default.
- Fixed a target semantics bug: the 512x256 Neural3D preview is a side-by-side
  camera preview, but the browser objective is a single source-view image-space
  overfit. `converge11` crops wide preview videos to the left source-view pane,
  so the default Neural3D target is now 128x128 and matches the model contract.
- Fixed the Gaussian metric/render aspect mismatch: training distances now
  scale x by target aspect, and render fits the target aspect inside the wide
  WebGPU canvas rather than stretching it.
- Activated the previously-unused `posRadius.z` parameter as a temporal center
  with a soft temporal gate in train and render. This is a major anti-blur fix:
  before this, every splat contributed to every frame with static color.
- Exposed temporal support as a UI hyperparameter. Narrow support reduced smear
  but could starve background/view-change gradients. The initial default was
  `0.26`; after the live 512-splat temporal sweep, `converge23` changes the
  default to `0.30` with a 0.14-0.32 slider.
- Matched training support to render support with a 3-sigma cutoff, so training
  no longer optimizes invisible far-field Gaussian tails.
- Made the animation frame loop recover across mode reset/dispose-create races;
  before this, changing to Dynamic splats-style could leave the page `Ready`
  while continuous `Start` did not advance, even though manual `Step` still
  worked.
- Added a `Shader Mode` selector:
  - `World Tubes-style shared motion`: linear velocity plus a harmonic time
    basis.
  - `Dynamic splats-style velocity`: linear per-splat screen-space velocity.
- Raised the splat-count range to expose 96-768 splats. The initial default was
  384; after the live 384/512/768 capacity sweep, `converge22` changes the
  default to 512 as the best current quality/throughput point.
- Replaced the more-random initial state with deterministic target-grid/color
  initialization from the preloaded frames. The grid is now aspect-proportional
  and colors are time-local mixed with average color.
- Kept the init boundary explicit: no COLMAP, pycolmap, VGGT, or point-cloud
  geometry seed runs in the browser yet.
- Added a small EMA to the displayed stochastic sample loss so the sidebar does
  not overemphasize per-step sample noise.
- Bumped browser module query strings to `20260707-converge14` to avoid stale
  module caching in the in-app browser.
- Added deterministic sparse validation readbacks in `converge16`:
  - `Grid Loss` evaluates a fixed source-view grid from the current GPU params.
  - `Motion Loss` weights the same grid by target-vs-mean-background energy so
    moving regions cannot be hidden by the static background.
  - The first reload made the real issue visible: `Grid Loss 0.000186` but
    `Motion Loss 0.044978`.
- Added a motion-biased training sampler in `converge17`:
  - the loader packs high-energy frame/pixel samples against the mean background
  - the train shader receives a `motionSamples` storage buffer
  - each sample is about 75% motion-focused and 25% uniform
  - empty motion-sample sets now stay empty for the train config; the storage
    buffer still receives a 4-byte fallback so WebGPU binding remains valid
- Added a visible `Motion Px` stat; the Neural3D source-view crop produced
  `2018` packed moving frame/pixel samples.
- Ran a short hyperparameter sweep:
  - Dynamic splats, 384 splats, temporal `0.26`, samples `96`, LR `0.45`:
    step `165`, motion improvement `0.000443`
  - same config with LR `0.90`: step `190`, motion improvement `0.001885`
  - LR `0.90` with temporal `0.18`: step `200`, motion improvement `0.001667`
  - decision: make LR `0.90` the browser default and leave temporal support at
    `0.26`
- Fixed a real reset-time browser error: ResizeObserver could call
  `renderOnce()` after `trainer = new DynamicSplatWebGpuTrainer(...)` but before
  `trainer.init(...)` set `dataset` and `device`. `converge18` now builds a
  local `nextTrainer`, assigns it globally only after init, disposes it on init
  failure, and guards render/train on initialized trainer state.
- Added motion-aware initialization in `converge19`:
  - most splats still use the aspect-aware target grid
  - the last 38% of splats are seeded from the high-motion frame/pixel buffer
  - motion-seeded splats use local frame time centers, tighter radii, and higher
    starting opacity
  - these are deliberately late in fixed draw order so they composite over the
    coarse grid/background residual
- Changed the meaning of visible `Motion Loss` in `converge20`:
  - before `converge20`, `Motion Loss` was a grid-weighted validation proxy
  - after `converge20`, it is direct MSE over up to 4096 packed high-motion
    frame/pixel samples
  - do not compare `converge20` motion-loss values numerically against older
    `converge17`-`19` values without noting the metric change
- Added `converge21` throughput visibility:
  - the sidebar now reports `Steps/s`, computed from actual trainer step deltas
    once per second
  - FPS remains the render-loop rate and should not be used as the optimizer
    throughput number
  - the JS trainer fallback `learningRate` default now matches the UI default
    `0.90`
- Added `converge22` default capacity change:
  - 384 splats keeps higher `Steps/s` but is visibly capacity-limited in the
    motion metric
  - 768 splats helps motion loss but is much slower
  - 512 splats nearly matched 768's true motion loss in the live probe while
    preserving a much better train-step rate
- Added `converge23` temporal-support default change:
  - `0.18` and `0.22` support starve the true motion metric
  - matched `0.26` is solid, but `0.30` improves the longer motion-loss result
  - default changes to `0.30`
- Added `converge24` sampler control and `converge25` default sampler change:
  - the hardcoded 75% motion / 25% uniform train-sample split is now exposed as
    a `Motion Mix` slider
  - the WGSL train config carries this as `motionSamplePermil`
  - the longer 95% motion-mix probe beat the old 75% default at similar step
    counts, so the browser default changes to 95% motion / 5% uniform
- Added `converge26` capacity default change:
  - the earlier 512-splat default was measured under the old sampler
  - under the 95% motion sampler, a matched 768-vs-512 retest showed 768
    reaching lower true motion loss at similar observed step rates
  - default changes to 768 splats, but the upper slider bound stays 768 until a
    1024+ capacity/speed probe exists
- Added `converge27` model-health diagnostics:
  - `Motion Cov` reports mean dynamic alpha coverage on packed motion samples
  - `Active` reports splats with opacity above 5%
  - the diagnostic showed the old path improved motion loss while losing
    dynamic coverage, which matched the user's "not a lot of splats" concern
- Added `converge28` motion-aware init fix:
  - motion-seeded splat fraction increases from 38% to 48%
  - motion-seeded radii and opacity are initialized slightly higher
  - this raises initial coverage and improves true motion loss, with a small
    grid-loss tradeoff

## Verification

- Syntax checks passed:
  - `node --check web/dynaworld_browser_trainer/app.js`
  - `node --check web/dynaworld_browser_trainer/dataset.js`
  - `node --check web/dynaworld_browser_trainer/trainerWebGpu.js`
- Server check:
  - `curl -I http://127.0.0.1:8080/web/dynaworld_browser_trainer/` returned
    `200 OK`.
- In-app browser reload showed:
  - script `./app.js?v=20260707-converge17`
  - dataset `Neural3D coffee_martini preview (source-view crop)`
  - GPU `apple`
  - target canvas 128x128
  - 384 splats
  - `2018` motion frame/pixel samples
  - temporal support `0.26`
  - 96 samples per step
  - status `Ready.`
- World Tubes-style and dynamic splats-style mode smoke:
  - both modes reset cleanly to step 0 and status `Ready.`
  - World Tubes-style source-crop smoke reached step 151 with finite displayed
    loss about `0.00013`
  - Dynamic splats-style source-crop smoke reached step 192 with finite
    displayed loss about `0.00016`
  - 768-splat Dynamic splats-style smoke reached step 98 with finite displayed
    loss about `0.00009`, but continuous training is only about `8 fps`
  - no browser warnings or errors
- `converge17` validation/motion-sampler smoke:
  - initial 384-splat metrics: `Grid Loss 0.000186`, `Motion Loss 0.044978`
  - World Tubes-style 384-splat short run reached step `191`,
    `Grid Loss 0.000183`, `Motion Loss 0.044205`, displayed sample loss about
    `0.00694`
  - Dynamic splats-style 384-splat short run reached step `367`,
    `Grid Loss 0.000181`, `Motion Loss 0.043216`, displayed sample loss about
    `0.00709`
  - Dynamic splats-style 768-splat short run reached step `128`,
    `Grid Loss 0.000177`, `Motion Loss 0.043898`, displayed sample loss about
    `0.00689`, about `8 fps`
  - browser console warnings/errors: `[]`
- `converge18` verification:
  - syntax checks passed again for `app.js`, `dataset.js`, and
    `trainerWebGpu.js`; `curl -I` returned `200 OK`
  - in-app reload loaded `./app.js?v=20260707-converge18` and
    `styles.css?v=20260707-converge18`
  - default LR showed `0.9`, dataset/GPU/motion samples remained
    `Neural3D coffee_martini preview (source-view crop)` / `apple` / `2018`
  - Dynamic splats-style short run: step `221`, `Grid Loss 0.000180`,
    `Motion Loss 0.042676`, motion improvement `0.002287`, no new console
    warnings/errors after reload
  - World Tubes-style short run: step `178`, `Grid Loss 0.000180`,
    `Motion Loss 0.042843`, motion improvement `0.002135`, no new console
    warnings/errors after mode reset
- `converge19` motion-aware init verification:
  - syntax checks passed for `app.js`, `dataset.js`, and `trainerWebGpu.js`;
    `curl -I` returned `200 OK`
  - in-app reload loaded `./app.js?v=20260707-converge19` and
    `styles.css?v=20260707-converge19`
  - initial World Tubes-style metrics: `Grid Loss 0.000219`,
    `Motion Loss 0.037862`
  - initial Dynamic splats-style metrics: `Grid Loss 0.000219`,
    `Motion Loss 0.037912`
  - Dynamic splats-style short run reached step `341`, `Grid Loss 0.000157`,
    `Motion Loss 0.027288`, motion improvement `0.010624`, no new console
    warnings/errors
  - World Tubes-style short run reached step `143`, `Grid Loss 0.000165`,
    `Motion Loss 0.027732`, motion improvement `0.010130`, no new console
    warnings/errors
- Longer `converge19` and sample-count checks:
  - Dynamic splats-style 384 splats ran to step `1126`, `Grid Loss 0.000146`,
    old grid-weighted `Motion Loss 0.025342`, no new console warnings/errors
  - Dynamic splats-style 768 splats ran to step `321`, `Grid Loss 0.000141`,
    old grid-weighted `Motion Loss 0.024954`, about `6-10 fps`, no new console
    warnings/errors
  - 32 samples/step was faster but noisy: final old grid-weighted `Motion Loss
    0.028316`
  - 64 samples/step was close but not clearly better on the longer run: final old
    grid-weighted `Motion Loss 0.026485`
  - decision: keep 96 samples/step as the safer default
- `converge20` true motion-sample metric verification:
  - syntax checks passed for `app.js`, `dataset.js`, and `trainerWebGpu.js`;
    `curl -I` returned `200 OK`
  - Dynamic splats-style 384 splats loaded
    `./app.js?v=20260707-converge20`, started at true `Motion Loss 0.009779`,
    reached step `520`, `Grid Loss 0.000153`, true `Motion Loss 0.007004`, no
    new console warnings/errors
  - World Tubes-style 384 splats started at true `Motion Loss 0.009814`, reached
    step `304`, `Grid Loss 0.000160`, true `Motion Loss 0.007175`, no new
    console warnings/errors
- `converge21` throughput/readout verification:
  - in-app reload loaded
    `app.js?v=20260707-converge21` and `styles.css?v=20260707-converge21`
  - Dynamic splats-style 384 splats on the Apple WebGPU adapter reached step
    `158` while running with `Steps/s 16.7`, `Grid Loss 0.000164`,
    true `Motion Loss 0.007612`, LR `0.9`, 96 samples/step, and `2018` motion
    frame/pixel samples
  - after pause the stat settled to `Steps/s 0.0`
  - browser console warnings/errors after reload: `[]`
- Capacity/default sweep:
  - 384 splats, Dynamic splats-style, LR `0.9`, 96 samples/step:
    true `Motion Loss 0.009779 -> 0.006904` by step `648`, with `Steps/s`
    roughly `16.7-20.7`, no warnings/errors
  - 768 splats, same config:
    true `Motion Loss 0.009557 -> 0.006711` by step `339`, but only
    `7.0-7.9` steps/s, no warnings/errors
  - 512 splats, same config:
    true `Motion Loss 0.009794 -> 0.006685` by step `553`, with `11.6-13.5`
    steps/s, no warnings/errors
- `converge22` post-edit smoke:
  - in-app reload loaded
    `app.js?v=20260707-converge22` and `styles.css?v=20260707-converge22`
  - default World Tubes-style boot showed 512 splats, `Grid Loss 0.000208`,
    true `Motion Loss 0.009774`, LR `0.9`, 96 samples/step, and `2018` motion
    frame/pixel samples
  - Dynamic splats-style reset showed 512 splats, `Grid Loss 0.000206`, true
    `Motion Loss 0.009794`
  - short Dynamic splats-style run reached step `174`, `Grid Loss 0.000148`,
    true `Motion Loss 0.007406`, `Steps/s 12.0`, no warnings/errors
- Temporal-support sweep at 512 splats:
  - `0.18`: true `Motion Loss 0.009794 -> 0.007093` by step `390`, no
    warnings/errors
  - `0.22`: true `Motion Loss 0.009794 -> 0.006991` by step `385`, no
    warnings/errors
  - matched `0.26`: true `Motion Loss 0.009794 -> 0.006877` by step `392`, no
    warnings/errors
  - `0.30`: true `Motion Loss 0.009794 -> 0.006794` by step `375`; longer run
    reached `0.006591` by step `574`, no warnings/errors
- `converge23` post-edit smoke:
  - in-app reload loaded
    `app.js?v=20260707-converge23` and `styles.css?v=20260707-converge23`
  - default World Tubes-style boot showed 512 splats, temporal `0.30`,
    `Grid Loss 0.000216`, true `Motion Loss 0.009968`, LR `0.9`, 96
    samples/step, and `2018` motion frame/pixel samples
  - Dynamic splats-style reset showed 512 splats, temporal `0.30`,
    `Grid Loss 0.000214`, true `Motion Loss 0.009983`
  - short Dynamic splats-style run reached step `237`, `Grid Loss 0.000145`,
    true `Motion Loss 0.007114`, `Steps/s 12.6` while running, no warnings/errors
- Post-`converge23` LR/sampler probes at 512 splats, temporal `0.30`, Dynamic
  splats-style:
  - constant LR `0.90`, 96 samples/step, old 75% motion mix: initial true
    `Motion Loss 0.009983`, then step `810`, `Grid Loss 0.000134`, true
    `Motion Loss 0.006431`
  - decaying LR from `0.90` to `0.45` around step `360` was negative: final
    step `835`, true `Motion Loss 0.006523`
  - 128 samples/step was also not a better default: final step `713`, true
    `Motion Loss 0.006481`, with lower throughput than the 96-sample run
- `converge24`/`converge25` motion-mix verification:
  - 75% mix reached step `436`, `Grid Loss 0.000139`, true
    `Motion Loss 0.006756`
  - 90% mix reached step `451`, `Grid Loss 0.000139`, true
    `Motion Loss 0.006623`
  - 95% mix reached step `444`, `Grid Loss 0.000139`, true
    `Motion Loss 0.006608`
  - longer 95% mix run loaded `app.js?v=20260707-converge24`, reached step
    `831`, `Grid Loss 0.000135`, true `Motion Loss 0.006320`, no browser
    warnings/errors
  - decision: promote 95% Motion Mix as `converge25`, while keeping 512
    splats, temporal `0.30`, LR `0.90`, and 96 samples/step
- `converge25` post-edit smoke:
  - in-app reload loaded
    `app.js?v=20260707-converge25` and `styles.css?v=20260707-converge25`
  - default boot showed 512 splats, temporal `0.30`, LR `0.90`,
    samples/step `96`, Motion Mix `95%`, GPU `apple`, and no browser
    warnings/errors
  - Dynamic splats-style short run reached step `263`, `Grid Loss 0.000144`,
    true `Motion Loss 0.006886`, no browser warnings/errors
- `converge26` 95%-sampler capacity retest:
  - 768 splats / Dynamic splats-style / temporal `0.30` / LR `0.90` /
    96 samples/step / Motion Mix `95%`: initial true `Motion Loss 0.009754`,
    then step `522`, `Grid Loss 0.000135`, true `Motion Loss 0.006201`, no
    browser warnings/errors
  - matched 512-splat rerun under the same settings: initial true
    `Motion Loss 0.009983`, then step `565`, `Grid Loss 0.000139`, true
    `Motion Loss 0.006505`, no browser warnings/errors
  - decision: promote 768 splats as `converge26`
- 64 samples/step at 768 splats is neutral:
  - 768 splats / Dynamic splats-style / temporal `0.30` / LR `0.90` /
    Motion Mix `95%` / 64 samples/step reached step `568`,
    `Grid Loss 0.000134`, true `Motion Loss 0.006218`, no browser
    warnings/errors
  - this is close to but not clearly better than the 96-sample 768 run
    (`0.006201` by step `522`), so the default stays 96 samples/step
- `converge26` post-edit smoke:
  - in-app reload loaded
    `app.js?v=20260707-converge26` and `styles.css?v=20260707-converge26`
  - default World Tubes-style boot showed 768 splats, temporal `0.30`, LR
    `0.90`, samples/step `96`, Motion Mix `95%`, GPU `apple`, no browser
    warnings/errors
  - Dynamic splats-style short run reached step `192`, `Grid Loss 0.000142`,
    true `Motion Loss 0.006806`, no browser warnings/errors
- `converge27` diagnostics:
  - boot loaded `app.js?v=20260707-converge27`, 768 splats, Motion Mix `95%`,
    no warnings/errors
  - initial World Tubes-style diagnostics: `455/768` active splats, motion
    coverage `39.6%`, true `Motion Loss 0.009720`
  - initial Dynamic splats-style diagnostics: `455/768` active splats, motion
    coverage `39.8%`, true `Motion Loss 0.009754`
  - Dynamic splats-style run reached step `415`, `Grid Loss 0.000136`, true
    `Motion Loss 0.006318`, motion coverage `29.9%`, no browser
    warnings/errors
  - diagnosis: the old init/training path was improving motion MSE while
    reducing dynamic coverage on the very samples we care about
- `converge28` motion-aware init verification:
  - boot loaded `app.js?v=20260707-converge28`, 768 splats, Motion Mix `95%`,
    no warnings/errors
  - initial Dynamic splats-style diagnostics: `504/768` active splats, motion
    coverage `63.3%`, true `Motion Loss 0.011483`
  - short Dynamic splats-style run reached step `470`, `Grid Loss 0.000160`,
    true `Motion Loss 0.005853`, motion coverage `41.6%`, no browser
    warnings/errors
  - extended run reached step `854`, `Grid Loss 0.000149`, true
    `Motion Loss 0.005459`, motion coverage `38.2%`, no browser warnings/errors
  - decision: keep `converge28`; it is a better motion-region default despite a
    small grid-loss tradeoff
- `converge29`/`30` diagnostic UI:
  - surfaced `Peak Alpha`, `Mean Opac`, and `Mean Radius` from the existing
    validation readback
  - made the desktop rail independently scrollable, then bumped the browser
    asset version again so the app reloads the CSS/module graph cleanly
  - no training math changed in this pass
  - rationale: the visible symptom is "not converging" / "not many splats"; the
    current browser path already defaults to 768 splats, but its simplified WGSL
    backward is all-splats-per-sample and effectively quadratic in splat count,
    so the next trace should first show whether coverage falls because
    primitives shrink, fade, or merely specialize
  - boundary rechecked: the Shader Mode menu selects two parameterizations
    inside one simplified browser trainer, not the native Metal
    shared-backward/tape/tiled compositor; initialization is deterministic
    target-grid/color plus motion-pixel seeding, not COLMAP/pycolmap/VGGT
- `converge31` motion-support guard:
  - audited the source-over alpha/color/center/radius/time-center derivative
    signs and did not find an obvious sign error
  - added a small motion-sample-only coverage hinge: if dynamic coverage on a
    motion sample falls below `0.50`, a weight-`0.20` alpha-gradient term flows
    through opacity, center/motion, radius, and temporal center
  - color gradients still come only from RGB reconstruction loss
  - this is a targeted support-preservation test for the simplified WGSL
    objective, not a Metal shared-backward/tile compositor port
  - browser test: Dynamic splats-style reached step `468`, `Grid Loss
    0.000244`, true `Motion Loss 0.006782`, motion coverage `53.7%`, mean
    opacity `8.5%`, mean radius `0.0147`, and no browser warnings/errors
  - read: this is too strong for the current default; it preserves support but
    lags the previous `converge28` step-470 motion loss (`0.005853`)
- `converge32` weakened motion-support guard:
  - changed the hinge to target `0.44`, weight `0.08`, so it should act only as
    a late guard once motion coverage is near the range we wanted to protect
  - browser test: Dynamic splats-style reached step `169`, `Grid Loss
    0.000200`, true `Motion Loss 0.007038`, motion coverage `52.5%`, mean
    radius `0.0142`, no browser warnings/errors
  - continued trace reached step `445`, `Grid Loss 0.000189`, true
    `Motion Loss 0.006263`, motion coverage `48.4%`, mean radius `0.0141`
  - extended trace reached step `861`, `Grid Loss 0.000189`, true
    `Motion Loss 0.005914`, motion coverage `47.0%`, mean radius `0.0140`, no
    browser warnings/errors
  - read: keep `converge32` as the support-health default for now; it gives up
    some raw motion MSE versus `converge28` (`0.005459` at step `854`) but avoids
    the same coverage collapse (`47.0%` vs `38.2%`)
- `converge33` mode-label and default check:
  - reloaded the app and confirmed the current default is World Tubes-style
  - World Tubes-style reached step `295`, `Grid Loss 0.000196`, true
    `Motion Loss 0.006440`, motion coverage `49.8%`, mean radius `0.0141`, no
    browser warnings/errors
  - extended World Tubes-style reached step `629`, `Grid Loss 0.000194`, true
    `Motion Loss 0.005938`, motion coverage `47.4%`, mean radius `0.0140`, no
    browser warnings/errors
  - read: keep World Tubes-style as the default under the weakened support guard
    because it effectively ties Dynamic splats-style (`0.005914` / `47.0%` at
    step `861`) while staying closer to the project thesis
  - renamed the UI label from `Shader Mode` to `Motion Model`; these are still
    two parameter branches inside one simplified WGSL trainer, not two full
    native shader-family ports

## Next useful work

Do not treat this as Metal parity. The next real port is to pick one proven
Metal lane and translate its renderer contract to WebGPU deliberately:

1. STAR/WorldTubes direct-atomic style: tile/bin support, depth/alpha ordering,
   fixed-order VJP, then shared camera-family traces.
2. Dynamic 3DGS fast-mac style: depth-aware alpha compositor, projected
   Gaussian params, and a matched browser loss.
3. Init bridge: export a small saved point/splat bundle from the native trainer
   or a geometry initializer, then let the browser fine-tune rather than
   starting from image-space target-grid seeds.

## 2026-07-07 follow-up: `converge34` visual diagnostics

User complaint was still visual non-convergence, so I rechecked the live browser
page before changing math. The boot state was World Tubes-style, `504/768` active
splats, `Motion Loss 0.011522`, `Motion Cov 63.1%`, `Peak Alpha 10.4%`,
`Mean Radius 0.0147`, and only `2018` packed motion pixels. A short step-65 run
already dropped true `Motion Loss` to `0.007950`, so the app was learning, but
the screen made that hard to judge because the target pane was tiny/letterboxed
next to a much larger render pane.

`converge34` keeps training math unchanged and changes the debugging surface:

- `Target View` menu with `RGB target` and `Motion residual`
- residual view draws amplified `abs(frame - mean_background)` so the sparse
  moving person/glass region is visible
- desktop workbench uses equal target/render columns
- cache bumped to `20260707-converge34`

Browser verification:

- reload confirmed `app.js?v=20260707-converge34` and
  `styles.css?v=20260707-converge34`
- target view selector exposes `rgb` and `motion_residual`
- equal desktop columns reported as `469.5px 469.5px`
- switching to residual updated the title to `frame 2 residual`
- post-edit World Tubes-style trace reached step `268`, `Grid Loss 0.000193`,
  true `Motion Loss 0.006505`, `Motion Cov 50.2%`, `Peak Alpha 9.2%`,
  `Mean Opac 8.5%`, `Mean Radius 0.0141`, no browser warnings/errors

One longer browser-automation wait timed out at the CDP command layer while the
app kept training; a recovery read paused the run cleanly at step `268`. Treat
that as automation friction, not a WebGPU training failure.

## 2026-07-07 follow-up: `converge38`/`39` result diagnostics and temporal floor

Added a result-side diagnostic after `converge34` made the target residual
visible but left the render pane in RGB. `Result View` now has:

- `RGB result`
- `Dynamic layer`
- `Alpha support`

The first implementation tried to sample the mean background storage buffer from
the splat fragment to draw a more literal residual. On the in-app Apple WebGPU
path this blacked out the render pane while the trainer still reported `Ready`.
Backing out the fragment background-buffer read and the extra fragment-position
argument restored RGB rendering in `converge38`. Keep this in mind before adding
fragment storage-buffer diagnostics again.

The `converge38` alpha-support screenshot showed the useful truth: the browser
model does find the moving person, but it also leaves a broad speckled dynamic
layer across the background. That makes the issue more specific than "not
converging":

- not missing support
- not random init failure
- not no motion learning
- likely broad temporal/background support plus imperfect residual/color
  isolation

Math smell found: the temporal gate floor was too high. At default temporal
support `0.30`, the old floor was `sigma*0.70 = 0.21`, meaning every splat kept
about 21% temporal opacity in every frame. `converge39` lowers this to
`sigma*0.30` clamped to `0.035..0.12`, so the default tail is now `0.09`.

Browser results:

- `converge38` boot: `Motion Loss 0.011522`, `Motion Cov 63.1%`
- `converge39` boot: `Motion Loss 0.011099`, `Motion Cov 59.7%`
- `converge39` short World Tubes-style trace: step `72`, `Grid Loss 0.000214`,
  true `Motion Loss 0.007788`, `Motion Cov 53.9%`, `Peak Alpha 9.7%`,
  `Mean Opac 8.5%`, `Mean Radius 0.0144`
- support screenshot pixel check changed modestly (`frac>30` in result crop
  `27.1% -> 26.2%`)

Artifacts:

- `output/browser_trainer/converge38_alpha_support_result_viewport2_step68.png`
- `output/browser_trainer/converge38_dynamic_layer_result_step68.png`
- `output/browser_trainer/converge39_alpha_support_result_step72.png`

Read: `converge39` is a small real improvement, not the final fix. The next
useful browser-side math change should target dynamic-layer sparsity/color
isolation, for example opacity decay/sparsity on non-motion support or a cleaner
residual-targeted objective, before spending more time on temporal/support
knobs.

## 2026-07-07 follow-up: `converge40`/`41`/`42` sparsity objective

The next pass targeted the broad dynamic layer rather than more motion-support
preservation. I added:

- `Static Cov`, a validation metric for dynamic alpha leaking onto low-motion
  grid samples
- a low-motion alpha penalty in the WGSL train kernel
- global opacity decay before the opacity-logit update

`converge40` used opacity decay `0.055`. It made the dynamic layer sparser, but
the decay was too blunt:

- boot: `Motion Loss 0.011099`, `Motion Cov 59.7%`, `Static Cov 2.8%`
- step `35`: `Motion Loss 0.008492`, `Motion Cov 54.7%`, `Static Cov 2.7%`,
  `Active 476/768`, `Mean Opac 8.2%`
- step `124`: `Motion Loss 0.007333`, `Motion Cov 47.3%`, `Static Cov 2.6%`,
  `Active 414/768`, `Mean Opac 7.4%`
- step `239`: `Motion Loss 0.007012`, `Motion Cov 42.4%`, `Static Cov 2.4%`,
  `Active 368/768`, `Mean Opac 6.5%`

That final row fell below the 44% motion-support guard, so v40 is rejected as a
default.

`converge41` lowered opacity decay to `0.025` while keeping the low-motion alpha
penalty:

- boot: `Motion Loss 0.011099`, `Motion Cov 59.7%`, `Static Cov 2.8%`
- step `143`: `Motion Loss 0.007236`, `Motion Cov 48.9%`, `Static Cov 2.7%`,
  `Active 461/768`, `Mean Opac 7.9%`
- step `294`: `Motion Loss 0.006751`, `Motion Cov 44.6%`, `Static Cov 2.6%`,
  `Active 406/768`, `Mean Opac 7.3%`, `Grid Loss 0.000182`, `Peak Alpha 7.8%`

The alpha-support crop comparison also moved in the desired direction:

- v39 crop: `meanMax 21.912`, `frac>30 0.2618`, `frac>80 0.0538`,
  `frac>140 0.0203`
- v41 crop: `meanMax 20.229`, `frac>30 0.2438`, `frac>80 0.0436`,
  `frac>140 0.0164`

This is still broad support visually, but v41 is the better trade: it reduces
background activity without collapsing below the support guard.

`converge42` keeps v41 train constants and thins `Static Cov` validation to one
quarter of low-motion grid samples to reduce validation/readback cost. Reload
verified `app.js?v=20260707-converge42` and
`styles.css?v=20260707-converge42`. Boot metrics were `Motion Loss 0.011099`,
`Motion Cov 59.7%`, `Static Cov 2.8%`, `Active 504/768`, `Mean Opac 8.5%`. A
one-step smoke exercised the 96-byte train uniform path and reached step `1`,
`Motion Loss 0.010983`, `Static Cov 2.8%`, `Active 503/768`.

Artifacts:

- `output/browser_trainer/converge39_alpha_support_result_step72.png`
- `output/browser_trainer/converge41_alpha_support_result_step294.png`

Read: the browser trainer now has an explicit sparsity/control metric and v42 is
the current safer default, but the honest next proof is a longer v42-vs-v32/v33
trace. Avoid repeating pure support knob sweeps unless the alpha-support view
shows a new failure mode.

## 2026-07-07 follow-up: `converge43` dedicated static samples

Audit after v42 found a sampling bug in the objective shape: the low-motion
alpha penalty existed, but default training was still 95% motion samples. That
meant static cleanup mostly arrived through the small 5% uniform tail rather
than through intentional low-motion samples.

`converge43` adds:

- `computeStaticSamples(...)` in `dataset.js`, using the same `0.00045`
  frame-vs-background energy threshold as the static alpha penalty
- a `staticSamples` storage buffer at train binding `7`
- `staticSampleCount` and `staticSamplePermil` in the existing 96-byte train
  uniform, reusing the old pad slots
- a default 8% static sample reserve; the motion sample rate is clamped to
  leave room for that reserve
- a `Static Px` sidebar stat

In-app reload verified:

- `app.js?v=20260707-converge43`
- `styles.css?v=20260707-converge43`
- `Motion Px 2018`
- `Static Px 16384`
- boot `Motion Loss 0.011099`
- boot `Static Cov 2.8%`

Trace:

- step `153`: `Grid Loss 0.000192`, `Motion Loss 0.007153`, `Motion Cov
  48.5%`, `Static Cov 2.7%`, `Peak Alpha 8.5%`, `Active 459/768`,
  `Mean Opac 7.8%`, `Mean Radius 0.0143`
- step `184`: `Grid Loss 0.000189`, `Motion Loss 0.007054`, `Motion Cov
  47.6%`, `Static Cov 2.6%`, `Peak Alpha 8.4%`, `Active 448/768`,
  `Mean Opac 7.7%`, `Mean Radius 0.0143`
- step `259` after pausing: `Grid Loss 0.000179`, `Motion Loss 0.006803`,
  `Motion Cov 45.5%`, `Static Cov 2.6%`, `Peak Alpha 8.0%`,
  `Active 420/768`, `Mean Opac 7.4%`, `Mean Radius 0.0142`

Browser console warnings/errors were empty. One browser-control call timed out
while waiting on a long trace and reset the automation session; the page kept
training and was paused cleanly after reconnecting. Treat that as automation
friction, not app/runtime failure.

Read: v43 makes the static cleanup term less accidental and is comparable to
v41 at a similar support state, but it is not a visual convergence proof. The
next useful browser check is a saved RGB/alpha-support comparison of v43 versus
the prior support-health v32/v33 behavior. If that still looks broad, stop
working this as sampler/decay tuning and move to real tile/depth/alpha
compositing or better geometry/init.

## 2026-07-07 follow-up: `converge44` static-reserve A/B control

After v43, the next blocker was measurement control: the 8% static reserve was
hardcoded, so comparing against the old v42-style sampler required source edits.
`converge44` exposes the reserve as a `Static Mix` slider:

- default `Static Mix` is `8%`
- range is `0%..16%`
- `Motion Mix` now displays the effective motion share after static reservation
- requested motion `95%` plus static `8%` displays effective motion `92%`
- setting `Static Mix` to `0%` restores effective motion `95%`, matching the
  v42-style sampler

In-app verification:

- reload loaded `app.js?v=20260707-converge44` and
  `styles.css?v=20260707-converge44`
- boot showed `Motion Px 2018`, `Static Px 16384`, `Motion Mix 92%`,
  `Static Mix 8%`, `Motion Loss 0.011099`, `Static Cov 2.8%`
- one-step smoke reached step `1`, `Motion Loss 0.010985`, `Motion Cov 59.6%`,
  `Static Cov 2.8%`, `Active 503/768`
- browser logs were empty
- setting `Static Mix` to `0%` changed labels to `Motion Mix 95%`,
  `Static Mix 0%`; restored to `8%` afterward

Read at the time: v44 did not prove a better visual result by itself; it gave
the browser trainer the control needed for the next `Static Mix 0%` versus `8%`
proof. That proof is now recorded in the follow-up below and supersedes this
as an open next step.

## 2026-07-07 follow-up: matched v44 static mix control and `converge45`

Ran the matched static-reserve control in the in-app browser on the Apple
WebGPU adapter, World Tubes-style, 768 splats, LR `0.90`, 96 samples/step,
temporal support `0.30`, requested motion mix `95%`.

Artifacts:

- `outputs/browser_trainer/2026-07-07_v44_static_mix_ab/static_mix_0_step274_metrics.json`
- `outputs/browser_trainer/2026-07-07_v44_static_mix_ab/static_mix_8_step271_metrics.json`
- `outputs/browser_trainer/2026-07-07_v44_static_mix_ab/static_mix_8_support52_step297_metrics.json`

Matched v44 result:

- `Static Mix 0%`: step `274`, `Grid Loss 0.000182`,
  `Motion Loss 0.006794`, `Motion Cov 45.0%`, `Static Cov 2.6%`,
  `Peak Alpha 7.9%`, `Active 414/768`, `Mean Opac 7.4%`,
  `Mean Radius 0.0143`.
- `Static Mix 8%`: step `271`, `Grid Loss 0.000183`,
  `Motion Loss 0.006822`, `Motion Cov 45.3%`, `Static Cov 2.6%`,
  `Peak Alpha 7.9%`, `Active 415/768`, `Mean Opac 7.4%`,
  `Mean Radius 0.0142`.

Read: static reserve is not the convergence culprit. Both arms are essentially
tied and settle near the hidden `motionCoverageTarget=0.44`. The user's
"not a lot of splats" read is real, but it is mostly the objective/guard
balance plus simplified renderer/init, not the new low-motion sample reserve.

`converge45` exposes that hidden target as a `Support Guard` slider:

- default `Support Guard 52%`
- range `40%..60%`
- `app.js` forwards `motionCoverageTarget` into `trainer.trainStep(...)`
- `trainerWebGpu.js` uses that value instead of hardcoding `0.44`
- assets bumped to `app.js?v=20260707-converge45` and
  `styles.css?v=20260707-converge45`

In-app v45 reload verified `Motion Mix 92%`, `Static Mix 8%`,
`Support Guard 52%`, `Motion Px 2018`, `Static Px 16384`, and boot metrics
matched v44 before training.

v45 trace:

- step `29`: `Motion Loss 0.008904`, `Motion Cov 56.9%`, `Active 494/768`
- step `84`: `Motion Loss 0.007901`, `Motion Cov 53.5%`, `Active 476/768`
- step `144`: `Motion Loss 0.007412`, `Motion Cov 51.1%`, `Active 461/768`
- step `224`: `Motion Loss 0.007172`, `Motion Cov 49.3%`, `Active 430/768`
- step `297`: `Grid Loss 0.000200`, `Motion Loss 0.007060`,
  `Motion Cov 48.2%`, `Static Cov 2.7%`, `Peak Alpha 8.1%`,
  `Active 406/768`, `Mean Opac 7.3%`, `Mean Radius 0.0146`

Read: v45 preserves a few more motion-coverage points but costs motion MSE
versus v44 (`0.007060` vs `0.006822` at similar horizon). Keep it because the
browser UI now lets us choose the tradeoff, but do not continue this as another
support-target sweep. The next real move is renderer/init parity:
depth/tile/alpha/transmittance behavior, shared-backward/tape-style reuse, and
exporting trained init/data bundles into the browser path.

Automation caveat: metric JSONs are reliable. The saved viewport screenshots in
the same folder are not reliable canvas captures after reload because the
browser screenshot helper returned viewport clips rather than element pixels;
use browser-side visual inspection or a proper Playwright element-screenshot
route for future image artifacts.

## 2026-07-07 follow-up: `converge46` frame-motion centroid init

Math audit after v45 found a real initialization mismatch: motion-seeded splats
were placed on high-motion pixels with local time centers, but their velocity
started as tiny random noise. That means the browser optimizer had to discover
coherent motion from sparse source-view samples, even though the loaded frames
already contain a weak image-space motion cue.

`converge46` adds a lightweight source-view motion prior:

- estimate per-frame motion centroids from target-vs-mean-background residual
  energy above `0.0006`
- finite-difference those centroids into normalized image-space velocities
- for motion-seeded splats, initialize `motion.xy` from half that velocity,
  clamped to `[-0.10, 0.10]` with small jitter
- back-solve `posRadius.xy` so the tube center still lands on the selected
  high-motion frame/pixel at the splat's `timeCenter`
- keep this entirely in the browser init path; it is not COLMAP/SfM/VGGT

Verification:

- `node --check web/dynaworld_browser_trainer/app.js`
- `node --check web/dynaworld_browser_trainer/dataset.js`
- `node --check web/dynaworld_browser_trainer/trainerWebGpu.js`
- `curl -I http://127.0.0.1:8080/web/dynaworld_browser_trainer/` returned
  `200 OK`
- in-app reload loaded `app.js?v=20260707-converge46`
- boot showed `Motion Mix 92%`, `Static Mix 8%`, `Support Guard 52%`,
  `Motion Px 2018`, `Static Px 16384`, `Motion Loss 0.011103`,
  `Motion Cov 57.1%`, `Static Cov 2.9%`, `Active 504/768`

Trace:

- step `39`: `Motion Loss 0.008461`, `Motion Cov 53.4%`, `Active 488/768`
- step `95`: `Motion Loss 0.007707`, `Motion Cov 50.6%`, `Active 473/768`
- step `168`: `Motion Loss 0.007223`, `Motion Cov 48.4%`, `Active 455/768`
- step `246`: `Motion Loss 0.007055`, `Motion Cov 47.4%`, `Active 422/768`
- step `290`: `Grid Loss 0.000211`, `Motion Loss 0.007036`,
  `Motion Cov 47.0%`, `Static Cov 2.8%`, `Peak Alpha 8.2%`,
  `Active 407/768`, `Mean Opac 7.4%`, `Mean Radius 0.0146`

Saved metrics:

- `outputs/browser_trainer/2026-07-07_v46_motion_init/motion_centroid_init_support52_step290_metrics.json`

Read: v46 is a small fit win over v45 at a similar horizon (`0.007036` vs
`0.007060`) but a support loss (`47.0%` vs `48.2%`). It is useful evidence that
the initialization was too random, but it is not the convergence fix. The next
mechanism should be real renderer/export parity or a better per-splat local
motion/geometry initializer, not another global support guard tweak.

## 2026-07-07 follow-up: `converge47` local residual-match motion init

v46's global centroid prior was too coarse: it improved fit slightly but lost
support. The next image-space initializer step was to estimate per-splat local
motion from nearby residual pixels rather than from a single frame centroid.

`converge47` changes the motion-seeded init:

- each selected high-motion sample keeps its frame/pixel/time center
- for adjacent frames, search a `7px` local window around the selected pixel
- score candidate matches by residual color similarity, small spatial cost, and
  a small residual-energy reward
- finite-difference the best previous/next local matches into an image-space
  velocity
- blend `75%` local velocity with `25%` frame-centroid fallback
- initialize `motion.xy` from the blended velocity and back-solve `posRadius.xy`
  so the World Tubes-style center still lands on the selected frame/pixel

Verification:

- `node --check web/dynaworld_browser_trainer/app.js`
- `node --check web/dynaworld_browser_trainer/dataset.js`
- `node --check web/dynaworld_browser_trainer/trainerWebGpu.js`
- `curl -I http://127.0.0.1:8080/web/dynaworld_browser_trainer/` returned
  `200 OK`
- in-app reload loaded `app.js?v=20260707-converge47`
- boot showed `Motion Mix 92%`, `Static Mix 8%`, `Support Guard 52%`,
  `Motion Px 2018`, `Static Px 16384`, `Motion Loss 0.010755`,
  `Motion Cov 59.3%`, `Static Cov 2.8%`, `Active 504/768`

Trace:

- step `31`: `Motion Loss 0.008415`, `Motion Cov 55.9%`, `Active 494/768`
- step `76`: `Motion Loss 0.007657`, `Motion Cov 53.2%`, `Active 476/768`
- step `141`: `Motion Loss 0.007212`, `Motion Cov 50.9%`, `Active 461/768`
- step `217`: `Motion Loss 0.006950`, `Motion Cov 49.0%`, `Active 435/768`
- step `279`: `Grid Loss 0.000194`, `Motion Loss 0.006885`,
  `Motion Cov 48.1%`, `Static Cov 2.7%`, `Peak Alpha 8.2%`,
  `Active 414/768`, `Mean Opac 7.4%`, `Mean Radius 0.0145`

Saved metrics:

- `outputs/browser_trainer/2026-07-07_v47_local_motion_init/local_motion_init_support52_step279_metrics.json`

Read: this is the first clean positive local browser result after v44. It
beats v45 fit at matched support (`0.006885` / `48.1%` versus `0.007060` /
`48.2%`) and beats v46's global centroid prior on both fit and support
(`0.007036` / `47.0%`). It still does not make the browser trainer a full
Metal/STAR/WorldTubes renderer port; the remaining gap is depth/order/tile
compositing, richer local geometry, and eventually source/heldout camera
training/exported bundles.

## 2026-07-08 follow-up: `converge48` looping preview and source/target camera strip

The user asked whether the preview could show multiple camera angles and loop
over time rather than appearing static. The current browser trainer remains a
single source-view image-space objective, so this is not true novel-view
rendering. The local Neural3D preview clip is a side-by-side camera preview,
though, so the UI can show both cropped panes as visual context while keeping
training on the source pane.

Changes:

- bumped browser assets to `app.js?v=20260707-converge48` and
  `styles.css?v=20260707-converge48`
- added `Loop Time` and `Loop Speed` controls; loop time is enabled by default
  and advances `timeSlider` inside the existing requestAnimationFrame loop
- added a source/target camera strip under the target canvas; both mini-canvases
  draw the same looped time as the main target/result panes
- `loadPresetDataset()` now loads the default source-view crop and, when
  available, attaches a target-view preview crop via `dataset.previewViews`
- did not change train math, splat params, shader gradients, or validation
  metrics

Resolution/performance clarification:

- `ffprobe` reports the local preview mp4 as `512x256`, i.e. two square
  `256x256` camera panes side by side
- the browser trainer crops one pane and decodes the training target to
  `128x128x8`
- visible canvas resolution can be much larger, but the current train cost is
  mostly splat count and samples-per-step: one compute worker per splat, each
  worker loops samples and recomputes `eval_model(...)` over all splats
- target/canvas resolution is comparatively cheap for training at the current
  fixed sample count, though render fragments and CPU validation still scale
  with pixels

Verification:

- `node --check web/dynaworld_browser_trainer/app.js`
- `node --check web/dynaworld_browser_trainer/dataset.js`
- `node --check web/dynaworld_browser_trainer/trainerWebGpu.js`
- restarted `python3 -m http.server 8080` from repo root
- `curl -I http://127.0.0.1:8080/web/dynaworld_browser_trainer/` returned
  `200 OK`
- served HTML/assets contain `converge48`, `Loop Time`,
  `advancePreviewTime(...)`, and `previewViews`

Limitation:

- after the user interruption, the in-app browser connector reported zero
  attached tabs, so no fresh visual in-app browser smoke was possible in this
  pass. Do not claim visual verification until the connector can see the tab
  again.

## 2026-07-08 follow-up: `converge49` validation visual metrics and error map

The user asked whether validation has visual-difference metrics beyond scalar
loss, and whether standard losses such as SSIM or regularization techniques are
present.

Answer before the patch:

- validation had deterministic `Grid Loss`, direct packed-motion `Motion Loss`,
  motion/static coverage, peak alpha, active splats, mean opacity, and mean
  radius
- it did not expose MAE, PSNR, SSIM, or a prediction-error image
- training regularization was browser-specific: temporal gate, support guard,
  low-motion alpha penalty, static sample reserve, opacity decay, radius/opacity
  clamps
- SSIM/DSSIM, LPIPS, total variation, velocity/acceleration smoothness, and
  alpha-budget regularizers were not implemented as losses

Changes:

- bumped browser assets to `app.js?v=20260707-converge49`
- added `Val MAE`, `Val PSNR`, and `Val SSIM` sidebar stats
- `Val SSIM` is a global luma SSIM approximation over the same sparse
  validation grid as `Grid Loss`; it is validation-only, not a differentiable
  training loss
- added `Target View -> Validation error`, which throttles GPU-param readback,
  evaluates the current source-view prediction on CPU for the selected frame,
  and draws an RGB-error heat map in the target pane
- did not change training math

Verification:

- `node --check web/dynaworld_browser_trainer/app.js`
- `node --check web/dynaworld_browser_trainer/dataset.js`
- `node --check web/dynaworld_browser_trainer/trainerWebGpu.js`
- `curl -I http://127.0.0.1:8080/web/dynaworld_browser_trainer/` returned
  `200 OK`

Read: adding SSIM as a metric is cheap and useful. Adding SSIM/DSSIM as a train
loss should be tested separately because the current failure mode is dynamic
support and local motion fit; perceptual/global scalar improvements can hide
the same support collapse that earlier grid loss hid.
