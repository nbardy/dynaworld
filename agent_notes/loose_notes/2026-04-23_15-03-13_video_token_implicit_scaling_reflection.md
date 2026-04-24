# Video-Token Implicit Scaling Reflection

Context:
    User believes the failure is not "just implicit camera," which matches the
    evidence so far. User asked for a reflective long-form note, a precise
    explanation of how implicit camera decoding currently works, what benchmark
    data is ready, and whether increasing splat count to 64k is a promising
    next test.

Status:
    active reflection note

Scope:
    This is not a polished conclusion. It is a durable engineering note meant
    to capture the current model, weakened assumptions, competing branches, and
    next experiments.

---

## 1. Observed Facts

### 1.1 Current small-baseline facts

- `32px / 2fps / 23 frames / 512 splats` known-camera feed-in run `w3a14cnp`
  reached:
  - `Eval/Loss 0.06593`
  - `Eval/L1 0.06392`
  - `Eval/SSIM 0.51251`
  - `Eval/PSNR 19.98`
- `32px / 2fps / 23 frames / 512 splats` image-implicit run `mv4ggjq8`
  reached:
  - `Eval/Loss 0.06030`
  - `Eval/L1 0.05853`
  - `Eval/SSIM 0.59131`
  - `Eval/PSNR 20.54`

Inference:
    Implicit camera as a concept is not generically broken. On the small
    image-encoder baseline it works very well.

### 1.2 Current video-token implicit facts

- Time bug was real:
  - trainer used local-window `0..1` times
  - query decode time entered only after query-to-video attention
- That was fixed:
  - trainer now passes absolute normalized frame-index time
  - query decode time enters before query cross-attention
  - encoder now also receives explicit time/span
  - head tokens now get explicit decode-time offsets before camera/gaussian heads
- Zero-init head bug was real:
  - all splats started as identical gray blobs at the origin
  - this was fixed by reusing `GaussianParameterHeads`
- Yet the 128px video-token implicit baseline still trails badly:
  - `etcghanx` 200-step bounded-init probe:
    - `Eval/Loss 0.14872`
    - `Eval/L1 0.09697`
    - `Eval/SSIM 0.28856`
  - `op7l2ypp` 200-step wide-camera probe:
    - `Eval/Loss 0.15014`
    - `Eval/L1 0.10032`
    - `Eval/SSIM 0.30114`
  - `febn4gq6` 200-step explicit-encoder-time probe:
    - `Eval/Loss 0.15407`
    - `Eval/L1 0.10153`
    - `Eval/SSIM 0.27152`
    - `Eval/TemporalAdjacentL1Ratio 0.01729`
    - `Eval/TemporalToFirstL1Ratio 0.26962`
  - `2vllet1i` 100-step `65536`-splat explicit-time probe:
    - `Eval/Loss 0.15387`
    - `Eval/L1 0.10286`
    - `Eval/SSIM 0.28416`
    - `Eval/PSNR 16.41`
    - `Eval/TemporalAdjacentL1Ratio 0.01513`
    - `Eval/TemporalToFirstL1Ratio 0.19692`
    - `Camera/EvalRotationDeltaMeanDegrees 7.19`
    - `Camera/EvalTranslationDeltaMean 0.604`

Inference:
    Missing time was part of the problem, but not the whole problem. Fixing
    time correctness did not recover the baseline. Increasing to 64k splats
    improved SSIM only slightly relative to the 8192-splat explicit-time run,
    but it greatly increased camera motion magnitude. Capacity matters, but not
    in the simple "more splats immediately fixes reconstruction" sense.

### 1.3 Current known-camera comparison facts

- `128px / 4fps / 46 frames / 65536 splats` known-camera run `3piy8cww`:
  - `Eval/Loss 0.10858`
  - `Eval/L1 0.07233`
  - `Eval/SSIM 0.49289`
  - `Eval/PSNR 18.93`
- Closest stable `128px / 4fps / 8192 splats` known-camera comparison:
  - `jyw88o4r` summary logged no SSIM, but SSIM computed from logged
    render/GT videos is about `0.29315`
  - same video-based estimate gives:
    - `L1 ~0.08890`
    - `PSNR ~17.34`

Inference:
    The current implicit run `febn4gq6` at `SSIM 0.2715` is slightly behind the
    closest stable 8192-splat known-camera run and far behind the strong 65k
    known-camera baseline.

### 1.4 Historical scaling facts across size/fps

- `32px / 2fps / 512 splats` known-camera:
  - `SSIM 0.5125`
- `64px / 4fps / 512 splats` known-camera:
  - summary lacked SSIM; computed from logged final render/GT videos:
    - `SSIM ~0.1810`
    - `L1 ~0.15445`
    - `PSNR ~14.00`
- `128px / 4fps / 8192 splats` video-token implicit:
  - `SSIM 0.2715`

Important density math:

```text
32x32  = 1024 pixels;  512 splats   => 0.5 splats/pixel  = 1 splat / 2 px
64x64  = 4096 pixels;  512 splats   => 0.125 splats/pixel = 1 splat / 8 px
128x128 = 16384 pixels; 8192 splats => 0.5 splats/pixel  = 1 splat / 2 px
128x128 = 16384 pixels; 65536 splats => 4.0 splats/pixel = 4 splats / px
```

Inference:
    The strong 32px baseline and the current 128px/8192 setup actually have
    similar splat-per-pixel density, while the old 64px/4fps/512-splat setup
    was much more under-capacity spatially. So the current gap is not explained
    by gross splat density alone.

---

## 2. How Implicit Camera Currently Works

Observed fact:
    The implicit-camera path does not decode Pluecker rays. Pluecker/ray
    conditioning exists in the known-camera feed-in model, where the camera is
    already known and the encoder receives ray grids.

Files:
    `src/train/gs_models/implicit_camera.py`
    `src/train/gs_models/dynamic_token_gs_implicit_camera.py`
    `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`

### 2.1 Global camera decode

Current behavior:
    `GlobalCameraHead` reads one global camera token and predicts only:

    - FoV residual
    - radius residual

Formula:

```text
raw = net(global_camera_token)          # shape [..., 2]
fov = base_fov + tanh(raw[0]) * max_fov_delta
radius = base_radius * exp(tanh(raw[1]) * max_log_radius_delta)
```

Then it constructs a base orbit camera:

```text
make_orbit_camera(
    image_size=image_size,
    radius=radius,
    azimuth=0,
    elevation=0,
    focal=focal_from_fov,
)
```

Interpretation:
    The starting camera points at the origin. It does not yet have a learned
    position/direction token in free 6DoF coordinates. It starts from a fixed
    canonical orbit pose with learned FoV/radius.

### 2.2 Per-frame motion decode

Current behavior:
    `PathCameraHead` reads one path token per decoded frame and predicts:

    - axis-angle rotation delta in 3 values
    - translation delta in 3 values

Formula:

```text
raw = net(path_token)                   # shape [B, 6]
rotation_delta = tanh(raw[:3]) * max_rotation_radians
translation_delta = tanh(raw[3:]) * (base_radius * max_translation_ratio)
```

Then `compose_camera_with_se3_delta(...)` builds:

```text
delta_transform[:3,:3] = axis_angle_to_matrix(rotation_delta)
delta_transform[:3, 3] = translation_delta
camera_to_world = base_camera.camera_to_world @ delta_transform
```

Interpretation:
    Camera animation is not "adding raw values to a Pluecker ray." It is a
    learned SE(3) delta composed onto a canonical base camera.

### 2.3 Time dependence of camera animation

Image implicit baseline:
    The path token is time-conditioned before path decoding.

Video-token implicit baseline:
    The path token is time-conditioned in two places:

    1. decode-time query offset before query-to-video cross-attention
    2. direct `head_time_proj(decode_time)` offset before path/camera head

So yes:
    the per-frame camera offset is decoded from a time-conditioned token.

Important limitation:
    The global camera token is still intentionally treated as clip-global. Time
    is applied to path/splat tokens, not to the global camera token itself.

Possible implication:
    This is a reasonable decomposition if the clip shares one scene-level
    canonical camera family and only needs per-frame residual motion. It could
    be wrong if the current representation needs stronger time-dependent camera
    factorization than one base + residual path.

---

## 3. Current Belief

Current belief:
    The main failure is not "implicit camera" by itself. The small image
    implicit baseline falsifies that simple story. The deeper issue is likely in
    the video-token implicit scaling contract: clip compression, token sharing,
    camera/splat factorization, or optimization support for 128px/4fps
    full-scene reconstruction.

Confidence:
    medium

Evidence:
    - small image implicit works
    - time bugs were real but fixing them did not recover performance
    - init bug was real but fixing it did not recover performance
    - current temporal motion remains too smooth
    - camera deltas shrink during training instead of expressing strong motion

Could be wrong if:
    - 64k splats closes most of the gap quickly
    - a better time embedding scheme (e.g. sinusoidal/Fourier) materially fixes
      the problem
    - the main issue is still camera range/support, but not yet in the tested
      form

---

## 4. Hypothesis Branches

## Branch A: Capacity is still the dominant bottleneck

Hypothesis:
    8192 splats are not enough once a single latent world must explain the full
    128px/4fps clip with implicit camera. The clip-memory model may simply need
    a larger world asset.

Why it might be true:
    - The strong known-camera baseline uses 65536 splats.
    - 8192 must absorb both geometry/appearance mismatch and camera uncertainty.
    - The video-token implicit model has to decode all frames from shared memory
      rather than frame-local encoder features.

What would make it false:
    A 64k-splat implicit probe stays near the same SSIM/L1 and still shows
    overly smooth temporal metrics.

Cheap test:
    Run the current video-token implicit model with `gaussians_per_token=512`
    (128 tokens -> 65536 splats) for a 50-100 step smoke, then ideally 200+.

If supported:
    Treat the current 8192 results as under-capacity rather than architectural
    failure.

If invalidated:
    Look harder at the contract and not just size.

Update after `2vllet1i`:
    This branch is weakened but not dead. Capacity clearly changes behavior:
    predicted camera motion becomes much less collapsed. But photometric
    improvement after 100 steps is modest (`SSIM 0.2715 -> 0.2842`), so size
    alone is not enough.

## Branch B: Shared clip-memory is the deeper bottleneck

Hypothesis:
    The video encoder compresses too much into one clip memory, so decoding
    per-time splats from a shared token bank loses frame-specific detail even
    with explicit `t`.

Why it might be true:
    - Small image implicit sees each target frame directly in the encoder.
    - Video-token implicit must decode each frame from shared memory.
    - Temporal-adjacent similarity remains far too high even after explicit
      time fixes.

What would make it false:
    Large-capacity 64k splats with the same architecture closes most of the
    gap.

Cheap test:
    Compare 64k video-token implicit vs 64k known-camera and inspect whether
    temporal ratios move toward GT.

If supported:
    Consider architectural changes at the memory/query boundary, not just more
    splats.

If invalidated:
    The shared memory may be adequate and capacity/optimization may be enough.

## Branch C: Camera factorization is too restrictive

Hypothesis:
    One clip-global base orbit camera plus time-varying residual path is too
    restrictive for the current data distribution.

Why it might be true:
    - DUSt3R pose ranges on the 128px clip show larger changes than the
      original implicit camera default allowed.
    - Predicted translation/rotation statistics remain relatively small.
    - The model may choose smoother camera motion because the world tokens are
      doing too much.

What would make it false:
    64k splats improve heavily without changing camera factorization.

Cheap test:
    Compare camera trajectory magnitudes from implicit predictions against the
    known-camera trajectory on the same frames.

If supported:
    Revisit the camera latent decomposition, not only splat count.

If invalidated:
    Camera factorization is probably not the main obstacle.

## Branch D: Time embedding form matters, not just time presence

Hypothesis:
    Explicit scalar time is present now, but learned MLP projection may still be
    a weak parameterization compared with sinusoidal/Fourier features.

Why it might be true:
    - Continuous phase-sensitive phenomena can be easier to model with Fourier
      features than a shallow learned projector.
    - The model currently uses learned projectors for:
      - encoder tubelet time
      - encoder clip span
      - query decode time
      - final head decode time

What would make it false:
    Swapping to Fourier features has negligible effect on temporal ratios and
    reconstruction.

Cheap test:
    Replace scalar-time projectors with fixed sinusoid/Fourier features feeding
    a small linear layer or direct additive embedding.

If supported:
    Keep explicit absolute index time, but change the embedding family.

If invalidated:
    The missing performance is not an embedding-basis problem.

## Branch E: The project contract itself is being violated by observation/target overlap shape

Hypothesis:
    The current video-token implicit training is still too autoencoding-like for
    the framing-3 training contract, and the learned world representation is not
    being forced into the right quotient behavior.

Why it might be true:
    - Encoder sees the same frames it is asked to reconstruct.
    - This may teach the wrong decomposition even though the rasterizer enforces
      some geometric structure.
    - The strategic docs emphasize omitted-observation support sets.

What would make it false:
    A stronger autoencoding-style baseline with more splats converges well and
    gives competitive render quality, making this an acceptable pre-probe
    rather than a contract violation that blocks learning.

Cheap test:
    Keep architecture fixed and vary only the observation/target split to create
    held-out in-clip targets.

If supported:
    Rework the trainer contract before drawing architecture conclusions.

If invalidated:
    The current all-frames reconstruction can still be a useful overfit probe.

---

## 5. Backtracks

Previous strong assumption:
    "Single-frame reuse was probably the main blocker."

Status:
    weakened

Evidence:
    Motion bug existed and fixing it improved temporal-to-first metrics, but the
    main 128px reconstruction gap remained.

Replacement model:
    The time bug was necessary to fix, but it was not the dominant remaining
    cause.

Previous strong assumption:
    "The bad screenshot was probably the renderer."

Status:
    invalidated

Evidence:
    Zero-init Gaussian head bug explained the early degenerate blobs; renderer
    parity work already showed the renderer was not the sole cause.

Replacement model:
    The failure is upstream of rendering.

Previous strong assumption:
    "Implicit camera may just not scale."

Status:
    unresolved / weakened

Evidence:
    Small image-implicit scales well enough at 32px. But that is not the same
    architecture as video-token implicit at 128px.

Replacement model:
    The failure is likely about the video-token implicit scaling recipe, not the
    abstract idea of photometric implicit camera learning.

---

## 6. Benchmark-Ready Data Inventory

Observed locally ready:

- `test_data/test_video_small.mp4`
  - paired DUSt3R output:
    `test_data/dust3r_outputs/test_video_small_all_frames`
  - 32px / 2fps baseline path
- `test_data/test_video_small_64_4fps.mp4`
  - paired DUSt3R output:
    `test_data/dust3r_outputs/test_video_small_64_4fps_all_frames`
  - 64px / 4fps baseline path
- `test_data/test_video_small_128_4fps.mp4`
  - paired DUSt3R output:
    `test_data/dust3r_outputs/test_video_small_128_4fps_all_frames`
  - 128px / 4fps baseline path
- `test_data/test_video_384_3fps.mp4`
- `test_data/test_video_384_64_6fps.mp4`
- `test_data/test_video_384_128_6fps.mp4`
- `test_data/dust3r_outputs/smoke_test`

Observed scaffolded but not present:

- script:
  `src/train_scripts/build_100_clip_dataset.sh`
- intended output:
  `data/clip_sets/local_100_128_4fps_46f`
- current state:
  `data/` directory does not exist in the workspace right now

Inference:
    We have small local single-video benchmark assets ready right now. We do not
    yet have a built 100-clip benchmark dataset in the workspace.

---

## 7. Decision Implications

If 64k implicit improves a lot:
    - capacity is a major missing ingredient
    - keep current time contract
    - compare 64k implicit vs 65k known-camera directly
    - then decide whether remaining gap is camera or clip-memory

If 64k implicit improves only a little:
    - capacity is not the main blocker
    - prioritize architecture/trainer-contract changes
    - next likely axis: stronger world/target separation or different temporal
      memory/query design

If 64k implicit OOMs or becomes unusably slow:
    - consider native batch constraints, temporal chunking, or a smaller
      `frames_per_step`
    - but note that known-camera 65k already proved the renderer path can
      support this regime

If Fourier time helps after 64k still fails:
    - keep explicit index-time contract
    - treat time basis as a second-order but real contributor

---

## 8. Concrete Next Steps

1. Run a `65536`-splat video-token implicit probe on the same `128px / 4fps`
   clip with the current explicit-time contract.
2. Compare:
   - `Eval/L1`
   - `Eval/SSIM`
   - `Eval/TemporalAdjacentL1Ratio`
   - `Eval/TemporalToFirstL1Ratio`
   - camera trajectory stats
3. If the gap remains large, branch into:
   - camera factorization audit
   - clip-memory bottleneck audit
   - held-out target training-contract probe
4. Consider swapping learned scalar time projectors for fixed
   sinusoidal/Fourier embeddings only after the 64k capacity check, so the
   ablation order stays interpretable.

Update:
    We already ran the first 64k check (`2vllet1i`). The next clean step is not
    "more of the same 64k smoke forever"; it is to decide whether to:
    - continue the 64k run longer to test optimizer-time rather than size,
    - compare predicted camera trajectory against known-camera trajectory,
    - or move to a held-out-target contract probe.

---

## 9. Bottom Line

Current bottom line:
    The evidence does not support blaming "implicit camera" in general. The
    small implicit baseline is too good for that. The current failure looks more
    like a video-token implicit scaling problem. The cleanest next test is to
    raise world capacity to 64k splats under the corrected explicit-time
    contract and see whether the model was simply too small or whether the
    architecture still fails even with adequate capacity.

Addendum after `2vllet1i`:
    We ran that 64k test. Result: the model stays stable, camera motion becomes
    much larger, but reconstruction improves only marginally. So pure world
    capacity is not the whole answer. The current best guess is that the
    clip-memory / time / camera / splat factorization is still wrong or still
    under-optimized even when the world asset is large enough to move.

---

## 10. Implementation Addendum: Swappable Architecture Blocks

User preference:
    Prefer different architectures that share layers/building blocks rather than
    one class full of conditionals.

What was implemented:
    - shared time-conditioning blocks in
      `src/train/gs_models/time_conditioning.py`
      - `MLPTimeConditioner`
      - `SinusoidalTimeConditioner`
    - shared camera blocks in
      `src/train/gs_models/implicit_camera.py`
      - existing `PathCameraHead`
      - new `TimeConditionedPathCameraHead`
    - two swappable video-token implicit architectures:
      - `DynamicVideoTokenGSImplicitCamera`
      - `DynamicVideoTokenGSImplicitCameraSinusoidalTime`
    - config-boundary architecture selection in
      `src/train/train_video_token_implicit_dynamic.py`
      via `model.variant`

Current variants:

```text
learned_time_orbit_path
    current/default architecture
    learned MLP time conditioning
    base orbit camera + residual path head

sinusoidal_time_path_mlp
    sinusoidal input/query/head time conditioning
    dedicated time-conditioned path-camera MLP head
```

Important design note:
    This still does not implement "decode camera as Pluecker ray," because that
    is the wrong abstraction boundary. Pluecker is a ray representation derived
    from a camera; it is not the camera latent itself. The clean future split is:

    - camera parameterization architecture
    - ray-conditioning architecture

    not "camera = Pluecker."

Config examples:
    - current learned-time config:
      `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`
    - sinusoidal-time variant:
      `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats_sinusoidal_time.jsonc`

Verification:
    Both variants passed compile and random forward smoke with:
    - input clip `[1, 16, 3, 128, 128]`
    - `decode_times == input_times == linspace(0, 1, 16)`
    - output `xyz` shape `[16, 8192, 3]`
    - `16` decoded cameras
