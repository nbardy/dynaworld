# dynaworld

**Hollywood's Open Source World Model.**

Dynamic video => world tokens => compressed splats.

The first broad goal is training a `video => world token` model. World tokens
are scene-state tokens. They count as world tokens only if they can be decoded
to splats that stay consistent across novel camera angles.

DynaWorld starts as a modality-shift system. It lets you move between video,
world tokens, and splats, so you can use splats when they're best and video
when its best.

The first use case is camera change. You shot the footage. Now you want a
different angle. DynaWorld turns that video into a compact dynamic splat scene,
lets you move the camera inside it, then renders video again.

The second use case is special effects. Traditional physics pipelines or
algorithmic special effects can run on the splat representations. As powerful
as video diffusion models can be for special effects, sometimes mathematical
representations are better when we want the control of traditional generative
algorithms.

This is made to be used in conjunction with video diffusion for upscaling,
editing, final-pixel cleanup, and later generation. They are complementary, not
opposites. First make the tokens 3D-consistent. Then make the tokens
predictive.

You can shoot a clip, or generate one with video diffusion. Then DynaWorld
creates an exploratory version of that clip.

## Video `<=>` Video

A world model needs to be `Video <=> Video` so it can train self structured on
video, unsupervised or self supervised.

`Video <=> Video` is the only training data for world models that scales.
Everything else requires expensive labeling, and labels don't scale.

The training contract is simple:

1. encode video
2. emit world tokens
3. decode world tokens to splats
4. render source-camera video for loss and novel-camera passes for consistency
5. compare the source render to the video you started from

Splats sit in the middle as the compact intermediate. No fake 3D labels. No
synthetic ground truth. Render splats and compare directly against
ground-truth video.

If a token set only works from the camera that encoded it, it is not a world
token yet. It is a view cache.

The useful signal is in the preimage.

## Cheap Adapters

Modalities don't require pretraining. The implicit latents in video models are
the key. Decoding them to splats is cheap adapter training.

A lightweight world-token and splat head on a frozen video backbone. Not a new
foundation model.

## World-Token Generation

Generation is the follow-up goal, not the first proof. First train
`video => world tokens`: observed video goes in, stable scene tokens come out,
and those tokens decode to splats that can be rendered under new cameras.

Once those tokens are worth predicting, train models over the token space
itself. That can be autoregressive or diffusion: continue a video by predicting
future world tokens, condition on an image to initialize them, or condition on
text to generate them. The renderer remains the pressure. Generated tokens
should still decode to a coherent dynamic splat world, not just plausible
pixels.

The sequencing matters. Stage 2 should not be used to hide camera leakage in
stage 1. The base world tokens need to already hold up under camera changes.

The stronger training task is probably to force the model to render images it
didn't encode. Encode part of a clip, decode splats as a function of `t`, then
train on the GT of images it didn't encode. That forces data that is not in the
encode path to come from the world model, the 3D inductive bias in the splat
renderer, and the time-conditioned decoder.

## Core Beliefs

See `research_notes/README.md` for the long-form rationale.

1. World models are video models. A strong video backbone already carries geometry, motion, and lighting structure.
2. The first broad goal is `video => world tokens`, where world tokens decode
   to splats that stay consistent across novel cameras.
3. DynaWorld starts as a modality-shift system: `Video <=> splats`, with world
   tokens in the middle.
4. `Video <=> Video` is the training contract. It is the only training data for
   world models that scales. World tokens and splats sit in the middle.
5. The useful signal is in the preimage.
6. Static and dynamic are the same problem.
7. Foundations are sacred. Modalities don't require pretraining. A lightweight
   world-token and splat head on a frozen video backbone is cheap adapter
   training.
8. Supervision should stay in pixel space. Render splats and compare to video.
9. Memory should be spent on dynamic scene state, not luxury parameters.
10. Pure generation comes after the representation works: AR or diffusion over
    world tokens for continuation, text=>video, and image=>video.

## Phases

**Phase I - `video => world tokens => splats`.** Dynamic reconstruction, fast
camera editing, and the base representation. Where we are today.

**Phase II - world-token prediction.** AR or diffusion over world tokens. This
is the path to video continuation, image=>video, and text=>video without making
pixels the main state.

**Phase III - interaction.** Actions inside the world model. Agents and physics
handles that let you control the dynamic scene, not just re-view it.

## Progress

Current work is focused on the base world-token model through small
single-video overfit runs and tiny scene-diverse datasets. The goal is to keep
training loops working, fast, and convergent before moving to larger data.
Completed items here have been tested on small single-video overfit runs unless
noted otherwise.
Longer-form research notes live under `research_notes/`.
For agent onboarding, active experiments, and code-organization routing, start
with `PROJECT_INDEX.md`, `EXPERIMENTS.md`, and `CODE_ORGANIZATION.md`.

### Baselines

- [x] Top-level video to splat baseline, reproducing TokenGS as the reference baseline.
- [x] Implicit camera baseline, extending the TokenGS baseline.
- [x] First real Neural3D `coffee_martini` train2/holdout1 protocol executes
  World Tubes, dynamic 3DGS, and WorldFoam with real LLFF camera poses and
  separate train/heldout metrics. The matched seeds 17/29/43 table is complete
  at 128px/16f/40 steps/1024 primitives with offline W&B media and a promotable
  deterministic World Tubes policy. World Tubes leads heldout PSNR on this
  one scene/split; broader camera-triplet and scene coverage remains.
- [x] Unified paper-ablation training now shares one typed space-time sampler,
  progressive-stage schedule, aspect-preserving image contract, and cost
  ledger across World Tubes/STAR Metal, WorldFoam/PowerFoam Metal, and dynamic
  3DGS/fast-mac Metal. A staged 4-frame MPS smoke and an all-300-frame MPS
  smoke are green; the 600-step progressive and pixel-matched 300-step quality
  rows are configured but not yet benchmark results.
- [ ] Run the checked-in 512-wide full-temporal quality rows across seeds and
  camera/scene breadth. Native 2704x2028 training remains a separate streaming
  data/ray implementation task and is not claimed by the eager 512-wide path.
- [x] Gauged UVT theory iteration is closed into the World Tubes mainline.
  WorldFoam remains a parked retained-depth challenger with explicit reopen
  gates; future work is breadth or measured implementation bottlenecks, not
  another umbrella formalism.

### Renderer

- [x] Fast differentiable Gaussian rasterizer on Mac local GPU for local experimentation.
- [x] Differentiable renderer integration set up for trainer loops.
- [x] Debug metrics added to trainer loops.
- [x] STAR UVT source-view overfit runs through `src/train/train.py` with the
  direct-atomic path.
- [x] Gauged/projective STAR UVT now carries tested trace metadata for
  anisotropic screen precision and pixel-varying conditional depth. The CPU
  certificates consume/preserve these fields through support refresh and
  quadrature lowering, while production interval Metal still uses the scalar
  visibility path until the next kernel consumes depth planes directly.
- [x] WorldFoam Gate4 native-cutwalk fused-MSE has a clean local speed gate
  against matched STAR UVT at render64/site24. The strict wrapper/verifier
  passes for real `2/4/8/16f` rows and for an explicit synthetic repeated-16f
  `32f` speed smoke after the framebitmask bit-31 fix. Treat this as shader
  speed evidence, not STAR RGB-quality/system parity.
- [x] STAR UVT F32 feature tubes have a direct Metal smoke path through
  `FeatureToColor` and a first-class chunked 8f/64px overfit config.
- [x] STAR UVT F32 feature-tube scale probes now log tile max/p95/overflow and
  fixed-bin eligibility; cap-256 runs make 16384 valid and bracket 32768
  support pruning. `alpha>=1/72` is the current best zero-overflow 32768 row,
  while `alpha>=1/80`, `alpha>=1/96`, and unpruned 32768 still overflow.
- [x] STAR UVT F32 feature-tube reports now record
  `requested_render_mode`, `effective_render_mode`, and fallback requirement for
  `feature_direct_fixedbin`; this is a promotion guard around the current direct
  feature path, not a separate optimized fixedbin shader.
- [x] STAR UVT F32 feature tubes now have a real `feature_direct_gradcache`
  backward mode. It is a small win, not the final path: the 64f/256px/32768t/F32
  synthetic backward row moves `485.63ms -> 471.29ms`, and the first-class
  alpha-pruned row passes at `1.226s/step`.
- [x] STAR UVT F32 feature tubes now have a recorded trainable
  `feature_direct_gradcache_reduce` prototype. It passes parity and the
  20-step alpha-pruned row, but it is slower than plain gradcache, so it is a
  negative result rather than the default.
- [x] STAR UVT F32 feature tubes now have a vectorized
  `feature_direct_gradcache_reduce_vec4` follow-up and a sequential direct-mode
  matrix runner. Vec4 passes parity and helps one synthetic row, but the real
  cap256 trainer row is slower than gradcache. A fresh 512px/8192t
  no-pre-norm first-class rerun now selects vec4 reduce as the fastest current
  feature diagnostic (`2.49s/step`, `1.18s` backward versus gradcache
  `2.86s`/`1.33s`, same loss/PSNR), but it is still diagnostic because feature
  quality is far below RGB STAR.
- [x] STAR UVT F32 feature tubes now have a cached-bin sidecar diagnostic.
  `gradcache_cached_bins` reuses forward tile bins in backward and cuts the
  same-session synthetic 64f/256px/32768t/F32 renderer backward
  `1068.0ms -> 935.8ms`, but the first-class 512px/8192t/chunk2 trainer row
  does not improve end-to-end (`16.20s/step`, `10.24s` backward versus plain
  gradcache `16.21s/step`, `9.68s` backward in the same session). Keep it
  diagnostic, not default.
- [x] The sequential direct-mode matrix now covers cached-bin variants and
  512px. The 39-row 64f/32768t/F32 run passes for all modes at 128/256/512px.
  At 512px, `gradcache_cached_bins` is the fastest full-gradient direct total
  row (`1.979s`, `1.103s` backward), while `gradcache_skip_feature_grad`
  remains the fastest diagnostic (`1.714s`, `0.804s` backward). This reinforces
  the current plan: cached bins are mixed/noisy, and the next real shader needs
  a different feature-gradient accumulation path rather than another bin reuse
  patch.
- [x] STAR UVT F32 feature tubes now have a benchmark-only
  feature-gradient-only / two-pass split diagnostic. Tiny F4/F32 parity passes,
  but naive split-recompute is slower than full gradcache: at 256px it is
  `1.343s` total / `1.063s` backward versus `0.972s` / `0.692s`; at 512px it
  is `2.471s` / `1.613s` versus `2.467s` / `1.379s`. The reverse-order 512px
  check also stayed negative (`3.217s` / `1.821s` versus `2.066s` / `1.204s`),
  so the next path should be true fixedbin/tile-slot feature-gradient
  accumulation or native image-space VJP, not duplicate traversal.
- [x] STAR UVT F32 feature tubes now have a fixedbin/tile-slot accumulator
  budget gate. On the synthetic 64f/32768t/F32 rows, replacing per-pixel
  feature-gradient atomics with one atomic per tile slot and channel would cut
  write count by `128x`, but naive slot-wise prefix recompute costs `39.8x`
  at 256px and `10.8x` at 512px. A scalar f32 contribution-weight tape is
  `1.171GiB` at 256px and `1.195GiB` at 512px; a wrong per-channel tape would
  be `37-38GiB`. This makes the next viable shader a compact scalar
  prefix/weight tape or native VJP, not a per-channel tape or recompute kernel.
- [x] STAR UVT F32 feature tubes now have a feature-only tile-slot reducer
  isolation gate. The existing barrier-heavy scalar/vec4 reducer is correct as
  a feature-only accumulator, and vec4 is a real isolated win:
  `gradcache_feature_grad_only_reduce_vec4` cuts feature-only backward
  `532.8 -> 449.9ms` at 256px and `869.1 -> 774.8ms` at 512px. Full-gradient
  same-session refresh shows vec4 reduce helps synthetic 512px
  (`1284.2 -> 1108.0ms` backward), but two-pass still loses or ties because it
  duplicates traversal. Keep single-pass vec4 reduce live as a shader candidate;
  do not promote two-pass composition.
- [x] STAR UVT F32 feature tubes now have 512px support probes at 4096 and
  8192 tubes. Both pass with zero overflow under `feature_direct_gradcache`, but
  they are already `6.46-7.94s/step`, so 512px/32768t is blocked on speed.
- [x] STAR UVT F32 feature tubes now have a narrow benchmark-only RGB handoff
  prototype, `fused_first3_sigmoid_mse`. It passes parity and cuts synthetic
  backward to `309.31ms`, but it only covers
  `alpha * sigmoid(feature[:3]) -> mean MSE`, not the learned F32 colorizer.
  The matched `64f/512px/8192t/F32` rerun still passes and records
  `494.09ms` backward / `1152.58ms` total; this keeps the fused native handoff
  direction alive as a boundary proof, not as a trainer route.
- [x] STAR UVT F32 feature tubes now have a generalized linear sigmoid-MSE
  handoff benchmark with colorizer weight/bias gradients. It passes parity, but
  it is slower than gradcache (`615-619ms` backward versus `477.5ms`
  same-session gradcache), so it is a negative speed result rather than a
  trainer path.
- [x] STAR UVT F32 feature tubes now have an image-space-prep logit handoff
  benchmark. It passes parity, but it is also slower than gradcache (`595.2ms`
  renderer backward plus `60.2ms` prep versus `529.0ms` same-session
  gradcache), so it is another negative speed result.
- [x] STAR UVT F32 feature tubes now have a logit-handoff plus tile-slot
  reducer gate. `logit_handoff_reduce` and `logit_handoff_reduce_vec4` combine
  image-space logit/alpha prep with the stable-tile feature-gradient reducers.
  The 64f/32768t/F32 direct matrix passes F4/F32 parity and zero overflow at
  256px/512px. Vec4 improves synthetic backward at 256px
  (`571.7 -> 510.6ms`) and narrowly at 512px (`654.8 -> 642.3ms`), while total
  time improves `800.0 -> 744.1ms` and `2137.8 -> 1512.4ms`; scalar reduce
  regresses 512px backward (`654.8 -> 722.5ms`). Keep vec4 as a diagnostic
  native-VJP/tile-slot candidate, not a trainer default.
- [x] STAR UVT F32 feature tubes now have a matched 512px native-handoff
  comparison. At `64f/512px/8192t/F32`, `logit_handoff_reduce_vec4` has the
  best native backward (`386.26ms`) but pays `421.89ms` of Torch prep, while
  generalized `linear_sigmoid_mse` is slower than gradcache (`918.09ms` vs
  `522.02ms` backward). The next native gate needs to fuse the logit/RGB/loss
  prep or avoid dense image-space prep, not only call the current handoff.
- [x] STAR UVT F32 feature tubes now have a native-prep logit handoff gate.
  `logit_handoff_reduce_vec4_native_prep` moves the linear sigmoid-MSE
  `grad_logits`/`grad_alpha` prep to Metal and feeds `[T,H,W,3]` logits directly
  into the existing reverse traversal. At `64f/512px/8192t/F32`, serial
  warm3/repeat5 prep drops `413.64 -> 37.29ms`, prep+backward drops
  `826.35 -> 428.98ms`, and total drops `1446.53 -> 1108.50ms` with F4/F32
  parity and zero overflow. This is still a benchmark-only linear colorizer
  gate, not the hidden frozen-probe trainer route.
- [x] STAR UVT F32 feature tubes now have a hidden sigmoid-MSE native gate.
  `hidden_sigmoid_mse_star_only` fuses hidden `FeatureToColor` RGB/loss VJP into
  the Metal reverse traversal and passes F4/F32 parity with zero overflow, but
  it is not the next speed keeper: H32 scalar totals `317.54/610.90/2549.39ms`
  at `128/256/512px`, H64 at 256px is `817.27ms`, and vec4 reduce is slower than
  scalar. The next native work should avoid dense `[T,H,W,F]` support or use a
  compact visibility/prefix tape.
- [x] STAR UVT F32 feature tubes now have a sparse hidden sigmoid-MSE native
  gate. `direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins` reuses
  cached sparse bins and fuses hidden RGB/loss VJP for selected pixels only. It
  passes F4/F32 parity and zero overflow. At `64f/512px/8192t/F32`, H32
  sparse64 total drops `29.66 -> 18.47ms` (`1.61x`), H32 sparse128 drops
  `111.17 -> 64.17ms` (`1.73x`), and H64 sparse64 drops `45.09 -> 28.40ms`
  (`1.59x`). The trainer-wired pixel64 follow-up is mechanically correct but
  neutral: warm sparse loss+backward is `113.25ms` manual versus `116.27ms`
  native, with matching endpoint loss (`3.26e-08` diff), so trainer promotion
  still needs compact full-support/target-area visual gradients rather than this
  pixel64 branch alone.
- [x] STAR UVT F32 feature tubes now have a first trainer-style
  logit-handoff RGB-VJP profile. On a real 64f/512px/8192t checkpoint, linear
  RGB reconstruction through the no-pre-norm sigmoid `FeatureToColor` matches
  autograd gradients (`9.43e-09` max abs error, zero loss error) and is a small
  timing win (`1691.0 -> 1587.4ms`, `1.065x`) with zero overflow. The 8f/64px
  smoke is cleaner/faster (`78.8 -> 34.7ms`, `2.27x`). This proves the handoff
  can be trainer-compatible for linear RGB loss, but it does not cover
  target-grid V-JEPA MSE or the hidden64 frozen RGB-probe objective.
- [x] STAR UVT F32 feature tubes now have a target-grid/frozen-probe VJP bridge
  profile for the current keeper objective. The 64f/512px/8192t 1300-checkpoint
  row matches normal autograd with zero loss error and zero overflow. The first
  autograd-image bridge proves correctness but not speed (`1545.5ms` autograd
  versus `1594.3ms` bridge, `0.969x`, `2.57e-08` max grad error). The analytic
  target-grid/probe VJP follow-up keeps parity (`3.07e-08` max grad error) and
  gives a small repeat-5 win (`1510.6 -> 1477.2ms`, `1.023x`), so native/fused
  target-grid/probe VJP was worth a trainer gate. The trainer opt-in
  `feature_target.image_vjp_mode=analytic` now passes a matched 5-step 64f/512
  smoke from the 1300 checkpoint, but it is not a clear end-to-end speed
  promotion: autograd mean step is `1303.6ms`, warm analytic rerun is
  `1304.6ms` (`1259.2ms` no-first versus `1264.1ms`), with the backward bucket
  improving by `103.3ms` but loss/VJP work moving into the loss bucket. Keep it
  as an opt-in diagnostic until a longer or fused/native gate wins step time.
- [x] STAR UVT F32 feature tubes now have a sparse-pixel target-grid VJP
  trainer gate. The new opt-in
  `feature_target.image_vjp_mode=analytic_sparse_pixels` packs the nonzero
  target-grid image-gradient pixels and calls a sparse direct-atomic Metal
  backward using forward bins. The repeat-3 parity profile matches autograd
  (`4.61e-08` max grad error, zero loss error) and cuts bridge total
  `1245.9ms -> 920.5ms`, with sparse renderer backward `557.6ms -> 46.3ms`
  while sparse packing still costs `184.0ms`. The matched 5-step trainer smoke
  passes from the 1300 checkpoint, matches dense analytic loss/probe PSNR, and
  improves no-first step `1318.0ms -> 973.7ms` with only `65,536` sparse pixels
  per step (`0.390625%` of dense). This proved the sparse target-grid
  hypothesis and is superseded by the direct sparse-grid follow-up below.
- [x] STAR UVT F32 feature tubes now have direct sparse target-grid VJP packing
  for the same keeper objective. The new opt-in
  `feature_target.image_vjp_mode=analytic_sparse_grid` analytically maps the
  trilinear target-grid/probe gradient to sparse source pixels instead of
  materializing and scanning a dense image-gradient tensor. Dense-vs-sparse VJP
  parity passes exactly on the small CPU/MPS check, and the repeat-3 64f/512
  bridge keeps full-objective parity (`4.60e-08` max grad error) while cutting
  total to `760.6ms`. The matched 5-step trainer smoke passes from the same
  1300 checkpoint and supersedes sparse-pixel timing: no-first step
  `973.7ms -> 795.3ms`, no-first backward `254.5ms -> 88.6ms`, with the same
  `65,536` sparse pixels/step and matching loss/probe movement. A matched
  sparse-grid render-mode matrix keeps `feature_direct_gradcache_reduce_vec4`
  as the selected mode and improves the checked no-first row to `730.5ms`
  (`78.3ms` mean backward), ahead of gradcache (`759.4ms`) and direct atomic
  (`779.3ms`). This is now the backward-only reference for the current
  target-grid/frozen-probe diagnostic; sparse-forward below supersedes it
  end-to-end. The remaining speed target is native GPU target-grid/probe
  loss+VJP or scalar fixedbin/tile-slot accumulation.
- [x] STAR UVT F32 feature tubes now have sparse feature forward for the
  target-grid/frozen-probe diagnostic. The new opt-in
  `feature_target.image_vjp_mode=analytic_sparse_grid_forward` renders only the
  target-grid support pixels, folds sparse feature values back into the target
  grid for feature/probe loss, and reuses sparse-grid VJP for backward. The
  first forward profile was bit-exact and cut dense feature render
  `515.9ms -> 70.5ms` (`7.322x`) for the same `65,536` pixels, with a first
  5-step trainer row at `492.3ms` no-first / `413.7ms` last step. A follow-up
  128/256/512 matrix and isolated 512px repeat found timing is run-order
  sensitive but still valid: all rows pass with zero overflow, the 512px
  post-scale repeat is `598.2ms` no-first / `477.6ms` last step, and the
  sequential 512px matrix row is `973.0ms` no-first. This is the current
  selected target-grid/frozen-probe diagnostic, but not yet a stable hard
  speed baseline. The new repeat-3 512px timing gate gives the comparison
  surface for the next native shader: no-first step mean/min/max/stdev
  `504.9/411.0/626.4/110.3ms`, last-step `468.8/409.3/549.9/72.7ms`, and
  no-first backward `142.2/114.7/174.4/30.1ms`. Sparse-grid dense-forward
  remains the backward-only reference.
- [x] STAR UVT F32 feature tubes now have a batched target-grid/frozen-probe
  VJP path for sparse-forward. The preflight harness stacks all 32 frame chunks
  after sparse feature forward, computes target-grid MSE, hidden64 frozen
  RGB-probe VJP, and sparse-grid VJP in one batched MPS path, and matches the
  per-chunk loss/gradient packs (`7.45e-09` loss error, `6.55e-11` max feature
  grad error). The isolated loss+VJP component drops `38.0ms -> 4.8ms`
  (`7.99x`). The opt-in trainer mode
  `feature_target.image_vjp_mode=analytic_sparse_grid_forward_batched` now
  passes the same 64f/512px/8192t 5-step checkpoint gate with zero overflow and
  probe PSNR `21.965 -> 21.984`; repeat-3 gives no-first step
  mean/min/max/stdev `179.3/159.7/215.6/31.5ms`, no-first backward
  `72.0/60.8/90.2/15.9ms`, and no-first render `71.1/67.8/77.4/5.5ms`.
  The 100-step media/helper gate also passes from the same checkpoint:
  `0.886537 -> 0.880744` loss, `0.632124 -> 0.627122` feature loss,
  zero overflow, mean step/backward/render `399.9/176.9/125.2ms`, and last-20
  `262.9/109.4/94.0ms`; the contact sheet is valid but still blurry, so this
  is a speed/path promotion rather than a visual-quality promotion.
- [x] STAR UVT F32 feature tubes have a matched dense-analytic target-grid
  trainer render-mode matrix for the frozen-probe objective. The runner
  `research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py`
  benchmarks `feature_direct_atomic`, gradcache, cached-bins, scalar reduce,
  vec4 reduce, and the fixedbin request from the same 1300-step checkpoint.
  All rows pass and end at the same loss/probe PSNR, but reduce/vec4 do not
  win end-to-end on this target-grid trainer (`vec4` repeat-top no-first
  `1509.6ms` versus direct-atomic `1249.0ms`). The sparse-grid matrix above
  supersedes it for the selected speed path. `feature_direct_fixedbin` now
  reports `kernel_backward_mode=direct_atomic` and
  `requested_fixedbin_is_direct_atomic_alias=true`; it is an eligibility
  contract, not an implemented fixed-bin kernel.
- [x] STAR UVT F32 feature tubes now have a first-class backward breakdown
  script. The 512px `feature_direct_gradcache` rows show the renderer is only
  `16.9-22.1%` of backward, while `FeatureToColor`/loss backward is
  `77.9-83.1%`; the 256px/32768t/cap256 row has renderer around `36%` of
  backward. This makes dense colorizer/loss VJP a first-order speed target, not
  just the renderer shader.
- [x] STAR UVT F32 feature tubes now have a no-pre-norm 512px speed A/B and a
  20-step media/quality gate. The 2-step no-pre-norm row is `3.72s/step`,
  `1.59s` backward versus `7.94s/step`, `4.88s` backward for default
  pre-norm. The 20-step media A/B still favors no-pre-norm for speed
  (`7.37s/step`, `3.37s` backward versus `11.10s/step`, `7.07s` backward), but
  default pre-norm ends slightly better (`0.31742` loss / `4.984` PSNR versus
  `0.32053` / `4.941`), so no-pre-norm is not promoted as the quality default.
  An identity/no-pre-norm decoder diagnostic is faster again (`2.54s/step`,
  `1.17s` backward), but ends worse (`0.32446` loss / `4.888` PSNR), so simply
  removing sigmoid/pre-norm is a speed diagnostic, not the quality fix. A
  hidden-64 pre-norm decoder gives only a tiny quality lift (`4.987` PSNR) while
  slowing to `19.18s/step` and `13.77s` backward; reducing pre-norm sigmoid
  init gain from `4` to `2` is similar (`4.987` PSNR, `14.12s/step`), so naive
  decoder capacity/init tweaks do not close Gate 4.
- [x] The checked fast feature-tube overfit diagnostic is now
  `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
  star-feature-512-fast`. It now runs the 64f/512px/8192t target-grid V-JEPA
  `analytic_sparse_grid_forward_batched` plus
  `feature_direct_gradcache_reduce_vec4` config and writes RGB-probe
  contact-sheet/MP4 media. Use it for speed/path probing, not as the promoted
  source-view quality row. The helper also exposes `star-feature-512-visual`
  for the current compact target-area visual route (`930.6ms`, `6.023` full
  RGB on the current-build gate) and
  `star-feature-512-native-fullcell` for the promoted exact full-support native
  vec4 W^T baseline; compact native star-only is rejected because it freezes the
  colorizer and is slower, and compact manual-hidden64 is rejected because it
  preserves colorizer grads but regresses speed and feature/probe quality. The
  native colorizer-gradient vec4 W^T path now passes tiny parity but is also
  rejected in the compact trainer gate. The SIMD-reduced native colorizer
  follow-up fixes the direct-kernel atomic envelope (`297.2ms` native total
  versus `312.1ms` sparse-pixel baseline in one matched compact window), but
  the 5-step trainer is still rejected (`2908.9ms` mean step, `604.0ms` sparse
  visual backward, same feature/probe regression). The older RGB-target speed
  row is available as `star-feature-512-rgbfast`. The dynamic-gsplat fixed-512
  comparator now has a stronger 20-step media gate at the same
  `64f/512px/8192` active primitive scale: `2.940s` mean timed step,
  `1.926s` backward, and `5.587` final eval PSNR with smeared media, so it is
  not the current fast local route or quality escape hatch.
  The selected visual-quality gate keeps the route out of scale-up: dense full
  RGB is only `6.023` PSNR and the media remains sparse/streaked or blurry,
  versus the RGB STAR same-clip bracket at `12.444` PSNR. A trainable
  low-frequency RGB-grid bridge now works mechanically and is fast
  (`353.1ms` mean step, `289.9ms` no-first), but is also rejected as a visual
  route: it improves grid/probe PSNR (`22.028 -> 22.248` RGB-grid) while
  worsening feature loss (`0.625418 -> 0.630230`) and dense full RGB
  (`5.657` PSNR). Combining that RGB-grid loss with the compact target-area
  visual route also fails: the combined gate is slower (`1.648s` mean step),
  improves grid/probe/sparse metrics, but worsens feature loss to `0.630296`
  and lands at only `5.720` dense full RGB PSNR. A dense alpha diagnostic now
  localizes the failure: forcing alpha to one raises the rejected routes to
  `11.450-14.616` PSNR and compositing over target background reaches
  `20.149-25.562` PSNR, while alpha `>0.1` covers only `41.5-43.5%` of pixels.
  The direct alpha-to-one follow-up also fails: sampled alpha loss improves
  `0.752440 -> 0.738210`, but dense RGB stays `6.018`, alpha `>0.1` stays
  `43.1%`, and feature/probe losses regress. The phase-covered alpha retry
  fails as well: sampled alpha loss improves `0.751768 -> 0.739891`, but dense
  RGB falls to `6.014`, dense alpha `>0.1` falls to `43.0%`, and feature/probe
  losses regress. A target-aware black-hole coverage retry fails too: black-hole
  loss improves `0.262537 -> 0.256889`, but dense RGB stays `6.014`, dense alpha
  `>0.1` stays `43.0%`, and feature/probe losses regress. Target-background
  composition is informative but not a visual route: it raises forced-alpha PSNR
  to `14.891-14.899` and oracle composition to `27.105-27.443`, but dense RGB
  is only `5.666-5.748` and alpha `>0.1` stays `40.8-43.1%`. The alpha-sweep
  and patch4 support retry fails too: `16x` posthoc alpha gain only reaches
  `8.337-8.592` PSNR, while `4x4` support plus target-background alpha pressure
  lands at `5.698` dense RGB and regresses feature/probe. Raw-opacity bias
  rerendering fails as well: best bias `+4` only reaches
  `6.194/5.926/5.871` PSNR for compact/targetbg-alpha/patch4. Dense alpha-only
  support then fails as the direct visibility retry: it adds `0.834s` render,
  `0.125s` loss, and `0.859s` backward per step, regresses weighted loss
  `1.271702 -> 1.284505`, feature loss `0.625418 -> 0.626814`, probe PSNR
  `22.028 -> 21.861`, and dense full RGB to `5.647`, while alpha `>0.1` falls
  to `40.7%`. The follow-up alpha-only visibility profile proves the current
  dense-alpha implementation can be made cheaper without dense F32 images:
  sparse-pixel F1 alpha render plus cached F1 backward matches alpha exactly,
  matches geometry/opacity gradients within `4.7e-7`, and cuts the profiled
  all-chunk alpha render+backward envelope `1100.8 -> 634.6ms`. This is only a
  diagnostic speed gate. The trainer follow-up wires the same path behind
  `dense_alpha.render_mode=sparse_f1` and reproduces the same negative quality
  endpoint, while cutting mean step/backward `2558.6/1114.2 -> 873.3/370.0ms`
  and dense-alpha render/loss/backward
  `834.5/124.6/858.9 -> 276.0/22.0/303.7ms`. The next quality gate should
  change the visibility/support model, not add another grid-level colorizer,
  scalar alpha penalty, opacity schedule, dense alpha loss, or sparse support
  shuffle. A first CPU support-changing bridge now passes as a gradient proof:
  same-support dense alpha stays at `0.0` target alpha `>0.10` coverage from a
  zero-hit start, while a soft projected-tube coverage proxy sends
  center/velocity gradients and reaches `0.324`; this is a trainer-bridge
  candidate, not a visual-quality promotion. The first-class trainer port now
  passes as a mechanics gate from the sparse 1500 checkpoint: weighted loss
  improves `0.871986 -> 0.871864`, feature target loss `0.625418 -> 0.625379`,
  RGB-probe PSNR `22.0277 -> 22.0291`, and visibility-proxy loss
  `-4.20957 -> -4.20992` with center/velocity gradients seen. It is still not
  a scale or quality promotion because dense full RGB PSNR is only `5.640` and
  the proxy adds about `237ms` mean step cost. The dense support follow-up
  rejects the current center-only proxy as the scale-up bridge: it improves
  forced-alpha/oracle content (`11.722 -> 14.552` forced-alpha PSNR,
  `20.140 -> 25.834` target-background oracle), but alpha `>0.1` falls
  `41.1% -> 40.5%`. A 10x/20-step proxy run makes the proxy objective better
  but fails trainer loss (`0.834100 -> 0.844115`) and still leaves alpha `>0.1`
  at `40.6%`. The opacity/precision support-aware follow-up is wired and
  tested, but also rejected as the next scale bridge: it optimizes support proxy
  loss `3.4303 -> 3.3821` and sends raw opacity/precision gradients, while
  feature loss slightly worsens, proxy work costs `693.7ms`/step, and dense
  support barely changes versus center-only (`5.640 -> 5.643` dense RGB,
  `40.5% -> 40.6%` alpha `>0.1`). The next CPU mechanism gate is positive:
  fixed-budget birth/split reallocates `8/16` dead tubes onto target support,
  reaching `1.0000` target alpha `>0.10` where same-support alpha stays
  `0.0000` and the center proxy reaches `0.5784`; refinement keeps coverage
  and lowers background alpha `0.0479 -> 0.0072`. The first trainer port now
  exists as `support_birth_split.enabled` and passes a 64f/512px sparse-1500
  gate: it reallocates `32/8192` low-opacity tubes, keeps zero overflow
  (`100/71/128` max/p95/cap), passes 5 steps at `189.4ms` mean / `138.3ms`
  last, and lifts full RGB PSNR to `5.708`. This is a real support-changing
  primitive, not a quality claim yet. The dense-support diagnostic keeps that
  read: birth/split improves normal/forced-alpha/high-alpha support versus
  center/support rows (`5.708` normal, `14.606` forced-alpha, alpha `>0.5`
  `0.117`), but alpha `>0.1` is only `0.411` and target-background oracle falls
  to `25.234`. The uncovered-brightness target sampler is now implemented and
  tested; it selects low-alpha bright target points and passes the same 512px
  5-step gate at `187.4ms` mean step with full RGB PSNR `5.713`, but dense
  support still leaves alpha `>0.1` at `0.411` and forced-alpha PSNR at
  `14.579`. So birth/split is real progress, while target selection alone is
  not enough. The first sweep now records the next boundary: cap `128` overflows
  at `64+` births (`132/103/128` max/p95/cap for `64`, `196/167/128` for
  `128`), cap `256` clears `64/128` births, and wide radius `96px` is what
  moves coverage. Best safe cap-128 row is `low_alpha_n32_r96_cap128` with
  alpha `>0.1` `0.420`, dense normal PSNR `5.825`, forced-alpha PSNR `14.591`,
  oracle `24.226`, and zero overflow. That is a modest coverage move with an
  oracle tradeoff, not a quality promotion. The intermediate-radius follow-up
  confirms the tradeoff curve: uncovered `r64/r72/r80/r88` moves alpha `>0.1`
  `0.411 -> 0.413 -> 0.415 -> 0.417` while oracle falls
  `25.319 -> 25.187 -> 25.015 -> 24.802`; low-alpha `r80/r88` fails the loss
  gate despite zero overflow. The born-opacity sweep is also negative as a
  promotion lever: at `r80`, uncovered opacity `0.4/0.6/0.8/0.9` moves alpha
  `>0.1` only `0.414 -> 0.415` while oracle falls `25.177 -> 24.987`; at
  `r88`, opacity `0.2/0.4/0.6/0.8` moves alpha `0.414 -> 0.417` while oracle
  falls `25.242 -> 24.802`. Longer continuation is not justified until birth
  support shape changes the coverage/oracle tradeoff. The first support-shape
  attempt is now a negative: `trajectory_ellipse` birth support with along
  `88px`, across `24/32px`, precision `48px`, and opacity `0.4/0.6` passes
  all eight cap-128 rows with zero overflow, but alpha `>0.1` stays
  `0.408-0.409`, below the prior isotropic `0.411`. The next support experiment
  should split target points into multiple centers or spatial strata before
  birth, not widen one global fitted line. That multi-center gate is now the
  first real coverage move: `farthest_xy` with `K=8`, `32` births, `r64`, and
  cap `128` reaches alpha `>0.1` `0.4309` with zero overflow (`101/71/128`),
  alpha `>0.5` `0.1550`, forced-alpha PSNR `14.608`, and mean step/backward
  `181.1/63.5ms`. The tradeoff is target-background oracle dropping to
  `23.965`, so the next gate should sweep multi-center radius/opacity to recover
  oracle while preserving coverage. The `K=8` radius/opacity sweep is now run:
  best coverage is `r72/o0.8` with alpha `>0.1` `0.4318`, alpha `>0.5`
  `0.1636`, normal PSNR `5.871`, and oracle `23.670`; the selected balanced row
  is `r64/o0.4`, which keeps alpha `>0.1` at `0.4298`, recovers oracle to
  `24.805`, keeps forced-alpha PSNR `14.620`, and stays fast at
  `167.9/58.1ms` step/backward. Next run a 20-step media gate on
  `K=8/r64/o0.4`. That gate now passes: loss `0.903197 -> 0.897231`, feature
  loss `0.631571 -> 0.631083`, probe PSNR `21.681 -> 21.769`, full RGB PSNR
  `5.794`, zero overflow, last step/backward `147.3/54.3ms`, dense alpha
  `>0.1` `0.431158`, forced-alpha `14.631`, and oracle `24.851`. This holds
  the support gain after 20 steps. The matched `K=8/r72/o0.4` 20-step sibling
  also passes (`0.910099 -> 0.903088` loss, probe PSNR `21.601 -> 21.703`,
  full RGB PSNR `5.820`, zero overflow, `157.9/61.1ms` mean
  step/backward, dense alpha `>0.1` `0.432454`, oracle `24.668`), but it gives
  back feature/probe loss and oracle for only a tiny coverage gain. Keep
  `K=8/r64/o0.4` as the balanced 20-step default. The regenerated 50-step
  continuation then showed the cap-128 boundary: `K=8/r64/o0.4` improves
  loss/probe but overflows (`277` tiles, max `146/128`), `n16/r48` and
  `n16/r40` reduce this to two overflow tiles, and `K=8/n8/r40/o0.4` is the
  current cap-safe seed (`pass=true`, zero overflow, max `123/128`, loss
  `0.754568 -> 0.749460`, RGB-probe PSNR `24.372 -> 24.501`). The longer
  safe-row gate selects the 90-step checkpoint (`0.754568 -> 0.747006` loss,
  feature loss `0.608402 -> 0.606764`, RGB-probe PSNR `24.372 -> 24.552`,
  zero overflow, max `122/128`); the 100-step sibling stays fixed-bin but
  fails after late objective jumps. A checkpoint-aware 100-step tail schedule
  now passes (`0.749454` final loss, zero overflow, max `122/128`) but does not
  beat the selected 90-step checkpoint and leaves dense support essentially
  unchanged. The allocation follow-up is also measured: uniform `n16`,
  `K=16/n16`, and `K=16/n16/r32` still overflow by two tiles, while
  `K=12/n12/r40/o0.4` is cap-safe but does not beat the selected `K=8/n8`
  90-step checkpoint or move forced-alpha/oracle support. The first cap-aware
  bridge is also measured: cap-slack target scoring alone still overflows by two
  tiles, exact-fit tile repair drifts to one final overflow tile, and guarded
  repair (`K=16/n16/r40/o0.4`, guard `2`) passes fixed-bin with max `127/128`,
  loss `0.753847 -> 0.749102`, and dense normal/forced/oracle PSNR
  `6.486/14.021/21.571`. The first residual-cap-slack scorer also passes
  fixed-bin and slightly improves scalar losses (`0.753586 -> 0.748839`,
  feature `0.608503 -> 0.607558`, probe PSNR `24.404 -> 24.520`), but dense
  support remains flat at `6.486/14.019/21.579` normal/forced/oracle PSNR. Dense
  support improves versus the regenerated 1500 checkpoint, but is nearly flat
  across these cap-safe variants. The footprint-aware residual scorer is now
  measured too: it gives the best K16 scalar row (`0.752912 -> 0.748672`,
  feature `0.608350 -> 0.607417`, probe PSNR `24.420 -> 24.521`) but dense
  support stays flat at `6.481/14.021/21.576`. The first target-grid feature
  init handoff is now measured and is a small positive: target-group-mean
  feature init gives `0.752454 -> 0.748504`, feature `0.608332 -> 0.607351`,
  probe PSNR `24.433 -> 24.524`, and dense normal/forced/oracle
  `6.488/14.054/21.629`, but alpha `>0.1` remains `0.655`. Media remains
  coverage/visibility limited rather than RGB-STAR quality. The first
  support-target alpha bridge is measured too: the pointwise alpha term learns
  (`0.492962 -> 0.478448`) and nudges dense normal/forced PSNR to
  `6.508/14.084`, but alpha `>0.1` only reaches `0.657` and the oracle stays
  flat (`21.626`). The first support-target-area patch bridge is cheaper and
  also learns (`0.597970 -> 0.581641`), but lands on the same dense plateau:
  `6.507/14.085/21.627`, alpha `>0.1` `0.657`, with worse feature loss than
  target-init. The selected-patch follow-up exposed a real sparse-binner bug
  for chunk-shifted moving tubes: selected-only support rendered zero alpha
  despite analytic selected-tube alpha. The binfix now passes a focused MPS
  regression, and the first targetarea2 50-step rerun passes fixed-bin with
  zero overflow, max tile `110/128`, loss `0.889263 -> 0.863064`,
  support-target-area loss `0.253626 -> 0.217254`, and selected-patch
  normal/forced/oracle PSNR `6.644/19.452/26.994`. Dense transfer is now
  measured: whole-frame normal/forced/oracle PSNR is `7.269/14.736/21.439`,
  alpha `>0.1` reaches `75.4%`, and raw opacity bias only reaches `8.039`
  PSNR. The prefix tape now shows selected born support is not hidden on its
  own target rays: selected tubes carry `93.1%` weight share, are top
  contributor on `95.7%`, and are prefix-hidden on only `1.6%`. The
  prefix-alpha follow-up learns its local control surface (`0.198281 ->
  0.172906`, selected weight `0.4114 -> 0.4419`) and passes fixed-bin, but the
  fair 50-step dense row is essentially flat against binfix
  (`7.262/14.732/21.438`, alpha `>0.1` `75.4%`). The next STAR gate is broader
  support ownership/coverage or a different sampling distribution, not another
  local alpha-pressure-only train.
- [x] STAR UVT F32 feature tubes now have a selected-shader 128/256/512 scale
  gate for the no-pre-norm 8192-tube setup. Vec4 reduce is meaningful at 512px
  (`2.858s -> 2.491s` step), small at 256px (`1.112s -> 1.069s`), and a
  tie/slight backward loss at 128px. Low-res support is stricter: 128px needed
  cap256 plus `alpha>=1/32` to clear overflow at 8192 tubes.
- [x] STAR UVT F32 feature tubes now have a precomputed V-JEPA bridge audit.
  The original audit found that the old fast diagnostic did not use cached
  V-JEPA targets. That is now superseded: `star-feature-512-fast` launches the
  cached V-JEPA target-grid/frozen-probe batched path, while the older
  RGB-target `FeatureToColor` speed diagnostic is preserved as
  `star-feature-512-rgbfast`.
- [x] STAR UVT F32 feature tubes now have an opt-in cached-feature target smoke.
  The `rgb_pyramid` cache path runs through `VideoFeatureCache`, adapts
  `rgb_x1` from `[1,3,8,64,64]` to `[8,32,64,64]`, trains directly on
  `render.feature_image`, and passes the cache-hit rerun with loss
  `0.34006 -> 0.24809`, zero overflow, and `93.5ms/step`. This proves the
  target bridge contract, not real V-JEPA quality yet.
- [x] STAR UVT F32 feature tubes now have a real cached V-JEPA target smoke.
  `vjepa_torchhub` returns `vjepa_tokens` as `[1,1024,768]`; the STAR target
  adapter uses explicit `token_grid_shape=[4,16,16]`, truncates to F32,
  interpolates to `[8,32,64,64]`, and trains on `render.feature_image`. The
  8f/64px/512t cache-hit smoke passes with loss `1.00082 -> 0.90042`, zero
  overflow, and `181.1ms/step`. This is the real V-JEPA bridge gate, not the
  512px quality/scale gate.
- [x] STAR UVT F32 feature tubes now have a real cached V-JEPA 512px scale gate.
  The separate chunked 64f/512px/8192t/F32 V-JEPA-target config passes under
  `feature_direct_gradcache_reduce_vec4` with loss `1.000014 -> 0.999545`,
  zero overflow, and `3.74s/step` (`1.08s` backward, `1.73s` target chunk/loss).
  The first attempt exposed a 48 GiB target-prep mistake; channel adaptation now
  runs before dense grid upsampling. The current keeper then avoids keeping the
  full `[64,32,512,512]` target resident by streaming `[2,32,512,512]` target
  chunks from the channel-adapted `[32,32,16,16]` token grid. This is a
  scale/trainability gate, not a quality baseline.
- [x] STAR UVT F32 feature tubes now have a cached-target-layout follow-up for
  that V-JEPA gate. `feature_target.materialization=cached_chunks` precomputes
  the same adapted `[64,32,512,512]` target into 32 resident chunks
  (`2048MiB`, `2.04s` load/prep) and cuts the 5-step cache-hit row to
  `1.655s/step` (`0.770s` backward, `0.601s` render, `0.202s` target/loss),
  with the same loss curve and zero overflow. This is the short-run speed path,
  but the target-cache budget shows the scaling cliff: float32 adapted targets
  are `4GiB` at 128f/512px/F32 or 64f/512px/F64 and `8GiB` at 64f/1024px/F32.
- [x] STAR UVT F32 feature tubes now have a target-grid V-JEPA loss diagnostic.
  `feature_target.materialization=target_grid` keeps the channel-adapted
  `[32,32,16,16]` V-JEPA grid resident (`1.0MiB`) and downsamples rendered
  feature chunks before loss instead of materializing `[64,32,512,512]`
  targets. The matched 5-step row passes with loss `0.999935 -> 0.999467`,
  zero overflow, and `1.351s/step` (`0.705s` backward, `0.548s` render,
  `0.041s` target/loss, `0.138s` target load/prep). This is the fastest
  low-memory V-JEPA target diagnostic, but it changes the loss surface from
  dense render-grid MSE to coarse token-grid MSE and needs a longer media gate
  before promotion. The 20-step media follow-up passes and keeps feature loss
  monotonic (`0.999935 -> 0.997425`) at `1.451s/step`, but it is not RGB
  quality evidence because `rgb_loss_weight=0` and the colorizer is not trained.
  The first RGB-auxiliary control (`rgb_loss_weight=1.0`) trains the colorizer
  and decreases both feature loss (`0.999935 -> 0.997336`) and RGB loss
  (`0.338171 -> 0.335263`), but only improves RGB PSNR `4.709 -> 4.746` in
  20 steps while costing `2.000s/step`; this is not enough to promote the
  target-grid visual path. Raising the auxiliary to `rgb_loss_weight=10.0`
  barely helps RGB (`4.709 -> 4.750` PSNR) and slightly worsens feature loss
  versus aux1 (`0.997547` vs `0.997336`), so RGB weight alone is not the missing
  quality lever. Extending aux10 to 100 steps moves more clearly
  (`RGB PSNR 4.709 -> 5.109`, feature loss `0.999935 -> 0.964670`) at
  `1.876s/step`, so schedule length matters, but this is still far below RGB
  STAR quality and is not a promotion. A matched RGB-warm20 schedule
  (`feature=0/rgb=20` for 20 steps, then `feature=1/rgb=10`) is faster
  (`1.639s/step`) but worse than constant aux10 on final RGB PSNR (`5.046`)
  and feature loss (`0.973557`), so skipping feature loss early is a negative
  visual-control gate.
- [x] STAR UVT F32 feature tubes now have a standalone target-grid
  feature-to-RGB oracle. The hidden64 probe trains only `FeatureToColor` on the
  cached V-JEPA target grid `[32,32,16,16]`, reaching grid PSNR `23.401` and
  full upsampled PSNR `20.073` at `2.427ms/step` (`1.003ms` backward). This
  proves the cached target-grid features are decodable; the missing bridge is
  loading/freezing that decoder in STAR training or probe logging, not another
  RGB aux weight or feature-loss-skipping schedule.
- [x] STAR UVT F32 feature tubes now have the frozen target-grid RGB probe
  wired into the STAR trainer. The 20-step 64f/512px/8192t gate passes at
  `1.220s/step` with zero overflow, feature loss `0.999935 -> 0.998357`, and
  frozen-probe PSNR `13.985 -> 14.060`. This proves cheap integration and
  gradient flow, but it is not a quality promotion because the short-run visual
  gain is tiny. The matched 100-step gate moves more clearly: feature loss
  `0.999935 -> 0.970035` and frozen-probe PSNR `13.985 -> 14.641` at
  `1.268s/step`, cheaper than the 100-step RGB-aux10 row but still well below
  the standalone `20.073` PSNR target-grid oracle. The 300-step extension keeps
  moving: feature loss `0.999935 -> 0.811652` and frozen-probe PSNR
  `13.985 -> 16.560` at `1.355s/step`, so the objective is viable but still not
  at oracle quality. A checkpoint/no-media 300-step rerun matches that curve at
  `1.268s/step`, and a resumed 300-step continuation reaches feature loss
  `0.655366` plus frozen-probe PSNR `19.884` at `1.440s/step`. That nearly
  reaches the standalone full-video upsample number (`20.073`). A probe-emphasis
  600->800 continuation from that state reaches frozen-probe PSNR `21.425` at
  `1.512s/step`, with zero overflow, but feature loss drifts
  `0.655132 -> 0.703820`. That passes the standalone full-video number while
  still trailing the same-grid oracle (`23.401`). A scheduled 800->1000 balance
  continuation recovers feature loss `0.703862 -> 0.643852` at `1.308s/step`,
  but gives back a small amount of probe PSNR (`21.428 -> 21.382`) and is
  nonpassing on probe-loss decrease. A feature0.5/probe40 1000->1100 Pareto
  continuation passes the combined gate at `1.461s/step`, moves probe PSNR
  `21.384 -> 21.789`, and keeps zero overflow, but feature loss drifts
  `0.643823 -> 0.656728`. A 1100->1200 recover schedule then lowers feature
  loss `0.656765 -> 0.635093` at `1.521s/step`, but gives back a little probe
  PSNR (`21.792 -> 21.738`) and is nonpassing on probe-loss decrease. A short
  feature0.75/probe40 1200->1250 continuation then passes and restores probe
  PSNR `21.740 -> 21.929` at `1.523s/step`, but pushes feature loss back up
  `0.635066 -> 0.638799`. A feature1/probe40 1250->1300 continuation is the
  first current both-improving objective-balance row: feature loss
  `0.638803 -> 0.632192`, probe PSNR `21.933 -> 21.963`, zero overflow, and
  `1.285s/step`. The 1300->1400 extension keeps both metrics moving: feature
  loss `0.632124 -> 0.627129`, probe PSNR `21.965 -> 21.979`, zero overflow,
	  and `1.690s/step`. This still trails the same-grid oracle (`23.401`), but the
	  timing-control repeat reproduces the slower regime at `1.711s/step` with the
	  same zero-overflow `68/45/128` max/p95/cap tile read. The sparse-forward
	  batched-VJP version preserves the same 100-step objective movement while
	  cutting mean step/backward/render to `399.9/176.9/125.2ms` and the last-20
	  window to `262.9ms/step`; the probe media is valid but still blurry. The
	  effective-lr001 sparse-forward rerun keeps the dense lr001 endpoint at
	  `372.3ms/step` mean, `158.9ms` backward, feature loss `0.630549`, and
	  probe PSNR `22.034`, but gives up lr005's better feature loss and has a
	  noisy late timing window. The whole-graph profile gate then splits the
	  current target-grid plus frozen-probe objective:
  renderer backward is still the main backward bucket (`81.3-81.4%` of manual
  backward), but the isolated manual profile does not reproduce the 1300-source
  slowdown (`1565.9ms` manual total at global step 1250 versus `1504.0ms` at
  1300), so the remaining timing question is trainer-autograd/MPS variance
  versus a missing end-to-end trace, not tile overflow. The end-to-end trace is
  now in place: `step_timings_ms` is recorded in trainer JSONs, the 20-step
  traces show the 1300-source row is still slower after dropping the first
  optimizer/warmup step (`1850.7ms` vs `1705.3ms`), and the 1300-source trace
  has a late objective spike at global step `1318`. A chunk trace around
  `1317/1318/1319` shows the spike is distributed, not a single bad frame
  chunk: `27/32` chunks increase, with `44.5%` of the weighted-loss delta in
  frames `0-15`, and the elevated loss persists into `1319`. The reports are
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`,
  and
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`
  plus
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`.
  The LR checkpoint gate then confirms this is schedule-state sensitive: the
  original `lr=0.005` continuation fails, while `lr=0.001` continuations avoid
  the 1318 spike. The trainer now re-applies config LR after loading optimizer
  state, because the 1300-step checkpoint optimizer carried `0.005`; the
  corrected retained-optimizer run records loaded/effective LRs `[0.005] ->
  [0.001]`, passes with end loss `0.884576`, feature loss `0.631648`, probe
  PSNR `21.991`, and no-first timing `1384.4ms/step` / `748.9ms` backward.
  The reset-optimizer control also passes (`0.884902`, `0.631614`, `21.984`),
  so the next quality continuation should use the 1300 checkpoint with
  effective `lr=0.001`; this is not the renderer-speed fix. Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`.
  The actual 100-step effective-`lr=0.001` continuation then passes with media
  and checkpoint: feature loss `0.632124 -> 0.630549`, probe PSNR
  `21.965 -> 22.034`, mean `1463.8ms/step`, `778.4ms` backward, and zero
  overflow. It avoids the early `1318` spike but later has a smaller transient
  jump at `1377->1378` before recovering. Compared with the older `lr=0.005`
  1300->1400 row, it is faster and has better probe PSNR, but worse final
  feature loss (`0.630549` vs `0.627129`) and slightly worse weighted loss
  (`0.880942` vs `0.880751`). Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`.
  The matched effective-lr001 sparse-forward rerun preserves that dense lr001
  endpoint at `372.3ms/step` mean and `158.9ms` backward, but it keeps the same
  quality tradeoff and noisy late timing. Report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.md`.
  A matched checkpoint-selection gate from global step 1400 then selects the
  lr005-sparse state for further quality work: 50 effective-lr001 steps from
  that checkpoint pass to feature loss `0.625976` and probe PSNR `22.010`,
  while the lr001-sparse checkpoint fails after a `1444 -> 1445` objective jump
  and ends at feature loss `0.631770` / probe PSNR `21.843`. Report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_checkpoint_select_1400_to1450.md`.
  The selected 1450->1500 media gate then passes from the lr005-sparse lineage
  and writes probe media/checkpoint: loss `0.877762 -> 0.876224`, feature loss
  `0.625962 -> 0.625428`, probe PSNR `22.010 -> 22.027`, mean
  `315.8ms/step`, `130.2ms` backward, last-20 `254.0ms/step` /
  `108.2ms` backward, zero overflow. The media is valid but still blurry, so
	  this is a stability/continuation pass rather than a visual-quality promotion.
	  Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr005sparse_1450_to1500_media.md`.
	  A full-resolution autograd RGB-aux bridge from the sparse 1500 checkpoint is
	  a negative quality result, even when the trainable hidden64 colorizer is
	  initialized from the trained target-grid RGB probe: RGB loss improves
	  `0.272626 -> 0.259968` and RGB PSNR moves `5.644 -> 5.851`, but feature
	  loss worsens `0.625418 -> 0.626799`, frozen-probe PSNR drops
	  `22.028 -> 21.879`, trainable-colorizer media artifacts appear, and mean
	  step time jumps to `5206.6ms` (`16.5x` slower than the sparse 1500 row).
	  Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_autograd_rgbaux_probeinit_from1500_negative.md`.
	  The rendered-feature sparse-pixel RGB probe from the same sparse 1500
	  checkpoint answers the distribution-mismatch follow-up: training only a
	  hidden64 colorizer on actual rendered sparse pixels is fast and passes
	  (`0.168261 -> 0.099014` sparse-sample loss, `7.740 -> 10.043` sparse-sample
		  PSNR, `241.4ms/step`), but dense full-video PSNR is only `6.096` and the
		  media remains sparse-streaked. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels.md`.
		  The denser stratified64 follow-up samples `262,144` full-resolution
		  pixels/step (`4x` the prior rendered-feature probe) and still reaches only
		  `6.132` dense full-video PSNR at `331.5ms/step`, so target-grid sampling
		  bias is not the explanation. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64.md`.
		  The first native sparse visual VJP gate then moves the sparse RGB loss back
		  into STAR features (`model_grad_seen=true`, frozen colorizer) at
		  `336.8ms/step`, but it is quality-negative with full-video PSNR `5.739`.
		  Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe.md`.
		  The joint sparse visual VJP follow-up trains STAR and the hidden64
		  colorizer together (`model_grad_seen=true`, `colorizer_grad_seen=true`)
		  and raises dense full-video PSNR to `6.025`, but it still trails the
		  colorizer-only stratified diagnostic (`6.132`) and slows to
		  `729.4ms/step`. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe.md`.
		  The mixed target-grid/probe plus sparse visual VJP gate then preserves
		  feature/probe movement and improves sparse visual sample PSNR to
		  `6.036`, but dense full-video PSNR remains `6.024` at
		  `964.0ms/step`. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500.md`.
		  The patch2x2 same-pixel support follow-up samples contiguous `2x2`
		  patches on a `32x32` grid. It is faster (`619.5ms/step`) and improves
		  sparse visual sample PSNR to `6.179`, but feature-target loss worsens
		  and dense full-video PSNR drops to `6.000`. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500.md`.
		  The patch-mean64 visual-basis follow-up samples `1,048,576` sparse
		  visual pixels/step and pools them into `262,144` local-mean cells. It
		  restores feature/probe movement and dense full-video PSNR to `6.023`,
		  but costs `1124.6ms/step` and media remains sparse/high-frequency.
		  Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patchmean64_from1500.md`.
		  The target-area64 follow-up keeps the same `1,048,576` sparse
		  visual pixels/step and `262,144` loss cells, but compares against true
		  area-downsampled RGB target cells. It is slightly faster
		  (`1103.1ms/step`) and raises sparse visual PSNR to `6.064`, but dense
		  full-video PSNR remains `6.023` and media is unchanged. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500.md`.
		  The phased target-area64 follow-up cycles the same compact `2x2`
		  support across a `4x4` subcell schedule. It passes and raises sparse
		  visual PSNR to `6.077`, but dense full-video PSNR falls to `6.019` at
		  `1169.2ms/step`; fixed support position is not the quality blocker.
		  Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500.md`.
		  The full-cell8 target-area follow-up sends gradients through every
		  pixel in each `8x8` target-area cell (`16,777,216` visual pixels/step
		  into `262,144` loss cells). It is nonpassing: sparse visual PSNR rises
		  to `5.822`, but feature loss worsens, probe PSNR falls to `21.860`,
		  dense full-video PSNR falls to `5.722`, and mean step is
		  `7526.7ms` with `5702.6ms` in sparse visual loss construction.
		  Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500.md`.
		  The manual hidden64 VJP version matches that endpoint while cutting
		  sparse visual loss construction to `3803.6ms` and mean step to
		  `6414.0ms`, but it remains nonpassing and quality-negative. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500.md`.
		  The star-only manual hidden64 variant skips colorizer parameter
		  gradients and cuts mean step further to `5801.7ms`, but dense full-video
		  PSNR drops again to `5.648`; it is a speed diagnostic, not a promotion.
		  Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500.md`.
		  The fast-GELU derivative variant keeps colorizer gradients and uses a
		  sigmoid-GELU derivative in the manual VJP. It is rejected: mean step is
		  `6252.1ms`, dense RGB stays `5.722`, and the profile loss-side total is
		  worse than exact manual. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_fastgelu_from1500.md`.
		  The compact linear colorizer/manual VJP variant is the first affordable
		  full-cell8 mechanics gate: a standalone linear target-grid RGB probe
		  reaches only `16.980` full-video PSNR, but the full-cell8 trainer row
		  passes mechanically at `2064.4ms/step`, with sparse visual loss
		  construction down to `383.3ms`. It still is not a quality promotion:
		  feature loss slightly worsens, dense full-video RGB is only `5.668`,
		  and the weak linear decoder gives away too much visual capacity. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500.md`.
		  The hidden32 manual VJP follow-up keeps most of the hidden64 standalone
		  target-grid oracle (`19.704` full-video PSNR vs hidden64 `20.073`) but
		  still costs `4298.4ms/step` and `2136.1ms` sparse visual loss
		  construction on full-cell8 support; dense full-video RGB remains only
		  `5.678`. This rejects "just shrink the hidden decoder in Python" as the
		  next route. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500.md`.
		  The split manual-VJP subphase profiles show the target-area reduction is
		  not the big remaining loss-side cost (`~0.12-0.13s` full-step
		  extrapolated); exact GELU backward (`~1.34-1.44s`) and the first
		  hidden-layer matmul (`~0.75-0.89s`) dominate. Reports:
		  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_split_fullstep.md`
		  and
		  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_staronly_subphase_profile_fullstep.md`.
	  The first explicit optimizer-LR schedule gate (`0.001` until global step
	  `1375`, then `0.00025`) is a negative quality result even though the run
	  passes: it removes the `1377->1378` jump, but a comparable jump reappears at
  `1385->1386`, and the 100-step row is worse than static effective-lr001 on
  final weighted loss (`0.881602` vs `0.880942`), feature loss (`0.630803` vs
  `0.630549`), probe PSNR (`22.027` vs `22.034`), and timing (`1506.9ms` /
  `807.2ms` backward vs `1463.8ms` / `778.4ms`). The late trace intentionally
  stops just after the spike and is expected to fail the quality pass bit; it
	  confirms `26/32` chunks worsen at `1385->1386`, with summed weighted-loss
  delta `0.015248` and largest chunk at frame `0` (`0.001802`). Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md`.
	  Do not promote this schedule; checkpoint selection is now resolved in favor
	  of the lr005-sparse lineage, while new speed work should beat the
	  sparse-forward batched-VJP helper before replacing it.
  The STAR feature overfit trainer now also has opt-in checkpoint/resume for longer
  gates: `output.checkpoint` saves model,
  colorizer, optimizer, config, row, and losses; `train.resume_checkpoint` loads
  them as warm-start local steps, with optimizer resume on by default.
  `train.global_step_offset` now records explicit continuation steps for
  resumed schedules. The contract is smoke-tested on the real 8f/64px
  RGB-pyramid route and the 64f/512px frozen-probe route.
- [x] STAR UVT F32 feature tubes now have a normalized V-JEPA comparison report
  against cached Gaussian/token rows. The matched 64f/512px/8192 rows are:
  STAR V-JEPA streaming target `3.743s/step`, STAR V-JEPA cached-chunks target
  `1.655s/step`, STAR V-JEPA target-grid loss `1.351s/step` (`1.451s/step`
  for the 20-step media row, about `2.000s/step` with 20-step RGB aux,
  `1.876s/step` for the 100-step aux10 row, `1.639s/step` for the negative
  RGB-warm20 row, and `0.00243s/step` for the standalone feature-to-RGB
  oracle; `1.220s/step`, `1.268s/step`, `1.355s/step`, and `1.440s/step` for
  the 20-step, 100-step, 300-step, and resumed 300-step integrated frozen-probe
  rows; `1.512s/step` for the probe-emphasis 600->800 continuation; and
  `1.308s/step` for the nonpassing scheduled balance continuation; and
  `1.461s/step` for the passing feature0.5/probe40 1000->1100 continuation;
  `1.521s/step` for the nonpassing 1100->1200 recover schedule; and
  `1.523s/step` for the passing feature0.75/probe40 1200->1250 probe-recovery
  row; `1.285s/step` for the passing feature1/probe40 1250->1300 both-improving
		  row; `1.690s/step` for the passing feature1/probe40 1300->1400 dense
		  extension; `1.711s/step` for the matched dense timing repeat; and
		  `0.400s/step` mean / `0.263s/step` last-20 for the lr005 sparse-forward
		  batched-VJP 100-step media helper; and `0.372s/step` mean /
		  `0.539s/step` last-20 for the lr001 sparse-forward rerun),
		  plus the selected lr005-sparse 1450->1500 media gate at `0.316s/step` mean
		  / `0.254s/step` last-20 and the negative autograd RGB-aux probe-init bridge
		  at `5.207s/step`, plus the rendered-feature sparse-pixel RGB probe at
		  `0.241s/step`, the stratified64 rendered-pixel probe at `0.332s/step`,
		  the sparse visual VJP frozen-probe gate at `0.337s/step`, and the joint
		  sparse visual VJP gate at `0.729s/step`, plus the mixed
		  target-grid/probe+sparse visual VJP gate at `0.964s/step` and the
		  patch2x2 support gate at `0.620s/step`, plus the patch-mean64
		  visual-basis gate at `1.125s/step`, plus the target-area64
		  visual-basis gate at `1.103s/step`, plus the phased target-area64
		  visual-basis gate at `1.169s/step`, plus the full-cell8
		  target-area gate at `7.527s/step` and its manual hidden64 VJP variant
		  at `6.414s/step` plus star-only manual hidden64 at `5.802s/step`
		  and fast-GELU manual hidden64 at `6.252s/step`, plus the compact
		  manual-linear full-cell8 diagnostic at `2.064s/step` and manual
		  hidden32 full-cell8 at `4.298s/step`, plus the matched 512px native
		  handoff gate (`fused_first3` `1.153s` total, `logit_handoff_reduce_vec4`
		  `0.386s` native backward plus `0.422s` prep),
	  STAR RGB fast diagnostic
	  `2.491s/step`, Gaussian/token recon-only cached conditioning
  `3.460s/step`, and Gaussian/token prediction-side V-JEPA loss
	  `38.621s/step` with `36.762s` in backward. This confirms repeated STAR
	  target interpolation was removable, target-grid loss avoids the 2GiB cache,
	  and sparse-forward batched VJP is the current fast cached-V-JEPA helper path;
	  the old Gaussian V-JEPA-loss bottleneck is frozen V-JEPA backward on predicted
	  video.
- [x] STAR UVT Gate 4 now has a same-clip 512px quality bracket. RGB STAR
  direct-atomic reaches `12.44` PSNR in 20 steps on the test video, while the
  best feature STAR row reaches only `4.99` PSNR (`feature_meets_rgb_psnr=false`
  in `outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md`); the
  fastest feature row is the identity/no-pre-norm diagnostic at `2.54s/step`.
  The feature path is not a source-overfit quality replacement yet.
- [x] STAR UVT F32 feature tubes now have a native full-cell target-area visual
  VJP gate. `native_hidden64_target_area_star_only` uses bin-only tile setup,
  native hidden64 RGB cell sums, and native STAR backward from target-area cell
  gradients. Tiny parity passes, synthetic full-support timing wins at 128/256px
  and survives 512px native-only (`1874.4ms`) where all-at-once Torch OOMs; the
  matched first-class 512px/64f trainer row cuts full-cell8 star-only mean step
  `5801.7 -> 3496.0ms` with zero overflow and the same `5.648` full RGB endpoint.
  This is a speed/memory promotion, not a visual-quality promotion.
- [ ] Use the native target-area path as the new full-support visual-VJP speed
  baseline, then either reduce native reverse recompute or change the visual
	  objective/support before claiming the 512px feature path is a replacement for
	  projected F32 splatting. The first recompute follow-up, native hidden32
	  target-area, is rejected: it cuts mean step to `2464.6ms` but fails the gate
	  with probe PSNR `19.481` and full RGB `5.632`. The skip-feature-grad
	  diagnostic also says raw feature-gradient atomics are not the factor blocker:
	  hidden64 target-area backward only drops `594.9 -> 562.2ms` at 256px and
	  `1918.6 -> 1854.3ms` at 512px. The opposite feature-only split confirms
	  the same bottleneck: same-session full/feature-only/geometry-only backward
	  is `581.3/548.2/547.3ms` at 256px and `1919.7/2106.7/2174.0ms` at 512px,
	  so shared hidden64 recompute/traversal dominates simple output-gradient
	  masking. The recompute-only floor with all output-gradient atomics disabled
	  is still `571.3ms` at 256px and `2101.7ms` at 512px, so the shared
	  replay/hidden64 VJP envelope is the native target-area bottleneck.
	  Traversal-only drops backward to `194.9ms` at 256px and `742.2ms` at
	  512px, putting the hidden64 forward/VJP slice at roughly `376.5ms` and
	  `1359.6ms`. Hidden-forward-only splits that slice into forward
	  `150.6/450.6ms` and backward `225.8/909.0ms` at 256/512px, so W^T/GELU
	  feature VJP is the larger hidden subtarget. Hidden-preact-only narrows it
	  further: output+GELU prebackward is only `54.8/61.7ms`, while the F32
	  W^T feature-gradient matvec is `171.0/847.3ms`. The exact row-major W^T
	  follow-up is rejected as a trainable speed path: it passes full-gradient
	  parity but slows full native backward (`647.4 -> 711.5ms` at 256px and
	  `2040.5 -> 2161.6ms` at 512px), while only nudging the recompute-only
	  floor (`572.1 -> 555.8ms`, `1993.0 -> 1983.4ms`). The exact vec4 W^T
	  follow-up is the first positive W^T kernel reduction: same-build full
	  backward improves `675.9 -> 642.2ms` at 256px and `2408.1 -> 1804.7ms`
	  at 512px, with a 512px repeat at `1832.8ms`; recompute-only improves
	  `586.6 -> 518.3ms` and `2305.2 -> 1635.8ms`. The current-build trainer
	  A/B promotes `native_hidden64_target_area_star_only_vec4_wt` as the
	  preferred full-support native target-area star-only mode: mean step improves
	  `4262.1 -> 4071.0ms`, mean backward `3700.2 -> 3152.6ms`, and mean sparse
	  visual backward `2546.7 -> 1963.5ms`, with matched `5.648` full RGB. The
	  50-step promoted-mode gate passes and warms to `3359.2ms` mean /
	  `3072.1ms` last step with zero overflow and full RGB `5.732`, but the
	  compact target-area64 helper route is still faster and higher quality on
	  the fresh current-build gate (`930.6ms`, `6.023` full RGB), so full-cell8
	  native target-area is the exact full-support baseline rather than the
	  practical visual objective. A compact native star-only diagnostic is also
	  rejected (`2265.0ms` mean step, no colorizer gradients); native compact VJP
	  only matters if it preserves colorizer gradients and beats compact autograd.
	  The compact manual-hidden64 colorizer-gradient diagnostic also fails:
	  it sees colorizer grads, but costs `2007.4ms` mean / `1899.2ms`
	  no-first step, worsens feature/probe quality, and still trails compact
	  autograd (`991.9ms` mean / `787.7ms` no-first over the first five rows).
	  The native colorizer-gradient vec4 W^T implementation closes the missing
	  ABI gap and passes tiny parity for STAR plus hidden/output colorizer
	  parameter gradients, but fails the trainer gate too (`2738.7ms` mean step,
	  `1474.2ms` backward, same feature/probe regression). The follow-up
	  colorizer-gradient-only split pins the failure on naive colorizer parameter
	  atomics: compact-support direct native backward is `88.9ms` for star-only,
	  `536.6ms` for colorizer-grad-only, and `531.4ms` for full colorizer vec4
	  W^T, so another STAR W^T shuffle is not the next compact-native lever.
	  A Torch/MPS sidecar reducer is correct but also rejected: it beats native
	  atomics (`390.9ms` versus `752.8ms` same-window total) but still trails the
	  sparse-pixel baseline (`276.6ms`) because it duplicates sparse render and
	  target-area hidden replay. The same-pass SIMD-reduced colorizer follow-up
	  fixes the direct-kernel atomic envelope (`297.2ms` native total versus
	  `312.1ms` sparse-pixel baseline in the matched compact window), but the
	  trainer gate still rejects it: mean step `2908.9ms`, mean backward
	  `1363.0ms`, sparse visual backward `604.0ms`, and the same feature/probe
	  regression as the naive native colorizer route. Keep compact autograd as
	  the practical visual route until native also removes whole-graph
	  target-area overhead or changes the objective/support. The selected
	  visual-quality gate now explicitly fails scale-up: dense full RGB is
	  `6.023` PSNR, media is still sparse/streaked or blurry, and RGB STAR is
	  `12.444` PSNR on the same-clip bracket.
- [ ] Attach to a video diffusion model for video diffusion features.
- [ ] Run single-step Marigold-style features for maximum information extraction.

### Viewer

- [ ] Set up an HTML viewer that can open the token format, load the MLPs, bake splats, sort them, and render with WGPU.
- [x] Prototype a separate browser WebGPU trainer at `web/dynaworld_browser_trainer/`.
  It preloads the local Neural3D preview when available, exposes World
  Tubes-style and dynamic-splats-style WGSL approximation modes, trains
  source-view splat/tube residuals live, reports actual optimizer throughput as
  `Steps/s`, and now defaults to 768 splats plus a measured 95% motion-sample
  mix. It also reports motion coverage / active-splat diagnostics and uses the
  more motion-covered `converge28` initialization. The `converge29`/`30` UI
  adds peak motion alpha plus mean opacity/radius readbacks so shrink/fade
  failures are visible, then bumps the asset cache and makes the desktop rail
  independently scrollable. `converge31` tests a small motion-coverage hinge
  in the simplified WGSL train shader; its 50%/0.20 setting preserves support
  but slows motion loss, so `converge32` weakens it to a late 44%/0.08 support
  guard. The extended `converge32` trace reached step `861`, true `Motion Loss
  0.005914`, and motion coverage `47.0%`, keeping more support than
  `converge28` at a small MSE cost. `converge33` renames the selector to
  `Motion Model` and keeps World Tubes-style as the default after a same-guard
  comparison reached step `629`, true `Motion Loss 0.005938`, and motion
  coverage `47.4%`, effectively tied with Dynamic splats-style. This is not the
  full Metal shared-backward/tiled renderer port yet. `converge34` keeps the
  math unchanged, makes target/render equal-width on desktop, adds an RGB versus
  motion-residual target selector, and verifies a live post-edit World
  Tubes-style trace to step `268` / true `Motion Loss 0.006505` / motion
  coverage `50.2%` with no browser warnings/errors. `converge38`/`39` add
  result-side dynamic-layer and alpha-support views; those show the model does
  cover the moving person but also leaves broad background support active.
  `converge39` lowers the temporal gate floor from `sigma*0.70` to
  `sigma*0.30`, improving boot true motion loss `0.011522 -> 0.011099` and
  short-trace loss to `0.007788` by step `72` while reducing motion coverage to
  `53.9%`. `converge40` adds `Static Cov`, a low-motion alpha penalty, and
  opacity decay, but the first decay weight `0.055` is too aggressive: step
  `239` reached true `Motion Loss 0.007012` while motion coverage fell to
  `42.4%`. `converge41` lowers decay to `0.025`, reaching step `294`, true
  `Motion Loss 0.006751`, motion coverage `44.6%`, `Static Cov 2.6%`, and
  `Active 406/768`; `converge42` keeps that train math and thins the static
  coverage validation readback. `converge43` adds a dedicated low-motion sample
  buffer plus an 8% static sample reserve so the static alpha penalty is trained
  deliberately instead of through the tiny uniform tail; the first browser trace
  loaded `Static Px 16384` and reached step `259`, true `Motion Loss 0.006803`,
  motion coverage `45.5%`, `Static Cov 2.6%`, and `Active 420/768` with no
  browser warnings/errors. `converge44` exposes that reserve as a `Static Mix`
  slider so `0%` recovers the v42-style sampler and default `8%` gives effective
  `Motion Mix 92%`; the in-app smoke loaded v44 assets, stepped once, and had
  no browser warnings/errors. The matched v44 control then showed the static
  reserve is not the core convergence issue: `0%` reached step `274`, true
  `Motion Loss 0.006794`, and motion coverage `45.0%`, while default `8%`
  reached step `271`, true `Motion Loss 0.006822`, and motion coverage
  `45.3%`. `converge45` exposes the hidden support target as `Support Guard`
  and defaults it to `52%`; the first v45 trace reached step `297`, true
  `Motion Loss 0.007060`, motion coverage `48.2%`, `Static Cov 2.7%`, and
  `Active 406/768`. `converge46` adds a lightweight frame-motion centroid
  initializer for motion-seeded splats; its first trace reached step `290`,
  true `Motion Loss 0.007036`, motion coverage `47.0%`, `Static Cov 2.8%`,
  and `Active 407/768`. This is a small fit win but not a support win.
  `converge47` makes the motion prior local by residual-matching nearby pixels
  in adjacent frames; it reached step `279`, true `Motion Loss 0.006885`,
  motion coverage `48.1%`, `Static Cov 2.7%`, and `Active 414/768`. That is
  the current best browser source-view tradeoff, but the next fix should still
  move toward renderer parity/exported init rather than another pure
  support-preservation tweak. `converge48` is a UI/preview pass: preview time
  now loops by default, the side-by-side Neural3D source/target crops are shown
  together in a small camera strip, and the docs call out that training is still
  the 128x128x8 source-view crop rather than target-camera or novel-view
  supervision. `converge49` adds validation-oriented visual quality signals:
  sparse-grid MAE, PSNR, and global-luma SSIM readouts plus a throttled
  source-view validation-error heat map. SSIM is validation-only for now; train
  regularization remains the existing alpha/support/opacity/radius guard stack.
  `converge50`-`59` add per-parameter Adam moments, persistent absolute-gradient
  and contribution statistics, fixed-cap weak-slot recycling into localized
  high-residual motion samples, parameter-delta/recycle diagnostics, serialized
  GPU readback, and startup/reset guards. The WebGPU pipeline compiles and boots
  cleanly on the Apple adapter, but the measured Adam/density probes do not yet
  beat `converge47`; treat this as the paper-backed optimizer/density foundation,
  not a convergence or heldout-view promotion. `multicam67` replaces the old
  source-view crop with a thin adapter over the canonical Coffee Martini paper
  contract: `cam04`/`cam09` train, `cam06` validation-only, eight exact times
  spanning all 300 frames at `96x72`, and 768 SfM XYZRGB seeds. Shared 3D
  primitives now project through the calibrated cameras in WebGPU, all three
  views loop together, and train versus heldout loss/PSNR/global-SSIM-proxy are
  reported separately. A 132-step World Tubes-style smoke improved train loss
  `0.182629 -> 0.173302` and heldout loss `0.192498 -> 0.185356`; this remains a
  browser demo result, not native World Tubes parity or a baseline row.

### Pretraining Setup

- [x] Collect diverse single-camera video datasets for the first pretraining pass.
- [x] Collect multi-camera data for novel-view-synthesis finetuning.
- [x] Document the same-view and novel-view data-loader contract in `research_notes/data_contract.md`.
- [ ] Build the mixed trainer/sampler that alternates single-sequence same-view loss with multicam heldout-camera loss.
- [ ] Decide whether scene cuts should be marked and split during preprocessing.

### World Token Base Model

- [ ] Define the exported world-token contract: token shapes, time
  conditioning, decoder inputs, and the minimum novel-camera consistency checks.
- [ ] Train the direct base path: encode video, emit world tokens, decode
  splats, render source and novel cameras.
- [ ] Sort out how to handle time.
- [ ] Support longer videos and sliding-window training.
- [ ] Better support novel camera angles when training mostly from the input camera angle.
- [ ] Separate camera and video representations well enough that camera changes do not collapse into video-embed leakage.
- [ ] Test the direct path: pretrain on single-camera source video, then finetune on paired camera data so the model can encode one view, swap the camera token, and decode another view.
- [ ] Find pretraining pressure that encourages camera-token swapping behavior before paired-camera finetuning.
- [ ] Test same-video chunk mixing: encode two chunks from the same video, combine the first chunk's video tokens with the second chunk's camera token, and train against the second chunk's ground truth so shared video tokens learn to render under a different camera path and time.
- [ ] Turn each clip into "multi view" with a crop and perspective warp. Classic videography trick where you take high resolution footage, crop a corner, and rescale so the perspective looks like it is in the center of each frame. Feels valuable, but downsides: it is still from the same angle, and too much perspective warp is a bit cheap and does not exactly align to GT camera data.
- [ ] Try the crop variant without perspective warp: shift the rays so the crop is defined as a camera extrinsic. More honest, but then it is sort of learning crop shift only, and might not generalize as well to non-crop shifts where the shift is not the center of the camera.
- [ ] Try doing both crop variants plus chunk mixing together in pretrain and see if that is enough to get a good prior.
- [ ] Worry about the wrong task forcing the camera implicitly into image tokens if we try to hide camera position too much in non-principled ways.
- [ ] Try a BERT-like random masking dropout scheme in pretrain as an alternative to only chopping the video in half. Might be more robust, but worry that it will force the camera data to hide itself in the image tokens.
- [ ] Keep AR/diffusion generation out of the base-model proof until the world
  tokens pass source-camera and novel-camera consistency checks.

### Novel View Post-Training

- [ ] Second stage post-training: render novel passes and train a GAN on them, so we do a GAN for novel and non-novel views. It has to learn to make them both look the same.
- [ ] Maybe some sort of reward style training here as well. See Chopgrad ([arxiv 2603.17812](https://arxiv.org/abs/2603.17812)) which recently did really high quality correction.
- [ ] The plan is a bit of both: (1) give the model some prior off-camera capability in pretrain, (2) refine that in post training.

### Video Diffusion Bootstrap

- [ ] Evaluate whether score distillation is useful by noising rendered output images or using a differentiable diffusion technique to push gradients back into the renderer.
- [ ] Test direct video features from a single-step zero-SNR schedule.
- [ ] Work out how video diffusion features interact with windowing and memory limits.

### Future Directions

#### Auto-Research

- [ ] Set up auto-research swarm configs.
- [ ] Keep local Mac shader support fast enough that contributors can run this locally without cloud GPU cost.
- [ ] Set up "World Model at Home".
- [ ] Document how to contribute auto-research so users can run local experiments and contribute findings back.
- [ ] Investigate async training across users' home GPUs.

#### World-Token Generation

- [ ] Train an AR predictor over world tokens for video continuation.
- [ ] Train a diffusion predictor over world tokens for video continuation.
- [ ] Test image=>video by initializing or conditioning the world-token stream from an image.
- [ ] Test text=>video by conditioning world-token generation on text.
- [ ] Benchmark whether world-token decoding is a useful inductive bias for
  video generation itself versus standard video diffusion.

## Setup

```bash
uv sync
git submodule update --init --recursive
```

## Camera Prebake

From the repo root:

```bash
./src/train_scripts/get_camera.sh
```

Default inputs and outputs live under `test_data/`.

## Local Mac Data

The canonical data-loader and manifest contract is
`research_notes/data_contract.md`. It distinguishes the broad same-view
single-sequence loader from the calibrated multicam heldout-camera loader.

The default tiny split dataset is now 30 short mined-video clips from 30
distinct source videos: 20 train and 10 test.

```bash
./src/dataset_scripts/youtube_scene_distinct_30_seed.sh
```

That writes `data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_64_4fps_16f/`.

The older local multi-camera sample builder is still available for camera-path
debugging:

```bash
./src/train_scripts/build_local_mac_30_clip_dataset.sh --overwrite
```

## Train

Tiny 30-clip local baseline:

```bash
PYTHONPATH=src/train uv run python src/train/train.py \
  src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc
```

Current recommended local dynamic run:

```bash
PYTHONPATH=src/train uv run python src/train/train.py \
  src/train_configs/local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc
```

The retired prebaked-camera TokenGS trainer lives in git history. The active
known-camera path is the video-token trainer config:

```bash
src/train_configs/local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc
```

Token-only smoke:

```bash
PYTHONPATH=src/train uv run python src/train/train.py \
  src/train_configs/local_mac_overfit_video_token_smoke.jsonc
```

Fast-mac 128px implicit-camera smoke:

```bash
PYTHONPATH=src/train uv run python src/train/train.py \
  src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats.jsonc
```
