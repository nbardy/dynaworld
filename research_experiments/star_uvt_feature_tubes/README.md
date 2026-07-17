# STAR UVT Feature Tubes

Isolated scratch space for feature-valued STAR/UVT tube experiments. Keep this
directory separate from the RGB `star_uvt_v0` trainer until the dense feature
contract is proven.

Current question:

- Can UVT tubes carry F32 features instead of RGB, render a feature image, and
  let the existing image-space `FeatureToColor` decoder produce RGB for the
  reconstruction loss?
- Can that same STAR feature route be connected to cached V-JEPA targets? Yes,
  the current `star-feature-512-fast` helper now launches the cached V-JEPA
  target-grid/frozen-probe sparse-forward batched VJP route. The older RGB
  reconstruction-through-`FeatureToColor` speed row is kept as
  `star-feature-512-rgbfast`.

Current port plan:

- `2026-05-18_fast_shader_port_plan.md` records the fast shader lessons from
  the feature-splat forks and maps them onto a staged STAR UVT feature-tube
  implementation.

Current state:

- Dense feature-tube contract passes on CPU and MPS.
- The prototype now records shapes, finite checks, gradient reachability,
  tiny-overfit loss, full-vs-frame-chunked backward parity, and split timings.
- Evidence:
  `outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract.json`
  and
  `outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json`.
- The RGB `star_uvt_v0` renderer remains hardcoded to 3-channel `float3`
  color/gradient paths.
- The first feature-specific direct Metal calls now exist under
  `torch_gsplat_bridge_star_uvt.feature_rasterize`:
  `render_uvt_feature_tubes` and `direct_atomic_feature_backward`.
- Tiny F=4/F=32 forward/backward parity passes, and the first
  64f/256px/32768/F32 direct feature timing row is finite with zero overflow:
  `757.9ms` total, `190.1ms` forward, `567.8ms` backward.
- A trainable autograd wrapper now exists:
  `render_uvt_feature_tubes_autograd`. The first real-video mini overfit through
  `FeatureToColor` decreases loss on `test_data/test_video_small_128_4fps.mp4`
  from `0.18671` to `0.04197` in 20 steps at 8f/64px/512t/F32.
- The same path now launches through the first-class trainer dispatch as
  `arch=star_uvt_feature_overfit`. The checked-in smoke config is
  `src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc`.
  The current chunked first-class artifact improves loss `0.18602 -> 0.04167`
  with `frame_chunk_size=2` and zero overflow.
- First 64-frame first-class scale probes pass: `64f/256px/8192t/F32/chunk4`
  runs at `0.965s/step` with `0.736s` in backward, and
  `64f/512px/2048t/F32/chunk2` runs at `4.021s/step` with `3.070s` in
  backward. These are bottleneck probes, not quality baselines.
- 512px scale now reaches 4096 and 8192 tubes with zero overflow under
  `feature_direct_gradcache`: `4096t/chunk2` passes at `6.456s/step`
  (`4.209s` backward, max tile `18`, p95 `9`) and `8192t/chunk2` passes at
  `7.937s/step` (`4.883s` backward, max tile `33`, p95 `17`). This proves
  support headroom, not practical speed.
- `firstclass_backward_breakdown.py` now splits the real first-class graph into
  render forward, `FeatureToColor`/loss forward, `FeatureToColor`/loss
  backward to image gradients, and Metal renderer backward. The new split shows
  the 512px bottleneck is mostly image-space colorizer/loss backward: on
  `4096t/8192t` gradcache rows, renderer backward is only `22.1%/16.9%` of
  backward, while colorizer/loss backward is `77.9%/83.1%`. On the
  256px/32768t/cap256 row, renderer backward is about `36%` of backward.
- A no-pre-norm colorizer A/B is the first fast whole-graph lever at 512px:
  with `colorize.pre_norm=false`, the 512px/4096t and 8192t breakdown rows drop
  colorizer/loss backward to `317.1ms` and `400.6ms`. The actual
  512px/8192t/chunk2 trainer passes the 2-step gate at `3.715s/step` with
  `1.586s` backward and zero overflow, versus the default pre-norm row at
  `7.937s/step` and `4.883s` backward. The 20-step media A/B keeps the speed
  win (`7.366s/step`, `3.370s` backward versus `11.098s/step`, `7.070s`
  backward), but default pre-norm ends slightly better (`0.31742` loss /
  `4.984` PSNR versus `0.32053` / `4.941`), so no-pre-norm is a speed
  candidate, not a quality promotion.
- The native-prep handoff gate removes most of the Torch-side linear
  sigmoid-MSE prep tax around `logit_handoff_reduce_vec4`: at
  `64f/512px/8192t/F32`, prep drops `413.64 -> 37.29ms`, prep+backward drops
  `826.35 -> 428.98ms`, and total drops `1446.53 -> 1108.50ms` with F4/F32
  parity and zero overflow. This is a benchmark-only linear colorizer result;
  the hidden frozen-probe trainer loss still needs a native prep/tape path.
- The hidden sigmoid-MSE native gate covers that next shape mechanically:
  H32/H64 hidden `FeatureToColor` RGB/loss VJP is fused into Metal and passes
  F4/F32 STAR-gradient parity with zero overflow. It is not the speed answer.
  H32 scalar totals `317.54/610.90/2549.39ms` at `128/256/512px`, H64 at 256px
  totals `817.27ms`, and vec4 reduce is slower than scalar. The next native
  route should avoid dense `[T,H,W,F]` support or use visibility/prefix tape.
- The sparse hidden sigmoid-MSE native gate is the first positive port of that
  lesson. It reuses cached sparse bins and fuses hidden RGB/loss VJP only over
  selected pixels. At `64f/512px/8192t/F32`, H32 sparse64 total drops
  `29.66 -> 18.47ms`, H32 sparse128 drops `111.17 -> 64.17ms`, and H64 sparse64
  drops `45.09 -> 28.40ms`, all with parity and zero overflow. The fused native
  backward is heavier than sparse backward alone, but it removes the larger
  Torch hidden-VJP prep cost. Compare it against the selected sparse-forward
  batched target/probe VJP route before trainer promotion. The first trainer
  integration does that for pixel64 support and is effectively neutral: warm
  manual hidden64 star-only sparse loss+backward is `113.25ms`, native is
	  `116.27ms`, and final sparse loss matches within `3.26e-08`. The follow-up
	  native target-area gate now covers the expensive full-support basis: matched
	  star-only full-cell8 trainer time drops `5801.7 -> 3496.0ms/step`, and
	  512px native-only synthetic support survives where the Torch hidden-VJP
	  baseline OOMs. It is a speed/memory promotion, not a quality promotion,
	  because dense full RGB remains `5.648`. The hidden32 native follow-up cuts
	  mean step further to `2464.6ms`, but fails the trainer gate with probe PSNR
	  `19.481` and full RGB `5.632`, so decoder shrinkage is rejected. The
	  skip-feature-grad diagnostic shows raw feature-gradient atomics are only a
	  small hidden64 native target-area slice (`594.9 -> 562.2ms` backward at
	  256px, `1918.6 -> 1854.3ms` at 512px), so the next speed gate should reduce
	  hidden64 recompute or change support/objective rather than target feature
	  atomics alone. The exact row-major W^T follow-up preserves parity but is not
	  the trainable speed path: full native backward slows to `711.5/2161.6ms` at
	  256/512px versus canonical `647.4/2040.5ms`; recompute-only improves only
	  slightly to `555.8/1983.4ms` versus `572.1/1993.0ms`. The vec4 W^T
	  follow-up is the first positive exact W^T kernel reduction: same-build full
	  backward improves to `642.2/1804.7ms` versus canonical `675.9/2408.1ms`
	  at 256/512px, and recompute-only improves to `518.3/1635.8ms` versus
	  `586.6/2305.2ms`. The current-build trainer A/B now proves whole-step
	  speed too: canonical vs vec4 mean step is `4262.1 -> 4071.0ms`, mean
	  backward `3700.2 -> 3152.6ms`, and mean sparse visual backward
	  `2546.7 -> 1963.5ms`, so vec4 W^T is the preferred full-support native
	  target-area star-only mode. The 50-step promoted-mode gate passes and
	  warms to `3359.2ms` mean / `3072.1ms` last step with full RGB `5.732` and
	  zero overflow, but the compact target-area64 helper route is still faster
	  and better on dense RGB on the fresh current-build gate (`930.6ms`,
	  `6.023`), so full-cell8 native target-area remains a full-support speed
	  baseline, not the visual answer. Compact native star-only is also rejected
	  (`2265.0ms` mean step, no colorizer gradients); the useful native port must
	  preserve colorizer gradients and beat compact autograd. Compact manual
	  hidden64 keeps colorizer gradients, but still fails: first-five timing is
	  `2007.4ms` mean / `1899.2ms` no-first, feature/probe quality regresses, and
	  compact autograd remains ahead at `991.9ms` mean / `787.7ms` no-first.
	  Native colorizer-gradient vec4 W^T passes tiny parity for STAR and
	  colorizer parameter gradients, but the trainer gate rejects it too:
	  `2738.7ms` mean step, `1474.2ms` backward, same feature/probe regression.
	  The colorizer-gradient-only split explains why: direct compact native
	  backward is `88.9ms` star-only, `536.6ms` colorizer-grad-only, and
	  `531.4ms` full colorizer, so the compact-native blocker is per-pixel
	  colorizer parameter atomics rather than STAR feature/geometry gradients.
	  A Torch/MPS sidecar reducer is correct, but it still loses to the existing
	  sparse-pixel baseline (`390.9ms` versus `276.6ms`) because it duplicates
	  sparse render and hidden replay before the native star-only backward. The
	  same-pass SIMD-reduce follow-up fixes that direct-kernel atomic shape:
	  compact native colorizer total/backward becomes `297.2/239.2ms` in the
	  matched run versus sparse-pixel `312.1/31.6ms`. The 5-step trainer still
	  rejects it (`2908.9ms` mean step, `604.0ms` sparse visual backward, same
	  feature/probe regression), so compact autograd remains the practical visual
	  route until native removes whole-graph overhead or changes support/objective.
	  A matched dynamic-gsplat fixed-512 smoke at `64f/512px/8192` active
	  Gaussians is slower (`8.019s` step, `5.638s` backward; raster `0.362s`),
	  so the next STAR UVT work should stay on visual quality/scale, not switch
	  the fast route to dynamic gsplat. The selected visual-quality gate then
	  fails scale-up explicitly: dense full RGB is `6.023` PSNR, the media stays
	  sparse/streaked or blurry, and RGB STAR is `12.444` PSNR on the same-clip
	  bracket. The follow-up trainable RGB-grid low-frequency bridge proves the
	  actual colorizer can train through the fast target-grid sparse VJP path
	  (`353.1ms` mean step, `289.9ms` no-first), but rejects low-frequency grid
	  RGB alone as the visual fix: dense full RGB falls to `5.657` PSNR while
	  feature loss worsens to `0.630230`. Combining RGB-grid40 with compact
	  target-area support is rejected too: it improves grid/probe/sparse metrics
	  but slows to `1.648s` mean step, worsens feature loss to `0.630296`, and
	  lands at only `5.720` dense full RGB PSNR. The dense alpha diagnostic then
	  localizes the failure: forced alpha raises the rejected routes to
	  `11.450-14.616` PSNR and target-background oracle composition reaches
	  `20.149-25.562`, while alpha `>0.1` covers only `41.5-43.5%` of pixels.
	  The direct sampled alpha-to-one follow-up is also rejected: sampled alpha
	  loss improves `0.752440 -> 0.738210`, but dense RGB stays `6.018`, dense
	  alpha `>0.1` stays `43.1%`, and feature/probe losses regress. The
	  phase-covered alpha retry is also rejected: sampled alpha loss improves
	  `0.751768 -> 0.739891`, but dense RGB falls to `6.014`, dense alpha
	  `>0.1` falls to `43.0%`, and feature/probe losses regress. The target-aware
	  black-hole retry is rejected too: black-hole loss improves `0.262537 ->
	  0.256889`, but dense RGB stays `6.014`, dense alpha `>0.1` stays `43.0%`,
	  and feature/probe losses regress. Target-background composition is useful
	  but rejected: forced-alpha PSNR rises to `14.891-14.899` and oracle
	  composition reaches `27.105-27.443`, but dense RGB is only `5.666-5.748`
	  and alpha `>0.1` remains `40.8-43.1%`. The alpha-sweep/patch4 follow-up
	  is rejected too: `16x` posthoc alpha gain only reaches `8.337-8.592`
	  PSNR, and `4x4` support plus target-background alpha pressure lands at
	  `5.698` dense RGB while regressing feature/probe. Raw-opacity bias is
	  negative too: best logit bias `+4` only reaches `6.194/5.926/5.871`
	  PSNR for compact/targetbg-alpha/patch4. Dense alpha-only support is also
	  negative: the new `dense_alpha` trainer path costs `2.559s/step`, regresses
	  weighted/feature/probe losses, lands at `5.647` dense RGB, and lowers
	  alpha `>0.1` to `40.7%`. The alpha-only visibility profile then proves a
	  cheaper diagnostic implementation: `render_uvt_feature_alpha_all_pixels_with_bins`
	  uses sparse-pixel F1 alpha rendering plus cached F1 backward, matches dense
	  alpha exactly, matches gradients within `4.7e-7`, and cuts the all-chunk
	  alpha render+backward envelope `1100.8 -> 634.6ms`. This is not a quality
	  promotion. The trainer opt-in `dense_alpha.render_mode=sparse_f1` reproduces
	  the same negative endpoint but cuts mean step/backward
	  `2558.6/1114.2 -> 873.3/370.0ms` and dense-alpha render/loss/backward
	  `834.5/124.6/858.9 -> 276.0/22.0/303.7ms`. The next visual gate should
	  change dense visibility support/composition, not just add same-support
	  alpha pressure, opacity scheduling, dense alpha, or another sparse support
	  schedule. The first CPU support-changing proxy gate now proves the
	  gradient mechanism before a trainer port: from zero target hits,
	  same-support dense alpha remains at `0.0` target alpha `>0.10` coverage,
	  while a soft projected-tube coverage proxy sends center/velocity gradients
	  and reaches `0.324`. The first trainer port is now wired behind the
	  `visibility_proxy` config block and passes as a mechanics gate from sparse
	  step 1500: weighted loss `0.871986 -> 0.871864`, feature loss
	  `0.625418 -> 0.625379`, RGB-probe PSNR `22.0277 -> 22.0291`, and
	  center/velocity gradients are seen. It is still not a quality promotion:
	  dense full RGB PSNR is `5.640` and proxy work costs `237ms`/step on
	  average. The dense-support follow-up rejects this center-only proxy as a
	  scale bridge: forced-alpha/oracle content improves (`11.722 -> 14.552`
	  forced-alpha PSNR, `20.140 -> 25.834` target-background oracle), but alpha
	  `>0.1` falls `41.1% -> 40.5%`. A 10x/20-step retry fails trainer loss and
	  only reaches `40.6%` alpha `>0.1`, so the next support experiment needs an
	  explicit opacity/support term or support-density change. That
	  opacity/precision support-aware trainer port now exists, but it is not the
	  scale bridge either: it lowers support proxy loss `3.4303 -> 3.3821` and
	  sends raw opacity/precision gradients, while feature loss slightly worsens,
	  proxy work costs `693.7ms`/step, and dense support barely changes versus
	  center-only (`5.640 -> 5.643` normal PSNR, `40.5% -> 40.6%` alpha
	  `>0.1`). Treat
	  `outputs/benchmarks/2026-05-20_star_uvt_visibility_support_proxy_gate.md`
	  as plumbing proof; next work needs cheaper/fused support density,
	  opacity/support parameterization, or support birth/split. The first
	  fixed-budget birth/split CPU gate is positive:
	  `outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.md`.
	  With `16` tubes and `8` reallocated dead/miss tubes, same-support alpha
	  stays at `0.0000` target alpha `>0.10`, the center proxy reaches `0.5784`,
	  and birth/split reaches `1.0000`; refinement keeps `1.0000` while reducing
	  background alpha `0.0479 -> 0.0072`. The trainer-port mechanism now exists
	  behind `support_birth_split.enabled` and passes a 512px/64f sparse-1500
	  gate:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_trainer_gate.md`.
	  It reallocates `32/8192` low-opacity tubes, preserves the fixed budget,
	  keeps zero overflow (`100/71/128` max/p95/cap), passes 5 steps at
	  `189.4ms` mean / `138.3ms` last, and reaches dense RGB PSNR `5.708`.
	  This is a real trainer primitive, not a Metal quality claim. The dense
	  support diagnostic keeps that read:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_dense_support_diagnostic.md`.
	  Birth32 improves normal/forced-alpha/high-alpha support versus center and
	  support proxies (`5.708`, `14.606`, alpha `>0.5` `0.117`), but alpha
	  `>0.1` is only `0.411` and target-background oracle falls to `25.234`.
	  The uncovered-brightness sampler follow-up now passes too:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_trainer_gate.md`
	  and
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_dense_support_diagnostic.md`.
	  It selects genuinely low-alpha bright targets (`selected_alpha_mean=0.0209`)
	  and reaches dense RGB PSNR `5.713` at `187.4ms` mean step, but alpha
	  `>0.1` remains `0.411`. The first sweep now lives at
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row.md`,
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row_cap256.md`,
	  and
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_n32_radius_cap128.md`.
	  Cap `128` rejects `64+` births by overflow, cap `256` clears them, and
	  radius `96px` is the coverage lever. Best safe cap-128 row is
	  `low_alpha_n32_r96_cap128`: alpha `>0.1` `0.420`, normal PSNR `5.825`,
	  forced-alpha PSNR `14.591`, oracle `24.226`, max tile `100/128`. Next
	  test intermediate radius, not another same-support penalty. The
	  intermediate-radius follow-up is now complete:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128.md`.
	  It shows uncovered `r64/r72/r80/r88` raises alpha `>0.1`
	  `0.411/0.413/0.415/0.417` while oracle falls
	  `25.319/25.187/25.015/24.802`; low-alpha `r80/r88` fails loss decrease.
	  The born-opacity sweep is also complete:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r80_cap128.md`
	  and
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r88_cap128.md`.
	  It only moves along the same tradeoff curve: lower opacity recovers oracle
	  but gives back coverage, higher opacity does the reverse. Next change
	  support shape, not just radius or scalar opacity. The first support-shape
	  gate is now complete:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128.md`
	  and
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128_dense_support.md`.
	  It adds `trajectory_ellipse` birth support, but the 8-row cap-128 gate is
	  a clean negative: alpha `>0.1` stays `0.408-0.409`, below the prior
	  isotropic `0.411`, despite zero overflow. Next try multi-center or
	  stratified birth/split instead of expanding one global ellipse. The first
	  multi-center gate is now complete:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128.md`
	  and
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128_dense_support.md`.
	  `farthest_xy` with `K=8`, `32` births, `r64`, and cap `128` is the first
	  real coverage move, reaching alpha `>0.1` `0.4309` with zero overflow and
	  forced-alpha PSNR `14.608`; oracle falls to `23.965`, so next sweep
	  multi-center radius/opacity. That sweep is now complete:
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128.md`
	  and
	  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128_dense_support.md`.
	  Best coverage is `r72/o0.8` alpha `>0.1` `0.4318` with oracle `23.670`;
	  the selected balanced row is `r64/o0.4`, alpha `>0.1` `0.4298`,
	  forced-alpha `14.620`, oracle `24.805`, zero overflow, and
	  `167.9/58.1ms` step/backward. Next run a 20-step media gate for
	  `K=8/r64/o0.4`. That gate is now complete with checked-in config
	  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step_media.jsonc`.
	  It passes with loss `0.903197 -> 0.897231`, probe PSNR
	  `21.681 -> 21.769`, zero overflow, last step/backward `147.3/54.3ms`,
	  dense alpha `>0.1` `0.431158`, forced-alpha `14.631`, and oracle
	  `24.851`. The matched `K=8/r72/o0.4` 20-step sibling also passes with
	  zero overflow, loss `0.910099 -> 0.903088`, probe PSNR
	  `21.601 -> 21.703`, full RGB PSNR `5.820`, mean step/backward
	  `157.9/61.1ms`, dense alpha `>0.1` `0.432454`, and oracle `24.668`.
	  That is a tiny coverage/normal-PSNR win with worse feature/probe loss and
	  worse oracle, so `K=8/r64/o0.4` remains the balanced default. It is a
	  positive support gate but not final visual quality.
- A same-session no-pre-norm 512px/8192t rerun selected
  `feature_direct_gradcache_reduce_vec4` as the current fastest feature-tube
  diagnostic: the 20-step media row passes with identical loss/PSNR to
  gradcache (`0.32053` / `4.941`) and improves mean timing from
  `2.858s/step`, `1.327s` backward to `2.491s/step`, `1.184s` backward. This
  older RGB-target route is now available as
  `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
  star-feature-512-rgbfast`.
- The selected-shader scale gate at 128/256/512 shows that vec4 reduce is a
  high-resolution speed choice, not a universal default. It ties or slightly
  loses backward at 128px, gives only a small 256px win, and gives the clear
  512px win above. It also shows lower resolution can be harder for fixedbin
  validity: 128px/8192t needed cap256 plus `alpha>=1/32` after cap128/default
  alpha, cap256/default alpha, and cap256/`alpha>=1/72` all overflowed.
- The precomputed V-JEPA bridge audit originally found that the old
  `star-feature-512-fast` did not use cached V-JEPA features. That has been
  superseded: `star-feature-512-fast` now points at the 100-step sparse-forward
  batched VJP target-grid config, while `star-feature-512-rgbfast` preserves the
  older RGB-target speed diagnostic. The generic cached-target adapter plus real
  V-JEPA smoke and 512px scale configs remain the bridge lineage.
- The first bridge smoke now passes. Config
  `src/train_configs/star_uvt_feature_testvideo_8f_64_rgbpyramid_target_gradcache_reduce_vec4_10step.jsonc`
  loads cached `rgb_x1`, adapts `[1,3,8,64,64]` to `[8,32,64,64]`, disables RGB
  loss, and trains directly against `render.feature_image`. The cache-hit rerun
  passes with loss `0.34006 -> 0.24809`, mean `93.5ms/step`, `43.0ms`
  backward, zero overflow, and model-gradient flow present. This is a contract
  smoke only; real V-JEPA target quality is still unmeasured. The adapter also
  accepts token-shaped cached features with explicit
  `feature_target.token_grid_shape=[T,H,W]`; the real V-JEPA smoke below uses
  that path.
- The real V-JEPA target smoke now passes. Config
  `src/train_configs/star_uvt_feature_testvideo_8f_64_vjepa_target_gradcache_reduce_vec4_10step.jsonc`
  loads cached `vjepa_tokens` `[1,1024,768]`, uses explicit
  `token_grid_shape=[4,16,16]`, truncates to F32, interpolates to
  `[8,32,64,64]`, standardizes channels, and trains against
  `render.feature_image`. The cache-hit row passes with loss
  `1.00082 -> 0.90042`, mean `181.1ms/step`, `53.8ms` backward, and zero
  overflow. This proves the real cached V-JEPA target bridge at smoke scale;
  it is not the selected 512px scale/quality gate.
- The real V-JEPA target 512px scale gate now passes with chunked target
  materialization. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_chunkedtarget_lr005_5step.jsonc`
  loads cached `vjepa_tokens` `[1,8192,768]`, uses
  `token_grid_shape=[32,16,16]`, truncates to F32 before grid upsampling, and
  streams `[2,32,512,512]` target chunks from a `[32,32,16,16]` channel-adapted
  source instead of keeping the full `[64,32,512,512]` target resident. The
  cache-hit row passes with loss `1.000014 -> 0.999545`, mean `3.743s/step`,
  `1.077s` backward, `1.734s` target chunk/loss, and zero overflow. The
  channel-before-grid adapter fixed an initial 48 GiB interpolation temporary.
  This is a scale/trainability gate, not a quality baseline.
- The cached-target-layout follow-up now passes for that same 512px V-JEPA
  target. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_cachedchunks_lr005_5step.jsonc`
  uses `feature_target.materialization=cached_chunks`, precomputes the adapted
  target into 32 resident chunks (`2048MiB`, `2.044s` load/prep), and cuts the
  cache-hit row to `1.655s/step`, `0.770s` backward, `0.601s` render, and
  `0.202s` target/loss, with the same loss curve and zero overflow. This is the
  exact dense render-grid-loss reference; the next risk is memory ceiling.
- The target-grid V-JEPA loss follow-up now passes and is the current
  speed/memory diagnostic. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_5step.jsonc`
  uses `feature_target.materialization=target_grid`, keeps only the
  channel-adapted `[32,32,16,16]` V-JEPA target grid resident (`1.0MiB`), and
  downsamples rendered feature chunks before loss. The row passes with loss
  `0.999935 -> 0.999467`, mean `1.351s/step`, `0.705s` backward, `0.548s`
  render, `0.041s` target/loss, `0.138s` target load/prep, and zero overflow.
  It avoids the 2GiB dense target cache but changes the objective from dense
  render-grid MSE to coarse token-grid MSE. The 20-step media follow-up also
  passes with feature loss `0.999935 -> 0.997425`, mean `1.451s/step`,
  `0.722s` backward, `0.630s` render, and `0.037s` target/loss. It writes media,
  but RGB PSNR/media are not quality evidence because `rgb_loss_weight=0` and
  the colorizer is not trained.
- The RGB-aux1 target-grid probe now passes as the first visual-control row.
  Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.jsonc`
  uses `rgb_loss_weight=1.0` and logs component losses. It decreases feature
  loss `0.999935 -> 0.997336` and RGB loss `0.338171 -> 0.335263`, with
  colorizer gradients present, but RGB PSNR only moves `4.709 -> 4.746` in
  20 steps. Mean timing is `2.000s/step`, `1.114s` backward, `0.586s` render,
  and `0.052s` target/loss. This is a useful control, not enough quality
  improvement to promote target-grid visual training.
- The RGB-aux10 target-grid probe is a weak negative control for pure weighting.
  Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.jsonc`
  improves RGB loss slightly more (`0.338171 -> 0.334961`, PSNR
  `4.709 -> 4.750`) but ends with slightly worse feature loss than aux1
  (`0.997547` vs `0.997336`) and still costs about `2.0s/step`. Larger RGB
  weight alone is not the quality lever.
- The 100-step RGB-aux10 schedule probe now passes. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.jsonc`
  reaches feature loss `0.964670` and RGB PSNR `5.109` from the same
  `4.709` RGB start, with mean timing `1.876s/step`, `1.033s` backward,
  `0.580s` render, and `0.043s` target/loss. This says schedule length matters;
  it still does not close the large gap to RGB STAR quality.
- The matched RGB-warm20 schedule probe is a negative visual-control gate.
  Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.jsonc`
  uses the new `feature_target.weight_schedule` support: 20 steps of
  `feature=0/rgb=20`, then 80 steps of `feature=1/rgb=10`. It passes and is
  cheaper (`1.639s/step`, `0.872s` backward), but ends worse than constant
  aux10 on both RGB PSNR (`5.046` vs `5.109`) and feature loss (`0.973557` vs
  `0.964670`).
- The standalone target-grid feature-to-RGB probe now passes and resolves the
  decodability question. Config
  `src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.jsonc`
  trains a hidden64 `FeatureToColor` directly on the cached `[32,32,16,16]`
  V-JEPA target grid, with downsampled RGB supervision at `[32,3,16,16]`. It
  reaches grid PSNR `23.401` and full-video upsampled PSNR `20.073` at
  `2.427ms/step`, with `1.003ms` backward and checkpoint
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`.
  This says the target-grid features are visually decodable; the next bridge is
  to load/freeze this decoder in STAR training or canonical probe logging.
- The frozen RGB-probe STAR integration gate now passes. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.jsonc`
  loads the hidden64 checkpoint above, freezes it, and adds
  `rgb_probe_loss_weight=10.0` against the downsampled RGB target grid. It
  passes with zero overflow at `1.220s/step`, `0.572s` backward, feature loss
  `0.999935 -> 0.998357`, and frozen-probe PSNR `13.985 -> 14.060`. This is
  the plumbing/speed proof, not a visual-quality promotion.
- The 100-step frozen RGB-probe STAR follow-up passes and was the first stronger
  visual diagnostic for the target-grid route. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.jsonc`
  uses the same frozen checkpoint and reaches feature loss
  `0.999935 -> 0.970035`, frozen-probe loss `0.039944 -> 0.034350`, and
  probe PSNR `13.985 -> 14.641` at `1.268s/step`, `0.630s` backward. It is
  cheaper than the 100-step RGB-aux10 row and moves probe PSNR more, but still
  trails the standalone `20.073` PSNR feature-to-RGB oracle.
- The 300-step frozen RGB-probe extension keeps closing the same gap. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.jsonc`
  reaches feature loss `0.999935 -> 0.811652`, frozen-probe loss
  `0.039944 -> 0.022079`, and probe PSNR `13.985 -> 16.560` at
  `1.355s/step`, `0.681s` backward. This is the current keeper diagnostic for
  target-grid visual training, but it is still not oracle quality.
- Longer feature-overfit gates can now use opt-in checkpoint/resume. The trainer
  writes `output.checkpoint` with model/colorizer/optimizer/config/row/losses and
  loads `train.resume_checkpoint` as warm-start local steps; optimizer resume is
  enabled by default. The real 8f/64px RGB-pyramid smoke wrote and resumed
  `/tmp/star_uvt_checkpoint_resume_smoke/*.pt` with zero overflow.
- The same checkpoint/resume path now passes at the real frozen-probe scale. The
  checked 300-step checkpoint/no-media config matches the keeper curve at
  `1.268s/step`, feature loss `0.811652`, and probe PSNR `16.560`, then the
  resumed 300-step media config loads that checkpoint with optimizer state and
  reaches feature loss `0.810827 -> 0.655366`, probe PSNR `16.576 -> 19.884`,
  zero overflow, and `1.440s/step`. This nearly reaches the standalone
  full-video upsample number (`20.073`). The follow-up probe-emphasis config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.jsonc`
  resumes that 600-step checkpoint with `train.global_step_offset=600`,
  `feature_target.loss_weight=0.25`, and `rgb_probe_loss_weight=40.0`. It
  passes 200 more local steps with probe PSNR `19.888 -> 21.425`, zero overflow,
  and `1.512s/step`, but feature loss drifts `0.655132 -> 0.703820`. This
  passes the standalone full-video upsample number while still trailing the
  same-grid oracle (`23.401`). The scheduled balance follow-up resumes the
  800-step checkpoint for global steps 800-1000, with `feature=1/probe=10`
  then `feature=0.5/probe=20`; it recovers feature loss
  `0.703862 -> 0.643852` at `1.308s/step`, but gives back a little probe PSNR
  (`21.428 -> 21.382`) and is nonpassing on probe-loss decrease. That makes the
  next gate objective balance rather than decodability. The feature0.5/probe40
  1000->1100 Pareto follow-up passes the combined gate at `1.461s/step`, moves
  probe PSNR `21.384 -> 21.789`, and keeps zero overflow, but feature loss
  drifts `0.643823 -> 0.656728`. The 1100->1200 recover schedule is
  nonpassing but useful: feature loss recovers `0.656765 -> 0.635093` at
  `1.521s/step`, while probe PSNR gives back a little (`21.792 -> 21.738`).
  The short feature0.75/probe40 1200->1250 follow-up passes and restores probe
  PSNR `21.740 -> 21.929` at `1.523s/step`, but feature loss rises
  `0.635066 -> 0.638799`. The feature1/probe40 1250->1300 follow-up is the
  first current both-improving balance row: feature loss `0.638803 -> 0.632192`,
  probe PSNR `21.933 -> 21.963`, zero overflow, and `1.285s/step`. The
	  1300->1400 extension keeps both improving to feature loss `0.627129` and
	  probe PSNR `21.979`, but slows to `1.690s/step` on the older dense path; a
	  matched timing repeat is `1.711s/step` with the same zero-overflow
	  `68/45/128` max/p95/cap tile count. The sparse-forward batched-VJP helper row
	  preserves the same 100-step movement at `399.9ms/step` mean and
	  `262.9ms/step` last-20, with valid but blurry probe media. The
	  effective-lr001 sparse-forward rerun preserves the dense lr001 endpoint at
	  `372.3ms/step` mean, `158.9ms` backward, feature loss `0.630549`, and
	  probe PSNR `22.034`, but gives up lr005's better feature loss and has noisy
	  late timing.
  The whole-graph profile for this objective shows renderer backward is
  `81.3-81.4%` of manual backward, but the isolated split does not reproduce
  the trainer slowdown (`1565.9ms` manual total at step 1250 versus `1504.0ms`
  at step 1300). The end-to-end trainer trace adds per-step timing samples and
  does reproduce the slowdown after dropping the first optimizer/warmup step:
  `1850.7ms` from step 1300 versus `1705.3ms` from step 1250, with a late
  objective spike at global step `1318`. The chunk trace around `1317-1319`
  shows the spike is distributed (`27/32` chunks worsen, `44.5%` of weighted
  delta in frames `0-15`) and persists into `1319`. The
  continuation-chain/profile/trace reports are
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md` and
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`
  and
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`
  and
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`.
  The follow-up LR checkpoint gate confirms the spike is schedule-state
  sensitive. After the trainer was fixed to re-apply config LR after loading
  optimizer state, the retained-optimizer `lr=0.001` row records loaded/effective
  LRs `[0.005] -> [0.001]`, removes the 1318 spike, and passes with end loss
  `0.884576`, feature loss `0.631648`, probe PSNR `21.991`, no-first
  `1384.4ms/step`, and `748.9ms` backward. The reset-optimizer `lr=0.001`
  control also passes (`0.884902`, `0.631614`, `21.984`) but is slower here.
  Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`.
  The 100-step effective-lr001 continuation from 1300 also passes with
  media/checkpoint: feature loss `0.632124 -> 0.630549`, probe PSNR
  `21.965 -> 22.034`, mean `1463.8ms/step`, and `778.4ms` backward. It avoids
  the early 1318 jump but later has a smaller transient at `1377->1378`; versus
  the older lr005 1300->1400 row it is faster and better on probe PSNR, but
  worse on final feature loss (`0.630549` vs `0.627129`) and slightly worse
  weighted loss (`0.880942` vs `0.880751`). Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`.
  The matched effective-lr001 sparse-forward rerun preserves that dense lr001
  endpoint at `372.3ms/step` mean and `158.9ms` backward, but it keeps the same
  quality tradeoff and noisy late timing. Report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.md`.
  The 1400 checkpoint-selection gate then selects the lr005-sparse state for
  further quality work: it passes 50 effective-lr001 steps to feature loss
  `0.625976` and probe PSNR `22.010`, while lr001-sparse 1400 fails after a
  `1444 -> 1445` jump and ends at feature loss `0.631770` / probe PSNR
  `21.843`. Report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_checkpoint_select_1400_to1450.md`.
  The selected lr005-sparse 1450->1500 media gate then passes and writes the
  next checkpoint/media: loss `0.877762 -> 0.876224`, feature loss
  `0.625962 -> 0.625428`, probe PSNR `22.010 -> 22.027`, mean
  `315.8ms/step`, `130.2ms` backward, last-20 `254.0ms/step` /
	  `108.2ms` backward, zero overflow. The contact sheet is still blurry, so this
	  is a stability/continuation pass rather than a quality promotion. Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr005sparse_1450_to1500_media.md`.
	  A full-resolution autograd RGB-aux probe-init bridge from that sparse 1500
	  checkpoint is a negative quality result: it loads the STAR model, skips the
	  checkpoint colorizer/optimizer, initializes the trainable hidden64 colorizer
	  from the trained target-grid RGB probe, and runs 20 RGB-aux steps. RGB loss
	  improves `0.272626 -> 0.259968`, but feature loss worsens
	  `0.625418 -> 0.626799`, frozen-probe PSNR drops `22.028 -> 21.879`,
	  trainable-colorizer media artifacts appear, and mean step time is `5.207s`
	  (`16.5x` slower than sparse 1500). Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_autograd_rgbaux_probeinit_from1500_negative.md`.
	  The rendered-feature sparse-pixel RGB probe from the same sparse 1500
	  checkpoint then trains on actual rendered feature pixels and passes its
	  sampled loss gate at `241.4ms/step` (`0.168261 -> 0.099014` sparse loss,
		  `7.740 -> 10.043` sparse PSNR), but dense full-video PSNR is only `6.096`
		  and the media remains sparse-streaked. This rules out the simple
		  target-grid-vs-rendered-distribution explanation as the whole fix. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels.md`.
		  The denser stratified64 follow-up samples `262,144` full-resolution
		  pixels/step (`4x` the previous rendered-feature probe) and still reaches
		  only `6.132` dense full-video PSNR at `331.5ms/step`, so target-grid
		  sampling bias is not the explanation. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64.md`.
		  The first native sparse visual VJP gate updates STAR parameters from sparse
		  RGB loss (`model_grad_seen=true`, frozen colorizer) at `336.8ms/step`, but
		  is quality-negative at `5.739` full-video PSNR. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe.md`.
		  The joint sparse visual VJP follow-up trains STAR and the hidden64
		  colorizer together (`model_grad_seen=true`, `colorizer_grad_seen=true`)
			  and raises full-video PSNR to `6.025`, but it still trails the
			  colorizer-only stratified diagnostic (`6.132`) and costs
			  `729.4ms/step`. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe.md`.
			  The mixed target-grid/probe plus sparse visual VJP follow-up preserves
			  feature/probe movement and raises sparse visual sample PSNR to `6.036`,
			  but dense full-video PSNR remains `6.024` while step time rises to
			  `964.0ms`; this is a mechanics pass, not a quality promotion. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500.md`.
			  The patch2x2 same-pixel support follow-up is faster (`619.5ms/step`)
			  and raises sparse visual sample PSNR to `6.179`, but feature-target
			  loss worsens and dense full-video PSNR drops to `6.000`; this rejects
			  contiguous sparse patch support as the visual fix. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500.md`.
			  The patch-mean64 visual-basis follow-up samples `1,048,576` sparse
			  visual pixels/step and pools them into `262,144` local-mean cells. It
			  passes and restores feature/probe movement with dense full-video PSNR
			  `6.023`, but costs `1124.6ms/step` and still has sparse/high-frequency
			  media. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patchmean64_from1500.md`.
			  The target-area64 follow-up keeps the same support but compares
			  against true area-downsampled RGB target cells. It is slightly faster
			  (`1103.1ms/step`) and raises sparse visual PSNR to `6.064`, but dense
			  full-video PSNR remains `6.023` and media is unchanged. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500.md`.
			  The phased target-area64 follow-up cycles the same compact `2x2`
			  support across a `4x4` subcell schedule. It passes and raises sparse
			  visual PSNR to `6.077`, but dense full-video PSNR falls to `6.019`
			  at `1169.2ms/step`. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500.md`.
			  The full-cell8 target-area follow-up sends gradients through every
			  pixel in each `8x8` target-area cell (`16,777,216` visual pixels/step
			  into `262,144` loss cells). It is nonpassing: feature loss and probe
			  PSNR worsen, dense full-video PSNR falls to `5.722`, and mean step is
			  `7526.7ms` with `5702.6ms` in sparse visual loss construction. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500.md`.
			  The manual hidden64 VJP version matches that endpoint while cutting
			  sparse visual loss construction to `3803.6ms` and mean step to
			  `6414.0ms`, but it remains nonpassing and quality-negative. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500.md`.
			  The star-only manual hidden64 variant skips colorizer parameter
			  gradients and cuts mean step to `5801.7ms`, but dense full-video
			  PSNR drops to `5.648`, so it is only a lower-bound diagnostic.
			  Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500.md`.
			  The fast-GELU derivative variant keeps colorizer gradients but is
			  rejected: mean step is `6252.1ms`, dense RGB stays `5.722`, and the
			  profile loss-side total is worse than exact manual. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_fastgelu_from1500.md`.
			  The compact linear colorizer/manual VJP variant cuts the full-cell8
			  row to `2064.4ms/step` and sparse visual loss construction to
			  `383.3ms`, but its standalone linear probe reaches only `16.980`
			  full-video PSNR and the trainer row remains visually poor at `5.668`
			  dense RGB. It is a lower-complexity mechanics diagnostic, not a
			  quality route. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500.md`.
			  The hidden32 manual VJP follow-up adds a smaller hidden decoder under
			  the generic `manual_hidden` VJP mode. The standalone probe keeps most
			  hidden64 visual capacity (`19.704` full PSNR vs `20.073`), but the
				  trainer still costs `4298.4ms/step` with `2136.1ms` sparse visual
				  loss construction and dense RGB remains only `5.678`; do not promote
				  it beyond Pareto evidence. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500.md`.
				  The native target-area hidden64 follow-up bins once, computes native
				  cell RGB sums, and runs native STAR backward from target-area cell
				  gradients. It passes tiny parity, wins synthetic full-support timing
				  at 128/256px, survives 512px native-only where the Torch hidden-VJP
				  baseline OOMs, and cuts the matched star-only trainer row
				  `5801.7 -> 3496.0ms/step` while preserving the same `5.648` dense
				  RGB endpoint. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_native_gate.md`.
				  The native target-area hidden32 follow-up uses the generic
				  `native_hidden_target_area_star_only` alias and reduces native
				  recompute (`2464.6ms/step`, `1321.7ms` sparse visual backward), but
				  fails quality/pass gates (`19.481` probe PSNR, `5.632` dense RGB).
				  Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden32_gate.md`.
				  The skip-feature-grad diagnostic intentionally zeros raw feature grads
				  while keeping geometry/opacity parity; it only saves `3-6%` of
				  hidden64 native target-area backward, so raw feature atomics are not
				  the main bottleneck. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_skip_feature_grad_gate.md`.
				  The opposite feature-only split keeps only feature gradients and
				  confirms simple gradient masking is not enough: full/feature-only/
				  geometry-only backward is `581.3/548.2/547.3ms` at 256px and
				  `1919.7/2106.7/2174.0ms` at 512px. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_geometrysplit_gate.md`.
				  The recompute-only floor disables all output-gradient atomics and
				  still costs `571.3ms` backward at 256px and `2101.7ms` at 512px,
				  confirming shared replay/hidden64 VJP is the native bottleneck.
				  Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_recompute_floor_gate.md`.
				  Traversal-only skips hidden64 VJP as well and drops backward to
				  `194.9ms` at 256px and `742.2ms` at 512px, isolating the hidden64
				  forward/VJP slice as the largest removable piece. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_traversal_floor_gate.md`.
				  Hidden-forward-only splits that hidden slice into forward
				  `150.6/450.6ms` and backward `225.8/909.0ms` at 256/512px.
				  Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_forward_backward_split_gate.md`.
				  Hidden-preact-only splits that again: output+GELU prebackward is
				  only `54.8/61.7ms`, while F32 W^T feature-gradient reconstruction
				  is `171.0/847.3ms`. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_preact_wt_split_gate.md`.
				  The split manual-VJP subphase profiles show target-area reduction is
				  only `~0.12-0.13s`; exact GELU backward (`~1.34-1.44s`) and fc1
			  (`~0.75-0.89s`) dominate the loss-side path. Reports:
			  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_split_fullstep.md`
			  and
			  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_staronly_subphase_profile_fullstep.md`.
	  The first explicit optimizer-LR schedule gate (`0.001` until global step
	  `1375`, then `0.00025`) is a negative promotion result: it passes and removes
  the `1377->1378` jump, but a comparable jump reappears at `1385->1386`, and
  it ends worse than static effective-lr001 on weighted loss (`0.881602` vs
  `0.880942`), feature loss (`0.630803` vs `0.630549`), probe PSNR (`22.027`
  vs `22.034`), and timing (`1506.9ms` / `807.2ms` backward vs `1463.8ms` /
  `778.4ms`). The diagnostic 88-step late trace is expected to fail the quality
  pass bit because it stops just after the spike; it confirms `26/32` chunks
  worsen at `1385->1386`, summed weighted-loss delta `0.015248`, max frame-0
  chunk delta `0.001802`. Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md`.
	  Checkpoint selection is now resolved in favor of the lr005-sparse lineage;
	  new speed work should beat the sparse-forward batched-VJP helper before
	  replacing it.
- The cross-family V-JEPA comparison now has a generated report:
  `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`.
  It shows STAR V-JEPA streaming target at `3.743s/step`, STAR V-JEPA
  cached-chunks target at `1.655s/step`, STAR V-JEPA target-grid loss at
  `1.351s/step` (`1.451s/step` for the 20-step media row and about
  `2.000s/step` with 20-step RGB aux, `1.876s/step` for 100-step aux10,
  `1.639s/step` for the negative RGB-warm20 row, `0.00243s/step` for the
  standalone target-grid feature-to-RGB oracle, `1.220s/step`, `1.268s/step`,
  `1.355s/step`, plus `1.440s/step` and `1.512s/step` for the integrated
  frozen-probe continuation rows, `1.308s/step` for the nonpassing scheduled
  balance row, `1.461s/step` for the passing feature0.5/probe40 row, and
  `1.521s/step` for the nonpassing recover schedule, and `1.523s/step` for the
  passing feature0.75/probe40 row, and `1.285s/step` for the passing
	  feature1/probe40 row, plus `1.690s/step` / `1.711s/step` for the dense
	  1300->1400 extension and timing repeat, `0.400s/step` mean /
		  `0.263s/step` last-20 for the lr005 sparse-forward batched-VJP helper row,
		  and `0.372s/step` mean / `0.539s/step` last-20 for the lr001
		  sparse-forward rerun),
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
			  manual-linear full-cell8 diagnostic at `2.064s/step`, manual
			  hidden32 at `4.298s/step`, and the matched 512px native handoff
			  gate with `logit_handoff_reduce_vec4` at `0.386s` native backward
			  plus `0.422s` prep,
	  selected STAR RGB fast diagnostic at
  `2.491s/step`,
  Gaussian/token recon-only cached
  conditioning at `3.460s/step`, and Gaussian/token prediction-side V-JEPA loss
  at `38.621s/step`. The actionable STAR bottleneck moved from repeated target
  interpolation to target objective choice/native-VJP and dataset-scale
  validation; the Gaussian V-JEPA-loss bottleneck is frozen V-JEPA backward on
  predictions.
- Gate 4 same-clip quality bracket now fails feature promotion. RGB STAR
  direct-atomic on the same 64f/512px/8192t test video reaches `12.444` PSNR in
  20 steps; feature STAR reaches only `4.987` PSNR after the hidden-64
  diagnostic. The dynamic RGB and projected F32 rows in the bracket are
  speed-only references.
- The identity/no-pre-norm decoder diagnostic is the fastest 512px feature row
  (`2.536s/step`, `1.173s` backward), but quality drops to `4.888` PSNR. The
  simple decoder unclamp hypothesis is therefore speed-only, not the missing
  quality fix.
- The hidden-64 pre-norm decoder-capacity diagnostic barely improves best
  feature PSNR (`4.984 -> 4.987`) while slowing to `19.180s/step` and
  `13.769s` backward. Naive dense per-pixel decoder capacity is not a practical
  Gate 4 bridge.
- The pre-norm sigmoid gain-2 colorizer-init diagnostic is also negative:
  `4.987` PSNR at `14.119s/step` and `8.913s` backward. The local gain note was
  worth testing, but simple init gain does not close the quality gap.
- The cached-bin sidecar diagnostic is correct but not promoted. Reusing
  forward tile bins in backward cuts the same-session synthetic
  64f/256px/32768t/F32 renderer backward `1068.0ms -> 935.8ms`, but the
  first-class 512px/8192t/chunk2 row ties step time and has slower measured
  backward than plain gradcache (`16.20s/step`, `10.24s` backward versus
  `16.21s/step`, `9.68s`). Rebinning is not the whole 512px fix.
- The direct-mode matrix now includes cached-bin variants and a 512px block:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached/summary.md`.
  All 39 rows pass. At 512px, `gradcache_cached_bins` is the fastest
  full-gradient direct total row (`1.979s`, `1.103s` backward), while
  `gradcache_skip_feature_grad` remains the fastest diagnostic (`1.714s`,
  `0.804s` backward). Cached-bin wins are mixed across resolution, so keep the
  next shader target on fixedbin/tile-slot feature-gradient accumulation.
- The feature-gradient-only / two-pass split diagnostic is also recorded:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_two_pass_feature_grad_matrix_256_512_64f_32768t_f32/summary.md`.
  `gradcache_feature_grad_only` keeps only feature-gradient atomics, and
  `gradcache_two_pass_feature_grad` composes the geometry/opacity pass with
  the feature-only pass. Tiny F4/F32 parity passes, but naive split-recompute
  is slower than full gradcache: `1.343s`/`1.063s` versus `0.972s`/`0.692s` at
  256px and `2.471s`/`1.613s` versus `2.467s`/`1.379s` at 512px. Reserve
  "two-pass" for true fixedbin/tile-slot accumulation or native image-space
  VJP, not duplicate STAR traversal.
- The fixedbin/tile-slot budget gate is recorded at
  `outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_budget_128_256_512_64f_32768t_f32/summary.md`.
  It uses current forward bins to size a real accumulator. The theoretical
  atomic-write reduction is `128x`, but naive per-slot prefix recompute costs
  `39.8x` at 256px and `10.8x` at 512px. A scalar f32 contribution tape is
  about `1.2GiB` at 256/512px, while a per-channel f32 tape would be
  `37-38GiB`. The plausible tile-slot design is compact scalar
  weights/prefixes plus channel reduction, not per-channel weight storage.
- The tile-slot reducer isolation gate is recorded at
  `outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_reduce_matrix_256_512_64f_32768t_f32/summary.md`.
  `gradcache_feature_grad_only_reduce_vec4` improves isolated feature-gradient
  backward at both checked sizes (`532.8 -> 449.9ms` at 256px,
  `869.1 -> 774.8ms` at 512px). The full-gradient refresh at
  `outputs/benchmarks/2026-05-19_star_uvt_feature_full_reduce_refresh_256_512_64f_32768t_f32/summary.md`
  also shows a synthetic 512px single-pass win for
  `gradcache_reduce_feature_grad_vec4` (`1284.2 -> 1108.0ms` backward). Keep
  the single-pass vec4 reducer live; two-pass reducer composition is still not
  promoted because duplicate traversal remains the bottleneck.
- The 256px real-video tube-count bracket hits overflow after 8192 tubes:
  16384 overflows `736` tiles with max tile load `151` and p95 `123`, while
  32768 overflows `8160` tiles with max `274` and p95 `238`. Treat those as
  diagnostics, not valid quality/speed rows.
- `STAR_UVT_TILE_CAPACITY=256` is the current validity specialization: it makes
  16384 tubes zero-overflow, unpruned 32768 still overflows `216` tiles, and
  32768 becomes zero-overflow when paired with support pruning. The best
  current passing 20-step candidate is `32768t/alpha>=1/72/cap256`: loss
  `0.31889 -> 0.29290`, PSNR `4.96 -> 5.33`, mean step `1.321s`, backward
  `1.021s`, max tile `252`, p95 `209`. `alpha>=1/80` and `alpha>=1/96`
  improve loss slightly but overflow late, so they are not fixed-bin rows.
- The first-class trainer now accepts `feature_uvt.render_mode`. The
  `feature_direct_fixedbin` request records `requested_render_mode`,
  `kernel_backward_mode`, `requested_fixedbin_is_direct_atomic_alias`, and
  `mode_fallback_required`: unpruned `32768t/cap256` falls back after `216`
  overflow tiles, while `32768t/alpha>=1/72/cap256` records zero-overflow
  fixedbin eligibility. This is a reporting and promotion contract around the
  current direct feature path, not a separate optimized fixedbin kernel yet.
- `feature_direct_gradcache` is now the first actual feature fast-backward
  mode. It caches the pixel gradient vector inside the direct feature backward
  kernel for `feature_dim <= 64`. The serial synthetic A/B at
  `64f/256px/32768t/F32` improves backward `485.6ms -> 471.3ms`; the
  first-class real-video `alpha>=1/72/cap256` row passes at `1.226s/step` with
  `0.973s` backward. This is a modest baseline improvement, not the final
  feature-shader win.
- The benchmark-only `gradcache_skip_feature_grad` diagnostic intentionally
  skips only the per-channel `grad_feature` atomic writes. It keeps
  geometry/opacity gradient parity and cuts the same synthetic backward to
  `327.7ms`, while a nearby full-gradcache rerun measured `592.5ms`. Treat this
  as the proof that feature-gradient atomics are the next target; do not use the
  skip mode for training.
- The trainable `feature_direct_gradcache_reduce` prototype ports the v11-style
  per-tile/simd feature-gradient reduction into STAR UVT stable tiles. It passes
  F4/F32 parity and the 20-step real-video gate, but it is slower than plain
  gradcache on the target row (`523.8ms` synthetic backward versus `491.1ms`
  same-session gradcache; `1000.3ms` first-class backward versus `973.2ms`).
  Keep it as a negative result, not the default mode.
- The follow-up `feature_direct_gradcache_reduce_vec4` mode packs the
  per-channel reduction into `float4` SIMD reductions. It passes F4/F32 parity
  and improves one synthetic direct-kernel control (`484.7ms` backward versus
  same-session gradcache `528.2ms`), but it is slower in the real first-class
  cap256 row (`2.095s/step`, `1.413s` backward) than both gradcache
  (`1.807s`, `1.333s`) and scalar reduce (`1.890s`, `1.395s`). Keep it as a
  selectable diagnostic, not the default.
- The benchmark-only `fused_first3_sigmoid_mse` mode is the first RGB-gradient
  handoff proof in STAR UVT. It reconstructs the pixel feature/alpha inside the
  direct backward kernel and computes a narrow
  `alpha * sigmoid(feature[:3]) -> mean MSE` VJP locally. It is not the learned
  `FeatureToColor` path, but it passes parity and times at `309.3ms` backward
  on the synthetic 64f/256px/32768t/F32 row. The matched
  `64f/512px/8192t/F32` rerun passes and records `494.09ms` backward /
  `1152.58ms` total; this is still a boundary proof, not a trainer route.
- The generalized `direct_linear_sigmoid_mse_backward` handoff now exists as a
  benchmark surface with real `[3,F]` colorizer weights, bias, sigmoid MSE, and
  colorizer parameter gradients. It passes F4/F32 parity, including colorizer
  weight/bias grads, but it is slower than gradcache on the target row:
  `615-619ms` backward on two full-gradient runs versus `477.5ms` for a
  same-session gradcache rerun. The skip-colorizer-gradient diagnostic was
  noisy and did not produce a convincing mean win (`598.5-714.1ms` backward),
  so do not promote this in-tile generalized handoff to first-class training.
- The image-space-prep `direct_logit_handoff_backward` gate also passes F4/F32
  parity, but it is not faster: at 64f/256px/32768t/F32 it records `595.2ms`
  renderer backward plus `60.2ms` handoff prep (`835.6ms` total), versus
  `529.0ms` backward and `693.2ms` total for the same-session gradcache rerun.
  This keeps the next real shader target on optimized fixedbin/tile-slot
  feature-gradient accumulation rather than another dense colorizer handoff.
- The follow-up `logit_handoff_reduce` and `logit_handoff_reduce_vec4` gate
  merges that image-space handoff with the existing stable-tile feature-gradient
  reducers. The 64f/32768t/F32 direct matrix passes F4/F32 parity and zero
  overflow at 256px/512px. Vec4 improves synthetic backward at 256px
  (`571.7 -> 510.6ms`) and narrowly at 512px (`654.8 -> 642.3ms`), while scalar
  reduce regresses 512px backward (`722.5ms`). This keeps vec4 alive as a
  diagnostic native-VJP/tile-slot bridge, but it is not a first-class trainer
  default because the 512px total win includes forward/prep timing movement.
  The matched 512px/8192t native-handoff gate says the native side is promising
  only if prep is fused: `logit_handoff_reduce_vec4` has `386.26ms` backward
  but `421.89ms` Torch prep. The hidden sigmoid-MSE native gate then passes
  parity but rejects simple dense hidden fusion: H32 scalar is `610.90ms`
  total at 256px and `2549.39ms` at 512px, H64 256px is `817.27ms`, and
  vec4 reduce is slower than scalar.
- The first real-video RGB-VJP profile then exercises that bridge against a
  trainer-style linear RGB reconstruction loss. On the 64f/512px/8192t
  1300-step checkpoint, `logit_handoff_reduce_vec4` matches autograd gradients
  to `9.43e-09` max abs error with zero loss error and cuts the measured row
  `1691.0 -> 1587.4ms` (`1.065x`) with zero overflow. The 8f/64px smoke passes
  at `2.27x`. This is the first first-class-ish proof for linear RGB loss, but
  it still does not cover target-grid V-JEPA MSE or hidden64 frozen-probe VJP.
- The target-grid/frozen-probe VJP bridge profile then covers the current keeper
  objective directly. It matches normal autograd on the 64f/512px/8192t
  1300-checkpoint row (`2.57e-08` max grad error, zero loss error, zero
  overflow), but repeat timing is a slight negative (`1545.5ms` autograd versus
  `1594.3ms` bridge). This is a correct image-space VJP bridge and timing split,
  not a speed promotion. The analytic target-grid/probe VJP mode then removes
  the autograd image-VJP graph for the current hidden64 probe and target-grid
  MSE; repeat-5 timing is a small win (`1510.6 -> 1477.2ms`, `1.023x`) with
  `3.07e-08` max grad error and zero overflow. This justifies a trainer gate for
  analytic/native target-grid VJP. The trainer gate is now wired as
  `feature_target.image_vjp_mode=analytic` and passes a matched 5-step 64f/512
  gate, but trainer step time is only a tie: autograd mean step `1303.6ms`,
  warm analytic rerun `1304.6ms`, no-first `1264.1ms` versus `1259.2ms`. The
  backward bucket improves by `103.3ms`, but manual VJP time moves into the loss
  bucket, so this remains diagnostic.
- The sparse-pixel target-grid VJP follow-up is the first current-objective
  speed gate that survives the trainer loop. The new opt-in
  `feature_target.image_vjp_mode=analytic_sparse_pixels` packs nonzero
  target-grid image-gradient pixels and calls a sparse direct-atomic Metal
  backward with forward bins. Repeat-3 parity passes (`4.61e-08` max grad
  error, zero loss error), sparse renderer backward drops `557.6ms -> 46.3ms`,
  and bridge total drops `1245.9ms -> 920.5ms`; the remaining cost is dense
  Torch VJP plus sparse packing (`184.0ms`). The matched 5-step trainer smoke
  passes from the same 1300-step checkpoint, matches dense loss/probe PSNR, and
  cuts no-first step `1318.0ms -> 973.7ms` while visiting only `65,536` sparse
  pixels per 64f/512 step (`0.390625%` of dense). This proved the sparse
  target-grid hypothesis and is superseded by the direct sparse-grid follow-up
  below.
- The direct sparse-grid VJP follow-up now supersedes sparse-pixel packing for
  the current target-grid/frozen-probe diagnostic. The new opt-in
  `feature_target.image_vjp_mode=analytic_sparse_grid` analytically maps the
  trilinear target-grid/probe gradient to sparse source pixels and feeds those
  ids/values to the sparse direct-atomic backward. A small CPU/MPS parity check
  matches dense VJP exactly, and the full repeat-3 64f/512 bridge keeps parity
  (`4.60e-08` max grad error) while cutting total to `760.6ms`. The matched
  5-step trainer smoke passes from the 1300 checkpoint, matches dense/sparse
  loss and probe PSNR movement, and cuts sparse-pixel no-first step
  `973.7ms -> 795.3ms` with `88.6ms` no-first backward. The sparse-grid
  render-mode matrix keeps `feature_direct_gradcache_reduce_vec4` as the
  selected renderer under the new VJP path: no-first `730.5ms`, mean backward
  `78.3ms`, zero overflow, versus gradcache `759.4ms` and direct atomic
  `779.3ms`. Sparse-grid dense-forward remains the backward-only reference.
- The sparse-forward follow-up is the selected current target-grid/frozen-probe
  diagnostic, with repeat-aware timing. The new opt-in
  `feature_target.image_vjp_mode=analytic_sparse_grid_forward` renders only the
  target-grid support pixels, folds sparse feature values into the target grid
  for feature/probe loss, and reuses sparse-grid VJP for backward. The sparse
  forward profile is bit-exact against dense feature/alpha values and initially
  cut forward render `515.9ms -> 70.5ms` (`7.322x`) for `65,536` support pixels.
  The 128/256/512 scale matrix passes all rows with zero overflow and records
  no-first trainer step `379.2ms` / `494.2ms` / `973.0ms`; the isolated 512px
  repeat after scale is `598.2ms` no-first / `477.6ms` last step. This means the
  path is valid and useful, but not yet a stable hard timing baseline. The
  dedicated 512px repeat-3 timing gate passes all rows with zero overflow and
  gives no-first step mean/min/max/stdev `504.9/411.0/626.4/110.3ms`, last-step
  `468.8/409.3/549.9/72.7ms`, and no-first backward
  `142.2/114.7/174.4/30.1ms`. The batched target-grid/probe VJP path then
  becomes the selected opt-in trainer speed lever: all 32 chunks share one
  target/probe loss+VJP pass with `7.45e-09` loss error and `6.55e-11` max
  feature-grad error, cutting isolated loss+VJP `38.0ms -> 4.8ms` (`7.99x`).
  The 5-step optimizer harness reaches `173.1ms` no-first step with zero
  overflow, and the integrated trainer repeat-3 gate gives no-first step
  mean/min/max/stdev `179.3/159.7/215.6/31.5ms`, no-first backward
  `72.0/60.8/90.2/15.9ms`, and no-first render `71.1/67.8/77.4/5.5ms`. The
  100-step helper/media gate also passes with loss `0.886537 -> 0.880744`,
  feature loss `0.632124 -> 0.627122`, zero overflow, mean
  step/backward/render `399.9/176.9/125.2ms`, and last-20
  `262.9/109.4/94.0ms`, while writing a valid contact sheet, MP4, and
  checkpoint. The next real gate is visual-quality improvement, followed by
  native GPU target-grid/probe loss+VJP or scalar fixedbin/tile-slot
  feature-gradient accumulation only if it beats this repeat/100-step surface.
- The first `targetgrid_render_mode_trainer_matrix.py` gate against dense
  analytic VJP is now historical context. It reused the same 1300-step
  checkpoint and 5-step analytic-VJP config for `feature_direct_atomic`,
  gradcache, cached-bins, scalar reduce, vec4 reduce, and the fixedbin request;
  all rows passed and landed on the same loss/probe PSNR, but the repeat-top
  check showed no vec4/reduce trainer win (`feature_direct_atomic` no-first
  `1249.0ms`, cached-bins `1410.9ms`, vec4 `1509.6ms`, fixedbin-request
  `1422.6ms`). The sparse-grid matrix above supersedes it for the selected
  speed path. `feature_direct_fixedbin` still reports
  `kernel_backward_mode=direct_atomic`; it remains an eligibility/fallback
  surface until a real fixedbin/tile-slot kernel is added.
- `direct_feature_mode_matrix.py` is the reproducible direct-mode runner. The
  first sequential all-mode 128/256 matrix is
  `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_mode_matrix_128_256_64f_32768t_f32/summary.md`;
  it records JSON/log paths and avoids parallel MPS timing contamination.
- The first-class 8f smoke now writes inspection media:
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_contact.png`
  and
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_side_by_side.mp4`.
- The 64f/256px/32768t `alpha>=1/72/cap256` candidate also writes inspection
  media:
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_contact.png`
  and
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_side_by_side.mp4`.
- The matching `feature_direct_fixedbin` mode-contract rerun writes:
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_contact.png`
  and
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_side_by_side.mp4`.
- The `feature_direct_gradcache` rerun writes:
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_contact.png`
  and
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_side_by_side.mp4`.
- The identity/no-pre-norm decoder diagnostic writes:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_media.json`,
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_contact.jpg`,
  and
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_sbs.mp4`.
- The hidden-64 decoder-capacity diagnostic writes:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_media.json`,
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_contact.jpg`,
  and
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_sbs.mp4`.
- The gain-2 colorizer-init diagnostic writes:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_media.json`,
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_contact.jpg`,
  and
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_sbs.mp4`.
- The current fast feature-tube diagnostic writes:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_fast_overfit_reduce_vec4_summary.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature_selected_shader_scale_128_256_512.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_media.json`,
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_contact.jpg`,
  and
  `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_sbs.mp4`.
- `firstclass_scale_report.py` summarizes existing JSON artifacts or runs
  configs and emits a markdown/JSON table.
- `firstclass_backward_breakdown.py` attributes first-class cost without running
  optimizer steps. It reuses checked-in trainer configs and target videos, then
  manually times render forward, colorizer/loss forward, image-space backward
  to `grad_feature_image`/`grad_alpha`, and the Metal feature backward.
- `star_uvt_vjepa_vs_gaussian_comparison.py` normalizes the STAR V-JEPA target,
  selected STAR RGB feature, multicam cached-V-JEPA, and 300-clip Gaussian
  timing artifacts into one comparison report.
- `target_cache_budget.py` records the cached-chunks adapted-target memory
  budget and the target-grid alternative so larger V-JEPA target runs do not
  silently scale into multi-GiB resident targets.

Minimal path:

1. Dense feature prototype: `[N,F]` tube features -> `[T,F,H,W]` feature image
   plus alpha -> `FeatureToColor` -> RGB composition/loss.
2. If tiny overfits work, create a feature-specific STAR renderer fork with a
   `feature_dim` contract. Do not mutate the RGB `star_uvt_v0` kernels in place.
3. First Metal target should be direct atomic / `index_add` for speed probes;
   deterministic compact feature backward is a later promotion gate.
4. Port F32 fast-shader lessons only into feature-specific modes: gradcache,
   accum/reduce, fixedbin with explicit overflow fallback, and render/loss
   microbatching.

Tiny check:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py --smoke
```

Gate 0 contract:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py \
  --gate0-benchmark --frames 5 --height 16 --width 16 --tubes 24 \
  --feature-dim 32 --steps 8 --chunk-size 2 --device mps \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json
```

Direct feature Metal gate:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_256_32768_f32.json
```

Direct feature gradcache A/B:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_64f_256_32768_f32.json
```

Cached-bin sidecar A/B:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache_cached_bins --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 --timing-warmup 2 \
  --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_cachedbins_64f_256_32768_f32.json
```

Feature-gradient atomic diagnostic:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache_skip_feature_grad --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_64f_256_32768_f32.json
```

Trainable reduced feature-gradient prototype:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache_reduce_feature_grad --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_64f_256_32768_f32.json
```

Vectorized reduced feature-gradient prototype:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache_reduce_feature_grad_vec4 --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_serial_64f_256_32768_f32.json
```

Narrow fused RGB-MSE handoff prototype:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode fused_first3_sigmoid_mse --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_serial_64f_256_32768_f32.json
```

Generalized linear sigmoid-MSE handoff diagnostic:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode linear_sigmoid_mse --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_64f_256_32768_f32.json
```

Image-space-prep logit handoff diagnostic:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode logit_handoff --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_serial_64f_256_32768_f32.json
```

Logit-handoff tile-slot reducer gate:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes logit_handoff,logit_handoff_reduce,logit_handoff_reduce_vec4 \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_matrix_256_512_64f_32768t_f32
.venv/bin/python research_experiments/star_uvt_feature_tubes/logit_handoff_reduce_report.py
```

Real-video linear RGB-VJP profile:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc \
  --warmup 1 --repeat 2 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_64f512_from1300
```

Target-grid/frozen-probe VJP bridge profile:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc \
  --warmup 2 --repeat 5 --image-vjp-mode analytic \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_repeat5
```

Analytic VJP trainer smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_analyticvjp.jsonc
.venv/bin/python research_experiments/star_uvt_feature_tubes/targetgrid_analytic_vjp_trainer_report.py
```

Sparse-pixel VJP profile and trainer smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --image-vjp-mode analytic_sparse_pixels --warmup 1 --repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_profile
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsepixvjp.jsonc
```

Sparse-grid VJP profile and trainer smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc \
  --image-vjp-mode analytic_sparse_grid --warmup 1 --repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_profile
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py \
  --base-config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix
```

Sparse-forward profile and trainer smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_sparse_forward_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc \
  --warmup 1 --repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforwardvjp.jsonc
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/sparse_forward_scale_matrix.py \
  --sizes 128,256,512 --profile-warmup 1 --profile-repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_scale_128_256_512
```

Sequential direct-mode matrix:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --sizes 128,256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 1 --repeat 3 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached
```

Feature-gradient-only / two-pass split diagnostic:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes gradcache,gradcache_skip_feature_grad,gradcache_feature_grad_only,gradcache_two_pass_feature_grad \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_direct_two_pass_feature_grad_matrix_256_512_64f_32768t_f32
```

Fixedbin/tile-slot accumulator budget:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/tile_slot_accumulator_budget.py \
  --sizes 128,256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --warmup 1 --repeat 3 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_budget_128_256_512_64f_32768t_f32
```

Feature-only tile-slot reducer isolation:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes gradcache,gradcache_skip_feature_grad,gradcache_feature_grad_only,gradcache_feature_grad_only_reduce,gradcache_feature_grad_only_reduce_vec4,gradcache_two_pass_feature_grad,gradcache_two_pass_feature_grad_reduce,gradcache_two_pass_feature_grad_reduce_vec4 \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_reduce_matrix_256_512_64f_32768t_f32
```

Full-gradient reducer refresh:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes gradcache,gradcache_reduce_feature_grad,gradcache_reduce_feature_grad_vec4 \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_full_reduce_refresh_256_512_64f_32768t_f32
```

Autograd video overfit gate:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py \
  --video-path test_data/test_video_small_128_4fps.mp4 \
  --frames 8 --size 64 --tubes 512 --feature-dim 32 --steps 20 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_video_overfit_8f_64px_512t_f32_20step.json
```

First-class chunked trainer gate:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc
```

First-class scale report:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py \
  --result-jsons \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_8192t_f32_chunk4_3step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_cap256_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_cap256_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_cap256_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_vec4_alpha1_72_cap256_20step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_2048t_f32_chunk2_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_4096t_f32_chunk2_gradcache_2step.json \
    outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_2step.json \
    outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_cachedbins_prenorm_2step.json \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json \
  --out-md outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md
```

First-class backward breakdown:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_4096t_2step.jsonc \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step.jsonc \
  --warmup 1 --repeat 2 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.json \
  --out-md outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.md
```

Cached-bin first-class trainer gate:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_cachedbins_chunk2_8192t_prenorm_2step.jsonc
```

No-pre-norm 512px speed A/B:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_4096t_2step.jsonc \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step.jsonc \
  --colorize-pre-norm false --warmup 1 --repeat 2 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.json \
  --out-md outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.md

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 WANDB_MODE=offline .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_2step.jsonc
```

20-step media A/B:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_prenorm_20step_media.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_20step_media.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_identity_no_prenorm_20step_media.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_hidden64_prenorm_20step_media.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_prenorm_gain2_20step_media.jsonc
```

Gate 4 same-clip quality bracket:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_rgb_testvideo_64f_512_directatomic_8192t_20step_media.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_directatomic_chunk2_8192t_prenorm_20step_media.jsonc

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/gate4_quality_bracket_report.py \
  --rgb-star-json outputs/benchmarks/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_media.json \
  --renderer-csv outputs/benchmarks/2026-05-19_renderer_scaling_report.csv \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.json \
  --out-md outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md
```
