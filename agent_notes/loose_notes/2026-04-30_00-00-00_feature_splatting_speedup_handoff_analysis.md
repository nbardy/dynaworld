# Feature Splatting Speedup Handoff: Mapping External Recommendations to Our Stack

Date: 2026-04-30
Context: An engineering handoff doc arrived from outside the project citing
Spacetime Gaussians (CVPR 2024), Faster-GS / Taming 3DGS, and Feature 3DGS
(Zhou et al., CVPR 2024). It proposes kernel-level and architecture-level
speedups for D-channel feature splatting. We just landed F=32 alpha-aware
feature splatting on top of v5_features (yesterday's session,
`2026-04-29_22-00-00_feature_splatting_alpha_aware_composition_session.md`).
The relevant throughput baseline from Codex's handback (256x256, 2048 splats,
MPS, with alpha):

| F   | forward (ms) | backward (ms) | total (ms) |
|----:|------------:|-------------:|-----------:|
|  3  | 9.7         | 8.3          | 18.0       |
|  8  | 9.1         | 11.4         | 20.5       |
| 32  | 19.9        | 24.2         | 44.1       |

F=32 is ~2.4x slower than F=3 end-to-end. The backward grows ~2.9x from F=3
to F=32; the forward grows ~2.1x. Both legs scale roughly linearly in D, with
the backward leg the steeper slope.

## TL;DR — what to do, in priority order

1. **Cap F at 32, don't go higher** — the doc's D=16/32 register-spilling
   warning is consistent with our forward+backward cost ramp. F=32 is the
   ceiling we should plan around. Wins-now, zero work, just a planning constraint.
2. **Per-splat-major backward (Faster-GS thread inversion)** — algorithmic idea
   ports cleanly to Metal even though the CUDA microoptimizations don't. Our
   current backward IS pixel-major with SIMD-group consolidation; **the Faster-GS
   approach would invert that and the per-channel atomic pressure on `g_colors`
   would drop a lot.** This is the single largest plausible backward speedup.
   Codex-territory, multi-day kernel rewrite, not low risk. Do later, only if
   F=32 backward stays a real bottleneck.
3. **Scalarization trick for opacity gradient — already done** (see analysis
   below). No action.
4. **1x1 MLP decoder — already what we use.** The doc's recommendation matches
   `FeatureToColor`. The 3x3 conv variant is a worse fit for a tile-based
   rasterizer and we should skip it. Already done.
5. **Detach feature densification (stop-gradient)** — gated on a missing
   prerequisite (V-JEPA teacher distillation loss). Premature work today.
6. **Fused Adam in the kernel — skip.** PyTorch's `fused=True` Adam is already
   active (`train_video_token_implicit_dynamic.py:1037`). Writing a Metal-side
   fused optimizer would be a big rewrite for sub-millisecond gains at our scale.

## Section-by-section analysis

The doc's five subsections, in order.

### 1. Rasterization: per-Gaussian thread mapping in backward

#### Recommendation

Invert the standard 3DGS backward thread mapping. Assign threads to **Gaussians**
instead of pixels. Each Gaussian thread iterates its covered pixels (up to a
tile of 256), accumulates D-dim gradient in registers, and does a single
coalesced atomicAdd at the end. Cuts global atomic collisions on `g_colors`
by up to 256x at D=32.

#### Applicability to our stack

- **Metal: yes algorithmically, with caveats.** The Faster-GS thread inversion
  is a thread-mapping change, not a CUDA-specific microoptimization. Metal
  has threadgroup memory + atomic_float and supports the same pattern. The
  CUDA-specific bits (warp shuffles, cooperative groups, NVCC register pressure
  knobs) translate to SIMD-groups + threadgroup barriers + Metal compiler
  knobs but with different numerical and resource profiles.
- **Already done in v5_features? Partially, in a different form.** Spot-check
  of `csrc/metal/gsplat_v5_features_kernels.metal:556-685`
  (`v5_features_tile_fast_backward_saved`):
  - **Threads ARE pixel-major** (one threadgroup per tile, one thread per
    tile pixel). This is the standard 3DGS layout the Faster-GS paper attacks.
  - However, **per-splat per-pixel atomicAdds are NOT happening for
    means2d/conics/opacities.** The kernel uses `simd_sum` to reduce
    `(l_gmx, l_gmy, l_ga, l_gb, l_gc, l_gop)` across the 32 lanes of a
    SIMD-group, then a second-stage `simd_sum` across `GSP_SIMDGROUPS=8`
    SIMD-groups via `partial0/partial1/partial2`, and finally `simd_lane==0`
    issues **one atomicAdd per splat per simdgroup-reduction** for those six
    geometry fields. Net atomic frequency: 6 atomicAdds per splat per tile,
    not per pixel. That's already the algorithmic win the doc is selling
    for the geometry stream.
  - **The D-dim color gradient is NOT consolidated this way.** Each pixel
    thread that hits a splat calls `atomic_add_feature_grads` (line 644 ->
    line 131-143), which loops `f = 0..F-1` and issues an `atomic_fetch_add`
    per channel. That's **F atomicAdds per pixel per splat-hit**, with
    pixel-level collisions on `g_colors[g, f]`. At F=32 and a 256-pixel
    tile, that's up to 8192 atomicAdds per splat per tile.
  - **`dot_pixel_features` (line 642, line 105-119) is also a D-loop per
    pixel per splat-hit**, reading `grad_features[pix, f] * colors[g, f]`
    in pixel-thread registers. This isn't atomic but it dominates the D-scale
    forward+backward bandwidth.
- **Prerequisite:** none. The kernel exists and we can rewrite its backward.
  The risk is gradient correctness — we have alpha-output Test B verifying
  geometry gradients flow correctly, but a thread-mapping change reshuffles
  the entire reduction order and floating-point summation order, so we'd
  re-run all five Codex tests (A-E) plus the F=3-vs-v5 parity.

#### Estimated impact

- **F=32 backward: ~8.3 ms (F=3 baseline) -> ~24.2 ms (current).** The 16 ms
  extra at F=32 vs F=3 backward is essentially the D-loop in
  `dot_pixel_features` + `atomic_add_feature_grads`. If the per-splat-major
  rewrite cuts that linear-D term by, say, ~3x (Faster-GS claim is up to
  256x for the atomic count, but Metal SIMD-group atomics already amortize
  some of that), backward at F=32 could drop from ~24 ms to ~13-15 ms. Total
  end-to-end at F=32: ~44 ms -> ~33-35 ms (**~20-25% wall clock savings on
  the recon backward**).
- **F=3 backward: probably no change**, since the existing SIMD reduction
  already handles the geometry gradients with low atomic pressure and D=3
  is too small to benefit from atomic consolidation on the color stream.
- **Memory: marginal increase.** Per-splat-major needs threadgroup-shared
  staging of pixel gradients (`[tile_pixels, F]`). At F=32 + 16x16 tile +
  fp32, that's 32 KB per tile in threadgroup memory, well within Metal's
  32-64 KB threadgroup-storage budget on M-series chips.
- **Forward: no impact.** Forward is already pixel-major, no atomics.

#### Effort estimate

- Codex hours: ~2-4 days. This is a real kernel rewrite, not a wrapper change.
  All five backward kernels touched (`fast_backward_saved`, the overflow
  backward, plus the dispatching in `gsplat_metal.mm` and `bindings.cpp`).
  Schema strings probably stay the same (signatures don't change), but the
  kernel internals are a rewrite.
- Dynaworld engineer hours: ~half a day to rerun tests A-E + a 400-step
  F=32 alpha-aware run vs `3reqcya9` to verify no regression. Mechanical.
- Risk factors: numerical reproducibility (FP32 summation order changes,
  could shift gradients by ~1e-6 scale; tests would need slightly looser
  tolerances), gradient correctness (the alpha-stream synthetic-channel
  trick has to carry through unchanged), threadgroup memory budget
  (verify the `[tile_pixels, F]` staging actually fits at F=32 with our
  `GSP_THREADS=256`).

#### Verdict

**Do later, not now.** The F=32 backward is real but not yet a measured
training bottleneck — recon_backward is one of several costs in our 256x256
single-clip overfit, and the dominant term at our current scale is dataset
I/O + camera/loss math, not the rasterizer. Revisit when (a) we move past
single-scene overfit to multi-scene Tier 2 / Tier 3 horizons where the F=32
raster is run hundreds of thousands of times, OR (b) profiler shows
recon_backward >50% of step wall clock.

### 2. Feature dim recommendation: D=16 or D=32

#### Recommendation

D=16 or D=32 is the sweet spot. D > 32 causes register spilling on consumer
GPUs.

#### Applicability to our stack

- **Metal: yes.** Apple Silicon GPUs have similar register-pressure dynamics
  to NVIDIA's consumer chips; M-series cores have ~256 threadgroup registers
  and a single-thread limit much smaller. F=32 with our pixel-major backward
  already keeps `(l_gmx, l_gmy, l_ga, l_gb, l_gc, l_gop)` and the partial
  feature accumulators in registers; pushing to F=64 would force spilling
  to threadgroup memory and slow the inner loop noticeably.
- **Already done in v5_features?** F=32 is what we shipped. The kernel works
  at F=3, F=8, F=32 (Codex's `feature_contract_check.py` validates these).
  We have not benchmarked F=16, F=64, or F=128.
- **Prerequisite:** none.

#### Estimated impact

Not a speedup recommendation per se — it's a cap. The doc tells us
"don't go higher than 32." Our F=32 throughput numbers are consistent with
this: F=32 is already 2.4x slower than F=3 end-to-end and 2.9x slower in
backward, on the linear-D scaling track. F=64 would presumably continue
the trend (probably ~70-90 ms total) AND start triggering register
spills, so the ramp would steepen. No measurable impact for staying at
F=32, but a planning guardrail against future "let's try F=128" temptations.

#### Effort estimate

Zero — it's a constraint to remember, not a code change.

#### Verdict

**Already done.** Cap F at 32 and treat F=64+ as a research curiosity, not
a default knob to widen. No action.

### 3. Decoder architecture: 1x1 MLP vs 3x3 Conv

#### Recommendation

1x1 MLP (Spacetime style): pixel-wise, preserves geometric boundaries,
embarrassingly parallel, fuse-able into the rasterizer kernel.

3x3 Conv (Feature 3DGS style): spatial regularizer, smooths splatting
artifacts, must run in standard PyTorch (cannot be fused into a tile-based
rasterizer because of halo).

#### Applicability to our stack

- **Metal: applies the same way as CUDA.** This is an architecture choice
  upstream of the kernel.
- **Already done in v5_features?** N/A — colorize is a PyTorch
  post-rasterization step (`src/train/colorize.py:88,
  train_video_token_implicit_dynamic.py:1346-1359`). The colorize MLP is
  already a 1x1 Conv2d (`nn.Conv2d(self.input_dim, 3, kernel_size=1)`) +
  optional 1x1 hidden. So **we are already on the recommended path.**
- The 3x3 variant is something we have NOT tried.
- The doc's claim that 1x1 can be "fused into the rasterizer kernel" is
  CUDA-specific and would mean rewriting `add_weighted_features` to do
  `colors[g, :] -> linear_combine -> rgb` in-kernel. Possible in Metal but
  high complexity, and we'd lose the ability to swap colorize MLPs without
  rebuilding the .metallib. Not worth it at our scale.
- **Prerequisite:** none for the 1x1 path; 3x3 would be additive.

#### Estimated impact

- **1x1 MLP (current):** baseline. Roughly free relative to the rasterizer.
- **3x3 conv (untried):** would smooth splat-boundary artifacts at the cost
  of (a) ~9x compute in the conv layer (still trivial: a 3x3 32->3 conv on
  256x256 is single-digit milliseconds in PyTorch), (b) loss of pixel-wise
  independence (downstream halo means it can't be fused into a tile-based
  raster). Smoothing is a real concern for our scenes — splat clustering
  has been an issue — but yesterday's alpha-aware composition fix attacked
  that structurally and shouldn't be re-fought via post-conv smoothing.
- **Fuse 1x1 colorize into the kernel:** ~2-5 ms savings if done well, but
  rebuilds the .metallib for every change to colorize. Not worth it.

#### Effort estimate

- Status quo: zero.
- 3x3 conv switch: ~2 hours engineer work (add `kernel_size` knob to
  `FeatureToColor`, rerun an A/B at F=32 alpha-aware vs current 1x1).
  No reason to do it before we have a real artifact symptom to attack.

#### Verdict

**Already done (1x1 MLP). Skip the 3x3 variant** unless we observe specific
splat-boundary smoothing problems in renders. Don't fuse colorize into the
kernel — premature optimization that breaks our ability to iterate on the
MLP architecture.

### 4. Scalarization trick for opacity gradients

#### Recommendation

Standard 3DGS opacity grad accumulates D-dim vectors. The chain rule has the
term `f_i . grad_F_p` which is a scalar — compute it early and accumulate
scalars instead of vectors. Shared memory drops from O(pixels x D) to
O(pixels x 1).

#### Applicability to our stack

- **Metal: yes.**
- **Already done in v5_features? Yes — already done.** Spot-check of
  `gsplat_v5_features_kernels.metal:642-655`:
  ```
  float dot_gc = dot_pixel_features(grad_features, colors, pix, g, mi) + alpha_grad;
  float g_alpha = T_prev * (dot_gc - gT);
  ...
  l_gop = g_raw * (raw_alpha / max(opacity, mf.eps));
  ```
  `dot_gc` IS the scalarized `f_i . grad_F_p` collapsed to a single float
  per (pixel, splat) hit, computed inside the per-pixel loop. `l_gop`
  (the opacity local grad) is ALSO scalar — one float per pixel per splat
  hit, then `simd_sum`'d via `simd_sum(l_gop)` (line 662) into per-tile
  partials. The kernel never accumulates a D-vector for opacity. The doc's
  trick has already been applied to this kernel.
- **However:** the `dot_pixel_features` D-loop itself (line 105-119) and
  the `atomic_add_feature_grads` D-loop (line 131-143) are still the
  D-scaling bottleneck, and those are not scalarizable — they ARE the
  D-channel per-pixel work that drives backward cost. The "scalarization
  for opacity" claim addresses opacity, not the color/feature stream.
- **Prerequisite:** none. Already in.

#### Estimated impact

Zero new impact — it's already in the kernel. If we hadn't done it, it
would have been ~1-2 ms savings at F=32 backward and a notable shared-memory
budget reduction. Both already paid out.

#### Effort estimate

Zero.

#### Verdict

**Already done.** No action.

### 5. Detach feature densification (stop-gradient on opacity/position)

#### Recommendation

When distilling features, the loss flows through `f_i` AND through
opacity/position. Detach opacity and position from the feature loss for the
first 50% of training so the network learns features in place.

#### Applicability to our stack

- **Metal: applies the same way.** This is a Python-side stop-gradient
  decision at the loss assembly site, not a kernel change.
- **Already done in v5_features?** N/A — this is a trainer-side concern.
  Looking at `train_video_token_implicit_dynamic.py:1346-1370`, the recon
  loss is built as `L1(alpha * colorize(features) + (1-alpha)*bg, GT)`,
  with all gradients flowing through everything (means2d, conics, opacities,
  features, and the colorize MLP). We do not currently distill against an
  external feature teacher.
- **Prerequisite — and this is the blocker.** "Detach feature densification"
  only makes sense in the context of a **feature distillation loss**:
  `L_feat = ||rendered_features - teacher_features||`. The advice is "during
  L_feat backward, freeze geometry; only let the splat colors update." We
  have no teacher loss yet — the F=32 features are trained end-to-end against
  RGB reconstruction via `colorize(features)`. There's no separate
  feature-supervision pathway whose geometry-side gradient we'd want to
  detach. So the advice doesn't apply to today's setup.
- The natural prerequisite is: implement a V-JEPA-as-teacher distillation
  loss (per `BASELINES.md` the obvious next step). Once that lands, the
  detach decision becomes live.

#### Estimated impact

- **Today: zero.** Nothing to detach.
- **After V-JEPA teacher loss lands:** the doc's empirical claim is that
  detaching for the first 50% of training helps the feature head converge
  before geometry follows. Plausible, but the evidence is from Feature 3DGS
  / Spacetime which use very different teacher signals (CLIP, DINO). Worth
  ablating, but only after the teacher loss is in place.

#### Effort estimate

- Today: not applicable (would be premature).
- Future (after teacher loss exists): ~half a day. Wire up a per-loss-term
  `detach_geometry` flag on `recon_backward` and run a 200-step A/B at
  the natural step horizon.

#### Verdict

**Gated on missing prerequisite.** Don't pursue until V-JEPA teacher
distillation is implemented. Note this when we get to that work.

### 6. Fused Adam updates

#### Recommendation

Fuse the optimizer step into the end of the custom backward kernel.
Bypass PyTorch VRAM bandwidth.

#### Applicability to our stack

- **Metal: in principle yes, in practice no.** Writing a Metal kernel that
  reads gradients, applies Adam moments, and writes parameters back would
  require either (a) custom Metal Adam kernels for every optimized parameter
  group (means2d, conics, scales, rotations, opacities, colors, plus
  colorize MLP weights), or (b) a unified all-parameters kernel. Either is
  a lot of code for an Adam impl that PyTorch already provides.
- **Already done?** Partially — the current trainer uses
  `torch.optim.Adam(..., fused=True)` (`train_video_token_implicit_dynamic.py:1040`).
  PyTorch's fused Adam path on MPS already kernel-fuses the moment
  update + parameter step into a single Metal kernel call per parameter
  tensor, so we already get the bandwidth savings within Adam itself. What
  the doc proposes — fusing Adam into the *rasterizer backward kernel* —
  would skip the round-trip through `g_colors` etc., but that is a
  big rewrite specific to a custom rasterizer + custom optimizer.
- **Prerequisite:** would mean rewriting v5_features's backward to also
  emit parameter updates, not gradients.

#### Estimated impact

- **Realistic Metal-side fused Adam in the kernel:** sub-millisecond
  savings on the optimizer step for our 8192-splat configs (the optimizer
  step itself is already fast in PyTorch's fused path; the bandwidth saved
  by avoiding the round-trip through `g_colors` is ~F * num_splats * fp32
  = 32 * 8192 * 4B = 1 MB, which on MPS is sub-millisecond to traverse).
- **For the colorize MLP weights:** zero — those are tiny tensors
  (32 * 3 + 3 = 99 floats), Adam on them is already free.
- **Realistic at our scale:** essentially nothing to gain.

#### Effort estimate

- Multi-day kernel rewrite (touching every backward kernel + optimizer
  bookkeeping). High risk: a bug in fused Adam silently corrupts training
  and is hard to bisect against PyTorch's reference Adam.
- Lose the ability to swap optimizers (try AdamW, Lion, Sophia) without
  another kernel rewrite each time.

#### Verdict

**Skip.** The wins are too small to justify the rewrite cost, and PyTorch's
`fused=True` Adam is already paying out the within-Adam bandwidth savings.
The doc's recommendation here is target-specific to scenarios where the
optimizer step dominates and PyTorch overhead is the bottleneck — neither
is true for our setup.

## Prioritized roadmap

### Wins worth pursuing now

None of the doc's recommendations are "do today, low risk, high ROI" for
our current state. The two highest-impact ideas (per-splat-major backward,
detach feature densification) are gated on either a measured bottleneck
(per-splat-major) or a missing prerequisite (teacher distillation). The
medium-impact ones are already done.

### Wins gated on missing prerequisites

1. **Detach feature densification.** Wait until V-JEPA teacher distillation
   exists. Implementing detach without a separate feature loss is a no-op.
   When the teacher loss lands, plan a 200-step A/B with detach-during-first-50%
   vs no-detach.

### Wins worth pursuing later, gated on profile evidence

2. **Per-splat-major backward kernel rewrite.** Save this for when (a) the
   F=32 backward is a measured >50% chunk of step wall-clock, OR (b) we
   start running Tier 3 / multi-scene for many tens of thousands of steps
   where the cumulative raster cost dominates. Plausible 20-25% wall-clock
   savings on F=32 backward, multi-day Codex-territory kernel work.
   Re-validate via Codex's existing tests A-E + a 400-step parity vs run
   `3reqcya9`.

### CUDA-specific ideas that don't translate cleanly

3. **Fused Adam inside the rasterizer kernel.** Skip. The "bandwidth saved"
   argument depends on a CUDA-specific assumption (PyTorch optimizer overhead
   dominates) that doesn't apply to our `fused=True` MPS Adam. Multi-day
   rewrite for sub-millisecond gains.

4. **Fuse 1x1 colorize MLP into the rasterizer kernel.** Skip. We'd lose
   the ability to swap colorize MLPs without rebuilding the .metallib, in
   exchange for ~2-5 ms savings. Not worth it.

### Already done (don't re-recommend)

5. **Scalarization trick for opacity gradients.** Already in
   `gsplat_v5_features_kernels.metal:642-655`. The kernel computes
   `dot_gc = sum_f(grad_features[pix,f] * colors[g,f]) + alpha_grad` as a
   scalar inside the per-pixel loop and never accumulates a D-vector for
   opacity. `simd_sum(l_gop)` does scalar-only reduction.
6. **1x1 MLP decoder (Spacetime-style).** `FeatureToColor` is a 1x1
   Conv2d already.
7. **F=32 dim cap.** F=32 is what we shipped; the doc's D=16/D=32 sweet-spot
   guidance matches. F=64+ is research curiosity, not default.

### Things to NOT do (the doc suggests them, we should explicitly skip)

- **Fused Adam in the rasterizer kernel** (item 3 above). Multi-day rewrite,
  sub-millisecond gain, PyTorch already covers most of the same bandwidth
  savings.
- **3x3 conv decoder.** Adds halo compute, breaks pixel-wise independence
  of the colorize stage, and the splat-clustering symptom that smoothing
  would have masked has already been fixed structurally by alpha-aware
  composition (run `3reqcya9`). Don't reach for a smoothing decoder to fix
  a problem we already solved at the loss-composition layer.

## One-line summary

Of the doc's six recommendations: three are already done (scalarization,
1x1 MLP, F=32 cap), two are gated on missing prerequisites (detach
densification needs a teacher loss; per-splat-major rewrite needs profiler
evidence that backward is the bottleneck), and one (fused Adam in-kernel)
should be skipped at our scale. No work to do today; the highest-leverage
follow-up is the per-splat-major backward, which is a multi-day Codex
kernel rewrite to be triggered by profile data, not by this doc.
