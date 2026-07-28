# Browser WGSL Mode Fidelity Audit

Date: 2026-07-22 23:27:13 Asia/Seoul

Status: code-reading audit. No runtime files were edited. This note compares the
current browser implementation to the canonical renderer taxonomy and checked-in
Metal/reference implementations; it does not claim measured parity.

## Question And Verdict

Question: are the two modes in
`web/dynaworld_browser_trainer/trainerWebGpu3d.js` faithful implementations of
(A) World Tubes / STAR UVT with shared backward and (B) standard dynamic 3DGS?

Verdict:

1. **World Tubes-style is not World Tubes / STAR UVT parity.** It is a
   world-space isotropic point-splat model with linear plus single-frequency
   sinusoidal center motion. It does not construct UVT traces, marginalize the
   ray-depth fiber, compile camera-time traces, bin in UVT, certify support or
   order, or run a STAR UVT adjoint.
2. **Dynamic splats-style is not the repository's standard dynamic 3DGS
   baseline.** It is the same shared trajectory model with the sinusoidal center
   term disabled. Standard dynamic 3DGS in this repository stores independent
   anisotropic Gaussian state per frame and uses fast-mac's projected conics,
   depth ordering, tiled rasterization, and rasterizer VJP.
3. **The selector does not select two renderers or two backwards.** Both modes
   execute the same WGSL projection, fixed-array-order compositing, sampled-ray
   tape, gradient reduction, and Adam update. The only forward difference is a
   conditional harmonic center offset, and only that offset's gradient is
   conditional.
4. **The browser does have shared parameter gradients, but not STAR's claimed
   shared work.** One workgroup evaluates all 768 splats for one sampled ray,
   records an alpha/compositing tape, and the update kernel sums a dense
   `[sample, splat]` gradient array. This is a useful compact sampled-ray VJP.
   It neither compiles a trace once for many times nor shares projection,
   support, binning, visibility, or adjoint work across camera-time samples.

This agrees with the canonical taxonomy: the browser is a simplified demo and
"World Tubes-style" is explicitly not native parity
(`research_notes/renderer_lane_taxonomy.md:257-267`).

## Sources Audited

- Browser model, forward, VJP, optimizer, preview:
  `web/dynaworld_browser_trainer/trainerWebGpu3d.js:44-69,133-457,529-760`.
- Browser labels: `web/dynaworld_browser_trainer/index.html:27-28` and
  `web/dynaworld_browser_trainer/app.js:218-220`.
- Canonical lane definitions and mathematical fork:
  `research_notes/renderer_lane_taxonomy.md:39-52,71-95,257-267`.
- World-to-UVT projection contract:
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py:47-72,96-133,164-235`.
- UVT parameterization:
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/model.py:98-135,306-329`.
- STAR UVT tiled forward, visibility handling, and direct backward:
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal:126-135,191-195,1498-1515,1874-1950,3312-3451`.
- Projective trace evaluation:
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal:202-235`.
- Standard dynamic 3DGS state and objective:
  `research_experiments/gauge_fields/train_splat_baseline.py:144-224,327-345,389-421`.
- fast-mac tiled forward/backward:
  `third_party/fast-mac-gsplat/variants/v5/csrc/metal/gsplat_v5_kernels.metal:119-136,412-473,476-607`.
- Paper lane kernel identities:
  `research_experiments/paper_runner_suite/run_unified_paper_ablation.py:240-273`.

## Current Browser Equations

For primitive `i`, normalized time `t in [0,1]`, and
`tau = 2t - 1`, the browser stores 16 floats:

```text
x_i         in R^3       base world center
r_i         in R         isotropic world radius
v_i         in R^3       linear world velocity
t0_i        in [0,1]     temporal gate center
h_i         in R^3       one-cycle harmonic world offset
m_i         in [0,1]     static mixture (initialized 0.92, never optimized)
c_i         in R^3       RGB, directly clamped rather than logits
l_i         in R         opacity logit
```

The mode-dependent center is exactly:

```text
x_i(t; mode=dynamic) = x_i + v_i tau
x_i(t; mode=tubes)   = x_i + v_i tau + h_i sin(2 pi t)
```

The modes are initially identical because `h_i = 0`. They diverge only because
the tubes-style branch receives gradients for `h_i`; the dynamic branch does
not. Switching the UI mode does not reinitialize parameters or Adam moments, so
an interactive switch is not an isolated A/B.

For calibrated world-to-camera rows `R|d`, normalized intrinsics
`(fx,fy,cx,cy)`, and camera point `(X,Y,Z)`:

```text
(X,Y,Z) = R x_i(t) + d
mu_i(t,k) = (fx X/Z + cx, fy Y/Z + cy)
rho_i(t,k) = clamp(r_i fy/Z, rho_min, rho_max)
```

For normalized sample `p=(u,v)` and image aspect `a=W/H`:

```text
D_i = a^2 (u-mu_x)^2 + (v-mu_y)^2
G_i = exp(-0.5 D_i / rho_i^2)
o_i = sigmoid(l_i)
f(sigma) = clamp(0.30 sigma, 0.035, 0.12)
E_i(t) = exp(-0.5 (t-t0_i)^2 / sigma^2)
g_i(t) = m_i + (1-m_i) [f + (1-f) E_i(t)]
alpha_i = o_i G_i g_i(t), when D_i <= 9 rho_i^2; otherwise 0
```

Because `m_i` is initialized to `0.92` and its gradient slot is repurposed as a
statistic then zeroed before Adam, the temporal gate starts as:

```text
g_i(t) = 0.92 + 0.08 [f + (1-f) E_i(t)].
```

Thus every primitive has at least about 92% temporal support. The UI's
"temporal support" sigma controls only the remaining 8% at initialization. This
is materially different from a UVT Gaussian's learned temporal precision and
helps explain why the two displayed modes can look nearly identical.

### Browser compositing

The training kernel traverses primitive buffer order, not camera depth:

```text
C_0 = 0
T_0 = 1
C_{i+1} = (1-alpha_i) C_i + alpha_i c_i
T_{i+1} = (1-alpha_i) T_i
L_rgb = ||C_N - C_target||_2^2 / 3
```

This recurrence is source-over in draw order. It equals conventional
front-to-back splatting only if the buffer happens to be ordered back-to-front.
The SfM seed order is not a visibility order and changes with camera/time, so
the browser has no valid general occlusion ordering. The preview also draws
instances in fixed buffer order through hardware alpha blending.

The sampled objective additionally applies:

```text
L_motion_guard = w_m max(0, target_coverage - (1-T_N))^2
L_static_alpha = w_s sum_i alpha_i^2       on static-selected rays
```

but `sampleLosses` reports only `L_rgb`, not these gradient-producing terms.
The displayed sampled loss is therefore not the full optimized objective.

### Browser backward

For each sampled ray, the kernel stores `C_i`, `alpha_i`, and the suffix
transmittance after primitive `i`. Its compositing derivatives are structurally
the reverse-mode derivatives of the browser's own fixed-order recurrence:

```text
dL/dalpha_i = <dL/dC_N, (c_i - C_i) prod_{j>i}(1-alpha_j)>
dL/dc_i     = dL/dC_N alpha_i prod_{j>i}(1-alpha_j)
```

Each workgroup writes one full gradient record per `(sample, splat)`, then the
update kernel computes:

```text
grad_i = sum_s grad_{s,i}
theta_i <- Adam(theta_i, grad_i)
```

This costs `O(S*N)` parameter evaluations and `O(S*N*16 floats)` gradient tape
for `S` sampled rays and `N=768` splats. It is independent of full raster
resolution, which is valuable, but it is not STAR's tiled UVT shared backward.

### Browser VJP omissions and mismatches

The VJP is not the exact derivative of even the simplified browser forward:

1. `d alpha / d t0` uses `g_i(t)` where the exact Gaussian contribution is
   `(1-m_i)(1-f)E_i(t)`. It therefore leaks the temporal floor and static mass
   into the `t0` gradient.
2. `m_i` has no gradient and never changes.
3. `sigma` is a UI constant, not a learned precision, and has no gradient.
4. The center gradient differentiates projected center but omits the path
   `x_i -> Z -> rho_i -> alpha_i`. Perspective footprint changes therefore do
   not fully update world position.
5. The radius derivative is applied even when projected radius was clamped;
   the exact derivative through `clamp` is zero at either bound.
6. The hard `3 sigma` support boundary is treated as constant, as in a typical
   raster active-set approximation, but there is no support-staleness guard or
   fallback certificate.
7. There are no derivatives for anisotropic covariance, orientation, depth
   order, visibility changes, camera parameters, or density topology because
   those quantities do not exist in this browser model.

## Faithful World Tubes / STAR UVT Contract

The canonical method first maps a world primitive into a reusable sensor-time
trace. For affine UVT, with `y=(u,v,t)`, mean `m_i`, and symmetric positive
precision `Q_i`:

```text
delta_i(y) = y - m_i
q_i(y) = delta_i^T Q_i delta_i
alpha_i(y) = min(alpha_max, opacity_i exp(-0.5 q_i(y)))
depth_i(y) = depth0_i + beta_i^T delta_i(y)
```

For a linear image-plane trajectory `(vu,vv)` with spatial precision
`[[lu,luv],[luv,lv]]` and residual temporal precision `lt`, checked-in code
constructs:

```text
Q_i = [ lu,   luv,  -(lu vu + luv vv)
        luv,  lv,   -(luv vu + lv vv)
        *,    *,     lt + lu vu^2 + 2 luv vu vv + lv vv^2 ]
```

This makes the quadratic equivalent to a moving spatial Gaussian plus residual
temporal support:

```text
q_i(u,v,t)
  = [du-vu dt, dv-vv dt]^T Lambda_uv [du-vu dt, dv-vv dt]
    + lt dt^2.
```

For a world tube, `project_world_tubes_pinhole` projects world center and
velocity, propagates spatial covariance through the pinhole Jacobian, inverts
the resulting 2D covariance, and forms the UVT `Q_i`. The projective extension
stores homogeneous quadratic traces:

```text
hu(t)=au0+au1 t+au2 t^2
hv(t)=av0+av1 t+av2 t^2
hz(t)=az0+az1 t+az2 t^2
u(t)=hu(t)/hz(t), v(t)=hv(t)/hz(t), depth(t)=hz(t), |hz| > eps.
```

The method boundary in the taxonomy is earlier ray-fiber marginalization:

```text
T_i(y) = pi_* Gamma^* w_i
S_i = H_yy - H_yz H_zz^-1 H_zy.
```

The trace compiler must additionally retain conditional depth/variance and
support, denominator, near-plane, visibility/order, interval, and fallback
certificates. STAR bins traces into UVT tiles, depth-sorts stable tiles, detects
order instability, and uses per-sample ordering or configured fallback for
unstable regions. Its direct and deterministic backwards differentiate the
depth-ordered transmittance recurrence into `m`, `Q`, opacity, and color, then
reduce contributions per tube.

### What the browser shares with World Tubes

- A world-space base center and velocity can describe a linear world tube.
- Parameters are shared over time rather than duplicated per frame.
- Pinhole projection and alpha compositing are present in simplified form.
- A temporal Gaussian-like gate exists.
- The sampled-ray tape has the same generic alpha-compositing reverse-mode
  identity used by many splat renderers.

### What differs or is missing

- No fiber pushforward or Schur-complement marginalization.
- No affine/projective UVT trace representation (`m,Q,depth0,beta`).
- No projection compiler and no camera-family basis sharing.
- No anisotropic UV covariance or learned UVT precision.
- No conditional depth, depth variance, depth sort, order strata, or fallback.
- No UVT support bounds, tile bins, interval atlas, active windows, overflow
  handling, denominator certificates, exposure, or rolling shutter.
- No STAR VJP into `m,Q,opacity,color`, and no chain rule back to world-tube or
  camera-program parameters.
- No cross-time reuse: every sampled `(camera,time,pixel)` reprojects all
  primitives from world space.

Conclusion: **the browser tubes mode shares a trajectory intuition, not the
World Tubes representation or shared renderer/backward contract.**

## Faithful Standard Dynamic 3DGS Contract In This Repository

The checked-in paper baseline is `FreeDynamic3DGS` with `splat_mode=per_frame`.
For each frame `t` and primitive `i`, it stores independent:

```text
mu_{t,i} in R^3             world mean
s_{t,i} in R_+^3            anisotropic scales
q_{t,i} in unit quaternions orientation
o_{t,i} in (0,1)            opacity
c_{t,i} in (0,1)^3          RGB
```

Thus its storage is `T*N` Gaussian states, not one trajectory state per
primitive. fast-mac projects the full 3D covariance
`Sigma=R(q) diag(s^2) R(q)^T` through the camera Jacobian to a 2D conic,
bins candidates into image tiles, sorts by depth, and composites:

```text
alpha_i(p) = min(alpha_max,
                 o_i exp(-0.5 (p-mu2d_i)^T C_i (p-mu2d_i)))
C <- C + T alpha_i c_i
T <- T (1-alpha_i).
```

The fast-mac backward reverses this same ordered recurrence and accumulates
gradients for 2D means, conics, colors, and opacity; autograd continues through
projection to 3D means, scales, and quaternions.

The baseline objective is robust RGB L1 plus scale and adjacent-frame state
regularization:

```text
L = robust_L1(render,target)
  + lambda_scale mean(exp(log_scale))
  + lambda_time [MSE(delta mu)
                 + 0.1 MSE(delta log_scale)
                 + 0.01 MSE(delta opacity_logit)
                 + 0.01 MSE(delta rgb_logit)].
```

### What the browser shares with dynamic 3DGS

- World-space means are projected through calibrated pinhole cameras.
- RGB, opacity, a footprint size, and alpha compositing are optimized.
- Adam updates the parameters from image reconstruction gradients.

### What differs or is missing

- Browser has one shared trajectory per primitive; baseline has independent
  Gaussian state per frame.
- Browser has one isotropic radius; baseline has 3D anisotropic scales and
  quaternion orientation.
- Browser alpha has a temporal gate absent from per-frame 3DGS.
- Browser uses fixed buffer order; baseline uses camera-depth order.
- Browser samples rays and scans all splats; fast-mac tiles full images and
  scans only binned candidates.
- Browser trains RGB MSE plus support heuristics; baseline trains robust L1 plus
  scale and temporal state smoothness.
- Browser has fixed 768 primitives and no active-count schedule or densification;
  the paper protocol can match stage-wise primitive counts and raster budgets.

Conclusion: **the browser dynamic mode is a linear-trajectory isotropic dynamic
splat model, not the repository's standard per-frame dynamic 3DGS baseline.**

## Exact Current Mode Difference

Holding the shared parameter buffer fixed, the complete intended mode delta is:

| Component | Tubes-style (`mode=0`) | Dynamic-style (`mode=1`) |
| --- | --- | --- |
| World center | `x + v tau + h sin(2 pi t)` | `x + v tau` |
| Harmonic gradient | `dL/dh = dL/dx(t) sin(2 pi t)` | zero |
| Temporal gate | same | same |
| World projection | same | same |
| Footprint | same isotropic radius | same isotropic radius |
| Visibility/order | same fixed parameter order | same fixed parameter order |
| Sampled objective | same | same |
| Backward/tape | same | same |
| Adam state/update | same | same |
| Initialization | same SfM subsample | same SfM subsample |

The honest current UI names would therefore be "linear + harmonic trajectory"
and "linear trajectory". Calling this result a World Tubes versus dynamic 3DGS
ablation would be false.

## Smallest Honest Matched Browser A/B Plan

The smallest defensible A/B should compare **temporal representations while
holding camera/data, primitive budget, raster semantics, loss, optimizer,
sampling, initialization, and reporting fixed**. It should not attempt full
projective interval compiler parity in the first patch.

### Gate 0: fix the shared raster contract first

Before comparing representations, both lanes need the same correct image
operator:

1. Add camera-depth keys at each selected time and sort active primitive IDs
   front-to-back (or explicitly reverse the current source-over recurrence).
2. Use one shared alpha/transmittance VJP for both lanes and finite-difference
   it on a tiny ordered 3-splat, 2-time, 2-camera fixture.
3. Add the missing perspective-radius path and correct `t0` gate derivative.
4. Report the full objective separately from RGB reconstruction loss.
5. Freeze mode choice at reset and use separate parameter/moment buffers per
   run; never treat a mid-run selector change as an A/B.

Without Gate 0, any quality difference is confounded by invalid visibility and
an inexact VJP.

### Lane A: minimal affine STAR UVT / World Tubes browser lane

Use the checked-in affine trace contract directly:

```text
parameters per tube: m[3], Q[6] or factored (Lambda_uv, velocity_uv, lambda_t),
                     depth0, beta[3], opacity, RGB
sample: y=(pixel_x+0.5, pixel_y+0.5, centered_frame_time)
alpha: opacity exp(-0.5 (y-m)^T Q (y-m))
depth: depth0 + beta^T(y-m)
```

Implement UVT support bounds and a compact active-ID list for sampled rays.
Depth-sort those IDs at the sampled `y`, composite, and differentiate exactly
into `m,Q,depth/ordering-held-fixed,opacity,RGB`. For the first honest A/B,
visibility membership and sort order may be treated as a compiled active set,
provided the UI and note explicitly say **affine STAR UVT subset** and report
support/order changes. Projective camera traces, interval certificates, and
fallback are follow-up parity gates, not silently omitted features.

Initialize the affine traces by projecting the same SfM points and zero world
velocities into each train-camera chart. Because one UVT trace is camera-chart
specific, the minimal multicamera implementation must either:

- compile one trace family from shared world-tube parameters for each calibrated
  camera, then reduce all chart gradients back to shared world parameters; or
- label the first gate as a single-camera affine STAR comparison.

For the existing multicamera demo, the first option is the honest one.

### Lane B: minimal standard per-frame dynamic 3DGS browser lane

Store `T*N` states with the checked-in baseline fields:

```text
mu[T,N,3], log_scale[T,N,3], quat[T,N,4], opacity_logit[T,N,1], rgb_logit[T,N,3].
```

At sampled time `t`, project only that frame's `N` Gaussians, propagate the 3D
anisotropic covariance to a 2D conic, depth-sort active IDs, and use the same
shared compositing/VJP contract. Apply the repository's adjacent-frame
smoothness equation and scale regularizer. This is the smallest browser model
that deserves the "standard dynamic 3DGS" label.

### Matching contract

Use the existing canonical browser bundle adapter and paper-protocol semantics;
do not create another camera/split stack. Pin:

```text
dataset/split: Coffee Martini; cam04 + cam09 train, cam06 held out
times:         same exported eight exact source times
initial XYZ:   same SfM point subset and deterministic seed
active budget: same N per rasterized time
sampling:      same ordered list of (view,time,pixel) samples per optimizer step
loss:          same RGB term for primary comparison; regularizers reported apart
optimizer:     same Adam beta/epsilon and per-field LR policy
steps/time:    report both fixed optimizer steps and fixed wall-clock
metrics:       train and heldout PSNR, SSIM, L1/MSE; full objective separately
cost:          parameters, optimizer bytes, active pairs, sampled rays, wall time
```

Two budgets are necessary because the representations intentionally differ in
storage and sharing:

1. **Same active primitive count per rendered time:** measures rendering and
   fitting behavior at equal visible capacity.
2. **Same total parameter/optimizer bytes:** measures the benefit of temporal
   sharing against the `T*N` per-frame baseline.

Do not call equal `N` alone a same-capacity comparison: per-frame dynamic 3DGS
stores approximately `T` times as many states.

### Acceptance labels

Use these labels until stronger gates pass:

```text
Current code:       harmonic trajectory splats vs linear trajectory splats
After minimal port: affine STAR UVT subset vs per-frame dynamic 3DGS subset
After parity gates: World Tubes / projective STAR UVT vs dynamic 3DGS / fast-mac
```

The final label requires projective trace/camera-family lowering, support and
order certificates, unstable-region fallback, and gradient comparison to Metal
fixtures for World Tubes; and anisotropic projected-conic plus depth-ordered
forward/backward comparison to fast-mac fixtures for dynamic 3DGS.

## Cheap Falsification And Parity Tests

1. **Mode identity test:** reset with `h=0`, run both current modes without
   updates, and require bitwise-equal sampled predictions. A mismatch indicates
   an undocumented mode branch.
2. **Temporal gate finite difference:** compare analytic and central-difference
   gradients for `t0`, `m`, position `z`, and radius inside and at clamp bounds.
   The current `t0` and `z -> radius` checks should fail, confirming this audit.
3. **Order permutation test:** render two overlapping splats at distinct depths,
   permute storage order, and require the depth-sorted result to remain fixed.
   The current browser should fail.
4. **STAR affine fixture:** export tiny `m,Q,depth0,beta,opacity,color` tensors,
   identical sample points, and upstream RGB gradients to Metal and WGSL;
   compare forward RGB/alpha and all continuous parameter gradients.
5. **fast-mac fixture:** export tiny 3D means/scales/quaternions/opacities/colors
   and cameras; compare projected means/conics/depth, forward RGB/alpha, and
   parameter gradients between fast-mac and WGSL.
6. **Cross-time sharing test:** increase selected frame count at fixed sampled
   rays. A true compiled trace route should not redo world-to-camera projection
   and support construction independently for every frame; current browser
   work grows directly with sampled camera-time rays.
7. **A/B isolation test:** assert each lane owns separate parameters, optimizer
   moments, RNG/sample stream cursor, and reset seed; compare from step zero.

## Decision

Do not add a second marketing-level mode toggle on top of the current kernel.
First relabel the existing selector as the harmonic trajectory ablation, or
hide it behind an "experimental trajectory" label. The smallest meaningful
technical next step is Gate 0 plus tiny Metal/WGSL parity fixtures. Only then
build the affine STAR UVT subset and per-frame anisotropic dynamic 3DGS subset
behind separate parameter layouts while preserving the canonical browser data
adapter.

Current confidence: high on the representation and backward mismatch because
the relevant equations and dispatch are explicit in code. Medium on the exact
smallest performant WebGPU data structure because candidate-list construction,
sorting strategy, and browser atomic capabilities need a measured prototype.
