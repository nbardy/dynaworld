# Browser Anisotropic 3D Gaussian Design

Date: 2026-07-23 00:24:30 Asia/Seoul

Status: design-only note. No runtime files were edited. This is the smallest
mathematically honest extension of the active sampled-ray browser trainer from
world-space isotropic spheres to anisotropic 3D Gaussians. It does not claim
parity with fast-mac, Dynamic 3DGS, or World Tubes, and it does not repair the
existing fixed storage-order visibility approximation.

## Decision

Represent each primitive by a positive-definite covariance in world space:

```text
Sigma_w = R(q) diag(exp(2 ell_x), exp(2 ell_y), exp(2 ell_z)) R(q)^T.
```

Project `Sigma_w` through the calibrated camera and the local Jacobian of the
pinhole map. Train and render the resulting 2D conic. Store three log-scales
and one unit quaternion, optimize both analytically, and renormalize the
quaternion after each Adam step.

Explicit rejection: **do not add a learned screen-space `(sigma_x, sigma_y,
angle)` to each splat.** Such an ellipse has no camera-independent world-space
meaning. It can fit one view while producing unrelated footprints in another,
so it is fake anisotropy for this calibrated multicamera trainer. A projected
ellipse is valid only when it is derived from one shared 3D covariance and the
selected camera.

## Scope And Honesty Boundary

This design makes the primitive footprint and its derivatives honest under the
first-order perspective projection used by standard Gaussian splatting. It
retains these active-trainer approximations:

- the Gaussian support set is detached at the `3 sigma` boundary;
- visibility/order changes are detached;
- the current fixed parameter order remains wrong for general multicamera
  visibility until a separate depth-sorted compositor lands;
- temporal motion changes the mean but not covariance or orientation;
- RGB remains degree-zero and view independent;
- the pinhole Jacobian is evaluated at the Gaussian mean rather than integrating
  the exact projective image of a finite 3D Gaussian.

Therefore the correct claim is "world-space anisotropic Gaussian footprint
with analytic sampled-ray VJP," not "full 3DGS parity."

## Current-To-Proposed State ABI

The current state is 16 floats. The proposed state is 24 floats, aligned as six
`vec4<f32>` records:

```text
struct Splat {
    centerStatic: vec4<f32>,   // xyz = base world mean, w = static mixture m
    velocityTime: vec4<f32>,   // xyz = linear world velocity, w = gate center t0
    harmonicPad: vec4<f32>,    // xyz = harmonic world offset, w unused
    logScalePad: vec4<f32>,    // xyz = ell = log world standard deviations, w unused
    rotation: vec4<f32>,       // raw quaternion r = (x,y,z,w)
    colorOpacity: vec4<f32>,   // rgb and opacity logit
};
```

This preserves all active parameters while replacing one scalar radius with
three log-scales and four quaternion components. Padding to 24 rather than 22
floats keeps every field naturally aligned and lets moments and sample-gradient
buffers reuse the same struct.

Forward always uses:

```text
q = r / max(||r||, epsilon_q).
```

Initialize `r=(0,0,0,1)`. The exact compatibility initialization for an old
radius `rho` is:

```text
ell_x = ell_y = ell_z = log(max(rho, sigma_world_min)).
```

With identity rotation this reproduces the old sphere before antialiasing and
clamp differences. Use this initialization for the first equivalence test.
Only after parity should exported local-neighborhood PCA initialize anisotropic
scales and orientation.

## Coordinate Convention

The dataset stores normalized image coordinates. Let:

```text
a = image_width / image_height
y = (a u, v)
```

The transformed coordinate `y` measures both axes in image-height units. It
replaces the current ad hoc distance `a^2 du^2 + dv^2` with a full covariance
in the same metric.

For camera world-to-camera rotation `R_c`, translation `d_c`, normalized
intrinsics `(f_x, f_y, c_x, c_y)`, and time-dependent world mean `mu_w(t)`:

```text
p = (X,Y,Z) = R_c mu_w(t) + d_c
mu = (a (f_x X/Z + c_x), f_y Y/Z + c_y).
```

Reject `Z <= near`. The Jacobian from camera coordinates to `y` is:

```text
A = a f_x
B = f_y

J = [ A/Z,   0,   -A X/Z^2
        0, B/Z,   -B Y/Z^2 ].
```

## World And Camera Covariance

Define positive world standard deviations:

```text
s_k = exp(ell_k)
d_k = s_k^2 = exp(2 ell_k)
D = diag(d_x, d_y, d_z)
Q = R(q)
Sigma_w = Q D Q^T
Sigma_c = R_c Sigma_w R_c^T.
```

The projected 2D covariance is:

```text
C = J Sigma_c J^T + sigma_filter^2 I_2.
```

`sigma_filter` is an antialias floor in image-height units, for example
`0.3 / image_height`. It must be a named constant/config field and must match
train, CPU validation, and render. It guarantees positive definiteness even
for tiny projected Gaussians.

For symmetric:

```text
C = [c00 c01; c01 c11]
det = c00 c11 - c01^2
K = C^-1 = (1/det) [c11 -c01; -c01 c00].
```

Reject or conservatively regularize a primitive if `det` is non-finite or
below a scale-aware threshold. Do not silently replace `K` with a screen-space
axis-aligned fallback during training; that would change the represented
primitive.

## Sampled-Ray Forward

For sample `y_s=(a u_s,v_s)`:

```text
delta = y_s - mu
qform = delta^T K delta
G = exp(-0.5 qform)
o = sigmoid(opacity_logit)
g = existing temporal gate
alpha = o g G.
```

The detached active set is:

```text
qform <= 9.
```

This is the anisotropic `3 sigma` ellipse. The compositing recurrence and its
existing `bar_alpha` can remain unchanged for this isolated extension. Clamp
`alpha` only if the forward, backward, CPU validation, and render all use the
same cap; if capped, its local derivative is zero on the saturated branch.

## Analytic Backward Through The Conic

Let `bar_x` mean `dL/dx`. After the compositor and regularizers produce
`bar_alpha`, for an uncapped active Gaussian:

```text
bar_qform = -0.5 bar_alpha alpha
v = K delta
bar_delta = 2 bar_qform v
bar_mu = -bar_delta
bar_K = bar_qform delta delta^T.
```

Since `K=C^-1`:

```text
bar_C = -K^T bar_K K^T
      = -bar_qform v v^T.
```

Because `C=J Sigma_c J^T + sigma_filter^2 I` and all covariance adjoints are
symmetrized before use:

```text
bar_Sigma_c = J^T bar_C J
bar_J = 2 bar_C J Sigma_c.
```

Numerically enforce:

```text
bar_C       = 0.5 (bar_C + bar_C^T)
bar_Sigma_c = 0.5 (bar_Sigma_c + bar_Sigma_c^T).
```

### Mean And Perspective-Jacobian Path

The mean path contributes:

```text
bar_X += bar_mu_x A/Z
bar_Y += bar_mu_y B/Z
bar_Z += -bar_mu_x A X/Z^2 - bar_mu_y B Y/Z^2.
```

Let `G=bar_J`. Only `J00`, `J02`, `J11`, and `J12` depend on `(X,Y,Z)`:

```text
bar_X += G02 (-A/Z^2)
bar_Y += G12 (-B/Z^2)
bar_Z += G00 (-A/Z^2) + G02 (2 A X/Z^3)
       + G11 (-B/Z^2) + G12 (2 B Y/Z^3).
```

This covariance-through-depth term is required. Omitting it repeats the
current scalar-radius bug where world `Z` receives only the projected-center
path.

Map camera-mean gradient back to world:

```text
bar_mu_w = R_c^T bar_p.
```

Then reuse the existing trajectory chain:

```text
bar_base     += bar_mu_w
bar_velocity += (2t-1) bar_mu_w
bar_harmonic += sin(2 pi t) bar_mu_w  // harmonic mode only.
```

### Covariance To Log-Scales

Write:

```text
Bq = R_c Q
Sigma_c = Bq D Bq^T.
```

For column `b_k` of `Bq`:

```text
bar_ell_k = 2 d_k b_k^T bar_Sigma_c b_k.
```

This derivative includes the `s=exp(ell)` chain and keeps scales positive
without post-update absolute values.

### Covariance To Rotation

First:

```text
bar_Bq = 2 bar_Sigma_c Bq D
bar_Q = R_c^T bar_Bq.
```

For normalized quaternion `q=(x,y,z,w)`, use:

```text
Q = [1-2(y^2+z^2), 2(xy-zw),     2(xz+yw)
     2(xy+zw),       1-2(x^2+z^2), 2(yz-xw)
     2(xz-yw),       2(yz+xw),     1-2(x^2+y^2)].
```

Given `H=bar_Q`, the Euclidean gradient with respect to normalized `q` is:

```text
g_x = -4x(H11+H22) + 2y(H01+H10) + 2z(H02+H20) + 2w(H21-H12)
g_y = -4y(H00+H22) + 2x(H01+H10) + 2z(H12+H21) + 2w(H02-H20)
g_z = -4z(H00+H11) + 2x(H02+H20) + 2y(H12+H21) + 2w(H10-H01)
g_w =  2z(H10-H01) + 2y(H02-H20) + 2x(H21-H12).
```

If the stored raw quaternion is `r`, `q=r/||r||`, then:

```text
bar_r = (I - q q^T) g / max(||r||, epsilon_q).
```

This tangent projection is not optional: treating normalized `q` as four
independent values gives the wrong derivative.

### Existing Opacity And Color Paths

The existing paths remain:

```text
bar_color = bar_output_color * alpha * suffix_transmittance
bar_logit = bar_alpha * g * G * o (1-o).
```

Temporal-gate derivatives should use the already corrected dynamic Gaussian
core, independently of covariance.

## Optimizer And Update Contract

Adam gains moment slots for `ell.xyz` and `rotation.xyzw`. Suggested first
implementation:

```text
ell_new = clamp(Adam(ell, bar_ell), ell_min, ell_max)
r_trial = Adam(r, bar_r)
q_new = normalize_or_identity(r_trial)
```

Store `q_new` back into the rotation field. This is an ambient Adam step plus a
unit-sphere retraction. It is not a Lie-group optimizer, but it is a standard,
small, mathematically coherent update for this prototype.

Do not clamp individual quaternion components. Do not optimize a 3x3 matrix
and Gram-Schmidt it. Do not canonicalize quaternion sign every step unless the
first and second moments are transformed consistently; `q` and `-q` represent
the same rotation, but flipping stored coordinates under unchanged moments
creates an optimizer discontinuity.

Use a separate scale LR. Start below position LR, then tune from gradient/update
quantiles. A safe debugging invariant is:

```text
max(abs(delta ell)) <= 0.01 per step
angle(q_new, q_old) <= 0.01 rad per step
```

These are guards for first smokes, not final hyperparameters.

The old scalar radius moment cannot be migrated meaningfully into three scale
moments. A state-layout change should reinitialize Adam moments and restart the
run.

## Render Path

The vertex shader must derive the ellipse from the same `C` as training. For a
2x2 SPD covariance, a stable Cholesky factor is enough:

```text
l00 = sqrt(max(c00, epsilon))
l10 = c01 / l00
l11 = sqrt(max(c11 - l10^2, epsilon))
L = [l00 0; l10 l11]
C = L L^T.
```

For unit quad coordinate `z in [-1,1]^2`, form a `3 sigma` covering
parallelogram:

```text
offset_metric = 3 L z
offset_uv = (offset_metric.x / a, offset_metric.y)
```

Convert `offset_uv` to NDC in the existing vertex path. Pass
`local = 3 z` to the fragment shader and evaluate:

```text
alpha = opacity * temporal_gate * exp(-0.5 dot(local,local)).
```

This works because screen offset is `L local`, so its Mahalanobis norm is
`dot(local,local)`. The parallelogram overdraws corners but the fragment value
is exact for the projected covariance. An eigenvector-aligned rectangle is not
required for correctness and is more expensive.

Train, CPU validation, and render must share:

- quaternion convention and normalization;
- `sigma_filter`;
- aspect-height coordinate transform;
- near plane;
- determinant guard;
- `3 sigma` support;
- alpha cap semantics.

## Memory And Performance Implications

State and sample-gradient records grow from 16 to 24 floats, a 50% increase.
At 768 splats and 96 samples:

```text
old sample-gradient tape = 96 * 768 * 64 bytes  = 4.72 MB
new sample-gradient tape = 96 * 768 * 96 bytes  = 7.08 MB.
```

Parameter ping-pong and two Adam moment buffers remain small. Workgroup alpha
and suffix tapes need not grow. The likely cost is register pressure and conic
math, not storage capacity.

Prediction for the current all-pairs kernel:

- training: 15-35% slower at fixed samples/splats;
- preview: 5-20% slower, depending on fragment overdraw;
- CPU validation: 2-4x slower unless conics are cached per camera/time/splat.

Cache `(mu, K, depth, bounds)` for validation across its pixel grid. Do not
cache across optimizer snapshots.

## Initialization Plan

Phase 1, exact compatibility:

1. Convert every old scalar radius to three equal log-scales.
2. Set quaternion identity.
3. Hold scale and rotation LR at zero.
4. Require sphere-forward and sphere-backward parity.

Phase 2, trainability:

1. Enable scale gradients with rotation frozen.
2. Enable quaternion gradients after scale finite differences pass.
3. Keep the old SfM centers and RGB.
4. Lower initial opacity/radius only in a separate ablation.

Phase 3, geometry-aware initialization:

1. Compute local 3D covariance from train-visible PLY neighbors.
2. Clamp eigenvalues in world units before taking logs.
3. Convert eigenvectors to a quaternion with a deterministic handedness.
4. Compare against isotropic compatibility init at matched count and wall time.

Do not infer per-splat orientation independently from one camera image. That is
the rejected screen-space shortcut in another form.

## Implementation Checklist

1. Add a standalone CPU reference for quaternion-to-covariance, projection,
   conic evaluation, and VJP before changing WGSL.
2. Define the 24-float ABI once and update parameter, moment, gradient, and
   readback byte counts from that constant.
3. Add WGSL helpers for safe quaternion normalization, rotation matrix,
   covariance projection, symmetric 2x2 inverse, and conic evaluation.
4. Replace scalar-radius support with `delta^T K delta <= 9` in training.
5. Add analytic `bar_C`, `bar_J`, perspective-depth, log-scale, and quaternion
   paths to the sampled-ray VJP.
6. Add scale/quaternion moments and quaternion retraction to the update shader.
7. Render Cholesky-oriented billboards from the same projected covariance.
8. Port the same math to CPU validation and preview-error evaluation.
9. Keep fixed-order compositing behavior unchanged for this isolated A/B, but
   retain the separate depth-order correctness blocker in the UI/docs.
10. Benchmark kernel-only and live-worker throughput at 768 splats and 96
    samples before changing count or validation cadence.

## Required Tests

### Algebraic tests

- SPD: projected `C` has positive eigenvalues for random finite cameras,
  centers, scales, and quaternions.
- Isotropic rotation invariance: equal scales produce identical `C` for every
  quaternion.
- Quaternion sign invariance: `q` and `-q` produce identical forward values.
- Camera equivariance: jointly rotating world mean/covariance and camera leaves
  the projected conic unchanged.

### Finite-difference tests

- all three log-scales;
- all four raw quaternion components away from zero norm;
- world `x,y,z`, specifically checking the `z -> J -> C -> alpha` path;
- opacity logit and temporal gate remain unchanged from their reference;
- points near but not on the detached `qform=9` boundary.

Use central differences in float64 CPU code first. Compare WGSL against that
reference on saved tiny fixtures.

### Compatibility tests

- Equal scales plus identity quaternion reproduce the old scalar Gaussian in
  aspect-height coordinates.
- CPU, WGSL train forward, and render agree on `mu`, `C`, `K`, `qform`, and
  alpha for the same primitive/camera/time/sample.
- A two-camera fixture proves one shared 3D covariance projects to different,
  geometrically consistent ellipses.
- A deliberately fitted screen-space ellipse must fail that two-camera
  consistency test; preserve this as a regression test against fake
  anisotropy.

### Runtime gates

- one-step worker smoke with finite parameters and moments;
- 100-step synthetic anisotropic target with decreasing loss;
- sphere compatibility run with scale/rotation frozen;
- Coffee Martini clean-start 768-splat matched-wall-time A/B;
- parameter quantiles: scales, anisotropy ratio, quaternion norm, update angle,
  determinant, projected major/minor radius, active count, and alpha coverage.

## Failure Modes And Responses

### Quaternion does not learn

Possible cause: scales remain nearly equal, making rotation unidentifiable.
This is mathematically expected. Inspect anisotropy ratios before increasing
rotation LR.

### Scale explodes while loss falls

Possible cause: broad alpha coverage is compensating for missing capacity or
wrong visibility order. Tighten log-scale bounds and inspect heldout/error maps;
do not interpret larger ellipses as successful geometry.

### Tiny determinant or NaN conics

Possible cause: extreme scales, invalid quaternion normalization, near-plane
means, or insufficient filter covariance. Fail the fixture, log the primitive,
and fix the source. Do not silently switch to an axis-aligned ellipse.

### Render and training diverge

Possible cause: one path uses normalized `(u,v)` while the other uses the
aspect-height metric, or Cholesky/local coordinates are transposed. Compare
saved `C`, `L`, and alpha values before touching optimizer rates.

### Quality does not improve

Anisotropy may not be the current dominant blocker. Fixed visibility order,
only 768 primitives, frozen temporal/static allocation, and no density control
remain independent ceilings. The correct negative conclusion would be that
anisotropic footprint capacity alone is insufficient, not that screen-space
fake anisotropy should replace it.

## Acceptance Boundary

Call this extension complete only when:

1. CPU finite differences validate center, log-scale, and raw-quaternion VJPs.
2. Equal-scale compatibility matches the old sphere within float tolerance.
3. Train, validation, and render use one covariance/conic convention.
4. Quaternion norms remain finite and near one after optimizer steps.
5. A two-camera test demonstrates camera-consistent projected anisotropy.
6. The result is still described as a browser prototype with fixed-order
   visibility, not as Dynamic 3DGS or World Tubes parity.
