# World-tube rotation audit and minimal curved-motion model

**Time:** 2026-07-23 13:13:18 +0900

**Context:** Follow-up audit of Codex task
`019f8956-1f8b-74c3-ab08-e15c57eabb6a` ("Polishing world tubes").
The user asked whether that task completed, whether it wrote into DynaWorld,
whether its formulas were sound, whether natively curved tubes were tackled,
and whether time-varying covariance rotation was the simpler missing model.

## Evidence inspected

- The completed task turns and their file-change records.
- `research_notes/spacetime_gaussian_representation/04_formulation_catalog.md`.
- `research_notes/spacetime_gaussian_representation/06_shader_boundary_and_depth_fiber.md`.
- `research_notes/spacetime_gaussian_representation/07_concern_status_and_build_plan.md`.
- `agent_notes/loose_notes/2026-07-23_02-42-32_spd4_build_and_status_plan.md`.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py`.
- Repository search for time-varying world covariance, rotation, spline, and
  curved-center implementations.

No GPU workload or benchmark was launched. This is a source-and-math audit.

## Observed status

The task completed its requested audit/handoff work, not the proposed renderer
or trainer implementation.

It wrote DynaWorld documents, principally:

- `research_notes/spacetime_gaussian_representation/06_shader_boundary_and_depth_fiber.md`;
- `research_notes/spacetime_gaussian_representation/07_concern_status_and_build_plan.md`;
- `agent_notes/loose_notes/2026-07-23_02-23-17_depth_fiber_and_shader_boundary.md`;
- `agent_notes/loose_notes/2026-07-23_02-42-32_spd4_build_and_status_plan.md`;
- updates to `research_notes/spacetime_gaussian_representation/README.md`.

The inspected task's final turns changed documentation only. They did not add
the proposed `WorldAtom`, full-SPD(4) trainer, `FiberTrace` ABI, rotating
covariance model, or curved world-tube implementation.

The active restricted world-tube code still stores:

```text
x0, velocity, t0, precision_xy[2], lambda_t, opacity, color
```

It has affine center motion and a fronto-parallel two-axis footprint. It has no
world-z width, full spatial SPD(3), spatial quaternion/rotation, time-dependent
covariance, or curved centerline.

## Audit of the prior mathematics

### Claim A: a strict joint SPD(4) Gaussian has affine conditional motion

**Status:** correct.

For

```text
Sigma4 = [[C + c v v^T, c v],
          [c v^T,       c  ]],
```

with `C in SPD(3)` and `c > 0`,

```text
X | T=t ~ N(x0 + v (t-t0), C).
```

This parameterization is lossless for strict SPD(4). The spatial conditional
covariance `C` is constant. The space-time cross covariance encodes affine
translation, not a time-varying physical rotation.

### Claim B: affine UVTZ factorization into UVT plus conditional depth

**Status:** correct under the written affine-chart and measure assumptions.

For precision blocks

```text
H = [[P, r],
     [r^T, h]],
```

completion of the square gives

```text
S       = P - r r^T / h
beta    = -r^T / h
var_z   = 1 / h.
```

Thus the UVT marginal plus affine conditional Gaussian depth reconstructs the
joint Gaussian locally. The prior note correctly warns that perspective makes
these quantities nonlinear functions of sensor coordinates and requires local
compilation/certification.

### Claim C: one SPD(4) atom supplies physical rotation over time

**Status:** false if interpreted that way; the prior documents themselves do
not make this false claim.

The six orientation degrees of freedom of one 4D ellipsoid describe a fixed
orientation in spacetime. After conditioning on time, its spatial covariance
is constant. A space-time tilt creates affine center motion; it does not rotate
the conditional 3D ellipsoid as time advances.

The prior formulation catalog correctly places time-varying rotation in the
separate "fully dynamic conditional Gaussian tube" class, but the build plan
deferred that class until after a strict SPD(4) implementation.

## Minimal time-rotating Gaussian

Let `tau = t - t0`. A direct rotating conditional Gaussian is

```text
rho(x,t) =
    a g(t) exp(
      -1/2 (x-m(t))^T C(t)^-1 (x-m(t))
    ),

C(t) = R(t) C0 R(t)^T.
```

For constant angular velocity,

```text
R(t) = R0 exp(tau [omega]_x),
```

where `[omega]_x` is the 3x3 skew matrix of angular velocity. This preserves
orthogonality and positive definiteness exactly. With fixed eigenvalues it
adds only three angular-velocity parameters per independently rotating splat,
although axis symmetries make some rotations unidentifiable.

This object is not a single joint Gaussian in `(x,t)`: the matrix exponential
makes its exponent nonlinear in time. That is not a defect. It is a
continuous-time transported 3D Gaussian and may be closer to the desired scene
model than forcing everything into one SPD(4) atom.

## Rotation of covariance versus natively curved centerlines

These are different effects.

### Per-splat covariance rotation

```text
m_i(t) = x_i0 + v_i tau
C_i(t) = R_i(t) C_i0 R_i(t)^T
```

This twists an anisotropic splat while its center remains on a straight line.
It can fix changing local orientation or footprint. It does not make the tube
centerline curved.

### Shared rigid-object motion

For splats belonging to one object,

```text
m_i(t) = c(t) + R(t) (x_i0 - c0)
C_i(t) = R(t) C_i0 R(t)^T.
```

Even with constant angular velocity, off-axis centers follow circular or
helical arcs. This is a genuinely curved world tube without a separate spline
per splat. It also preserves object coherence and uses one pose trajectory for
many splats.

This shared continuous-time `SE(3)` model is the most economical interpretation
of "time-based rotation" when the scene motion is object-like. Independent
per-splat angular velocities are a weaker structural prior and can shear a
rigid object apart.

### Curved translation unrelated to rotation

If residual center motion remains after shared rigid transport, use the next
smallest extension:

```text
c(t) = c0 + v tau + 1/2 a tau^2
```

or a low-knot spline. This should be added only if a constant-twist `SE(3)`
model leaves a measured motion residual.

## Backtrack on the previous build priority

**Previous recommendation:** implement a full strict SPD(4) world atom and
lossless `FiberTrace` before adding rotation or splines.

**Status:** mathematically coherent but strategically unproven.

It is appropriate if the immediate objective is a canonical finite 4D Gaussian
and exact trace-factorization paper. It is not the cheapest test of the user's
hypothesis that the quality gap came from fixed spatial orientation and rigid
linear motion.

The existing benchmark compared a very restricted tube—only two spatial
precisions and no 3D orientation—against a much freer per-frame 3DGS model.
That result cannot tell us whether full static SPD(3), time-varying rotation,
shared object rotation, or curved translation is the decisive missing degree
of freedom.

## Recommended falsification ladder

Keep renderer, data split, active splat count, training pixel budget, and
appearance fixed. Add one degree class at a time:

1. **W0 current:** affine center + `precision_xy`.
2. **W1 full static shape:** affine center + full world `C0 in SPD(3)`.
3. **W2 independent rotation:** W1 + `R_i(t)=R_i0 exp(tau[omega_i]_x)`.
4. **W3 shared rigid pose:** object/group `SE(3)` transport of centers and
   covariances.
5. **W4 curved residual:** W3 + quadratic or low-knot translational residual.
6. **W5 scale/deformation:** time-varying log eigenvalues only if W4 is still
   insufficient.

For the first experiment, do not modify the STAR compiler. Sample `m_i(t)` and
`C_i(t)` at each training frame and render through the existing ordinary 3DGS
path. This isolates representation quality cheaply. If W2 or W3 wins, then
compile it into piecewise STAR trace cells.

## Measurements and falsification

Measure held-out PSNR, SSIM, LPIPS, checkpoint bytes, active splats, wall time,
peak memory, and temporal consistency. Also record:

- average and 95th-percentile angular displacement;
- covariance eigenvalue condition numbers;
- center-trajectory residual after affine motion;
- center-trajectory residual after shared rigid motion;
- number of time segments required to approximate the rotating model within a
  fixed screen-space error.

The rotation hypothesis is weakened if W2 does not materially improve held-out
quality over W1 at matched bytes and compute. The shared-rigid hypothesis is
supported if W3 improves both held-out quality and trajectory residual relative
to independent rotation. Curved translation is justified only if W3 leaves a
large systematic center residual that W4 removes.

## Current conclusion

The earlier task produced a useful and mostly sound mathematical audit, but it
did not complete the proposed system and did not test natively curved or
time-rotating world tubes.

The simplest high-value next model is not necessarily full SPD(4). It is a
full 3D Gaussian transported by continuous-time rotation—preferably a shared
`SE(3)` object pose when motion is coherent. That model rotates covariance and
automatically curves off-axis splat trajectories. A short matched ablation can
determine whether this was the missing ingredient before more compiler or
representation machinery is built.
