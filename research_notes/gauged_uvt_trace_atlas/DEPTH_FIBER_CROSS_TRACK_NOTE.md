# Depth Fiber Cross-Track Note

Date: 2026-07-05

Purpose: preserve the shared "depth fiber" math across the World Tubes and
WorldFoam paper tracks. This note is deliberately cross-track: it explains the
common camera-ray bundle object, then separates what World Tubes does with the
fiber from what WorldFoam does with it.

## One-Sentence Answer

Yes, the depth fiber is useful in both tracks.

```text
World Tubes:
    use the depth fiber as the dimension to integrate/summarize into UVT
    footprints, while retaining conditional depth/order certificates.

WorldFoam:
    keep the depth fiber as the primary visibility/transmittance axis.
```

The same mathematical object drives both papers, but the operator order differs.

```text
World Tubes:
    early pushforward along z, then visibility repair.

WorldFoam:
    delayed pushforward; render transmittance along z before collapse.
```

This is the cleanest way to say why both papers exist.

## Shared Object

Let:

```text
B = Omega x T
y = (u,v,t) in B
pi: E_Gamma -> B
F_y = pi^{-1}(y)
Gamma: E_Gamma -> M
M = R^3 x R
```

`E_Gamma` is the camera-ray bundle over sensor time. Each fiber `F_y` is the
ray-depth domain for one sensor sample.

A local camera gauge is a trivialization:

```text
chi_l: E_Gamma | C_l -> C_l x Z_l
chi_l(e) = (y,z)
```

where `z` is the depth/fiber coordinate. It can be:

```text
ordinary depth
inverse depth
log depth
projective depth
orbit parameter
row-time coupled rolling-shutter coordinate
object-local ray coordinate
```

The camera map in gauge `l` is:

```text
Gamma_l(y,z) = Gamma(chi_l^{-1}(y,z))
```

and the fiber measure must carry a Jacobian:

```text
dmu_y(e) = J_l(y,z) dz
```

That Jacobian is not bookkeeping. It is the condition under which ordinary
depth, log depth, inverse depth, projective/orbit gauges, and rolling-shutter
gauges are the same physical trace in different coordinates.

## Gauge-Invariant Fiber Trace

For a density-like world primitive `rho_i`, the camera-program trace is:

```text
Trace_Gamma[rho_i](y)
    = pi_* Gamma^* rho_i
    = integral_{F_y} rho_i(Gamma(e)) dmu_y(e)
```

In local gauge:

```text
bar_rho_i^l(y)
    = integral_{Z_l}
        rho_i(Gamma_l(y,z)) J_l(y,z) dz
```

If another gauge uses `z_b = phi_y(z_a)`, then:

```text
J_a(y,z_a) dz_a = J_b(y,z_b) dz_b
```

so:

```text
integral rho_i(Gamma_a(y,z_a)) J_a(y,z_a) dz_a
=
integral rho_i(Gamma_b(y,z_b)) J_b(y,z_b) dz_b
```

This is the mathematical reason "gauged camera space" is not hand-wavy. The
gauge changes local expressions and conditioning; it does not change the
underlying rendered trace when the measure is correct.

## In World Tubes

World Tubes starts with a spacetime primitive and usually wants a compact
sensor-time footprint.

```text
rho_i(x)
    -> Gamma^* rho_i(y,z)
    -> pi_* Gamma^* rho_i(y)
```

For a spacetime Gaussian under a local affine camera gauge:

```text
rho_i(x) = a_i exp[-1/2 (x - m_i)^T Lambda_i (x - m_i)]
Gamma_l(y,z) ~= x0 + J_y delta_y + J_z delta_z
```

the pulled-back precision is:

```text
H = J^T Lambda_i J

H = [H_yy H_yz
     H_zy H_zz]
```

Fiber marginalization gives the UVT footprint precision:

```text
S = H_yy - H_yz H_zz^{-1} H_zy
```

and a conditional depth/fiber model:

```text
z_hat_i(y) = z0 + H_zz^{-1}(g_z - H_zy (y - y0))
Var(z | y) = H_zz^{-1}
```

This precision/mean summary is not the complete marginal by itself. For a
scalar untruncated fiber with locally constant measure factor `J_0`, the UVT
amplitude also contains `J_0 sqrt(2 pi / H_zz)` after completing the square.
Clipped fibers or varying Jacobians need a residual certificate or quadrature.

So in World Tubes, the depth fiber becomes:

```text
Schur-complement dimension
conditional depth model
depth variance / interval certificate
visibility gauge atlas input
pairwise depth-difference field
finite-exposure / rolling-shutter integration coordinate
backward-support certificate
```

World Tubes can push most of the primitive down to `(u,v,t)` for speed. It
cannot discard the depth fiber completely, because visibility is not determined
by the pure UVT marginal.

## In WorldFoam

WorldFoam does not primarily want a depth-marginalized footprint. It wants a
lifted opacity field over sensor time and depth:

```text
sigma_l(y,z)
```

For bounded world cells `F_j` with density `sigma_j`, pull each cell into the
camera gauge:

```text
rho_{j,l}(y,z)
    =
    1_{Gamma_l(y,z) in F_j}
    sigma_j(Gamma_l(y,z))
    J_l(y,z)
```

Aggregate opacity:

```text
sigma_l(y,z) = sum_j rho_{j,l}(y,z)
```

Then visibility is optical depth along the fiber:

```text
tau_l(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds
T_l(y,z)   = exp(-tau_l(y,z))
I(y)       = integral T_l(y,z) sigma_l(y,z) c_l(y,z) dz
```

In WorldFoam, the depth fiber is not auxiliary metadata. It is the rendering
axis:

```text
support intervals live on z
cell entry/exit events live on z
transmittance prefixes live on z
density gradients need prefix/suffix state along z
fallbacks are depth-layer or ray-fiber local
```

That is why WorldFoam should not be forced into the World Tubes Schur story.
The Schur complement is the algebraic jewel for Gaussian tubes. Foam's jewel is
retained-fiber transmittance with sparse event structure.

## The Non-Commutation Principle

The deepest shared lesson is:

```text
visibility does not commute with depth marginalization.
```

Consider two translucent layers with opacities `alpha_1, alpha_2` and colors
`c_1, c_2`.

If layer 1 is in front:

```text
I_12 = alpha_1 c_1 + (1 - alpha_1) alpha_2 c_2
```

If layer 2 is in front:

```text
I_21 = alpha_2 c_2 + (1 - alpha_2) alpha_1 c_1
```

Subtract:

```text
I_12 - I_21 = alpha_1 alpha_2 (c_1 - c_2)
```

Therefore, two scenes can have the same depth-marginalized UVT opacity/color
moments but produce different rendered colors when order changes.

Consequence:

```text
World Tubes:
    needs conditional depth/order certificates after UVT marginalization.

WorldFoam:
    avoids that specific loss by keeping sigma(y,z) and rendering along z.
```

This is the operator-ordering reason the tracks should remain separate.

## Shared Backward Lesson

Both tracks benefit from fiber-aware adjoints.

World Tubes, with fixed local compositing order:

```text
dI/dc_i     = T_i alpha_i
dI/dalpha_i = T_i (c_i - I_behind_i)
```

WorldFoam, in continuous depth:

```text
I(y) = integral T(z) sigma(z) c(z) dz
```

Variation:

```text
delta I(y)
    =
    integral T sigma delta c dz
    +
    integral T (c(z) - I_behind(y,z)) delta sigma(z) dz
```

So WorldFoam backward needs:

```text
front transmittance prefix
behind-radiance suffix
local density/color basis derivatives
cell/event endpoint derivatives or fixed-topology refresh discipline
```

The depth fiber is useful for sharing derivatives because it gives a stable
axis for prefix/suffix computation across many frames or exposure samples.

## What Breaks If The Fiber Is Thrown Away

World Tubes failure:

```text
pure UVT alpha/color marginal cannot certify occlusion order
crossing depth events become invisible to the cache
finite exposure blends the wrong order when order changes during shutter
gradient attribution can be wrong near disocclusion
```

WorldFoam failure:

```text
transmittance cannot be represented as a simple UVT opacity without losing
where opacity occurs along the ray
cell-ray event sparsity is lost
prefix/suffix adjoints become unavailable
the method degenerates into generic dense volume or sorted splat replay
```

## Current Organization

World Tubes track:

```text
research_notes/gauged_uvt_trace_atlas/README.md
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
research_notes/gauged_uvt_trace_atlas/02_gaussian_fiber_pushforward/README.md
research_notes/gauged_uvt_trace_atlas/05_visibility_strata/README.md
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_EXPERIMENT_PLAN.md
```

WorldFoam track:

```text
research_notes/worldfoam_paper/README.md
research_notes/worldfoam_paper/WORLDFOAM_HANDOFF.md
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md
research_notes/worldfoam_paper/scientist_notes/
research_notes/worldfoam_paper/proofs/
```

Bridge:

```text
research_notes/gauged_uvt_trace_atlas/08_worldfoam_bridge/README.md
research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
```

## Next Tests

World Tubes:

```text
synthetic crossing-layer test where UVT marginals match but order differs
visibility gauge atlas certificate test over pairwise depth roots
finite-exposure crossing test where order flips during shutter
```

WorldFoam:

```text
same scene rendered as retained sigma(y,z)
compiled prefix transmittance vs per-frame WorldFoam replay
prefix/suffix VJP finite-difference check
event-density curve showing when z-layer complexity destroys reuse
```

The depth fiber is not an optional philosophical layer. It is the coordinate on
which both correctness and reuse live.
