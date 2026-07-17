# What Is This? What Is The Goal?

Date: 2026-05-24

This note is the first-read contract for the Gauged UVT Trace Atlas idea. It
starts from the user-facing intent, then makes the renderer claim precise enough
to test.

## What Is This?

This is a camera-program compiler for dynamic rendering.

The object being compiled is not a frame, a video, or a set of ordinary
per-frame splats. The object being compiled is:

```text
4D spacetime world primitives
    pulled through a known camera program
    onto the camera ray bundle
    then integrated or summarized along each sensor fiber
    into reusable viewport-time traces.
```

In short:

```text
world spacetime primitive -> UVT sensor-time trace
```

The output of compilation is a trace atlas:

```text
K_Gamma = {sensor-time charts, active trace sets, support bounds,
           visibility/order data, derivative data, fallback certificates}
```

The atlas lives over:

```text
B = Omega x T
y = (u, v, tau)
```

where `Omega` is the 2D viewport and `T` is sensor time. A rendered frame is a
slice of this object. A finite-exposure frame is an integral through this
object. A rolling-shutter image is a row-coupled sampling or integration through
this same object.

This is why the right noun is trace atlas, not video cache.

## What Is The Goal?

The goal is fast rasterization across time from 4D spacetime primitives:

```text
many 2D viewport rasters across time
from one known or low-dimensional camera program
with shared projection, binning, support, visibility, and backward work.
```

The core performance target is:

```text
output pixels still cost O(F H W)
world-side work must grow sublinearly with F
```

Here:

```text
F = number of frames or temporal samples
H, W = image height and width
```

The renderer cannot avoid writing `F H W` output samples. The win condition is
that it avoids repeating the expensive world-side work for every frame:

```text
per-frame baseline:
    T(F) ~= F * (project + bin + sort + backward_replay) + F H W * shade

trace-atlas target:
    T(F) ~= compile(Gamma, W) + eval(F H W) + small_refine(F)
```

The useful claim is not "video rendering is free." The useful claim is:

```text
project/bin/support/visibility/backward replay are amortized over sensor time.
```

For a smooth camera path and coherent scene, the compile object should scale
with the complexity of traces and visibility strata, not with the number of
requested frame samples.

## Formal Object

Let world spacetime be:

```text
M = R^3 x R
```

Let the dynamic world be a measure or atom field:

```text
W_theta = sum_i w_i(theta_i)
```

where each `w_i` may be a 4D Gaussian, a deforming 3D Gaussian, a foam cell, an
instance-local atom, or another local spacetime primitive.

Let the sensor-time base be:

```text
B = Omega x T
y = (u, v, tau)
```

Let the camera program define a ray/fiber bundle:

```text
pi: E_Gamma -> B
pi^{-1}(y) = F_y
```

`F_y` is the depth/fiber domain over a sensor sample. A local gauge is a
trivialization:

```text
chi_a: E_Gamma | C_a -> C_a x D_a
chi_a(e) = (y, z_a)
```

The camera program maps bundle points into world spacetime:

```text
Gamma: E_Gamma -> M
Gamma_a(y, z_a) = Gamma(chi_a^{-1}(y, z_a))
```

The invariant trace of a world primitive is:

```text
Trace_Gamma[w_i](y) = pi_* Gamma^* w_i
```

More explicitly, for a density-like primitive:

```text
bar_rho_i(y) = integral_{F_y} rho_i(Gamma(e)) dmu_y(e)
```

or in a gauge:

```text
bar_rho_i^a(y) = integral_{D_a} rho_i(Gamma_a(y, z_a)) J_a(y, z_a) dz_a
```

The renderer stores local approximations of:

```text
alpha_i,a(y)      opacity / density footprint
c_i,a(y)          color or feature trace
z_hat_i,a(y)      conditional fiber/depth statistic
U_i,a(y)          uncertainty and approximation certificate
```

over chart domains `C_a subset B`.

## Rasterization Interpretation

The atlas is a rasterization data structure over sensor time.

A traditional dynamic Gaussian renderer does, per frame:

```text
world primitive at tau_k
    -> project into screen
    -> estimate support
    -> bin into tiles
    -> sort or approximate visibility
    -> shade/composite
```

The trace-atlas renderer does, over a camera-time interval:

```text
world primitive over T
    -> trace through camera bundle
    -> estimate support in (u, v, tau)
    -> bin into tile-time cells
    -> compile visibility/order over cells
    -> shade/composite slices
```

So the raster primitive is no longer only a 2D splat. It is a sensor-time
footprint:

```text
alpha_i(u, v, tau)
```

with conditional visibility data:

```text
z_hat_i(u, v, tau), uncertainty_i(u, v, tau)
```

Frames are slices:

```text
I_k(u, v) = I(u, v, tau_k)
```

Exposure images are time integrals:

```text
I_k(u, v) = integral w_k(u, v, tau) I(u, v, tau) d tau
```

The phrase "2D render rasters across time" means the renderer evaluates a
piecewise analytic 3D base-domain object and emits ordinary 2D images as
slices or integrals.

## Clean Derivative Condition

The atlas must be trainable. That means "fast forward only" is not enough.

Inside a chart with fixed visibility order, the rendering map should be
differentiable:

```text
theta_i -> trace parameters phi_i,a -> I(y) -> L
```

For a loss over sensor time:

```text
L = integral_B ell(I(y), I_star(y)) dy
```

the gradient should be expressible as:

```text
dL/dtheta_i
    = sum_a integral_{C_a}
        A(y)^T dI(y)/dphi_i,a dphi_i,a/dtheta_i dy
```

where:

```text
A(y) = dL/dI(y)
```

The backward pass target is the same as the forward target:

```text
do not replay full per-frame projection/binning/sorting for every frame
when the trace atlas already encodes the reusable geometry.
```

The derivative contract has four parts:

1. Trace derivatives are clean inside a chart.
2. Chart transitions are explicit and differentiable where used.
3. Visibility strata are handled as piecewise-smooth regions with stable order
   or bounded commutation error.
4. Non-smooth/chaotic regions fall back to a local reference path with an
   explicit cost and error certificate.

This is why residual checks and fallback are not the theory. They are the proof
obligation that tells us where the smooth chart theory is valid.

## Memory-Bandwidth Reuse Condition

The atlas must reuse memory traffic, not merely arithmetic.

The expensive repeated traffic in a per-frame renderer includes:

```text
primitive reads
projection parameter reads
tile-list writes
tile-list reads
sort scratch
visibility/depth scratch
backward replay scratch
gradient accumulation scratch
```

The atlas should store trace records once per valid sensor-time chart:

```text
TraceRecord {
    primitive_id
    chart_id
    support_bound_uvt
    opacity_trace_params
    color_trace_params
    depth_trace_params
    derivative_sidecars
    error_certificate
}
```

Tile-time cells should reference these records by interval-compressed lists:

```text
TileTimeIndex {
    tile_uv
    tau_interval
    trace_record_range
    order_or_order_graph
    fallback_flag
}
```

The memory target is:

```text
do not store per-frame bins unless a local fallback requires it.
```

For a smooth orbit, a primitive that remains coherent across many frames should
produce a small number of trace records, not `F` copies.

## Sublinear Growth Condition

Define:

```text
C_proj(F)    projection/support/binning cost over F frames
C_sort(F)    visibility ordering cost over F frames
C_back(F)    backward replay geometry cost over F frames
C_pix(F)     unavoidable per-output pixel cost
```

The per-frame baseline has:

```text
C_proj(F) = O(F N)
C_sort(F) = O(F * active_tile_sort)
C_back(F) = O(F * backward_replay)
C_pix(F)  = O(F H W K)
```

The trace-atlas target is:

```text
C_proj(F) = O(N * A_trace)
C_sort(F) = O(A_visibility)
C_back(F) = O(A_adjoint + F H W local_derivatives)
C_pix(F)  = O(F H W K_eval)
```

where:

```text
A_trace      = number of required charts per primitive
A_visibility = number/complexity of visibility strata
A_adjoint    = number/complexity of adjoint chart records
```

The desired regime is:

```text
A_trace << F
A_visibility << F * per_frame_sort_complexity
```

or more generally:

```text
C_world(F) = O(F^rho), rho < 1
```

for the non-pixel world-side work over useful path segments.

If chart count grows linearly with frames, the method has collapsed back into
ordinary per-frame rendering. That is a falsification condition.

## Revolving Camera Condition

A revolving camera is not a special exception. It is the camera program that
forces the correct geometry.

For an orbit, a world point or primitive center projects as:

```text
h(t) = K(t) [R(t) | T(t)] X(t)
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
```

The rational denominator `h_z(t)` is part of the chart. A global affine UVT
tube is the wrong object for a wide orbit. The right object is an atlas:

```text
{C_a, chi_a, transition_ab}
```

where each chart chooses a gauge that keeps the trace simple:

```text
projective homogeneous gauge
inverse-depth gauge
object-local gauge
foam-local gauge
ordinary depth gauge
```

The compiler should first change gauge or transition chart. It should split
only when the chosen gauge cannot preserve low residuals, monotone fiber order,
or bounded support. It should fall back only when even local charting is not
worth the complexity.

This is the answer to "shouldn't rich math handle revolving cameras?": yes.
The rich math is the bundle/gauge/atlas structure. Fallback is a guardrail for
pathological local events, not the main handling of the orbit.

## Visibility Condition

Depth integration alone does not solve visibility.

Each trace must carry conditional fiber/depth information:

```text
z_hat_i(y)
Var[z_i | y]
```

Visibility changes live on strata:

```text
z_hat_i(y) = z_hat_j(y)
```

Inside a sensor-time cell, the compiler can store:

```text
stable total order
stable partial order
commutable ambiguous pairs
fallback flag
```

For two alpha-composited traces, an unresolved order swap is acceptable only
when the visible error is bounded:

```text
|Delta I_ij(y)| <= alpha_i(y) alpha_j(y) |c_i(y) - c_j(y)|
```

The visibility goal is not to make sorting vanish everywhere. It is:

```text
compile stable order over regions
bound harmless ambiguity
fall back only for important chaotic regions
```

## WorldFoam Interpretation

WorldFoam is the same object with a different primitive basis.

A foam cell is a persistent world support region:

```text
F_i subset M
```

The camera-program preimage is:

```text
Gamma^{-1}(F_i) subset E_Gamma
```

Pushing along `pi` gives a sensor-time support trace:

```text
pi(Gamma^{-1}(F_i)) subset B
```

So foam cells can be compiled into the same tile-time atlas. The difference is
that their support bounds may come from cell intersection geometry rather than
Gaussian Schur complements.

## Minimum Acceptance Criteria

The idea is not "good to go" until it passes gates like these:

1. Projective trace evaluator matches Torch and runs in Metal.
2. Local chart fitting reports residuals that stay bounded across orbit
   windows.
3. Support bounds cover dense sampled traces without exploding tile count.
4. Visibility atlas stores stable or bounded-order cells for most active
   regions.
5. Fallback fraction stays low enough that amortization survives.
6. Forward rendering matches dense per-frame/per-sample reference.
7. Backward gradients match a reference path on synthetic cases.
8. Measured non-pixel world-side cost grows sublinearly with requested frames.
9. Memory stays below a useful multiple of the primitive representation.
10. Real STAR UVT / WorldFoam integration improves an actual multi-frame,
    finite-exposure, rolling, or orbit workload.

## Current Prototype Slice

The current prototype slice is now a small compiler stack, not just one
evaluator. The first Metal hook remains:

```text
projective_trace_eval(coeffs, times, eps) -> [N, S, 4]
```

It tests one essential claim:

```text
the renderer can evaluate homogeneous camera-time trace charts on Metal.
```

The compiler-side stack now includes:

```text
projective samples
    -> local affine/quadratic UVT chart fit
    -> residual / denominator / validity certificate
    -> split accepted local windows
    -> support bounds
    -> visibility sidecars
    -> visible-swap bounds
    -> tile-time records and cells
    -> CPU atlas reference rendering
```

The concrete helper surface is:

```text
fit_projective_trace_polynomial(...)
split_projective_trace_windows(...)
bound_projective_trace_windows(...)
make_projective_trace_visibility_sidecar(...)
compare_projective_trace_depth_order(...)
bound_projective_trace_visible_swap_cost(...)
bin_projective_trace_support_bounds(...)
assemble_projective_trace_tile_time_atlas(...)
render_projective_trace_tile_time_atlas_reference(...)
```

The first existing-renderer bridge is:

```text
projective_trace_windows_to_uvt_tubes(...)
render_projective_trace_uvt_bridge_reference(..., use_window_gates=True)
render_projective_trace_uvt_bridge_metal_gated(...)
render_uvt_tubes_gated(...)
direct_atomic_backward_gated(...)
direct_backward_projective_trace_uvt_bridge_metal_gated(...)
pack_projective_trace_tile_time_bins(...)
render_projective_trace_tile_time_atlas_metal(...)
direct_backward_projective_trace_tile_time_atlas_metal(...)
```

This lowers accepted degree-1 chart windows into the existing STAR UVT q-UVT
contract and uses explicit interval gates to keep split chart segments from
leaking. The current q-UVT Metal path now has native shader-side interval gate
buffers for those split affine segments, including direct atomic VJP coverage
and a one-step bridge-level loss-decrease smoke.

The first nonlinear projective atlas-cell forward renderer now exists too:
packed tile-time cells carry exact per-entry active intervals, and Metal
evaluates degree-2 homogeneous projective traces directly at each sample. The
matching direct VJP now differentiates color, opacity, and homogeneous
projective coefficients against a Torch autograd oracle. A coefficient-only
trainability smoke now holds color fixed, updates homogeneous coefficients with
that VJP, and verifies rendered MSE drops.

The first frame-count scaling benchmark for packed projective atlas cells now
exists. On a deterministic 45-degree orbit fixture:

```text
4 -> 64 frames
dense per-frame project/bin pairs: 35 -> 555
ideal interval atlas entries:      13 -> 13
fixed-slab tile_t=4 entries:       13 -> 208
```

This is exactly the distinction the theory needs: the compiled atlas object is
sublinear in world-side project/bin work, while a fixed-slab schedule expands
that same interval linearly in the number of temporal slabs.

The first interval-compressed projective cell Metal forward path now consumes
that object directly. `render_projective_trace_cell_interval_atlas_metal(...)`
packs spatial tile entries once with per-entry `[active_start, active_stop)`
frame intervals and dispatches `render_projective_trace_cell_interval_tiles`
over output samples. On the same 4 -> 64 frame fixture, the interval Metal path
matches the slab image sums and scales `24.8067ms -> 29.3612ms`, while the slab
path scales `20.0995ms -> 37.2617ms`.

The matching interval-compressed projective cell direct VJP now exists too.
`direct_backward_projective_trace_cell_interval_atlas_metal(...)` calls
`direct_projective_trace_cell_interval_backward` over the same spatial tile bins
and explicit active intervals. Focused tests match Torch autograd for color,
opacity, and cell trace coefficients; a one-step coefficient trainability smoke
keeps color fixed and verifies Metal-rendered MSE drops after native interval
VJP updates. The latest tiny scaling artifact also times interval backward:
`6.7827ms -> 35.6822ms` over `4 -> 64` frames, while interval entries remain
`13 -> 13`. That is not sublinear total backward rendering, because output
pixels still grow, but it proves the derivative path consumes the same
interval-compressed sensor-time object.

The next hard gate after interval forward/backward was a real trainer producer
for nontrivial projective/gauge-domain intervals, so the optimizer loop uses
actual chart segments instead of only the degenerate full-video interval. That
gate now has a first trainer-harness proof. The helper
`render_projective_cell_interval_atlas_metal_backward(...)` wraps the
interval-compressed cell forward and direct VJP in a PyTorch autograd function.
The focused smoke builds split projective windows, lowers them to
`ProjectiveTraceCellTraceAtlas` rows with multiple active intervals, runs a
loss through the interval cell renderer, backprops through the interval direct
VJP, and takes a loss-decreasing optimizer step on cell trace coefficients. The
next lifecycle gate now has a first concrete form:
`projective_trace_cell_atlas_coverage_report(...)` detects when live cell
trace coefficients move outside compiled tile-time coverage,
`projective_trace_cell_atlas_visibility_report(...)` detects when live depths
no longer match compiled front-to-back order, and
`rebin_projective_trace_cell_atlas(...)` rebuilds support/depth metadata while
preserving the differentiable coefficient/opacity/color tensors. Focused tests
move a trace into a new tile and flip two traces' depth order without changing
screen support; both stale cases are detected, rebinned, and repaired. The
trainer harness now also exposes
`refresh_projective_cell_interval_atlas_if_stale(...)`, and a Metal smoke moves
an MPS coefficient tensor with an optimizer step, refreshes metadata, renders
through the interval-compressed autograd path, and verifies gradients still
flow into the same tensor. `ProjectiveCellIntervalTrainerState` now owns the
atlas/config/times/refresh cadence for trainer-style loops and calls refresh
from `after_optimizer_step()`. Ambiguous near-tie visibility is now explicit:
strict refresh raises, opt-in refresh marks affected cells as
`visibility_ambiguous_depth`, and the Metal fast path rejects those cells. The
CPU/Torch reference cell-atlas renderer now executes fallback cells by sorting
marked tile/sample regions by live evaluated depth, and the atlas/trainer state
report fallback fraction and reasons. Refresh now uses continuous screen/tile
support roots to split moving traces into time-local tile runs, then tries
continuous visibility-root splitting and sampled visibility-stratum splitting
before fallback. Crossing depth order becomes stable time-run cells without
replacing live tensors; exact visibility roots on frame samples become
singleton cells so fallback is localized to the true tie/event sample. A
continuous sensor-time partition now merges support roots, visibility roots,
and caller-supplied exposure/shutter split times into intervals independent of
frame-index cells; it can now lower those intervals into normalized
finite-exposure quadrature and per-row rolling-shutter quadrature schedules.
Those schedules now feed differentiable CPU/Torch reference oracles that
evaluate fractional sensor times, live-sort by depth, composite, and accumulate
sample weights for finite exposure or row-wise rolling shutter. They also lower
to sample-indexed interval atlases that can render through the existing
interval Metal kernel; rolling shutter now batches unique row sample times into
one schedule with a `[sample,row]` weight matrix and a row-weighted interval
Metal kernel that writes the final rolling image directly. Mixed fallback
forward rendering now patches whole fallback tile/sample regions with the
live-depth reference before exposure or rolling accumulation, while keeping
non-fallback regions on interval Metal. The trainer state now has
`fallback_render_mode="mixed"` so those fallback patches can remain
differentiable while fast regions keep native interval Metal VJP.
The first-class STAR UVT feature trainer now rejects
`projective_interval.enabled` unless a real `ProjectiveTraceCellTraceAtlas`
producer is explicit. A first compatible producer now exists for exact
isotropic affine UVT tubes:

```text
uvt_tubes_to_projective_trace_cell_atlas(...)
make_projective_cell_interval_atlas_from_uvt_tubes(...)
```

It completes the UVT quadratic in spatial variables, extracts the moving
screen center as the gauge trace, lowers that into cell-polynomial atlas rows,
and runs the existing support/visibility event compilers. This is not the full
WorldFoam/general-splat producer yet; it intentionally rejects anisotropic
footprints and pixel-varying depth. Continuous temporal opacity is now stored
as a quadratic `opacity_time_coeffs` payload and consumed by the CPU/Torch
reference renderer plus the native interval Metal forward/backward path. The
remaining gap is routing this producer through the real trainer loop, extending
the trace/native representation for anisotropy and pixel-varying depth, setting
production budget defaults, and adding cheaper/native fallback VJP.
