# Screen-Time Compiler Follow-Up Audit

**Status:** useful independent confirmation and clarification of the existing
World Tubes compiler; not a new representation or a separate paper.

**Audit date:** 2026-07-26

**Source intake:** external follow-up dump, SHA-256
`9297e456845d7c285e705962631aff2f20c30bf40d56627d10b2215b163407c5`.

## Bottom line

The dump makes one important distinction correctly:

\[
\boxed{\text{shared parameters} \ne \text{shared geometric work}.}
\]

A dynamic primitive can be compactly parameterized and still be expanded,
projected, binned, sorted, and differentiated independently at every frame.
For example, a fiberwise family

\[
\rho_\theta(x,t)
=
\exp\!\left[-\tfrac12 x^\top P_\theta(t)x
+q_\theta(t)^\top x+r_\theta(t)\right]
\]

can use \(O(1)\) world parameters per primitive while a direct renderer still
does \(\Theta(NT)\) primitive-frame projection work.

The proposed remedy—compile a world primitive once into a continuous
screen-time support and event record—is already the central World Tubes / STAR
UVT design in this repository. The follow-up is therefore valuable as:

1. an independent re-derivation of the correct compiler boundary;
2. a clean statement of the unavoidable image-output lower bound;
3. a warning not to call parameter sharing computational sharing;
4. a useful paper-exposition pass.

It does not replace or supersede the native \(\mu_4+\operatorname{SPD}(4)\)
world object, Gauged UVT Trace Atlas, or World Tubes paper method.

## Exact overlap with the existing method

The follow-up proposes:

\[
\text{world atom}
\longrightarrow
\text{continuous screen-time footprint}
\longrightarrow
\text{support/order event atlas}
\longrightarrow
\text{frame evaluation}.
\]

The repository already expresses this as:

\[
\text{world primitive}
\xrightarrow{\Gamma^*}
\text{camera-ray pullback}
\xrightarrow{\pi_*}
\text{UVT trace}
\longrightarrow
\text{visibility/support atlas}
\longrightarrow
\text{raster and VJP}.
\]

Here:

- \(\Gamma\) is the camera ray program;
- \(\Gamma^*\) pulls the world field onto sensor-time-depth coordinates;
- \(\pi_*\) integrates or eliminates the ray-depth fiber for World Tubes;
- the remaining UVT object is evaluated over requested output samples;
- visibility strata, support intervals, fallbacks, and adjoint records preserve
  the information that Gaussian marginalization alone cannot preserve.

The external phrase “push the entire atom into screen-time” is consequently a
good informal description of \(\pi_*\Gamma^*\), but not a new operator.

## The output lower bound and the right scaling claim

If the required output is \(T\) images with \(P\) pixels each, explicitly
writing those images costs

\[
\Omega(PT)
\]

operations or memory transactions in any ordinary raster API. No
representation can make the complete explicit output asymptotically sublinear
in \(T\).

The meaningful target is narrower. Write total work as

\[
W
=
W_{\mathrm{compile}}
+W_{\mathrm{events}}
+W_{\mathrm{evaluate}}
+W_{\mathrm{output}}.
\]

A useful idealization is

\[
W_{\mathrm{compiled}}
=
O(NK+S K_{\mathrm{tr}}+B+H+PT),
\]

where:

- \(N\) is the number of world primitives;
- \(K\) is a bounded trace-fitting or camera-chart complexity;
- \(S\) is the actual primitive-gauge-tile-event trace-record count;
- \(K_{\mathrm{tr}}\) is the payload per trace record;
- \(B\) is persistent bin/tile incidence payload;
- \(H\) is residual shading/compositing work;
- \(PT\) is the unavoidable explicit image work.

A replay renderer instead has a structural term such as

\[
W_{\mathrm{replay}}
=
O(NT+B_T+H_T+PT).
\]

The claim is only a temporal win when measured event/trace complexity is
smaller than replay:

\[
NK+S K_{\mathrm{tr}}+B=o(NT+B_T)
\]

on the camera programs and scene distributions under test. Neither \(S\) nor
\(B\) is automatically sublinear in \(T\). A camera or visibility schedule can
force \(\Theta(T)\) distinct events, one event can emit several tile/trace
records, and a pathological continuous program can produce still more local
subdivisions. Full rendering is normally also lower-bounded by
\(\Omega(PT+H)\), not output writes alone.

Therefore the paper-safe statement is:

> World Tubes factors repeated primitive projection and structural visibility
> work through a continuous screen-time atlas. Its non-output cost scales with
> accepted trace and event complexity rather than being forced to scale with
> primitive-frame count.

It is not safe to claim unconditional sublinear total rendering in time.

## When a 4D Gaussian projects to an exact UVT Gaussian

Let \(Z\in\mathbb R^4\) be jointly Gaussian:

\[
Z\sim\mathcal N(\mu_4,\Sigma_4).
\]

If one globally affine observation chart maps spacetime to sensor-time:

\[
Y=LZ+d,\qquad Y=(u,v,t),
\]

then

\[
Y\sim\mathcal N(L\mu_4+d,L\Sigma_4L^\top).
\]

Equivalently, if the camera-ray construction is affine and depth is eliminated
from a joint Gaussian, the UVT precision follows by the relevant Schur
complement. This is exact closure.

The qualification matters:

- a moving perspective camera is not generally one global affine map in
  \((x,t)\);
- the image of a Gaussian under a rational perspective map is not generally a
  Gaussian;
- a local Jacobian projection gives an approximation, not an identity;
- the repository's retained-ray/projective atlas can use exact ray equations
  and certified local trace fits, so “ordinary 3DGS Jacobian approximation” is
  not the only implementation.

Thus the external statement “full 4D Gaussians become exact screen-time
Gaussians” is true only under its affine-chart hypothesis.

## Static world matter does not imply a static screen-time footprint

If a world density is static,

\[
\rho(x,t)=\rho(x),
\]

but the camera program changes with time, then

\[
(\Gamma^*\rho)(u,v,t,s)
=
\rho\!\left(o(u,v,t)+s\,d(u,v,t)\right)
\]

still changes with \(t\). A static world Gaussian is an extrusion in world
time, but not normally an extrusion in UVT.

This invalidates any unqualified claim that a static Gaussian produces a
constant screen footprint through time. That holds only for a static camera
and time-independent appearance, or in another explicitly restricted camera
family.

## Visibility cannot be reduced to center-depth crossings

For two point-like contributors, a center-depth crossing can identify an order
swap. For extended volumetric support, however:

- both contributors can overlap over a depth interval;
- their local emitted colors may differ;
- the transfer operators do not commute;
- the first contact, overlap topology, and exit events can differ from center
  crossings.

A root schedule for representative depths is therefore a useful broad phase,
not a complete visibility theorem. World Tubes must retain order certificates
and fallbacks; WorldFoam retains the ray-depth fiber and evaluates ordered
optical transfer directly.

## What shared backward actually means

Let a compiled trace have coefficients \(a_\theta\) shared over a camera-time
domain, and let output sample \(y_k=(u_k,v_k,t_k)\) evaluate

\[
I_k=F(a_\theta;y_k).
\]

Then

\[
\frac{\partial L}{\partial\theta}
=
\left(\frac{\partial a_\theta}{\partial\theta}\right)^\top
\sum_k
\left(\frac{\partial F}{\partial a}(a_\theta;y_k)\right)^\top
\frac{\partial L}{\partial I_k}.
\]

This shares:

- construction of \(a_\theta\);
- its derivative map;
- support/event metadata;
- topology and visibility tapes when certified.

It does not eliminate per-sample evaluation or the sum over output pixels.
“Differentiate projection once” should be read as differentiating a shared
coefficient record once, followed by unavoidable sample interactions.

## Relationship to the native world object

The screen-time compiler does not require abandoning the native world object.
For the canonical finite-lifetime Gaussian,

\[
\mathcal G=(\mu_4,\Sigma_4,\alpha,a),
\]

the world representation is camera-independent. The UVT trace is a compiled
camera-program-specific artifact. Keeping those levels separate avoids making
the scene itself an accidental function of one training camera path.

The same compiler interface can accept richer objects:

- full \(\operatorname{SPD}(4)\) Gaussians;
- bounded semidefinite persistent tubes;
- finite-element WorldFoam fields;
- compact convex-potential atoms;
- other bounded world cells.

The closure and event rules differ by object, but the camera-program boundary
does not.

## Claim classification

| Follow-up claim | Classification |
|---|---|
| Fiberwise \(P(t),q(t),r(t)\) shares parameters but not raster work | Correct and important clarification |
| Explicit \(T\times P\) output costs \(\Omega(TP)\) | Correct lower bound |
| Compile continuous screen-time support and events | Existing World Tubes / Gauged UVT method |
| Full 4D Gaussian always maps to exact screen-time Gaussian | False without an affine observation hypothesis |
| Curved atoms map to swept conics | Useful representation description; exactness depends on camera/object family |
| Visibility can be handled by continuous order events | Correct program, but event count can be linear and representative-depth roots are incomplete |
| One shared backward removes frame dependence | Overstated; it shares coefficients/tapes, not output evaluation |
| A new paper is implied | No; this strengthens the existing World Tubes paper |

## Paper changes worth keeping

The follow-up should influence the World Tubes paper in four concrete ways:

1. Add the decomposition “parameter compactness versus structural compute
   compactness.”
2. State the \(\Omega(PT)\) output lower bound early.
3. State all scaling in trace/event variables and report their empirical death
   curves.
4. Put the affine closure hypothesis beside every exact Gaussian pushforward
   statement.

## Falsification and engineering checks

1. **Affine closure fixture.** Sample one \(\operatorname{SPD}(4)\) Gaussian,
   apply a known affine world-to-UVT map, and compare the analytic mean and
   covariance with Monte Carlo projection.
2. **Perspective non-closure fixture.** Repeat with a wide-FOV perspective
   camera and quantify the residual of the best Gaussian UVT fit.
3. **Static-world/moving-camera fixture.** Keep the primitive fixed and move
   the camera; assert that UVT coefficients change.
4. **Event death curve.** Plot \(B/(NT)\), \(E/(NT)\), compile bytes, fallback
   fraction, and total time for \(T=2,\ldots,300\).
5. **Replay theorem.** On certified cells, require compiled and per-frame
   same-representation color and gradients to agree within declared numeric
   tolerances.
6. **Adversarial visibility.** Use long overlapping colored volumes whose
   centers never cross but whose support ordering changes; representative
   center-depth metadata must fail or enter fallback.

## Decision

Do not open a third “screen-time compiler” paper. Fold the useful distinctions
and lower-bound language into World Tubes. Keep the native world representation
camera-independent, and judge temporal reuse by measured trace/event
complexity rather than by parameter count or slogans about processing all
frames at once.
