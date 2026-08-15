# WorldFoam Stratified Lagrangian Optical-Connection Audit

Date: 2026-08-05

Status: audited mathematical intake and falsification plan. This document does
not claim a new runtime, lower temporal rank, native shader parity, or measured
memory reduction. It fixes the scientist note's multiplication convention,
qualifies its theorems, records counterexamples, and identifies the smallest
experiment that can decide whether the connection is computationally useful.

Source intake:

```text
/Users/nicholasbardy/.codex/attachments/
  2492c6e4-bcde-416a-80b0-711c5a6101da/pasted-text.txt
SHA-256:
  965c7a1a28343914dd348a88afa1b30a976dabd6dbf80fb48a1076ad878334c5
```

The unified relationship to the prior translated optical-depth measure and
the corrected physical-`U`/group-completion-`U_tilde`/signed-`K_F` ABI split
is recorded in `WORLD_FOAM_MEASURE_CONNECTION_SYNTHESIS_2026-08-05.md`.

## Executive Verdict

The constrained Lagrangian optical connection is the best new mathematical
hypothesis in the intake. It gives an exact identity:

```text
covariant time variation of ordered ray transfer
  = depth integral of transported optical curvature
  + moving-endpoint flux.
```

For an independently specified, orientation-preserving material flow, zero
curvature is equivalent to exact reuse of **every transported depth
subinterval**. This is a real theorem and a useful correctness diagnostic.

It is not yet a speed theorem. WorldFoam already compiles the total affine
transfer, which is invariant under many monotone depth rearrangements. A
curvature source can have the same or greater approximation rank than the
transfer, and the flow plus endpoint transports may cost more than they save.
The next decision must therefore compare three representations under identical
primal and tangent certificates:

```text
direct transfer U
vs flow-corrected transfer U_tilde
vs transported curvature source K_F.
```

The rest of the intake is valuable mainly as organization:

| Intake formulation | Audited status | Action |
| --- | --- | --- |
| Stratified ray-fiber bundle | Correct local geometry; mostly existing | Keep a qualified local-triviality lemma; run a cross-pixel reuse census before any patch compiler. |
| Lagrangian optical connection | Strongest genuinely new theorem/hypothesis | Seal the corrected equations and then build one small CPU oracle. |
| Factorization cosheaf | Ordered refinement is correct; the claimed cosheaf is not formally defined | Call it multiplicative interval transport or a monoid-valued concatenation functor. |
| Jet stack | Jet product is existing algebra; corrected order-`q` seam lemma is useful | Add the lemma; treat a coarser transfer cover as future work. |
| Optimizer monodromy | Real simple roots have trivial order monodromy | Keep the real-root lemma; defer braid/2-stack machinery. |

No renderer, shader, or public method should be renamed around bundles,
cosheaves, stacks, or holonomy. The method remains **WorldFoam**. Open camera
rays use ordered parallel transport; holonomy is reserved for closed loops.

## 1. Assumptions and Executable Convention

### 1.1 Regularity and scope

The classical identities below assume one regular chart with:

- finite optical depth, hence `beta > 0` when group inverses are used;
- `A_z,A_t` continuously differentiable in the chart interior;
- finitely many simple, distinct owner boundaries;
- continuously differentiable clipped endpoints;
- strict non-owner gaps on the interiors of active runs;
- nonzero ray speed and nonzero cut denominators; and
- no birth, death, repeated root, full-fiber tie, or simultaneous event.

For P0 fields, `A_z` is piecewise smooth. Distributional statements then use
bounded-variation fields with one-sided traces. A depth flow `w` should be at
least Lipschitz in depth to generate a non-crossing, orientation-preserving
flow. Event seams require separate one-sided statements.

### 1.2 WorldFoam scans near to far

The repository defines

```text
compose(front, back) = front(back(c)).
```

Let camera depth increase from the near endpoint `a` to the far endpoint `b`.
For `a < s < b`, the executable transfer convention is

$$
U_t(b,a)=U_t(s,a)U_t(b,s).
$$

Consequently,

$$
\partial_bU_t(b,a)=U_t(b,a)A_z(t,b),
\qquad
\partial_aU_t(b,a)=-A_z(t,a)U_t(b,a).
$$

The scientist attachment instead used the left-ordered ODE
`partial_z U=A_z U`. Its curvature theorem is internally meaningful in that
opposite convention, but its sandwich order, endpoint factors, gauge
correction, and holonomy sign cannot be copied into this repository.

An executable order sentinel is a front red segment and a rear blue segment:

$$
g_R=\left(\tfrac12,\tfrac12 e_R\right),\qquad
g_B=\left(\tfrac12,\tfrac12 e_B\right),
$$

$$
g_Rg_B
=\left(\tfrac14,\tfrac12e_R+\tfrac14e_B\right).
$$

Any oracle producing the blue-heavy reversed moment has used the wrong scan.

## 2. The Ray Fibration and Its Regular Strata

Let the sensor-time base be

$$
B\subset\mathbb R^2_{u,v}\times I_t.
$$

For one tracked pixel, use `B=I_t`. Define the clipped ray-depth space

$$
\mathcal R
=\{(b,z):b\in B,\ z_-(b)\le z\le z_+(b)\},
\qquad
\pi:\mathcal R\to B,
\quad
\pi(b,z)=b.
$$

The physical evaluation map is

$$
\Gamma:\mathcal R\to\mathbb R^3\times I,
\qquad
\Gamma(b,z)=(x_b(z),t_b).
$$

Pulling the kinetic cell partition back through `Gamma` divides each fiber
into ordered owner intervals. The radiance state is a second affine bundle
over `mathcal R`; it is not the ray bundle itself:

$$
\mathcal E\longrightarrow\mathcal R\overset\pi\longrightarrow B.
$$

Both bundles are topologically unremarkable on an interval. The content lies
in the optical connection, its stratification, and ordered interval
composition.

### 2.1 Owner discriminant

Let active boundaries be locally defined by

$$
h_r(b,z)=0.
$$

The regular owner base consists of points where:

$$
\partial_z h_r(b,z_r)\ne0,
$$

$$
z_-(b)<z_1(b)<\cdots<z_R(b)<z_+(b),
$$

all competitor inequalities are strict, and no ray-collapse, full-fiber tie,
or simultaneous event occurs. Call the complement `Delta_owner`.

This is deliberately not called the minimal transfer discriminant. A
zero-width owner event can be invisible to the rendered transfer, while a cut
denominator guard can fail without a physical topology event.

### 2.2 Qualified local-triviality lemma

**Lemma.** Suppose a reference fiber has finitely many simple separated roots,
continuous clipped endpoints, and uniform strict competitor gaps on compact
run interiors. Then there is a neighborhood `V` of the reference base point
and an owner-preserving homeomorphism

$$
\Phi:V\times\mathcal R_{b_0}\to\pi^{-1}(V),
\qquad
\pi\circ\Phi(b,z)=b.
$$

If the predicates are analytic, each cut graph is analytic. Transfer
coefficients are analytic only if the camera map, physical fiber Jacobian,
material fields, and endpoints are analytic as well.

**Proof.** The implicit-function theorem gives unique local cut graphs
`z_r(b)`. Strict separation and strict owner gaps persist after shrinking the
neighborhood. Map each reference interval affinely onto its corresponding
moving interval. The interval maps join continuously and preserve order and
owner labels. The resulting trivialization is generally only piecewise smooth
in depth at the cuts. `square`

For a canonical fiber coordinate `s`, with `z=phi_b(s)`, the pulled-back
generator is

$$
\widetilde A_s(b,s)
=A_z(b,\phi_b(s))\,\partial_s\phi_b(s).
$$

This is coordinate invariance of the optical measure, not a new renderer.

### 2.3 Possible sensor-time sharing

A patch in `(u,v,t)` could share owner identities, boundary identities, event
ordering, and a canonical depth coordinate. It still needs pixel-dependent
event locations, segment lengths, and transfer values. No theorem bounds the
number of connected patches; occlusion fronts can fragment the base almost
track by track.

The only justified next action is a reuse census. Canonicalize each track by
owner-word sequence, active boundary site IDs, event-source order, chart count,
and selected rank. Region-grow adjacent identical signatures and report
template-only and full-numeric reusable bytes separately. Stop if any of these
hold:

- connected patch count exceeds half the track count;
- median patch size is below four tracks;
- shareable bytes are below 25% of compiled artifact bytes; or
- estimated exact compile work improves by less than `2x`.

## 3. Exact Affine Optical Transport

Write homogeneous radiance as `q_hat=(q,1)` with `q in R^3`. Define

$$
T(\beta,m)=
\begin{bmatrix}
\beta I_3&m\\
0&1
\end{bmatrix},
\qquad
\beta>0,
\quad
m\in\mathbb R^3.
$$

It acts by `q -> m+beta q`, and

$$
T(\beta_1,m_1)T(\beta_2,m_2)
=T(\beta_1\beta_2,m_1+\beta_1m_2).
$$

Thus

$$
G_{\mathrm{Aff}}\cong\mathbb R_{>0}\ltimes\mathbb R^3,
$$

with inverse

$$
T(\beta,m)^{-1}=T(\beta^{-1},-\beta^{-1}m).
$$

Physical emission-absorption occupies a contraction semigroup, normally
`0<beta<=1`, with additional color-cone constraints. The inverse exists in the
group completion but is generally not a physical optical segment. Numerical
underflow to `beta=0` is outside the group and cannot enter a holonomy test.

The Lie algebra elements are

$$
X(a,b)=
\begin{bmatrix}
aI_3&b\\
0&0
\end{bmatrix},
$$

with

$$
[X(a,b),X(c,d)]=X(0,ad-cb).
$$

The exponential is

$$
\exp\!\left(\ell X(a,b)\right)
=
\begin{bmatrix}
e^{a\ell}I_3&\frac{e^{a\ell}-1}{a}b\\
0&1
\end{bmatrix},
$$

using the removable limit `ell b` when `a=0`.

For one P0 material with world extinction `rho` and color `c`, the coordinate
depth generator must include physical ray speed:

$$
v_z(t,z)=\|\partial_z\Gamma(t,z)\|,
$$

$$
A_z
=v_zX(-\rho,\rho c)
=X(-\lambda,\eta),
\qquad
\lambda=v_z\rho,
\quad
\eta=\lambda c.
$$

Then

$$
\exp\!\left(\ell X(-\lambda,\lambda c)\right)
=
\begin{bmatrix}
e^{-\lambda\ell}I_3&(1-e^{-\lambda\ell})c\\
0&1
\end{bmatrix}
$$

for coordinate-constant `lambda`. Omitting `v_z` confounds a depth or camera
reparameterization with material change.

## 4. Repo-Native Curvature-Variation Theorem

Introduce a horizontal generator `A_t(t,z)` and define the right-ordered
curvature

$$
\boxed{
F^R_{tz}
=\partial_tA_z-\partial_zA_t+[A_t,A_z].
}
$$

### 4.1 Fixed endpoints

**Theorem.** For fixed `a<b`,

$$
\boxed{
\partial_tU
-UA_t(t,b)
+A_t(t,a)U
=\int_a^b
U_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds.
}
$$

**Proof.** Let `P(z)=U_t(z,a)` and

$$
V(z)=\partial_tP(z)-P(z)A_t(t,z)+A_t(t,a)P(z).
$$

Using `partial_z P=P A_z` gives

$$
\partial_zV=V A_z+P F^R_{tz},
\qquad
V(a)=0.
$$

Variation of constants yields

$$
V(b)=\int_a^bU_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds.
\qquad\square
$$

The prefix is on the left and the suffix on the right. Differently colored
segments make this order observable.

### 4.2 Four-scalar component oracle

Write

$$
U=T(\beta,m),
\quad
A_t(a)=X(\alpha_a,\nu_a),
\quad
A_t(b)=X(\alpha_b,\nu_b),
\quad
F=X(f,g).
$$

For depth `s`, write

$$
U_t(s,a)=T(\beta_p,m_p),
\qquad
U_t(b,s)=T(\beta_s,m_s).
$$

The independently differentiated covariant derivative is

$$
D_t\beta=\dot\beta-\beta\alpha_b+\beta\alpha_a,
$$

$$
D_tm=\dot m-\beta\nu_b+\nu_a+\alpha_am.
$$

The transported curvature predicts

$$
K_\beta=\int_a^b\beta_p\beta_s f\,ds,
$$

$$
K_m=\int_a^b\beta_p(g+f m_s)\,ds.
$$

The suffix moment `m_s`, not the prefix moment, is an order-sensitive test of
the implementation.

### 4.3 Moving endpoints

For `a=a(t)` and `b=b(t)`, define

$$
B_a=A_t(t,a)+\dot a A_z(t,a),
\qquad
B_b=A_t(t,b)+\dot b A_z(t,b).
$$

Then

$$
\boxed{
\frac{dU}{dt}
=U B_b-B_aU
+\int_a^bU_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds.
}
$$

Endpoint flux must not be silently absorbed into the interior distributional
term unless the field has explicitly been extended outside the clip interval.

Let endpoint transports solve

$$
\dot H_a=H_aB_a,
\qquad
\dot H_b=H_bB_b.
$$

with reference conditions

$$
H_a(t_0)=H_b(t_0)=I.
$$

Define

$$
\widetilde U=H_a U H_b^{-1}.
$$

Then

$$
\boxed{
\frac{d\widetilde U}{dt}
=H_a
\left[
\int_a^bU_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds
\right]
H_b^{-1}.
}
$$

This is the corrected gauge-corrected transfer for the repository convention.

## 5. Constrained Lagrangian Optical Connection

Choose a depth velocity `w(t,z)` and horizontal vector

$$
H=\partial_t+w\partial_z.
$$

Impose the constrained connection

$$
A_t=-wA_z.
$$

Because `[A_t,A_z]=0`,

$$
\boxed{
F^R_{tz}=\partial_tA_z+\partial_z(wA_z).
}
$$

For `A_z=X(-lambda,eta)`,

$$
F^R_{tz}
=X\!\left(
-\left[\partial_t\lambda+\partial_z(w\lambda)\right],
\partial_t\eta+\partial_z(w\eta)
\right).
$$

Thus flatness is the pair of continuity equations

$$
\partial_t\lambda+\partial_z(w\lambda)=0,
$$

$$
\partial_t\eta+\partial_z(w\eta)=0.
$$

When `eta=lambda c`, the second follows from extinction conservation plus
color advection,

$$
\partial_tc+w\partial_zc=0.
$$

### 5.1 Exact coherent-motion theorem

Let `w` be continuously differentiable in depth (with sufficient continuity
in time) and generate orientation-preserving `C^1` diffeomorphisms `phi_t`
normalized by

$$
\partial_t\phi_t(s)=w(t,\phi_t(s)).
$$

$$
\phi_{t_0}(s)=s.
$$

Then

$$
\boxed{
F^R_{tz}=0
\iff
A_z(t,\phi_t(s))\partial_s\phi_t(s)=A_z(t_0,s)
\quad\text{for every }s.
}
$$

Consequently every transported subinterval is exactly reusable:

$$
U_t(\phi_t(s_1),\phi_t(s_0))
=U_{t_0}(s_1,s_0).
$$

The forward implication is the conservation law along the flow. Conversely,
if every transported subinterval transfer is invariant, shrink a subinterval
to a point; its infinitesimal generator is invariant, yielding the pullback
identity and hence zero curvature.

For merely depth-Lipschitz `w`, the flow is orientation-preserving and
bi-Lipschitz, but the displayed Jacobian identity should be read almost
everywhere (or as equality of pulled-back optical measures), not as an
everywhere pointwise `C^1` statement.

For a whole clipped ray, flatness plus endpoints following the same flow
guarantees constant transfer:

$$
\dot a=w(t,a),
\qquad
\dot b=w(t,b).
$$

Without endpoint advection, constancy is not guaranteed because endpoint flux
remains. Endpoint advection is sufficient, not necessary in every special
field: two non-advected endpoints can move together through a homogeneous
generator while preserving their separation and total transfer.

### 5.2 Why “flat iff one total transfer is reusable” is false

On `z in [0,1]`, take `A_t=w=0` and

$$
A_z(t,z)=\left(1+\varepsilon t\cos(2\pi z)\right)X_0,
$$

where `X_0=X(-1,c)` and `|epsilon t|<1`. All generators commute, so

$$
U(t)=\exp\!\left(\int_0^1A_z(t,z)\,dz\right)=\exp(X_0)
$$

is exactly constant, while

$$
F^R_{tz}=\varepsilon\cos(2\pi z)X_0\ne0.
$$

The transported curvature cancels after integration. Flatness is necessary
only for path independence or reuse of **all subintervals**, not equality of a
single total ray transfer.

Flatness also does not remove endpoint flux. For

$$
A_z(t,z)=f(z-t)X_0,
\qquad
w=1,
$$

the interior curvature vanishes, but the transfer on fixed clips `[0,1]`
changes as optical mass crosses the endpoints.

### 5.3 A one-track scalar flow is not a general 3D correspondence

Let a world-space scene flow be `V(x,t)`. A scalar depth flow along one fixed
pixel track is physically realizable only if

$$
V(\Gamma(t,z),t)
=\partial_t\Gamma(t,z)+w(t,z)\partial_z\Gamma(t,z).
$$

Generic camera or object motion has a transverse component and moves material
between pixels. The full ray bundle needs a horizontal field

$$
H
=\partial_t+v_u\partial_u+v_v\partial_v+w\partial_z,
$$

with

$$
D\Gamma(H)=(V,1).
$$

A single depth-dependent `w` can carry separated layers with different axial
velocities when one Lipschitz, single-valued, non-crossing flow interpolates
them. It cannot represent transverse motion, two velocities at the same
`(t,z)`, crossings, splitting/merging, or layer exchange across pixels. Exact
one-track reuse therefore requires a strong invariance condition; general
coherent motion requires a shared, compact sensor-depth flow and cross-pixel
transport.

This connects the idea to canonical/deformation/scene-flow dynamic rendering.
It does not establish novelty by terminology alone.

### 5.4 Gauge cheating

An unconstrained connection can encode the answer. In the repository
convention, choosing

$$
A_t(t,b)=U(t)^{-1}\dot U(t),
\qquad
A_t(t,a)=0
$$

makes the covariant derivative of the total transfer vanish. Locally, with
`g(t,z)=U_t(z,a)`, the choice

$$
A_t=g^{-1}\partial_tg
$$

makes the right connection flat.

Even after imposing `A_t=-wA_z`, any smooth positive scalar density of fixed
total mass can be flattened by a per-ray flow. One solution is

$$
w(t,z)
=\frac{C(t)-\int_a^z\partial_t\lambda(t,s)\,ds}{\lambda(t,z)},
$$

with `C=0` for zero near-end flux. Color introduces additional constraints,
but opacity curvature alone is vacuous if `w` is freely fitted per ray.

Admissible `w` must therefore be generated by a capacity-bounded shared scene
or camera motion model, use a declared basis independent of requested frames,
and have all parameter, fitting, storage, and gradient costs charged to the
method. Joint RGB supervision is legitimate; an unconstrained `A_t` or
per-ray/per-frame `w` conditioned on `U` or targets as an answer table is not.
The first falsifiable capacity gate is

$$
\operatorname{DoF}_t(w)
\le
\operatorname{DoF}_t(\text{site motion} + \text{camera motion}),
$$

with a direct-`U` control given the same added temporal DoF and retained bytes.

## 6. Distributional Curvature and Closed-Loop Holonomy

### 6.1 Moving P0 interfaces

For an interface `z=r(t)`, write

$$
A_z=A_z^-1_{z<r}+A_z^+1_{z>r},
\qquad
[A_z]=A_z^+-A_z^-.
$$

The general singular curvature for a BV connection is

$$
\boxed{
F_{\mathrm{sing}}
=\left([wA_z]-\dot r[A_z]\right)\delta_{z=r}.
}
$$

Only when `w` has one continuous trace at the boundary does this reduce to

$$
(w(r)-\dot r)[A_z]\delta_{z=r}.
$$

Its repo-ordered transported contribution is

$$
\boxed{
K_{\mathrm{sing},r}
=U_t(r,a)
\left([wA_z]-\dot r[A_z]\right)
U_t(b,r).
}
$$

An unrestricted flow that interpolates every cut velocity can erase every
singular term. The interpretation “boundary curvature equals correspondence
mismatch” is meaningful only for independently constrained flow.

The discontinuous-`w` case is a BV connection diagnostic outside the `C^1`
flow theorem in Section 5.1; it is not evidence for one globally admissible
orientation-preserving material flow.

A moving noncommuting-boundary sentinel is

$$
U=e^{(r-a)A^-}e^{(b-r)A^+},
$$

for which, at `w=0`,

$$
\dot U=-\dot r\,U_t(r,a)[A_z]U_t(b,r).
$$

### 6.2 Genuine holonomy

Holonomy belongs to a closed ray-time plaquette, not an open camera ray. With
right temporal scans satisfying

$$
\partial_tH_z=H_zA_t,
\qquad
H_z(t_0,t_0)=I,
$$

a positively oriented repo-convention plaquette may be written

$$
\operatorname{Hol}^R_+(R)
=H_aU_{t_1}H_b^{-1}U_{t_0}^{-1}.
$$

For a small rectangle,

$$
\operatorname{Hol}^R_+(R)
=I+F^R_{tz}\,\Delta t\,\Delta z+o(\Delta t\Delta z).
$$

Reversing loop orientation flips the sign. The loop written in the scientist
attachment traverses the opposite orientation under its own convention and
therefore has `I-F Delta t Delta z`, not the stated plus sign. Constant
noncommuting `A_z,A_t` is the sign test.

On a simply connected flat chart,

$$
U_{t_1}=H_a^{-1}U_{t_0}H_b.
$$

This is the proper geometric role of holonomy: it measures obstruction to
path-independent identification of neighboring open-ray transports.

## 7. Curvature-Source Representation: Hypothesis, Not Result

Define

$$
K_F(t)
=H_a
\left[
\int_a^bU_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds
\right]
H_b^{-1}.
$$

Then

$$
\frac{d\widetilde U}{dt}=K_F(t),
$$

and

$$
\widetilde U(t)
=\widetilde U(t_0)+\int_{t_0}^tK_F(\tau)\,d\tau.
$$

If `K_F_hat` approximates `K_F` with uniform matrix error `epsilon_F` over
duration `T`, then

$$
\|\widetilde U-\widehat{\widetilde U}\|_\infty
\le T\varepsilon_F.
$$

Recovering `U` introduces condition factors from `H_a` and `H_b`; errors in
the fitted flow, endpoint transports, or curvature quadrature add separately.
Additive integration in the group completion can leave the physical
contraction/color cone, so reconstructed transfers require the same fail-closed
cone certification as the current Lie atlas.

There is an additional representation boundary. In general,

$$
\widetilde U=H_aUH_b^{-1}
$$

is a valid affine **group-completion** element but not a physical optical
transfer: its attenuation can exceed one and its moment can be signed. The
current atlas requires `kappa>=0` and `0<=v_c<=kappa`, so it cannot consume
`U_tilde` unchanged. The first oracle must use an unrestricted `beta>0`
affine-group chart and apply the physical cone only after reconstructing `U`.
Likewise, `K_F=dU_tilde/dt` is a signed tangent matrix, not a transfer, and
must not be passed through the current transfer decoder or cone ABI.

### 7.1 Why low curvature does not automatically lower rank

- A derivative can have equal or greater approximation rank than its
  antiderivative; `sin(omega t)` and `omega cos(omega t)` are the elementary
  warning.
- Monotone depth reparameterization already leaves the total ordered transfer
  invariant when optical density is transformed correctly. An advected slab
  may have `J_U=1` before any curvature quotient.
- Curvature can cancel in depth, as the cosine counterexample shows.
- The flow, endpoint transports, and their selected tangents may carry the
  same complexity supposedly removed from `U`.
- Gauge correction may be useful even when integrating `K_F` is not. The
  oracle can reuse the current atlas's approximation family for `U_tilde`, but
  needs the unrestricted group-completion chart described above rather than
  the production physical-cone ABI.

The correct cost comparison is therefore

$$
\operatorname{cost}(U)
\quad\text{vs}\quad
\operatorname{cost}(w,H_a,H_b,\widetilde U)
\quad\text{vs}\quad
\operatorname{cost}(w,H_a,H_b,K_F,\text{integration}).
$$

It must include fitting, storage, reconstruction, cone certification, forward
work, selected-tangent rank, gradients through `w`, and endpoint conditioning.

### 7.2 Systems boundary and first ablation

For a fixed certified atlas, distinguish track-chart rows `R_row,b` from
ordered owner/run entries `W_run,b`. The direct route dispatches
`sum_b R_row,b J_U,b` forward threads, but both forward and reverse scan the
ordered words. Its ordered-word and sample work are therefore

$$
W_U^{\mathrm{word}}
=2\sum_bJ_{U,b}W_{\mathrm{run},b},
\qquad
W_U^{\mathrm{sample}}
=\Theta\!\left(
\sum_rF_rJ_{U,r}
+\sum_rN_{\mathrm{fb},r}J_{U,r}^2
+PF
\right).
$$

The connection can improve these terms only by lowering certified ragged node
counts `J`; it does not remove ordered depth `W_run`, fallback work, unavoidable sample/output
work, or the already frame-density-independent compiled reverse theorem.

A discrete transported-curvature product still has tangent multiplication

$$
(G,K)\otimes(H,L)=(GH,KH+GL),
$$

so for ordered pieces

$$
K_{\mathrm{total}}
=\sum_iG_{<i}K_iG_{>i}.
$$

That remains `Theta(R)` per temporal node unless a product tree or cached
subinterval transports are retained, which introduces its own memory/update
cost. Sparse curvature support alone does not erase prefix/suffix transport.

Therefore the first computational ablation is **direct approximation of
`U_tilde` with the existing atlas's approximation family but an oracle-local,
unrestricted affine-group chart**. It decides whether the independently
specified flow removes temporal complexity without first adding a curvature
integrator. It does not reuse the production physical-cone ABI unchanged.
Only if that comparison is favorable should signed tangent `K_F` be evaluated
as a candidate runtime representation.

## 8. Ordered Interval Factorization Without Category Overclaim

For an ordered adjacent partition

$$
[a,b]=J_1\cup\cdots\cup J_k,
$$

the fundamental-solution property gives

$$
U_{[a,b]}=U_{J_1}\star\cdots\star U_{J_k}.
$$

Associativity makes this invariant under ordered refinement and
parenthesization. The precise safe language is **multiplicative interval
transport** or a **monoid-valued functor on oriented interval
concatenations**.

The attachment's assignment `F(J)=U_J in G` is not a factorization algebra:
it assigns a computed element rather than an algebraic object, and disjoint
subintervals that do not cover `J` omit the transfer through their gaps. For
constant extinction on `[0,3]`, the transfers of `[0,1]` and `[2,3]` multiply
to attenuation `exp(-2 rho)`, not the full `exp(-3 rho)`.

For a truly local update

$$
G=P\star M\star S,
\qquad
M\mapsto M',
$$

exact repair is

$$
G'=P\star M'\star S.
$$

A balanced tree updates `s` contiguous leaves in `O(s+log R)`, but adds about

$$
16J(R-1)\ \text{bytes}
$$

of float32 internal transfers per compiled row, plus indices, and changes
floating-point parenthesization. Dense world/camera updates touch broad
support, so this provides neither a temporal-scaling theorem nor an automatic
memory win over the current prefix-only `O(JR)` replay.

Numerical transfer is stratumwise analytic; that does not make it a
constructible sheaf, whose stalks are locally constant. The structural owner
program may be locally constant on a stratum. Current code has no transition
morphisms, descent data, conflict objects, or 2-morphism coherence, and it
fails closed on simultaneous/full-fiber events. Do not build a software
“sheaf” or “stack.”

## 9. Tangent Semigroup and Corrected Seam Theorem

### 9.1 Pointwise selected jets

For a transfer and one directional derivative, define

$$
(g,\dot g)\odot(h,\dot h)
=\left(gh,\dot g\,h+g\,\dot h\right).
$$

This product is associative by differentiating `(gh)k=g(hk)`. In affine
coordinates,

$$
\dot\beta
=\dot\beta_1\beta_2+\beta_1\dot\beta_2,
$$

$$
\dot m
=\dot m_1+\dot\beta_1m_2+\beta_1\dot m_2.
$$

For `k` selected directions the pointwise state has `4(1+k)` scalars. This is
useful proof algebra but should not become stored forward-jet state for general
training; reverse cotangents plus sparse VJPs remain more memory efficient.

### 9.2 One-sided order-`q` seam proposition

Let `q>=1` be an integer and `x=t-e`. Suppose `P_+`, `P_-`, `S_+`, `S_-`,
`ell_+`, `ell_-`, `X_+`, and `X_-` denote one-sided `C^q` germs with `C^q`
extensions to one common neighborhood of `x=0` (equivalently, interpret the
comparisons below as matching one-sided Taylor/Peano jets). Suppose they
satisfy

$$
G_\pm(x)=P_\pm(x)e^{\ell_\pm(x)X_\pm(x)}S_\pm(x),
$$

$$
P_+(x)-P_-(x)=o(|x|^q),
\qquad
S_+(x)-S_-(x)=o(|x|^q),
$$

$$
\ell_\pm(x)=a_\pm x^q+o(|x|^q),
\qquad
X_\pm(x)\to X_\pm^0.
$$

Here `a_-` uses the same signed coordinate `x=t-e`. Then

$$
G_+(x)-G_-(x)
=x^qP_0(a_+X_+^0-a_-X_-^0)S_0+o(|x|^q).
$$

The traces agree through order `q-1`, and

$$
\boxed{
G_+^{(q)}(e)-G_-^{(q)}(e)
=q!P_0(a_+X_+^0-a_-X_-^0)S_0.
}
$$

An absent segment contributes zero. The scientist note's
`q! a P(X_+-X_-)S` is only the special case with common exterior germs and
`a_+=a_-=a`. Neither a bare big-`O` nor little-`o` asymptotic, without the
declared one-sided `C^q`/Peano regularity, is enough to guarantee classical
`q`-th derivatives.

### 9.3 Correct event filtration

For a declared transfer norm and selected direction family `D`, record the
observable seam defects

$$
\delta_e^{(0)}=\|G_+(e)-G_-(e)\|,
$$

$$
\delta_e^{(1)}(D)
=\sup_{v\in D,\ \|v\|\le1}
\|D_vG_+(e)-D_vG_-(e)\|.
$$

Higher cumulative-jet defects are defined analogously. These are diagnostics,
not topology labels: an owner rewrite can have zero primal defect and a
nonzero geometry or provenance-action defect.

Define

$$
\Sigma_{\le q}^D
=\{e:\text{some required transfer derivative in direction family }D
\text{ through order }q\text{ fails to glue}\}.
$$

For seams caused solely by owner events,

$$
\boxed{
\Sigma_{\le0}^D
\subseteq\Sigma_{\le1}^D
\subseteq\cdots
\subseteq\Delta_{\mathrm{owner}}.
}
$$

If a defect compares exactly the `q`-th derivative rather than the cumulative
jet, the sets need not be nested. Material, geometry, camera, and provenance
directions require separate filtrations. Approximation-induced chart splits
need not be owner events at all.

### 9.4 Coarser transfer cover

The one genuinely new compiler idea in the jet formulation is a two-level
atlas:

```text
exact owner subcharts and provenance
              |
              v
coarser transfer charts and sample-to-node schedules.
```

Owner charts remain exact for geometry and provenance, while adjacent transfer
charts may merge only when primal and every required tangent action glue to
tolerance. Current source assumes one fixed owner word and `[J,R]` lengths per
transfer chart, so this is not implemented. Nodes spanning seams would require
ragged heterogeneous words or interpolation over existing owner-chart nodes.

Before implementing that branch, census `delta_e^(0)` and `delta_e^(1)(D)`
over every supported generic event. If required tangent defects remain large
at nearly every owner seam, a two-level cover cannot remove meaningful work.

Promotion requires a reduction in actual work, not chart count alone:

$$
\sum_d\widetilde J_d^2,
\qquad
\sum_dF_d\widetilde J_d,
\qquad
\sum_d\widetilde J_dR_d
$$

must fall without larger certificate cost or native divergence. The existing
continuous certificate covers material actions, not the needed geometry/event
jets, so this remains future work.

## 10. Parameter Discriminant and Real-Root Continuation

Let `theta in Theta` denote world/camera parameters and let
`h_alpha(t;theta)` be the actual predicate registry. A conceptual parameter
discriminant includes repeated roots, relevant shared roots, endpoint roots,
degree loss/root-at-infinity, identically zero predicates, and non-root guard
failures.

Using all pairwise resultants is too conservative: unrelated predicates can
share harmless complex roots. The current local Bernstein/root-tube
certificates over the exact operational registry are more useful than building
a global algebraic discriminant.

Define the regular simple-root space

$$
\mathcal E
=\{(\theta,t,\alpha):
h_\alpha(t;\theta)=0,
\ \partial_th_\alpha(t;\theta)\ne0\}.
$$

Under finite-root and endpoint-separation assumptions, projection to regular
parameter space is locally a finite covering. This is the mathematical content
of the existing simple-root continuation oracle.

**Real-order lemma.** On a connected regular region, if every relevant root
remains real, simple, interior, distinct, and the total root count is fixed,
then

$$
\tau_1(\theta)<\cdots<\tau_E(\theta)
$$

are global continuous labels. Real-time ordering has trivial monodromy because
two real roots cannot exchange order without colliding.

Complex braid monodromy is real mathematics but not a physical event-dispatch
mechanism here. A future warm in-place program updater could be tested by
continuing around a closed parameter loop and comparing its endpoint to a
fresh canonical semantic compile. Raw digests are insufficient because root
isolator refinements and internal IDs may differ. No such updater exists, and
current simultaneous events fail closed, so program monodromy and 2-stack
coherence are deferred.

The parked coherence obligations are nevertheless explicit. Disjoint local
rewrites should commute,

$$
T_aT_b=T_bT_a,
$$

while a supported family of adjacent pure-order swaps would need the braid
relation

$$
T_iT_{i+1}T_i=T_{i+1}T_iT_{i+1}.
$$

Births, deaths, clipping events, and full-fiber ties need their own coherence
cells. These equations are future correctness tests, not authorization to
continue through currently unsupported simultaneous events.

## 11. Unified Safe Formulation

The least inflated description that retains all sound content is:

> WorldFoam is a stratified ray bundle with fiberwise, right-ordered affine
> optical transport. Its owner program is locally constant on regular strata;
> its transfer is stratumwise analytic; selected tangents form an associative
> jet semigroup. A compact, independently constrained scene/camera flow induces
> a horizontal optical connection whose curvature measures the residual
> failure of transported subinterval reuse.

This avoids pretending that a custom “ordered direct image,” factorization
cosheaf, constructible stack, or braid runtime has been implemented.

### 11.1 Do not conflate the paper's two connections

The current paper draft already defines a **cell-frame adjacency connection**.
Its transport maps compare tangent frames across witnessed radical faces, and
its closed-loop product is a diagnostic for inconsistency of a spatial cell
complex.  That object lives on the cell-adjacency graph and acts on tangent
frames.

The connection in this audit is instead the **ray-fiber optical connection**

$$
\Omega_{\mathrm{opt}}=A_z\,dz+A_t\,dt,
$$

on the affine radiance-state bundle over a ray-time strip.  Its vertical open
paths render radiance, and its closed ray-time plaquettes diagnose the failure
of neighboring optical transports to be path independent.  The two objects
have different bases, fibers, structure groups, curvatures, and operational
tests.  If this analysis enters the manuscript, use the qualified names
`ray-fiber optical connection` and `cell-frame adjacency connection`; never
write an unqualified `connection curvature` or `holonomy` where either could
be meant.

## 12. Decisive Oracle Before Any Runtime Branch

If the current paper/runtime gates are closed first, a source-only
`kinetic_optical_curvature_oracle.py` may consume one exact stable chart:

- exact cuts `z_r(t)` and cut velocities;
- ordered owner generators, including physical ray Jacobians;
- one independently generated low-dimensional scene/camera flow;
- exact total transfer and selected sparse parameter directions; and
- explicit near/far endpoint trajectories.

It must keep three representation types distinct:

- physical transfer `U`, using the current physical cone;
- group-completion `U_tilde`, using an unrestricted `beta>0` affine chart;
- signed tangent `K_F`, using a generic four-component tangent basis.

It should produce:

1. bulk curvature `partial_t A_z+partial_z(w A_z)`;
2. general singular terms `([wA_z]-dot r[A_z]) delta_r`;
3. moving-endpoint flux;
4. the repo-ordered transported curvature integral;
5. an independently differentiated covariant derivative;
6. small-plaquette holonomy with an orientation receipt;
7. equal-tolerance ranks `J_U`, `J_U_tilde`, and `J_F`;
8. the same ranks for selected material and geometry tangents;
9. flow, endpoint, reconstruction, and certificate bytes/work; and
10. reconstructed-`U` physical-cone and endpoint-conditioning reports.

The word `rank` in this gate means certified approximation complexity, not an
unqualified polynomial degree. Fix a compact chart interval `I`, one declared
approximation family `A_J`, a normed selected-direction set `D`, and tolerances
`epsilon_0,epsilon_1`. For any represented quantity `X`, define

$$
\boxed{
J_X(\epsilon_0,\epsilon_1;D)
=\min\left\{
J:\exists\widehat X_J\in\mathcal A_J,
\ \sup_{t\in I}\|X-\widehat X_J\|\le\epsilon_0,
\ \sup_{\substack{t\in I,\ v\in D\\\|v\|\le1}}
\|D_vX-D_v\widehat X_J\|\le\epsilon_1
\right\}.
}
$$

For `U_tilde` and `K_F`, the certificate is evaluated after reconstructing the
same physical transfer `U`; flow, endpoint-transport, quadrature/integration,
and cone-certification error consume the same end-to-end error budget. Compare
total certified bytes and work, not the three raw `J_X` values in isolation.
Record both maximum rank and total ragged node count; a lower maximum with more
charts is not a computational win.

`U_tilde` itself is not required to satisfy the physical contraction cone, but
must stay inside the nonsingular affine group (`beta_tilde>0`). `K_F` is not
assigned a transfer cone at all. Additive reconstruction from `K_F` must fail
closed if it crosses `beta_tilde<=0` or reconstructs an unphysical `U`.

Required correctness fixtures:

- front-red/back-blue ordering;
- moving noncommuting boundary;
- advected differently colored slabs with moving clips;
- boundary mismatch with `dot r != w`;
- material evolution with zero singular and nonzero bulk curvature;
- discontinuous `w` requiring `[wA_z]-dot r[A_z]`;
- constant noncommuting connection for holonomy orientation;
- cosine-depth curvature cancellation;
- flat translating density with fixed endpoint flux; and
- sideways camera motion that violates scalar-depth-flow realizability.

At least these two fixtures are numerically pinned so an oracle cannot pass on
commuting or scalar-flow-compatible cases:

1. On `[a,b]=[0,3]`, use a moving interface `r(t)=1+t`, `w=0`,
   `A^-=X(-1,e_R)`, and `A^+=X(-2,2e_B)`. Then
   `[A^-,A^+]=X(0,2e_R-2e_B) != 0`, and the independent derivative must equal
   `-U(r,a)[A_z]U(b,r)` at `t=0`.
2. On a clipped positive-depth interval `0<z_min<=z<=z_max`, use the static
   pinhole ray map `Gamma(u,v,t,z)=z(u,v,1)`, the central fixed track
   `(u,v)=(0,0)`, and world flow `V=e_x`. No scalar `w` can satisfy
   `V=partial_t Gamma+w partial_z Gamma` on that track, although the full
   sensor-depth lift represents it with `v_u=1/z`.

The advected slab is an identity test, not evidence of compression: direct
`U` should already be constant.

Initial numerical gates are an identity residual no larger than `1e-9` in
well-conditioned float64 fixtures, reconstructed forward error no larger than
`1e-5`, and normalized selected-gradient error no larger than `1e-4`. These do
not replace the continuous certificate.

Do not promote a curvature runtime unless, on representative real nontrivial
charts and under the same continuous primal/tangent norm:

- total retained payload and ordered-word work
  `sum_b J_b W_run,b` improve by at least `2x` against both direct `U` and
  direct `U_tilde`;
- predicted and then measured request time improves by at least 20%;
- the flow uses no requested-frame or hidden per-ray answer table;
- all flow and endpoint gradients are included;
- reconstruction remains inside the physical cone; and
- the benefit survives geometry directions, not only a frozen-flow material
  surrogate.

Any per-frame flow state, tangent-rank regression, failed conditioning, or
sub-20% request-time gain kills the runtime branch. Curvature remains a theorem
and correspondence diagnostic.

## 13. Relation to Current WorldFoam Work

### Already present

- exact four-scalar affine transfer and ordered word composition;
- compact Lie-chart temporal approximation;
- continuous primal and selected material-action certification;
- event/root charting and restricted simple-root continuation;
- prefix-only constant-state word VJP;
- frame-density-independent compiled word work at fixed program complexity;
- streamed `B_p x K` target/sample processing; and
- source-written fused owner-local reverse without a `[J,W]` length-cotangent
  output. The fixed-camera coordinator can now select this fused reverse
  explicitly, and the combined CPU updater preserves the selected-mode receipt;
  staged sparse remains the default and these newest source/tests are unrun.

### New mathematical results worth retaining

- the corrected repo-native curvature-variation identity;
- the constrained-flow flatness theorem for every transported subinterval;
- the general P0 boundary-curvature measure;
- the explicit counterexample to the one-total-ray converse;
- the scalar-flow realizability limitation and full sensor-depth lift;
- the corrected one-sided order-`q` seam proposition;
- the cumulative event-regularity filtration; and
- the real simple-root trivial-monodromy lemma.

### Not implemented and not a current blocker

- curvature-source compilation;
- sensor-time patch compilation;
- a two-level owner/transfer atlas;
- balanced-tree warm numeric repair;
- program monodromy;
- simultaneous-event coherence; and
- sheaf, stack, or 2-stack runtime objects.

The immediate paper path remains unchanged in substance. The coordinator and
CPU-update source seam is now written, but it must first pass its focused
CPU/fake-native gates on a safe host. Then build and attest fused v1, match
staged/fused float64 forward and gradient oracles, run poison/fence and native
allocator gates, add the missing production-trainer/evaluator routing, and run
the fixed-dataset `F=8/64/300` scaling matrix plus public quality rows.
Moving/projective/gauged-camera support must either land later or be explicitly
narrowed in the implementation claim.

## 14. Branches, Backtracks, and Stop Rules

| Branch considered | Why it looked promising | What changed the conclusion |
| --- | --- | --- |
| Rename the renderer as a fiber-bundle/holonomy method | Unifies camera rays, depth, and transport geometrically | The ray bundle is locally trivial and open-ray transport already exists; topology adds no runtime. |
| Compile `K_F` instead of `U` immediately | Coherent motion can have zero curvature | `U` may already be constant; derivative rank need not fall; flow/endpoints add state and gradients. |
| Treat interval products as a factorization cosheaf | Ordered local-to-global composition resembles `E_1` structure | The proposed assignment is an element, omits gaps, and lacks actual sheaf/factorization data. |
| Merge every optically invisible owner seam | Could reduce chart count | Geometry/provenance tangents can remain discontinuous, and the native ABI assumes one fixed word per chart. |
| Use braid monodromy for root tracking | Polynomial roots can braid in complex parameter space | Separated interior real roots have canonical sorted labels; current updater recompiles and simultaneous events fail closed. |
| Fit arbitrary `w` to make curvature small | Produces an apparently flat representation | It can hide the whole answer; `w` must come from independent bounded scene/camera motion. |

Stop theory expansion after the oracle decision. A failed computational gate
does not invalidate the curvature theorem; it means the theorem belongs in
analysis/diagnostics rather than the runtime or paper headline.

## 15. Literature Boundary and Citation Corrections

- Local owner triviality follows directly from the implicit-function theorem;
  the attachment's `math/0604428` citation is a mismatched paper and should be
  removed rather than used as authority.
- Ginot's factorization-algebra notes are legitimate background for locally
  constant factorization algebras and `E_n` structure, but they do not make the
  proposed element-valued assignment a factorization algebra:
  <https://arxiv.org/abs/1307.5213>.
- Woolf studies fundamental categories classifying constructible sheaves and
  cosheaves: <https://arxiv.org/abs/0811.2580>.
- The appropriate constructible-stack/exit-path 2-category reference is
  Treumann: <https://arxiv.org/abs/0708.0659>. This remains background, not an
  implemented WorldFoam construction.
- Complex braid monodromy of nonsingular univariate polynomials is genuine but
  does not overturn the real-order lemma:
  <https://arxiv.org/abs/2001.01634>.
- Canonical/deformation and scene-flow dynamic rendering are important novelty
  context for any Lagrangian quotient claim. See D-NeRF:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Pumarola_D-NeRF_Neural_Radiance_Fields_for_Dynamic_Scenes_CVPR_2021_paper.html>
  and Forward Flow:
  <https://openaccess.thecvf.com/content/ICCV2023/html/Guo_Forward_Flow_for_Novel_View_Synthesis_of_Dynamic_Scenes_ICCV_2023_paper.html>.

## Final Decision

The intake is useful and contains the best new mathematical hypothesis so far,
but it does not supersede the existing memory-light architecture.

Keep and prove:

```text
specified compact flow
  -> constrained horizontal optical connection
  -> curvature as the residual of transported subinterval reuse.
```

Falsify before implementing:

```text
does group-completion U_tilde or signed-tangent K_F reduce total
primal+tangent state and work
after charging flow, endpoints, reconstruction, and gradients?
```

Park the categorical superstructure and finish the fused fixed-camera
WorldFoam trainer path first.
