# WorldFoam Measure--Connection Synthesis

Date: 2026-08-05

Status: canonical synthesis of the translated optical-depth measure and the
new constrained Lagrangian optical-connection proposal. The mathematical
identities are audited. The curvature oracle, unrestricted group-completion
atlas, equal-certificate rank comparison, native runtime, and performance
claims are not implemented or measured.

Primary inputs:

- `WORLD_FOAM_PAPER_DRAFT.md`, especially the translated optical-depth-measure
  theorem;
- `WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md`;
- `WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`;
- the scientist attachment with SHA-256
  `965c7a1a28343914dd348a88afa1b30a976dabd6dbf80fb48a1076ad878334c5`.

This note is source- and mathematics-only. No Python, import, test, build,
Metal, MPS, CUDA, dataset, or training workload was run on this host.

## Executive Decision

The strongest unified formulation is:

```text
ordered optical field along depth
  -> translated optical-depth measure (kappa, nu)
  -> four-scalar affine transfer (beta, m)
  -> bounded shared scene/camera flow
  -> flow-covariant transfer U_tilde
  -> optical-curvature residual K_F.
```

These are layers of one method, not competing renderers.

- `(kappa,nu)` is the order-explicit proof and tangent object.
- `(beta,m)` is the compact exact runtime quotient.
- The existing event atlas preserves changing depth order and differently
  colored overlap.
- The new horizontal connection asks which temporal change is merely coherent
  transport and which change is irreducible after a declared correspondence.
- Curvature can lower temporal atlas complexity only if a bounded shared
  compact flow makes `U_tilde` or `K_F` cheaper under the same
  primal-and-tangent certificate.

The connection is the best new mathematical hypothesis in the latest memo. It
does not supersede the current memory-light trainer. It targets temporal rank
and chart growth, not the four-scalar executor state and not the unavoidable
sample/output slice.

The safe implementation order is:

```text
finish and validate direct U
  -> build a CPU float64 group-completion oracle for U_tilde
  -> compare U against U_tilde under one end-to-end certificate
  -> compute K_F in the oracle only
  -> consider a native branch only after a decisive measured win.
```

### Coverage of the scientist memo

No proposed mathematical branch is discarded silently:

| Memo branch | Canonical disposition |
| --- | --- |
| Stratified ray fibration / ordered direct image | Sections 1 and 3 here; qualified local-triviality theorem and cross-pixel patch kill test in Sections 2--3 of the full audit. |
| Lagrangian optical connection / curvature / holonomy | Sections 4--9 here contain the repo-ordered equations, flow lift, BV interface term, and three-way ABI split; Sections 4--7 of the audit contain proofs and counterexamples. |
| Factorization cosheaf | The exact interval-concatenation law is retained in audit Section 8. The unsupported categorical overclaim is rejected; no math needed by the renderer is lost. |
| Jet stack and seam defects | The associative selected-jet product, corrected one-sided order-`q` seam theorem, cumulative event filtration, and two-level-atlas hypothesis are retained in audit Section 9. |
| Optimizer monodromy | The real-simple-root trivial-monodromy theorem and the boundary of any future closed-loop diagnostic are retained in audit Section 10. Complex braid/2-stack machinery is deferred. |

The full theorem-by-theorem source is
`WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`; this file
is the executable synthesis with the existing translated-measure and
memory-light architecture.

## 1. Three Different Objects

### 1.1 Ray-depth bundle

Let the sensor-time base be

$$
B\subset\mathbb R^2_{u,v}\times I_t.
$$

The clipped ray-depth space is

$$
\mathcal R
=\{(b,z):b\in B,\ z_-(b)\le z\le z_+(b)\},
\qquad
\pi:\mathcal R\to B.
$$

With spatial ray map `x=Gamma_x(b,z)`, the physical spacetime evaluation is

$$
\Gamma(b,z)=(\Gamma_x(b,z),t_b).
$$

The kinetic cell partition pulls back to ordered owner intervals on every
fiber. This bundle is locally trivial on every regular chart; its topology is
not itself the computational contribution.

### 1.2 Optical radiance bundle

The radiance state is affine `R^3`, represented homogeneously in `R^4`. The
optical transfer group is four-dimensional. It is more precise to write

$$
\mathcal E\longrightarrow\mathcal R\longrightarrow B
$$

than to call the radiance fiber itself four-dimensional.

### 1.3 Owner/event program

The owner word, cut identities, provenance, material attribution, and sparse
geometry action are richer than the rendered transfer. Two different programs
can render the same `(beta,m)`. Exact owner topology must therefore remain
available for geometry gradients even when transfer charts can be coarsened.

## 2. Vertical Ordered Transport

### 2.1 Affine group

For rear radiance `q in R^3`, define

$$
T(\beta,m)
=
\begin{bmatrix}
\beta I_3&m\\
0&1
\end{bmatrix},
\qquad
q\mapsto m+\beta q,
\qquad
\beta>0.
$$

Composition is

$$
T(\beta_1,m_1)T(\beta_2,m_2)
=T(\beta_1\beta_2,m_1+\beta_1m_2).
$$

The inverse in the group completion is

$$
T(\beta,m)^{-1}
=T(\beta^{-1},-\beta^{-1}m).
$$

Physical emission--absorption occupies only a contraction/color cone inside
this group. For bounded RGB, a simple cone is

$$
0<\beta\le1,
\qquad
0\le m_c\le1-\beta.
$$

The Lie algebra elements are

$$
X(a,b)
=
\begin{bmatrix}
aI_3&b\\
0&0
\end{bmatrix},
$$

with

$$
[X(a,b),X(c,d)]=X(0,ad-cb).
$$

For a P0 interval with coordinate extinction `lambda` and emitted-density
vector `eta=lambda*c`,

$$
A_z=X(-\lambda,\eta).
$$

In an arbitrary camera depth gauge,

$$
\lambda=\rho\,\|\partial_z\Gamma_x\|,
\qquad
\eta=\lambda c.
$$

The physical ray-speed Jacobian is load-bearing.

### 2.2 Executable WorldFoam order

WorldFoam scans near to far. For `a<s<b`,

$$
\boxed{
U_t(b,a)=U_t(s,a)U_t(b,s).
}
$$

Consequently,

$$
\partial_bU_t(b,a)=U_t(b,a)A_z(t,b),
\qquad
\partial_aU_t(b,a)=-A_z(t,a)U_t(b,a).
$$

The latest scientist memo derives its main theorem in the opposite left-
ordered convention and later silently uses the repository order for interval
factorization. Its sandwiches and holonomy sign cannot be copied verbatim.

## 3. Translated Optical-Depth Measure

For a P0 word with optical depths `tau_r>=0` and colors `c_r`, define

$$
K_0=0,
\qquad
K_r=\sum_{q\le r}\tau_q,
\qquad
\kappa=K_R,
$$

and the vector measure

$$
d\nu(u)=c_r\,du,
\qquad
u\in[K_{r-1},K_r).
$$

Ordered concatenation is

$$
(\kappa_A,\nu_A)\odot(\kappa_B,\nu_B)
=
(\kappa_A+\kappa_B,
 \nu_A+S_{\kappa_A\#}\nu_B),
$$

where `S_a(u)=u+a`. Its Laplace image is

$$
\mathcal L(\kappa,\nu)
=
(\beta,m),
$$

$$
\beta=e^{-\kappa},
\qquad
m=\int_0^\infty e^{-u}\,d\nu(u).
$$

`L` is a monoid homomorphism:

$$
\mathcal L(A\odot B)
=
(\beta_A\beta_B,m_A+\beta_Am_B).
$$

The map is deliberately non-injective. `(kappa,nu)` remembers the ordered
color profile; `(beta,m)` retains exactly the declared action on arbitrary rear
radiance.

For one stable finite word, its tangent measure is

$$
\dot\nu
=
\sum_r\dot c_r1_{[K_{r-1},K_r)}du
+\sum_{r<R}(c_r-c_{r+1})\dot K_r\delta_{K_r}
+c_R\dot\kappa\delta_\kappa.
$$

Thus moving geometry becomes boundary-supported tangent mass in cumulative
optical depth. This is why a low-rank primal transfer need not have a low-rank
geometry/material tangent.

For `C` channels, `(beta,m)` is also the exact contextual quotient of a P0
word under arbitrary rear-radiance contexts. On the generic physical interior,
any `C^1` exact encoder/decoder needs at least `C+1` real coordinates. RGB
therefore needs at least four smooth coordinates. This validates the current
four-scalar ABI; it does not lower-bound owner-program, chart, or world state.

## 4. Horizontal Connection in Repo Convention

Introduce a horizontal generator `A_t(t,z)` and define

$$
\boxed{
F^R_{tz}
=\partial_tA_z-\partial_zA_t+[A_t,A_z].
}
$$

For fixed endpoints,

$$
\boxed{
\partial_tU
-UA_t(t,b)
+A_t(t,a)U
=
\int_a^b
U_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds.
}
$$

For moving endpoints, set

$$
B_a=A_t(t,a)+\dot aA_z(t,a),
\qquad
B_b=A_t(t,b)+\dot bA_z(t,b).
$$

Then

$$
\boxed{
\frac{dU}{dt}
=UB_b-B_aU
+\int_a^bU_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds.
}
$$

Let

$$
\dot H_a=H_aB_a,
\qquad
\dot H_b=H_bB_b,
\qquad
H_a(t_0)=H_b(t_0)=I,
$$

and define

$$
\widetilde U=H_aUH_b^{-1}.
$$

The flow-covariant derivative is

$$
\boxed{
\frac{d\widetilde U}{dt}
=H_a
\left[
\int_a^bU_t(s,a)F^R_{tz}(t,s)U_t(b,s)\,ds
\right]
H_b^{-1}
=:K_F(t).
}
$$

This is the exact relation between vertical ordered transport and temporal
curvature. It is a Duhamel/connection identity specialized to WorldFoam's
affine optical group and executable multiplication order.

## 5. Constrained Lagrangian Connection

Choose a depth velocity `w(t,z)` and horizontal vector

$$
H=\partial_t+w\partial_z.
$$

Impose

$$
A_t=-wA_z.
$$

The commutator vanishes pointwise and

$$
\boxed{
F^R_{tz}=\partial_tA_z+\partial_z(wA_z).
}
$$

With `A_z=X(-lambda,eta)`, define continuity residuals

$$
r_\lambda=\partial_t\lambda+\partial_z(w\lambda),
$$

$$
r_\eta=\partial_t\eta+\partial_z(w\eta).
$$

Then

$$
F^R_{tz}=X(-r_\lambda,r_\eta).
$$

Where `lambda>0` and `eta=lambda*c`, flatness is extinction conservation plus
color advection:

$$
\partial_t\lambda+\partial_z(w\lambda)=0,
$$

$$
\partial_tc+w\partial_zc=0.
$$

Use the `eta` equation directly through vacuum; division by `lambda` is not
valid there.

If `w` generates normalized orientation-preserving diffeomorphisms `phi_t`,

$$
\partial_t\phi_t(s)=w(t,\phi_t(s)),
\qquad
\phi_{t_0}(s)=s,
$$

then

$$
\boxed{
F^R_{tz}=0
\iff
A_z(t,\phi_t(s))\partial_s\phi_t(s)=A_z(t_0,s)
\quad\text{for every transported subinterval.}
}
$$

Consequently,

$$
U_t(\phi_t(s_1),\phi_t(s_0))
=U_{t_0}(s_1,s_0).
$$

For a whole clipped ray, endpoints must also follow the flow for this simple
constancy statement:

$$
\dot a=w(t,a),
\qquad
\dot b=w(t,b).
$$

This theorem is stronger than and different from equality of one total
transfer. Nonzero curvature may cancel after depth integration.

## 6. The Measure--Connection Bridge

Let

$$
d\mu_\lambda=\lambda(t,z)\,dz,
\qquad
d\mu_\eta=\eta(t,z)\,dz.
$$

On any transported subinterval, flatness is equivalent to

$$
\phi_t^*\mu_\lambda(t)=\mu_\lambda(t_0),
\qquad
\phi_t^*\mu_\eta(t)=\mu_\eta(t_0).
$$

Define cumulative optical depth

$$
q_t(z)=\int_{a(t)}^z\lambda(t,s)\,ds.
$$

The translated color measure is the pushforward of emitted-density measure by
this cumulative coordinate:

$$
\nu_t=(q_t)_\#\mu_\eta(t).
$$

When `lambda>0`, this reduces to `dnu(u)=c(u)du`. Therefore

$$
F^R_{tz}=0
\Longrightarrow
(\kappa,\nu)\text{ invariant on every transported subinterval}
\Longrightarrow
(\beta,m)=\mathcal L(\kappa,\nu)\text{ invariant there}.
$$

The converses must be separated:

- invariance of every transported generator/subinterval implies flatness;
- invariance of one total `(kappa,nu)` need not determine all subintervals;
- invariance of one total `(beta,m)` is weakest because `L` is non-injective.

This explains exactly why curvature cannot replace the translated-measure
proof. Curvature supplies a temporal correspondence residual; the measure
retains the depth-order information whose compact Laplace image renders RGB.

The boundary atoms in `dot nu` and the singular curvature below are two
coordinates on the same visibility phenomenon. The former is the ordinary
parameter tangent in cumulative optical depth. The latter is its covariant
residual after subtracting the declared material flow.

## 7. P0 Interfaces and Genuine Holonomy

At a moving interface `z=r(t)` with generator jump `[A_z]=A_z^+-A_z^-`, the
general BV singular curvature is

$$
\boxed{
F_{\mathrm{sing}}
=\left([wA_z]-\dot r[A_z]\right)\delta_{z=r}.
}
$$

Only when `w` has one continuous trace does it reduce to

$$
(w(r)-\dot r)[A_z]\delta_{z=r}.
$$

The repo-ordered transported contribution is

$$
K_{\mathrm{sing},r}
=U_t(r,a)
\left([wA_z]-\dot r[A_z]\right)
U_t(b,r).
$$

Endpoint flux remains separate unless the field is explicitly extended
outside the clipped interval.

Holonomy belongs to a closed ray-time plaquette. For right temporal scans, a
positive repo-oriented loop is

$$
\operatorname{Hol}^R_+(R)
=H_aU_{t_1}H_b^{-1}U_{t_0}^{-1},
$$

with small-area expansion

$$
\operatorname{Hol}^R_+(R)
=I+F^R_{tz}\Delta t\Delta z+o(\Delta t\Delta z).
$$

Open camera rays remain parallel transport, not holonomy.

## 8. Admissible Flow and Moving Cameras

A free per-ray `w` is invalid. It can absorb the answer and make many positive
scalar densities artificially flat. The flow must come from a bounded shared
scene/camera model, with every parameter, byte, evaluation, and gradient
charged. It may be jointly trained from RGB supervision; what is forbidden is
conditioning an unconstrained per-ray/per-time flow on the transfer or target
so that it stores the answer.

Let `V(x,t)` be a declared world-space velocity. For the full sensor-depth
bundle, use

$$
H
=\partial_t+v_u\partial_u+v_v\partial_v+w\partial_z.
$$

The spatial lift condition is

$$
\Gamma_{x,u}v_u
+\Gamma_{x,v}v_v
+\Gamma_{x,z}w
=V-\Gamma_{x,t}.
$$

Where the sensor-depth Jacobian is invertible,

$$
\boxed{
\begin{bmatrix}v_u\\v_v\\w\end{bmatrix}
=
[\Gamma_{x,u}\ \Gamma_{x,v}\ \Gamma_{x,z}]^{-1}
(V-\Gamma_{x,t}).
}
$$

On one fixed pixel track, scalar `w` is exact only if

$$
V-\Gamma_{x,t}=w\Gamma_{x,z}.
$$

The best axial projection is

$$
w_*
=\frac{\langle V-\Gamma_{x,t},\Gamma_{x,z}\rangle}
{\|\Gamma_{x,z}\|^2},
$$

with transverse lift residual

$$
r_\perp
=\|V-\Gamma_{x,t}-w_*\Gamma_{x,z}\|.
$$

Generic moving cameras and object motion have nonzero transverse residual.
Crossings, splitting, merging, and two velocities at one point also violate a
single orientation-preserving scalar flow.

Current kinetic site velocities are useful inputs for a compact global or
spatially bounded flow fit. Piecewise independent site velocities are not
automatically one continuous non-crossing flow. Their jumps are valid BV
curvature diagnostics, not proof of a global Lagrangian quotient.

A future flow receipt must bind at least:

- basis and coefficients;
- temporal capacity with the explicit gate
  `DoF_t(w) <= DoF_t(site motion + camera motion)`;
- retained bytes and evaluation work;
- supervision and conditioning path, with proof that capacity is shared and
  does not contain per-ray/per-frame `U` or target answer tables;
- lift residual;
- minimum orientation/noncrossing margin;
- endpoint policy; and
- every trainable-flow gradient.

## 9. U, U_tilde, and K_F Are Different ABIs

### 9.1 Direct physical transfer `U`

`U` lies in the physical contraction/color cone. The current affine-Lie atlas
uses

$$
\kappa=-\log\beta,
\qquad
v=\frac{\kappa}{1-\beta}m,
$$

with removable `kappa=0` limit and fail-closed constraints

$$
\kappa\ge0,
\qquad
0\le v_c\le\kappa.
$$

This is the current production representation.

### 9.2 Flow-corrected group element `U_tilde`

Because endpoint transports use inverses,

$$
\widetilde U=H_aUH_b^{-1}
$$

is generally in the affine group completion, not the physical cone. Its
attenuation is

$$
\widetilde\beta
=\frac{\beta_a\beta_U}{\beta_b}>0,
$$

which may exceed one, and its moment may be signed.

Therefore the current physical atlas cannot compile `U_tilde` unchanged. The
first oracle should use the unrestricted smooth group chart

$$
\chi(\beta,m)
=
\left(
-\log\beta,
\frac{(-\log\beta)m}{1-\beta}
\right),
$$

for every `beta>0`, with the removable identity limit, but without the
physical cone constraint. Only reconstructed

$$
U=H_a^{-1}\widetilde U H_b
$$

must satisfy the physical cone and the end-to-end primal/tangent certificate.

### 9.3 Curvature source `K_F`

`K_F=dU_tilde/dt` is a tangent matrix, not an optical transfer. It has four
signed affine-tangent components but cannot use the transfer decoder or
physical cone ABI. The first oracle should interpolate it as a generic signed
four-vector and integrate

$$
\widetilde U(t)
=\widetilde U(t_0)+\int_{t_0}^tK_F(\tau)\,d\tau.
$$

If `||K_F-K_F_hat||<=epsilon_F` over duration `T`, then

$$
\|\widetilde U-\widehat{\widetilde U}\|
\le
\|\widetilde U(t_0)-\widehat{\widetilde U}(t_0)\|
+T\epsilon_F.
$$

Reconstruction adds endpoint-transport errors and condition factors. Additive
integration can cross `beta_tilde<=0`; the oracle must fail closed.

A later structure-preserving alternative is the left-trivialized/body tangent
(equivalently, the generator acting on the right)

$$
\Xi_R=\widetilde U^{-1}K_F,
\qquad
\dot{\widetilde U}=\widetilde U\Xi_R,
$$

but that introduces another noncommutative temporal product integral and is
not the first experiment.

## 10. What Can Actually Become Cheaper

Use separate symbols for the two quantities that source code historically
called `R`: let `R_row,b` be track-chart rows in native block `b`, let
`W_run,b` be ordered owner/run entries in that block, let `F` be requested
samples, and let `J_X,b` be the certified temporal node count for
representation `X`. This distinction is load-bearing because the node-forward
kernel dispatches `R_row,b J_b` threads but each thread scans its complete
owner word. Thus

$$
N_{\mathrm{forward\ threads}}
=\sum_bR_{\mathrm{row},b}J_b,
$$

while the actual ordered-run work is

$$
N_{\mathrm{forward\ word}}
=N_{\mathrm{reverse\ word}}
=\sum_bW_{\mathrm{run},b}J_b.
$$

For one fixed-program material step,

$$
\boxed{
N_{\mathrm{ordered\ world}}
=2\sum_bW_{\mathrm{run},b}J_b.
}
$$

The source telemetry now keeps the forward thread count separate from the
forward ordered-run interaction count and requires the latter to equal the
material word-VJP count.

The current direct compiled path has schematic dominant state/work

$$
M_U
\supset
4\sum_bJ_{U,b}W_{\mathrm{run},b}
+32\sum_bR_{\mathrm{row},b}J_{U,b}
\quad\text{bytes},
$$

for float32 physical lengths, node transfers, and live node cotangents, and

$$
W_U^{\mathrm{word}}
=2\sum_bJ_{U,b}W_{\mathrm{run},b},
\qquad
W_U^{\mathrm{sample}}
=\Theta\!\left(
\sum_rF_rJ_r
+\sum_rN_{\mathrm{fb},r}J_r^2
+PF
\right).
$$

The exact constants vary with block/ragged layout. Fused geometry removes a
temporary `[J,W]` length cotangent; it does not remove the primal lengths.

`U_tilde` preserves the same structural form:

$$
W_{\widetilde U}^{\mathrm{word}}
=O\!\left(\sum_bJ_{\widetilde U,b}W_{\mathrm{run},b}\right),
$$

plus flow and endpoint state, reconstruction, cone checks, and their VJPs. It
can win only by lowering certified node/chart counts enough to pay for those
extras.

When endpoints follow a perfectly flat flow, `H_a=H_b=I` and direct `U` is
already constant. The advected slab is therefore an identity fixture, not
compression evidence.

A naive curvature source still requires prefix/suffix transports around each
source:

$$
K_{\mathrm{total}}
=\sum_iG_{<i}K_iG_{>i}.
$$

Its node cost remains `O(J_F W)` unless a separately proved sparse-source
algorithm exists. Sparse curvature support does not imply sparse training
gradients because prefix/suffix transports can depend on distant parameters.

The integration estimate also cuts both ways. To hold a global reconstructed
error `epsilon` fixed over duration `T`, the elementary bound can require

$$
\epsilon_F\le\epsilon/T.
$$

The tighter local tolerance can increase `J_F` with duration. Small or sparse
curvature therefore does not imply a smaller certified curvature atlas.

The connection does not improve the already proved fixed-program scaling in
requested frame density by itself. Its plausible target is smaller `J` and
slower chart/event growth as physical duration and coherent motion increase.

## 11. Exact First Oracle

Create one CPU float64 `kinetic_optical_curvature_oracle.py` only after the
current direct-`U` safe-host gates close. It should consume one exact stable
chart:

- exact cuts and cut velocities;
- ordered P0 generators with physical ray Jacobians;
- explicit endpoint paths;
- one sealed admissible compact flow receipt;
- exact direct `U`; and
- selected material, geometry, camera, and flow directions.

It should produce:

1. direct `U`;
2. endpoint transports `H_a,H_b`;
3. group-completion `U_tilde`;
4. bulk and singular curvature separately;
5. moving-endpoint flux;
6. signed tangent source `K_F`;
7. an independently differentiated covariant derivative;
8. plaquette holonomy with an orientation receipt;
9. lift residual and noncrossing margin;
10. equal-family/equal-tolerance `J_U,J_U_tilde,J_F`;
11. the same selected tangent ranks;
12. reconstructed-`U` cone/error;
13. endpoint conditioning; and
14. total payload and work estimates.

Required sentinels:

- front-red/rear-blue order;
- moving noncommuting boundary;
- advected differently colored slabs with moving clips;
- flat interior with fixed-endpoint flux;
- boundary mismatch `dot r != w`;
- material evolution with zero singular and nonzero bulk curvature;
- discontinuous `w` requiring `[wA_z]-dot r[A_z]`;
- constant noncommuting plaquette orientation;
- nonzero curvature with constant total `U`;
- transverse motion impossible for scalar `w`; and
- nonphysical `U_tilde` with physical reconstructed `U`.

The ablation rows are:

| Row | Representation | Purpose |
| --- | --- | --- |
| A0 | direct physical `U` | Current baseline. |
| A0c | direct physical `U` with the same added temporal DoF/bytes allowed to A1 | Capacity-matched control: separates quotient benefit from a larger model. |
| A1 | group-completion `U_tilde` with bounded shared flow | Tests whether canonicalization lowers total certified complexity. |
| A2 | signed `K_F` plus base `U_tilde(t0)` | Tests curvature-source rank and support. |
| C0 | `w=0` | Convention/endpoint control. |
| C1 | per-ray fitted `w` | Explicit cheating upper bound; never promotable. |

## 12. Promotion and Kill Criteria

Do not add a native connection ABI unless A1 first wins in the oracle. Do not
add a native `K_F` ABI unless A2 then beats both A0 and A1.

Kill the runtime branch if any of these hold:

- flow stores per-frame or hidden per-ray answer tables;
- flow capacity or conditioning contains per-ray/per-frame `U`, RGB, or target
  answer tables rather than one bounded shared scene/camera model;
- `DoF_t(w) > DoF_t(site motion + camera motion)` or the direct-transfer
  capacity-matched A0c control removes the apparent gain;
- lift or orientation/noncrossing checks fail;
- well-conditioned float64 identity residual exceeds `1e-9`;
- reconstructed forward error exceeds `1e-5`;
- normalized selected-gradient error exceeds `1e-4`;
- geometry/flow tangent rank regresses despite lower primal rank;
- endpoint inverses are ill-conditioned;
- reconstructed `U` leaves the physical cone;
- total retained payload and ordered-word work
  `sum_b J_b W_run,b` fail to improve by at least `2x`
  against both direct alternatives;
- measured request time later improves by less than 20%;
- the gain exists only on an advected slab where direct `U` is already
  constant.

`J_F/J_U` and curvature-support sparsity are explanatory diagnostics, not
independent promotion laws. Either may fail to improve while a future certified
algorithm still wins the decisive total-bytes, total-work, and measured-time
gates; the current prefix/suffix implementation simply offers no such shortcut.

If the branch fails, retain the connection as a theorem and diagnostic. That
still clarifies correspondence failure, endpoint flux, material evolution,
and visibility events.

## 13. Paper Claim Ladder

Claimable after mathematical proofreading:

> We derive a flow-covariant variation identity for WorldFoam's ordered affine
> transfer and characterize exact transported-subinterval reuse by flatness of
> a capacity-constrained shared Lagrangian optical connection.

Claimable after the correctness oracle:

> The decomposition separates endpoint flux, bulk material evolution, and
> interface/correspondence mismatch on noncommuting fixtures.

Claimable only after equal-certificate rank experiments:

> Flow correction reduces certified temporal representation complexity.

Claimable only after native measurements:

> Curvature-aware compilation reduces retained payload, memory, bandwidth, or
> request time.

Do not claim now:

- a holonomy renderer;
- general moving-camera correspondence from scalar `w`;
- handling of arbitrary crossings or topology change;
- lower rank merely because curvature is small;
- memory-light/sublinear scaling caused by the connection;
- a factorization cosheaf, constructible stack, or braid runtime; or
- a new native WorldFoam backend.

Keep the method name **WorldFoam**. Use `constrained Lagrangian ray-fiber
optical connection`, `flow-covariant ordered transfer`, and `optical curvature
residual`. Reserve `holonomy` for a closed ray-time plaquette.

The draft already contains a separate **cell-frame adjacency connection**.
If the ray-fiber oracle succeeds, make the ray-fiber optical connection the
main mathematical connection and demote cell-frame holonomy to an exploratory
diagnostic or appendix until its heldout-residual correlation gate succeeds.

## 14. Literature Boundary

Connections, parallel transport, curvature, holonomy, and Wilson-line
variation are standard differential geometry. Lagrangian/canonical dynamic
scene representations are also established. Useful primary context includes:

- Cattaneo, Cotta-Ramusino, and Rinaldi, *Loop and Path Spaces and Four-
  Dimensional BF Theories*: <https://arxiv.org/abs/math/9803077>;
- D-NeRF: <https://openaccess.thecvf.com/content/CVPR2021/html/Pumarola_D-NeRF_Neural_Radiance_Fields_for_Dynamic_Scenes_CVPR_2021_paper.html>;
- Nerfies: <https://openaccess.thecvf.com/content/ICCV2021/html/Park_Nerfies_Deformable_Neural_Radiance_Fields_ICCV_2021_paper.html>;
- Forward Flow: <https://openaccess.thecvf.com/content/ICCV2023/html/Guo_Forward_Flow_for_Novel_View_Synthesis_of_Dynamic_Scenes_ICCV_2023_paper.html>.

The defensible repo-level contribution is the rendering-specific combination:
the right-ordered affine optical specialization, transported-subinterval
flatness theorem, BV interface residual, measure/connection bridge, and
certified compiler comparison. Phrase this as `we derive`, not `we are the
first`, until a comprehensive novelty review is complete.

Factorization algebras and constructible stacks are legitimate background but
not current WorldFoam runtime objects. The attachment's `math/0604428`
local-triviality citation is unrelated and must not be used.

## Final Decision

This is the best new mathematical layer found in the latest scientist pass.
It unifies rather than replaces prior work:

```text
(kappa,nu) explains depth order and tangent boundary mass;
(beta,m) remains the exact compact executor;
the event atlas handles supported simple crossings and topology events while
simultaneous and full-fiber degeneracies remain fail-closed;
the Lagrangian connection identifies coherent cross-time reuse;
curvature records what that bounded shared flow cannot explain.
```

Finish the direct memory-light `U` trainer and safe-host validation first. Then
run the group-completion `U`/`U_tilde` oracle. The curvature theorem belongs in
the mathematics now; curvature compilation belongs in code only if the
equal-certificate experiment wins.
