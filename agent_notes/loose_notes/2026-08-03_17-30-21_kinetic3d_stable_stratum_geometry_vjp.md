# Kinetic 3D stable-stratum geometry VJP

Date: 2026-08-03

## Why this work exists

The kinetic 3D power-word frontend already provides the useful representation-level
reduction for WorldFoam:

```text
p_i(t) = p_i0 + t v_i
w_i(t) = sum_{k=0}^2 w_ik t^k
x(t,z) = o(t) + z d(t)
```

At fixed camera time, every site's squared distance along a ray has the same
quadratic term in `z`; ownership is therefore the lower envelope of affine
functions of `z`. The topology compiler can discover a sparse ordered owner word
and exact candidate event predicates without storing per-frame site tables.

What remained absent was a geometry reverse for a chart whose topology had already
been certified. The existing chart transfer bridge intentionally returned no
geometry gradients. That gap prevented the representation from demonstrating the
specific World-Tubes-like memory contract we care about: reduce sample residuals to
`J` compiler-node cotangents, then run geometry backward over the compact node/run
program rather than replaying all requested frames.

This session implements only that missing fixed-program derivative. It does not
claim the harder derivative of topology changes.

## Exact scope

New implementation:

```text
research_experiments/world_foam_lane2/kinetic_stable_stratum_vjp.py
```

New independent CPU tests:

```text
research_experiments/world_foam_lane2/test_kinetic_stable_stratum_vjp.py
```

The function accepts:

- Euclidean kinetic sites with affine position and degree-at-most-two power weight;
- affine origin and direction coefficients for each ray track;
- `J` compiler node times;
- one externally certified, fixed owner word per track;
- static P0 density and color per site;
- one transfer cotangent `[beta_bar, m_bar]` per track and compiler node;
- fixed `near` and `far` ray coordinates;
- an explicit nonempty continuous-topology certificate identifier.

It returns node transfers plus accumulated gradients for:

```text
p_i0, v_i, w_i0/w_i1/w_i2,
o_p0/o_p1, d_p0/d_p1,
rho_i, c_i.
```

The derivative holds fixed:

- owner topology and owner identities;
- topology/chart endpoints and event times;
- compiler node times;
- interpolation rank and any sample-to-node reduction;
- the Euclidean metric;
- the near/far ray-coordinate endpoints.

This is the classical derivative inside one smooth combinatorial stratum. It is not
a derivative through the map that selects the stratum.

## Forward geometry at one node

At node time `t`, define

```text
p_i = p_i0 + t v_i
w_i = sum_k w_ik t^k
o = o0 + t o1
d = d0 + t d1
s = ||d||.
```

For adjacent owners `i` and `j`, use the power-distance difference

```text
h_ij(z) = D_i(o + z d, t) - D_j(o + z d, t) = A_ij z + B_ij,
A_ij = 2 (p_j - p_i)^T d,
B_ij = 2 (p_j - p_i)^T o
       + ||p_i||^2 - ||p_j||^2 - w_i + w_j.
```

The active cut is

```text
z_ij = -B_ij / A_ij.
```

For ordered cuts `z_0=near, z_1, ..., z_R=far`, run `r` has coordinate
and physical lengths

```text
Delta z_r = z_{r+1} - z_r,
L_r = ||d|| Delta z_r.
```

The P0 segment transfer for owner `i_r` is

```text
tau_r = rho_i L_r,
beta_r = exp(-tau_r),
m_r = (1-beta_r) c_i.
```

Composition is front-to-back:

```text
(beta_a,m_a) o (beta_b,m_b)
    = (beta_a beta_b, m_a + beta_a m_b).
```

## Prefix-only ordered-transfer reverse

No suffix array is required. Immediately before run `r`, retain the current prefix
`(beta_pre,m_pre)`. Let `(beta,m)` denote the final transfer and let
`(beta_bar,m_bar)` be its cotangent. Differentiating through the ordered product gives

```text
tau_bar_r = m_bar^T (m_pre + beta_pre c_i - m)
            - beta beta_bar.
```

The material and length cotangents are

```text
L_bar_r       = rho_i tau_bar_r,
rho_bar_i    += L_r tau_bar_r,
c_bar_i      += beta_pre (1-beta_r) m_bar.
```

Then

```text
Delta_z_bar_r = ||d|| L_bar_r,
s_bar        += Delta z_r L_bar_r.
```

Each internal cut receives the right-end contribution of its left run and the
negative left-end contribution of its right run. This sweep stores only the running
prefix plus one cotangent per active cut.

## Implicit cut derivative

The useful simplification is not a Schur marginalization of depth. WorldFoam must
retain ordered depth segments. Instead, once an active ordered word is fixed, every
segment endpoint is an implicit scalar root. That scalar admits a cheap local VJP.

For one cut, let `x=o+zd` and receive `z_bar`. Since `h(z,theta)=0`,

```text
d z / d theta = -(partial_theta h) / (partial_z h),
partial_z h = A.
```

Set

```text
q = -z_bar / A.
```

The sparse pullback is

```text
p_i_bar += q 2(p_i-x)
p_j_bar += q 2(x-p_j)
w_i_bar -= q
w_j_bar += q
o_bar   += q 2(p_j-p_i)
d_bar   += q z 2(p_j-p_i).
```

The physical ray-speed term contributes separately:

```text
d_bar += s_bar d / ||d||.
```

Finally, the affine temporal parameterization gives

```text
p_i0_bar += p_i_bar
v_i_bar  += t p_i_bar
w_ik_bar += t^k w_i_bar
o0_bar   += o_bar
o1_bar   += t o_bar
d0_bar   += d_bar
d1_bar   += t d_bar.
```

Thus each active cut touches only its two incident sites and its ray track. There is
no dense site-by-site geometry Jacobian in the reverse.

## Work and memory accounting

Let `R_p` be the number of owner runs for track `p`, `S` the total site count, and
`J` the compiler node count.

The actual differentiable forward/reverse program uses:

```text
run work: O(J sum_p R_p)
cut work: O(J sum_p (R_p-1))
frame-by-run reverse state: zero
requested frame count used by this API: zero.
```

The sample-to-node reduction is upstream and must itself be implemented compactly;
this module does not prove that upstream property merely by omitting `F`.

There is also a deliberately strict node-local validation pass. It evaluates every
competitor against every claimed owner at left endpoint, right endpoint, and
midpoint. Its work is

```text
O(J S sum_p R_p).
```

That audit remains independent of requested frame count and allocates no
frame-by-run reverse tape, but it is not free. The returned accounting reports the
reverse and validation counts/scalings separately. A production compiler may reuse
an exact certificate's already computed margins rather than repeating this audit,
but removing it before such provenance is wired would weaken the current safety
contract.

## Stable-stratum trust gates

The VJP fails closed at a node when any of the following occurs:

1. ray speed is too small;
2. an active cut denominator `|A|` is too small;
3. the scale-normalized cut cosine is too small;
4. a coordinate or physical segment length is too small/nonpositive;
5. adjacent owners fail to tie at their claimed cut within a scaled tolerance;
6. any nonadjacent competitor reaches or undercuts the claimed owner at a checked
   endpoint/midpoint;
7. the caller omits continuous-certificate provenance.

At fixed `t`, pairwise power gaps are affine in `z`; checking both segment endpoints
is enough in exact arithmetic to establish the sign throughout the segment.
Midpoints are retained as a cheap numerical red-team check and as a clearer failure
location. These pointwise checks do not establish the sign continuously in `t`.

The continuous owner/event certificate therefore remains mandatory. Passing an ID
does not magically verify it; the ID records the external proof object that the
integration layer must resolve.

## Validation performed

The main positive fixture has:

- three moving sites with quadratic weights;
- two affine ray tracks;
- three compiler nodes;
- fixed owner word `(0,1,2)` on every track/node;
- nontrivial density, color, and transfer cotangents.

An independent differentiable PyTorch oracle reconstructs the cuts and ordered P0
product directly. The manual result matches autograd for all six gradient groups:

```text
positions0, velocities, weight coefficients,
ray coefficients, density, color.
```

The same fixture perturbs every parameter group simultaneously and compares the
manual gradient's directional contraction against a central finite difference. It
passes in float64 at tight tolerance.

Further tests prove:

- summing three one-node VJPs reproduces one three-node VJP;
- the public signature contains no frame-count input;
- accounting reports zero requested frames and no frame-by-run reverse state;
- a zero cut denominator fails closed;
- a deliberately omitted middle owner is caught by the all-competitor audit;
- promoting the observed owner gap to a stricter threshold rejects the same chart;
- missing continuous-certificate provenance is rejected.

CPU command:

```text
PYTHONPATH=research_experiments/world_foam_lane2 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_kinetic_stable_stratum_vjp.py \
  research_experiments/world_foam_lane2/test_kinetic_power_word_compiler.py \
  research_experiments/world_foam_lane2/test_rational_polynomial_roots.py -q
```

Observed result:

```text
23 passed in 1.33s
```

No GPU, MPS, dataset load, training run, or publication benchmark was launched.

## Branches considered and rejected

### Production autograd through the compiler

This would be expedient for a fixture but would preserve a large dynamic graph,
blur the fixed-topology contract, and make it easy for frame sampling to leak into
the geometry reverse. Autograd remains an independent test oracle only.

### Dense Jacobian of every cut against every site

Each cut depends directly on only two active sites. A dense site Jacobian destroys
the incidence sparsity the formulation exposes and is unnecessary. The implemented
VJP scatters only to the adjacent pair.

### Differentiate owner discovery or sorted-envelope operations

Those operations are nonsmooth at topology events. Pretending their discrete path
has an ordinary derivative would silently cross strata. This implementation makes
the frozen-program assumption a type/API-level input and fails on small margins.

### Treat node checks as a continuous certificate

Finite samples in time can miss an event. The kinetic compiler already derives
exact event predicates; the correct integration is to attach one of its continuous
proof objects, not to increase node density and call that certification.

### Remove the all-competitor audit to claim pure `O(JR)` total work

That would make the current standalone API trust an arbitrary owner word. The
honest statement is `O(JR)` differentiable work plus `O(JSR)` defensive validation.
Once a verified compiler certificate supplies equivalent margins, the repeated
audit can be elided by a separately tested integration path.

## What this falsifies and what it does not

The tests falsify the hypothesis that a fixed-word geometry reverse intrinsically
needs per-frame replay or a new Schur-complement-style elimination. Within a stable
stratum, scalar implicit cut roots plus an ordered prefix reverse are sufficient.

They do not establish:

- correctness at topology births/deaths, triple concurrence, or near/far events;
- derivatives of chart boundaries or adaptive compiler-node placement;
- an end-to-end sample-to-node adjoint;
- bounded spherical/cellular truncation used by every WorldFoam variant;
- a general shared-SPD or per-site anisotropic metric derivative;
- native Metal/CUDA integration or performance;
- full-scene trainability or quality.

## Next integration step

The immediate mathematical integration path is now concrete:

1. use an existing certified kinetic owner chart, not a handwritten word;
2. reduce loss cotangents from requested samples to its compact transfer nodes;
3. call this stable-stratum node VJP with the certificate's identity and margins;
4. map kinetic site/ray coefficient gradients into the trainer's parameterization;
5. add an optimizer-step smoke that stays inside the recorded trust region;
6. reject or recompile the chart when an update would cross a margin.

Only after that end-to-end CPU contract is sound should the compact program be
ported to a native backend. The remaining discovery problem is not a new formula
for ordinary fixed-stratum depth. It is how to manage chart recompilation and
nondifferentiable event boundaries during optimization without restoring
frame-linear geometry replay.
