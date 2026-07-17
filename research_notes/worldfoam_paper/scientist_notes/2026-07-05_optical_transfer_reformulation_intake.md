# Scientist Dump Intake: Optical-Transfer Reformulation

Date: 2026-07-05

Source:

```text
research_notes/world_foam_reformulation.md
```

Purpose: hard triage of the optical-transfer reformulation. This note records
what is genuinely useful, what is only vocabulary, what is dangerous unless
tested, and how the main WorldFoam lane should change.

## Verdict

The reformulation is genuinely interesting, but not uniformly good.

Hard split:

```text
high-value:        visibility monoid, transfer matrix/product integral,
                   commutator visibility theorem, monoid VJP, event closure
useful framing:    optical matter notation, ray-fiber lambda/eta naming,
                   owner-run event rasterizer, optical-depth basis
risky detours:     Magnus Foam as mainline, boundary flux before FD tests,
                   measure-theory vocabulary without implementation leverage
unsafe claims:     "strictly generalizes splatting" as a renderer-quality claim
```

The best new identity is:

```text
WorldFoam is a camera-compiled optical-transfer algebra over ray fibers.
```

That is stronger than saying:

```text
WorldFoam is sigma(u,v,t,z) plus a transmittance prefix.
```

The old sentence is physically correct. The new sentence explains the actual
renderer object: event elements that compose, commute or fail to commute,
compress, scan, and backpropagate.

## Promote Now

### K1. Visibility monoid

Keep as the central algebra:

```text
g = (beta, m)

g1 otimes g2
    =
    (beta1 beta2, m1 + beta1 m2)

decode((beta,m), B) = m + beta B
```

Why it matters:

```text
alpha compositing
volume transmittance
owner-run events
parallel prefix
direct VJP
compression criteria
```

all share one operator. This is the paper spine.

### K2. Transfer matrix / product integral

Keep as the mathematical headline:

```text
M(beta,m) =
    [ beta I_C   m ]
    [ 0          1 ]

A_y(z) =
    [ -lambda_y(z) I_C   eta_y(z) ]
    [ 0                  0        ]
```

Rendering is:

```text
[ I(y) ]     [ P exp int A_y(z) dz ] [ I_bg(y) ]
[ 1    ]  =  [                       ] [ 1       ]
```

This is elegant and unifying. The implementation should still be event
elements first, not a symbolic path-exponential engine.

### K3. Alpha compositing as atomic optical transfer

Keep the safe claim:

```text
splat with opacity alpha_i and color c_i
    -> g_i = (1 - alpha_i, alpha_i c_i)

sorted alpha compositing
    -> depth-ordered monoid scan of g_i
```

Safe wording:

```text
WorldFoam generalizes splat compositing at the ray-transfer equation level.
```

Unsafe wording:

```text
WorldFoam strictly dominates splatting as a trainable renderer.
```

That second sentence is not true until quality, speed, memory, and stability
all win under matched public conditions.

### K4. Commutator visibility theorem

Keep as a highlighted theorem:

```text
[A1,A2] color-column ~= lambda1 lambda2 (c1 - c2)
```

Discrete equivalent:

```text
Delta m = alpha_i alpha_j (c_i - c_j)
```

Interpretation:

```text
visibility/order error is opacity overlap times color contrast.
```

This is one of the few ideas in the dump that is both elegant and immediately
testable. It should drive:

```text
swap-error tests
interval split heuristics
compression bakeoffs
visibility-stress figures
```

### K5. Monoid scan VJP

Keep before boundary calculus.

For:

```text
h = a otimes b
beta_h = beta_a beta_b
m_h = m_a + beta_a m_b
```

VJP:

```text
bar m_a    += bar m_h
bar m_b    += beta_a bar m_h
bar beta_a += beta_b bar beta_h + dot(bar m_h, m_b)
bar beta_b += beta_a bar beta_h
```

For constant owner-run segment:

```text
beta = exp(-DeltaTau)
m = (1 - beta) c

bar DeltaTau = -beta bar beta + beta dot(bar m, c)
bar c        = (1 - beta) bar m
```

This is the first implementation-facing math to test. It is smaller, safer,
and more useful than jumping straight to moving-boundary gradients.

### K6. Event closure instead of Schur closure

Best sentence:

```text
WorldFoam does not have Schur closure. It has event closure.
```

Meaning:

```text
World Tubes:
    Gaussian pullback + fiber marginalization -> Schur-complement UVT footprint

WorldFoam:
    bounded cell pullback -> certified event intervals + optical transfer scan
```

Do not force the World Tubes Gaussian proof onto WorldFoam. Foam cells have
indicator boundaries and root events; their closure is sparse event structure,
not global Gaussian marginalization.

## Develop Behind Tests

### H1. Magnus / commutator compression

Interesting but dangerous.

First-order collapse:

```text
Lambda = int lambda dz
Q      = int lambda c dz
g      = (exp(-Lambda), (1 - exp(-Lambda)) Q / Lambda)
```

Second-order commutator moment:

```text
K = 1/2 int int lambda1 lambda2 (c1 - c2)
```

Hypothesis:

```text
split depth intervals where commutator energy is large.
```

This could be a real contribution. It is not the next implementation target.
First compare it against simple adaptive transmittance/error splitting at equal
memory. If it does not win, keep it as explanatory math only.

### H2. Interface flux adjoint

Potentially important:

```text
dI/ds =
    T(s) [
      lambda_A (c_A - I_behind)
      - lambda_B (c_B - I_behind)
    ]
```

This is a plausible path from fixed-topology VJP to moving-boundary gradients.
But sign conventions, face normals, near-parallel rays, sphere endpoints, and
support-event topology changes are all easy places to be wrong.

Gate before paper use:

```text
moving plane face finite difference
moving sphere endpoint finite difference
power-cell face crossing finite difference
near-parallel denominator stress
fixed-topology guard/fallback check
```

### H3. Optical-depth coordinate basis

Interesting rasterizer variant:

```text
s = tau(z) - tau(z_k)
m_k = int e^{-s} c(s) ds
```

If color is polynomial in optical depth, moments have closed form. This may
reduce quadrature or layer count, but it should not block monoid/VJP tests.

## Do Not Promote

### D1. Measure-theory vocabulary as contribution

`mu`, `nu`, and `c = dnu/dmu` are useful notation. They are not a contribution
unless they change a theorem, implementation, or experiment.

### D2. Magnus Foam as the first implementation

Magnus compression is a hypothesis. The first implementation should be:

```text
event optical elements
same-representation replay equivalence
monoid VJP finite differences
commutator prediction fixture
```

### D3. Boundary calculus as solved

Boundary flux is a research branch, not a finished theorem. Keep direct VJPs
inside fixed compiled topology until finite differences pass.

### D4. Splat dominance

The only currently safe splat statement is equation-level:

```text
alpha compositing is the atomic case of optical transfer.
```

Do not infer:

```text
better public quality
better trainability
lower memory
better speed across all scenes
```

from that equation.

## Paper Spine After This Intake

Use this method order:

```text
1. camera-gauged optical matter
2. ray-fiber lambda/eta pullback
3. visibility monoid
4. transfer matrix / product integral
5. alpha compositing as atomic case
6. event atlas rasterization / owner-run elements
7. commutator theorem and compression signal
8. monoid scan VJP
9. fixed-topology boundary flux as optional extension
10. event closure vs Schur closure
```

The main proof baseline remains:

```text
per-frame WorldFoam replay vs compiled WorldFoam event atlas
```

External STAR/GS rows position the result. They do not prove the compiler.

## Immediate Tests

Run before broad public-scene quality sweeps:

```text
1. Alpha equivalence:
   sorted splat compositing equals monoid scan of atomic elements.

2. Commutator prediction:
   two-layer and multi-layer order errors follow opacity overlap times color
   contrast.

3. Event replay equivalence:
   compiled event elements match per-frame owner-run replay under fixed
   certified intervals.

4. Monoid VJP:
   beta/m/DeltaTau/color/sigma/length finite differences match direct VJP.

5. Boundary flux:
   only after monoid VJP passes; validate moving faces and support endpoints.

6. Compression bakeoff:
   commutator-energy split vs simple adaptive splitting at equal memory/error.
```

## Final Critical Take

The good part of the reformulation is not "more fancy math." It is that it
turns WorldFoam from:

```text
density plus prefix scan
```

into:

```text
an optical-transfer algebra with a concrete event element, theorem-level
relation to alpha compositing, a visible noncommutativity/error criterion, and
a direct VJP.
```

That is worth preserving.

The trap is letting the beautiful branches outrun implementation evidence.
Magnus compression and boundary flux should be treated like hypotheses with
finite-difference gates, not like already-won paper contributions.
