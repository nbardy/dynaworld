# World Tubes: sober mathematical contribution and scaling assessment

The user asked to return to the existing work, apart from speculative gauge
redesigns, and identify the new mathematics and actual sublinear claim. This
note reads the current manuscripts, native/source paths, and retained orbit
artifact. No code changes, GPU jobs, new benchmarks, or evidence promotions.
The source files hashed during the September 6 audit are unchanged at this
review, including the native ordering and compiler paths discussed below.

## What the current propositions establish

| Proposition | Mathematical ingredient | Assessment |
| --- | --- | --- |
| Fiber-gauge invariance | Change of variables with the physical fiber measure | Standard identity, useful correctness contract |
| Conditional-Gaussian equivalence | Affine Gaussian pushforward, marginalization, Schur complement | Standard identity, useful footprint/depth construction |
| Fixed-cell compositing correctness | Adjacent alpha-layer swap identity plus triangle inequality | Standard algebra assembled into a scoped error guarantee |
| Fixed-topology compiled adjoint | Chain rule through compiler and evaluator | Standard identity, necessary for learning the shared world |

For example, swapping adjacent layers a,b changes their contribution by
`T_before * alpha_a * alpha_b * (color_a-color_b)`. A bound on this expression
supports controlled approximate order reuse. It does not prove that acceptable
orders are cheap to find, nor that many learned-scene cells satisfy the bound.

No demonstrated new foundational mathematical theorem is needed to explain
the current method. The candidate contribution is the constructive algorithm:
compile a known camera program and dynamic Gaussian world into reusable
sensor-time trace records, support/order domains, and a differentiable map
back to the same world parameters. A particular construction and its proven
cost/error guarantees could be new despite standard component identities.
This review does not establish exhaustive novelty clearance.

Local affine Gaussian ray-space footprints have prior art in
[EWA Volume Splatting](https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf).
Maintaining geometric structure until certificates fail is the established
[kinetic data structure](https://graphics.stanford.edu/~comba/papers/socg.pdf)
idea. Gaussian ordering reuse also appears in
[Neo](https://arxiv.org/abs/2511.12930). A specific dynamic-world compiler and
training adjoint must be distinguished from these, rather than claiming the
general idea of temporal reuse or gauge invariance as new.

## The mechanism that can genuinely share work

Consider a projected center u(t)=a+bt over a fixed physical interval. Its
description uses two coefficients whether queried at 10 times or 1,000 times.
With fixed tile width, tile-boundary crossings occur at algebraic times
`t=(boundary-a)/b`, not at every requested frame. If two scalar depth traces
are affine in time, their difference has at most one isolated crossing unless
identically equal. Fixed coefficients can therefore describe many queries
and a small number of support/order changes.

This example proves the possibility of query-density-independent descriptions
and events in a restricted family. It does not prove that every primitive,
camera, support ellipse, translucent overlap, and learned update admits the
same inexpensive construction. Near poles, non-polynomial trajectories,
approximation error, dense overlaps, and conservative splitting change the
problem. An event may touch many records, so counting events alone is not a
bound on all metadata or work.

The compiler's useful move is replacing many separately lowered frame states
with coefficients and valid intervals. That can remove repeated expensive
geometry transformations while still evaluating a cheaper function at every
requested time. Gauge changes are not the source of the asymptotic reuse.

## Simple cost model: amortization is not o(F) total rendering

Fix primitive count, image size, physical time interval, camera path, and error
tolerance. Let A be expensive per-frame geometric preparation, B the remaining
per-frame rendering/output work, a the cheaper compiled per-frame evaluation,
and C the one-time compilation cost. The idealized comparison is

```text
T_replay(F) = F(A+B)
T_compiled(F) = C + F(a+B).
```

When C remains bounded as F increases, shared preparation C is O(1) in sample
count and its amortized cost C/F decreases. Total compiled work is still
Theta(F) when a+B is positive. The asymptotic speedup is the constant
`(A+B)/(a+B)`, potentially large. For illustrative units A=9, B=1, a=0, C=9,
the change is from 10F to 9+F: useful, but both are linear as F tends to infinity.

An affine timing curve C+bF can also look sublinear on a finite log-log plot:
its local logarithmic slope is `bF/(C+bF)<1`, tending to one. Fixed overhead
and improved GPU occupancy create further finite-range effects. Neither a
large speedup nor a slope below one over a few samples proves total o(F) work.

Writing F dense images of P pixels requires Omega(FP) output operations under
ordinary finite-bandwidth computation. An exact VJP receiving arbitrary dense
image cotangents likewise has an Omega(FP) worst-case input-reading cost. An
analytic exposure integral, compressed output, or specially structured loss
could define a different query problem, but the current dense-frame comparison
does not establish such a result.

## More complete accounting for the actual compiler

Let R be retained trace/bin records, E the events, M the executed trace-time
evaluations, K the primitive-pixel interactions, D the cost of event discovery,
fitting, and validation, and S any live ordering cost. Then a useful accounting is

```text
T_forward = D(N,E,F,epsilon) + O(R+M+K+FP) + S
T_backward = O(FP+M+K) + V_C + ordering/recomputation overhead,
```

where V_C is the compiler VJP, possibly applied repeatedly by the current
chunking policy. These are bookkeeping models, not proved universal bounds.
The declaration R independent of F does not make D, M, K, or S independent of F.
Metadata counting also does not establish peak-memory savings against a replay
implementation that streams or recomputes per-frame intermediates.

The current draft at WORLD_TUBES_PAPER_DRAFT.md:1017-1018 gives a retained-record
count for compiled metadata. Reading that formula as a compilation-time bound
would be incorrect. The draft itself labels these accounting identities and
explicitly rejects a universal sublinear end-to-end claim at lines 1037-1042.

Concrete implementation limitations, with paths relative to
`third_party/fast-mac-gsplat/variants/star_uvt_v0/`:

- `torch_gsplat_bridge_star_uvt/projective_trace.py:752-795` evaluates supplied
  trace-time samples, fits polynomials, and checks residuals at those samples.
- The same file at 2303-2345 materializes a trace-time tensor and loops over
  sample membership for coverage validation. Compact output metadata can still
  require sample-linear discovery/validation in these paths.
- `csrc/metal/star_uvt_kernels.metal:650-692` scans candidates for the next
  depth-ranked primitive at each pixel/time. The enclosing interval renderer
  repeats that selection. Certificates do not yet eliminate this live sorting.
- The previous audit found repeated bin packing/upload and repeated compiler
  VJPs across time chunks. These remain engineering opportunities rather than
  demonstrated completed amortization.

The manuscript passage at WORLD_TUBES_PAPER_DRAFT.md:669-677 claiming direct
consumption of precompiled order still conflicts with this native path.

## Backward: what can be shared exactly

For fixed world parameters theta and topology kappa, write phi=C(theta;kappa)
and L=sum_f L_f(R_f(phi;kappa)). Then

```text
g_phi = sum_f (D_phi R_f)^T grad_I L_f
g_theta = (D_theta C)^T g_phi.
```

One can accumulate image contributions before applying the expensive compiler
VJP once. The summation and rendering adjoints still process frame-dependent
inputs. This is a valid way to reduce repeated work, not a new differentiation
identity or a sublinear algorithm for arbitrary dense residuals. Topology
changes and parameter updates invalidate simple reuse unless separately handled.

## What the complete orbit timing series actually says

Source: `outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json`.
Settings: 32x32 images, two warmups, three timed iterations. Times below are ms.

| F | Forward compiled | Forward replay | Backward compiled | Backward replay |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 10.09 | 20.06 | 39.14 | 32.59 |
| 16 | 39.30 | 42.04 | 40.71 | 36.71 |
| 32 | 33.01 | 47.54 | 39.11 | 30.15 |
| 64 | 25.78 | 219.53 | 22.86 | 144.30 |

The F=64 row does report 8.52x forward and 6.31x backward speedup. The earlier
rows are not a clean scaling curve: compiled forward is nonmonotonic, and
compiled backward is slower than replay at F=8,16,32. Thus the artifact gives
a promising endpoint result, not a reliable fitted power law or evidence of
uniform backward speedup. Inspecting only the final ratio overstates consistency.

The structural result is cleaner: compiled trace count stays at 8 while replay
trace count rises 16,32,64,128. Compiled interval entries are 99,135,156,153,
while replay entries are 224,436,856,1686. Trace state is fixed; all metadata is
not literally constant. No statistical confidence interval or new measurement
was constructed in this review.

The separate learned high-motion timing synchronization asymmetry documented
on September 6 remains a separate issue and should not be attributed to this
orbit timer. Neither result is proof of a full learned-scene training speedup.

## What would constitute the missing stronger mathematical result?

Specify a restricted primitive/camera family, physical near-plane margin,
support/rendering law, approximation tolerance, and event degeneracy assumptions.
Then construct the certificate/trace algorithm and bound both its retained
records and its discovery cost independently of query density where possible.
Include representation error, support/order error, and world-gradient error;
sampled residuals alone do not imply continuous guarantees. State explicitly
which sample-dependent evaluation and dense-output costs remain.

A theorem for that construction could be worthwhile. The current record-count
formula, standard chain rule, and favorable endpoint timing do not substitute
for it. Empirically, use one frozen learned world and camera interval, matched
outputs/physical gradient perturbations, repeated timing with symmetric
synchronization, event/record counts, and route-scoped peak memory. Include
ordinary batching/reuse and check whether total compile-inclusive savings persist.

## Recommended claim today

World Tubes is a candidate differentiable camera-program compiler that reuses
geometric preparation and compresses intermediate trace state across time.
Its core identities are established mathematics; originality would reside in
the particular construction and demonstrated guarantees/performance. Retained
fixtures show structural reuse and a favorable timing endpoint. They do not
prove a sublinear dense renderer, sublinear dense-loss backward, or universal
sublinear compilation. A substantial constant-factor acceleration remains a
valuable outcome if it survives fair, representative measurement.
