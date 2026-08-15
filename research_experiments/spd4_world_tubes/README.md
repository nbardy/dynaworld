# SPD(4) World Tubes reference slice

This package begins with a CPU/float64 mathematical oracle for compiling a
native spacetime Gaussian into the existing STAR UVT record. The later
trainable lane adds an opt-in float32 producer that lowers to the unchanged
production Metal ABI; the oracle itself remains the higher-fidelity
conditional-depth and retained-fiber reference.

The source atom is

\[
\rho(X)=a_{\rm peak}\exp\left(
-\tfrac12(X-\mu)^\top\Sigma^{-1}(X-\mu)
\right),\qquad X=(x,y,z,t),\quad\Sigma\in\mathrm{SPD}(4).
\]

The lossless conditional block chart stores a spatial Cholesky factor
\(C=L_xL_x^\top\), spacetime tilt \(v\), and temporal variance
\(c=\exp(2\ell_t)\):

\[
\Sigma=
\begin{bmatrix}
C+cvv^\top & cv\\
cv^\top & c
\end{bmatrix}.
\]

Every strict SPD(4) covariance has one such chart. In particular, conditioning
on time yields \(E[x\mid t]=x_0+v(t-t_0)\): affine motion is already encoded by
the geometry of one 4D volume, rather than added as a separate trajectory.

`AffineRayGauge` represents either an invertible world-to-`(u,v,depth,t)` map
or the equivalent ray bundle

\[
X=b+A(u,v,t)+d\,s.
\]

After exact affine pushforward, partitioning \(a=(u,v,t)\) from depth gives

\[
\begin{aligned}
C_a &= \Sigma_{aa}, & Q_a &= C_a^{-1},\\
E[d\mid a] &= \mu_d+\Sigma_{da}C_a^{-1}(a-\mu_a),\\
\operatorname{Var}[d\mid a]
&=\Sigma_{dd}-\Sigma_{da}C_a^{-1}\Sigma_{ad}.
\end{aligned}
\]

The six legacy adapter fields are `ma`, packed `q_uvt`, `depth0`,
`depth_beta`, `opacity`, and `color`. The float64 trace additionally retains
conditional depth variance and the pushed joint covariance. The fast STAR ABI
still has six fields, while the variance-aware hybrid carries
`depth_variance` beside that record and consumes it in its certificate and
fallback.

Opacity lowering is explicit:

- `peak_splat` preserves the historical
  \(\alpha=\rho\exp(-q/2)\) convention;
- `beer_lambert` uses projected optical thickness
  \(\tau=\rho\exp(-q/2)\) and
  \(\alpha=1-\exp(-\tau)\).

Beer--Lambert forward, support cutoff, exact q/opacity VJP, and a live Metal
parity gate are implemented for the static-view q-UVT RGB direct-atomic path.
Projective atlas Beer rendering remains fail-closed.

Amplitude semantics are explicit. A world peak density compiles to the
fiber-integrated UVT coefficient

\[
a_{\rm fiber}
=a_{\rm peak}\, (ds/dd)\sqrt{2\pi\operatorname{Var}[d\mid a]}.
\]

If the input is already marked `fiber_integrated`, the inverse conversion is
used only in that target gauge. Such an amplitude is camera/gauge-specific; it
is not a native world-density convention.

`certify_confidence_band_order` computes exact extrema of each affine
pairwise depth-band gap on a UVT box. Uncertified overlap is sent conceptually
to depth-resolved emission-absorption without assigning one discrete order to
thick overlapping atoms. That boundary now has a native Metal implementation:

- `retained_fiber_transfer.metal` implements fixed-midpoint optical transfer
  and its VJP;
- `retained_fiber_metal.py` exposes differentiable forward/VJP and a tile
  certificate compiler;
- `hybrid_transfer.py` renders certified tiles with fast STAR and only
  ambiguous tiles with retained depth;
- the production benchmark/trainer records fallback counts, reason bits,
  active-set counts, and minimum pair separation.

The quadrature is fixed and capped at 64 samples. Integration-bound and
certificate derivatives are detached compiler decisions. This is not yet an
adaptive quadrature/error certificate.

Run the CPU-only gate with:

```bash
uv run --with pytest python -m pytest tests/test_spd4_world_tubes.py -q
```

## Trainable parallel lane

The reference object now also has a float32/MPS-safe production chart in
`third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/spd4_world_atom.py`.
The original restricted model remains the default. The multicamera benchmark
and unified paper runner select the source explicitly:

```text
--uvt-world-representation legacy_tube
--uvt-world-representation full_spd4
```

Both sources compile to the same six-field fast STAR raster ABI, so their
static multicamera A/B uses the same Metal forward/backward. Full SPD(4) has
18 trainable scalars per atom (14 geometry plus RGB/opacity), versus 14 for the
legacy tube (10 geometry plus RGB/opacity). A separate
`--uvt-spd4-init-precision-z` control permits a near-planar legacy-lift
initialization instead of silently changing initial projected support.

The physical runner axes are:

```text
--uvt-alpha-mode {peak_splat,beer_lambert}
--uvt-amplitude-convention {fiber_integrated,peak_density}
--uvt-render-backend {
  metal_tile,
  retained_fiber_metal,
  hybrid_retained_fiber
}
```

`peak_density` is a native world extinction density. The compiler multiplies
it by the physical fiber Jacobian and
\(\sqrt{2\pi\operatorname{Var}(d\mid u,v,t)}\). The Jacobian is evaluated with
the reciprocal-frame identity

\[
\left\|\frac{\partial x}{\partial d}\right\|
=
\frac{\|r_u\times r_v\|}
     {|\langle r_d,r_u\times r_v\rangle|},
\]

avoiding a batched inverse in the training hot path.

The controlled CPU capacity fixture is:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  python research_experiments/spd4_world_tubes/run_capacity_gate.py
```

It uses three static camera charts whose conditional-covariance observation
matrix has rank six. With one atom in each lane and initial losses matched to
within 0.2%, native SPD(4) reached `1.16e-13` MSE while the restricted source
retained `2.07e-4` MSE. This is a synthetic representation-capacity
certificate, not a public-scene quality or speed result. Artifact:
`artifacts/foundation_gates/spd4_native_multiview_capacity_cpu.json`.

A bounded Coffee Martini MPS smoke also exercises same-atom and
matched-parameter rows. The four-row validated summary is
`artifacts/spd4_parallel_smoke/summary.json`. All rows used two optimizer
steps, four frames, the same 30,720-pixel raster budget, and zero tile
overflow; sampled driver memory was approximately 37.6 MB in every row.
These are end-to-end plumbing and memory checks, not convergence evidence.

A later 16-frame/40-step seed-17 bounded ladder provides the first short
convergence evidence. At matched trainable scalars, the corrected runs are:

| source / opacity | atoms | scalars | train wall | heldout PSNR | peak driver |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy / peak splat | 256 | 3,584 | 4.9020 s | 5.9865 dB | 63.36 MB |
| full SPD(4) / peak splat | 199 | 3,582 | 4.7512 s | 7.0054 dB | 46.60 MB |
| full SPD(4) / Beer fiber | 199 | 3,582 | 4.6758 s | 7.1333 dB | 46.60 MB |

The first native run incorrectly took about 40 seconds because every
projection performed an unused 4x4 inverse, an unused determinant, and several
synchronous MPS scalar validations. After the hot-path repair, an isolated
199-atom projection fell from about 119 ms to 4.4 ms. The old artifacts remain
on disk as superseded failure evidence; corrected result directories end in
`_optimized`. These are single-seed short-protocol results, not a public
multi-seed benchmark claim.

The retained/hybrid production smokes show both success and a live boundary:
with 16 atoms the hybrid routed `10/64` tiles to retained depth and matched the
full-retained heldout metric; with 199 atoms at the current same-depth
initialization it routed `64/64`. The certificate is correct but currently too
conservative/selective for that dense initialization.

Current hard limits remain explicit:

- `static_view`, `dynamic_first_order`, and `projective_first_order` are
  enabled for full SPD(4); the two moving modes share the tested homogeneous
  one-chart affine camera gauge, while `segmented` remains fail-loud and is
  not a claim of this non-piecewise slice;
- the moving compiler is a first-order camera-program chart, not an exact
  nonlinear pinhole projection over arbitrarily long time windows;
- conditional depth variance is consumed by the static retained/hybrid route,
  but not by the ordinary peak-splat fast ABI;
- the Metal forward uses conditional depth for hard ordering, while its
  piecewise VJP returns zero cotangents for `depth0` and `depth_beta`;
- the retained-fiber VJP differentiates atom fields but not compiled depth
  bounds, tile membership, or the certificate decision;
- fixed midpoint quadrature is not an adaptive/error-certified integrator;
- projective-atlas retained-fiber fallback remains unimplemented;
- the 199-atom dense initialization currently falls back on every tile;
- no multi-seed/public-scene convergence claim follows from the bounded rows.

Primary current evidence:

- `artifacts/foundation_gates/spd4_retained_fiber_cpu_metal.json`
- `artifacts/spd4_bounded_16f_40step/`
- `artifacts/spd4_retained_hybrid_smoke/`
- `agent_notes/loose_notes/2026-07-27_17-36-51_spd4_physical_renderer_and_bounded_training.md`
