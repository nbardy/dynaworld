# SPD(4) physical renderer completion and bounded training

- Time: 2026-07-27 17:36:51 +0900
- Lane: World Tubes / strict-SPD(4) production engineering
- Author/role: Codex coordinator, implementation, falsification, and bounded-run operator
- Objective: finish the parallel native-SPD(4) renderer slice far enough to
  train it beside the legacy tube, add physical Beer--Lambert and retained-depth
  visibility routes, run bounded comparisons safely on local MPS, and record
  both positive and negative results.
- Why attempted: the previous source/compiler gate proved that a native
  `mu4 + SPD(4)` atom can encode motion and conditional covariance, but the
  production shader still discarded conditional depth variance and used only
  peak-splat opacity. The user explicitly requested shader completion and
  side-by-side training rather than another theory-only pass.

## Inputs

- Native source/compiler:
  `research_experiments/spd4_world_tubes/`
- Production STAR variant:
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/`
- Coffee Martini bounded protocol:
  `src/train_configs/paper_protocols/coffee_martini_protocol_bounded_16f_40step.jsonc`
- Parameter-matched protocol:
  `src/train_configs/paper_protocols/coffee_martini_protocol_bounded_16f_40step_spd4_param_matched_199.jsonc`
- Base multicamera config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`
- Train cameras: `cam04`, `cam09`
- Heldout camera: `cam06`
- Seed: `17`
- Local host: Apple MPS, 24 GiB unified memory

No web/literature input was used in this engineering session. This was not an
isolated novelty lane; existing repository math and prior notes were used.

## Implemented renderer branches

The original runtime lane remains available as:

```text
--uvt-world-representation legacy_tube
```

The native source is parallel and opt-in:

```text
--uvt-world-representation full_spd4
```

The renderer axis is now:

```text
--uvt-render-backend {
  dense,
  metal_tile,
  retained_fiber_metal,
  hybrid_retained_fiber
}
```

The amplitude/alpha axes are:

```text
--uvt-alpha-mode {peak_splat,beer_lambert}
--uvt-amplitude-convention {fiber_integrated,peak_density}
```

The defaults remain the historical `legacy_tube + peak_splat + metal_tile`
behavior where selected by existing configs. No existing lane is silently
reinterpreted.

## Beer--Lambert derivation and implementation

For projected optical thickness

```math
\tau(a)=\rho\exp[-q(a)/2],
```

the physical single-primitive opacity is

```math
\alpha(a)=1-\exp[-\tau(a)].
```

The implemented derivatives are

```math
\frac{\partial\alpha}{\partial\rho}
=\exp[-\tau]\exp[-q/2],
```

```math
\frac{\partial\alpha}{\partial q}
=-\frac12\tau\exp[-\tau].
```

The shader returns zero derivative on the explicit alpha cap. Beer mode uses
`max_alpha=1`; peak-splat retains the historical `0.99` cap. Support culling
uses the exact Beer threshold

```math
\tau_{\rm threshold}=-\log(1-\alpha_{\rm threshold}).
```

The core q-UVT CPU, dense, Metal forward, and direct VJP paths now carry this
mode. Projective atlas Beer rendering remains fail-closed because the full
projective shader family has not been given the same physical contract.

## Native amplitude convention

For a world peak extinction density `rho_peak`, the camera compiler produces a
fiber-integrated peak optical thickness

```math
\tau_0
=\rho_{\rm peak}
  \left\|\frac{\partial x_{\rm world}}{\partial d}\right\|
  \sqrt{2\pi\,\operatorname{Var}(d\mid u,v,t)}.
```

For the time-preserving affine gauge with spatial rows
`r_u, r_v, r_d`, the physical fiber Jacobian is evaluated without a matrix
inverse:

```math
\frac{\partial x_{\rm world}}{\partial d}
=\frac{r_u\times r_v}
       {\langle r_d,r_u\times r_v\rangle},
```

and therefore

```math
\left\|\frac{\partial x_{\rm world}}{\partial d}\right\|
=\frac{\|r_u\times r_v\|}
       {|\langle r_d,r_u\times r_v\rangle|}.
```

This is the reciprocal-frame column of the inverse gauge, computed with cross
products and a dot product. A regression test monkeypatches
`torch.linalg.inv` to fail, proving the production path no longer depends on a
general 4x4 inverse.

`fiber_integrated` skips this conversion entirely. `peak_density` is the
native world convention; the same numeric initialization is not expected to
match the same center alpha in every view because the fiber scale is
camera/direction dependent.

## Retained-depth optical transfer

For each active atom at a fixed `(u,v,t)`, conditional depth is

```math
d\mid a \sim
\mathcal N\left(
  d_0+\beta^\top(a-\mu_a),
  \sigma_d^2
\right).
```

The fallback evaluates the depth-dependent density

```math
\lambda_i(d,a)
=\tau_i(a)
 \frac{\exp[-(d-\bar d_i(a))^2/(2\sigma_{d,i}^2)]}
      {\sqrt{2\pi\sigma_{d,i}^2}},
```

then performs ordered emission--absorption over depth samples. This consumes
the conditional variance that the six-field fast STAR record otherwise drops.
The native Metal implementation has an explicit VJP for UVT mean/precision,
conditional depth mean/slope/variance, optical thickness, and color.

The current integration is fixed midpoint quadrature, at most 64 samples.
Depth integration bounds and the fallback decision are compiled/detached. This
is a differentiable renderer with respect to atom fields, not an adaptive
quadrature/error certificate.

## Variance-aware tile certificate

The Metal tile compiler:

1. computes conservative optical-support AABBs;
2. records at most 256 active atoms per tile-time cell;
3. evaluates exact affine extrema of each pairwise conditional-depth band gap
   over the overlap box;
4. certifies the fast hard order only when the confidence bands remain
   separated by the requested gap;
5. sends active overflow, invalid records, or ambiguous colored overlap to the
   retained-fiber path.

Reason bits are:

```text
1 = active-set overflow
2 = invalid record
4 = ambiguous depth bands
```

The certificate is nondifferentiable by design, as are ordinary bin/order
decisions. Gradients flow through the branch selected for each pixel.

## Failure found during integration: under-dispatched Metal VJP

The initial retained-fiber Metal oracle passed on a `1x1`/`2x2` fixture but
returned zero gradients for a production `8x8` pixel outside the first few
threads.

Root cause:

- `torch.mps.compile_shader` chooses the dispatch grid from the first tensor
  argument;
- the VJP kernel's first tensor was `grad_ma`, shaped `[N,3]`;
- only `3N` pixel threads were launched, not `F*H*W`;
- the tiny oracle happened to have no pixel beyond that accidental grid.

Fix:

- move the `[F,H,W]` fallback mask to the first kernel argument for both
  forward and VJP;
- retain bounds checks;
- add a production-sized hybrid gradient test whose selected pixel lies beyond
  the old accidental grid.

After the fix, the integration fixture produced nonzero gradients:

```text
depth0 grad      [ 2.56514e-3, -2.56403e-3 ]
depth variance   [-6.37252e-5,  2.67570e-4 ]
```

This was a real shader-dispatch bug, not a symmetric zero-gradient scene.

## Failure found during bounded training: projection synchronization

The first parameter-matched SPD(4) run took `40.50 s` versus the legacy
`4.90 s`. This initially looked like an inherent cost of the 4D object.
Profiling falsified that interpretation.

Two hot-path mistakes were present:

1. every projection computed a batched 4x4 inverse solely for the optional
   peak-density conversion, even in `fiber_integrated` mode;
2. several `.item()` validation checks and an unused `torch.linalg.det`
   forced full MPS synchronization on every projection.

For 199 atoms, the isolated projection fell from approximately:

```text
119 ms -> 4.4 ms
```

after:

- skipping the unused amplitude conversion;
- replacing the required peak-density inverse with the reciprocal-frame
  formula above;
- retaining structural checks on MPS while deferring scalar numerical
  validation to the trainer's synchronized nonfinite-loss/rollback guard;
- preserving eager fail-loud numerical checks on CPU/reference calls.

The pre-fix outputs remain on disk and are explicitly superseded rather than
silently rewritten:

```text
artifacts/spd4_bounded_16f_40step/full_spd4_199_param_matched/
artifacts/spd4_bounded_16f_40step/full_spd4_199_beer_fiber/
artifacts/spd4_bounded_16f_40step/full_spd4_199_beer_peak_density/
```

## Corrected bounded 16-frame / 40-step results

All rows use the same two-train-camera/one-heldout-camera Coffee Martini split,
seed 17, four target frames per optimizer step, and `direct_atomic+index_add`.
These are short bounded engineering/convergence rows, not publication-quality
multi-seed results.

| Row | Atoms | Trainable scalars | Train wall (s) | Steady forward (s) | Train-view aggregate PSNR | Heldout PSNR | Peak driver bytes | Overflow note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy peak-splat | 256 | 3,584 | 4.9020 | 0.06208 | 7.6746 | 5.9865 | 63,356,928 | zero |
| full SPD(4) peak-splat, equal count | 256 | 4,608 | 4.5168 | 0.05376 | 12.8940 | 7.0888 | 63,373,312 | 42 eval tiles in `cam09`; use matched row for clean comparison |
| full SPD(4) peak-splat, parameter matched | 199 | 3,582 | 4.7512 | 0.05774 | 11.4993 | 7.0054 | 46,596,096 | zero |
| full SPD(4) Beer, fiber integrated | 199 | 3,582 | 4.6758 | 0.05788 | 12.0743 | **7.1333** | 46,596,096 | zero |
| full SPD(4) Beer, world peak density | 199 | 3,582 | 3.7387 | 0.04452 | 8.3113 | 6.5838 | 54,984,704 | zero; initialization is not center-alpha matched |

The clean parameter-matched comparison gives:

```text
full SPD(4) peak vs legacy:
  +1.0189 dB heldout
  -3.1% train wall
  -26.5% sampled peak driver bytes

full SPD(4) Beer fiber vs legacy:
  +1.1467 dB heldout
  -4.6% train wall
  -26.5% sampled peak driver bytes
```

Status: computational evidence on one seed and one short protocol. It does not
prove a public benchmark win or convergence superiority. It does falsify the
claim that the native SPD(4) source inherently costs about ten times more in
this runner.

Artifacts:

```text
artifacts/spd4_bounded_16f_40step/legacy_256/
artifacts/spd4_bounded_16f_40step/full_spd4_256_optimized/
artifacts/spd4_bounded_16f_40step/full_spd4_199_param_matched_optimized/
artifacts/spd4_bounded_16f_40step/full_spd4_199_beer_fiber_optimized/
artifacts/spd4_bounded_16f_40step/full_spd4_199_beer_peak_density_optimized/
```

## Retained/hybrid production smokes

| Row | Atoms | Train wall (s) | Steady forward (s) | Fallback tiles | Fallback fraction | Invalid/overflow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full retained | 16 | 0.8753 | 0.05706 | 64/64 | 1.0 | 0/0 |
| certified hybrid | 16 | 0.9559 | 0.07375 | 10/64 | 0.15625 | 0/0 |
| certified hybrid stress | 199 | 3.0194 | 0.29124 | 64/64 | 1.0 | 0/0 |

The 16-atom hybrid and full-retained runs produce the same heldout PSNR
(`5.868593`) to the recorded precision. This is evidence that the mixed branch
is wired consistently on that smoke.

The 199-atom stress row is an important negative result: the current exact
band-separation certificate is too conservative/selective at the dense,
same-depth initialization and falls back everywhere. A full-resolution hybrid
run was therefore not launched. The next certificate work should target one
of:

- better physical depth initialization;
- bounded color-commutator certificates for approximately commuting overlap;
- smaller certified support based on an explicit quadrature/error tolerance;
- learned depth separation if justified by the training objective.

Simply weakening the confidence radius without changing the retained
quadrature extent would invalidate the guarantee and is not an acceptable
tuning shortcut.

Artifacts:

```text
artifacts/spd4_retained_hybrid_smoke/retained_16x4f_2step/
artifacts/spd4_retained_hybrid_smoke/hybrid_16x4f_2step/
artifacts/spd4_retained_hybrid_smoke/hybrid_199x4f_2step/
```

## Verification

Native extension builds:

```text
CPython 3.14 build_ext --inplace: pass
CPython 3.11 venv build_ext --inplace: pass
```

Focused native Beer gate:

```text
15 passed
```

Combined SPD(4), unified runner, WorldFoam material, Beer, and retained-fiber
suite before the final projection optimization:

```text
142 passed, 4 skipped
```

Post-optimization SPD(4) suite:

```text
32 passed
```

Retained-fiber CPU/Metal gate:

```text
forward max abs error: 2.682209e-7
worst normalized VJP error: 1.266599e-7
32-vs-1024-sample error: 2.459586e-4
driver allocation: 27,754,496 bytes
status: pass
```

WorldFoam material evidence from the parallel hardening pass:

```text
57 passed, 3 skipped CPU/relevant suite
40 passed opt-in Metal suite
max Metal forward normalized error: 7.51e-8
max Metal VJP normalized error: 5.96e-8
```

## Claim status

- Proved lemma: the reciprocal-frame formula equals the physical spatial fiber
  Jacobian for a time-preserving affine camera gauge.
- Tested implementation fact: native mean+SPD(4), Beer alpha/VJP, conditional
  depth variance, tile certification, masked retained-fiber fallback, and
  trainer/runner metadata are wired for the static camera route.
- Computational evidence: the bounded single-seed SPD(4) rows beat the legacy
  row at matched trainable scalars without added train wall or driver memory.
- Computational evidence: the 16-atom hybrid uses retained depth on only
  15.6% of tiles and matches the full-retained smoke metric.
- Refuted interpretation: the initial 10x native-SPD(4) slowdown was inherent
  to the mathematical representation.
- Refuted engineering assumption: a tiny Metal VJP fixture was enough to
  validate dispatch extent.
- Unresolved: adaptive/error-certified depth quadrature.
- Unresolved: selective certification at dense same-depth initialization.
- Unresolved: exact nonlinear/projective-atlas Beer and retained-fiber wiring.
- Unresolved: multi-seed/public-scene convergence and quality.
- Unresolved: event-boundary derivatives and quadrature-bound derivatives.
- Unresolved: fair cross-view initialization for native world peak density.

## Precise next actions

1. Add a depth-initialization ablation before altering the certificate.
2. Derive and test a color-commutator error certificate that can accept
   overlapping bands when their transfer operators commute within a declared
   image-error tolerance.
3. Add adaptive quadrature with a forward and VJP error estimator; do not
   claim physical fallback completion until both pass.
4. Extend the existing projective/gauge atlas with a certified camera-chart
   remainder and carry the retained-fiber fallback through that route.
5. Run the 199-atom peak/Beer rows across seeds and at least three scenes on an
   adequate host before putting a quality claim in the abstract.
6. Keep legacy and native rows in separate output directories and one local
   MPS process at a time.

## Related predecessor notes

- `2026-07-27_04-56-24_spd4_worldfoam_four_pass_implementation_plan.md`
- `2026-07-27_15-55-31_spd4_paper_runner_axis.md`
- `2026-07-27_16-19-33_native_spd4_parallel_training_and_metal_smoke.md`
- `2026-07-27_16-24-18_ordered_ray_transfer_paper_and_shader_update.md`
- `2026-07-27_16-51-19_worldfoam_material_selection_and_paper_sync.md`

