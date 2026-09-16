# Transported Geometry and Reusable Differentiable Ray Operators

Research prepared for Nicholas Bardy, 8 September 2026.

The PDF contains 20 content pages plus two pages of linked primary references. The editable LaTeX source is `research.tex`. This package contains two reproducible CPU mathematical diagnostics, not a production rasterizer, GPU performance comparison, or trained reconstruction model.

## Read the research

Direction A (pages 3–5): a compiled spacetime-Gaussian control, including the actual STGS model and its published baseline.

Direction B (pages 6–9): transported convex constant-density material regions, exact overlapping-ray interval integration, and a full analytic reverse pass including affine deformation, optical density, motion, and camera parameters. Exact static ellipsoid overlap integration is existing work in EVER, not claimed here as new.

Direction C (pages 10–12): opaque algebraic solids, implicit root derivatives, a conditional finite visibility-event theorem, its expensive construction cost, and filtered-boundary derivatives.

Direction D (page 13): transported Fourier density and an analytic transmission operator; its order-insensitive shortcut is rejected for general colored occlusion.

Direction E (pages 14–17): geometry-backed ray operators, interpolation/transposed-adjoint factorization, explicit value and derivative error bounds, and a geometry-induced exponential/polynomial temporal compiler. Canonical world geometry is retained for novel viewpoints. Existing light fields, kinetic data structures, and empirical interpolation are credited.

Pages 18–20 cover full costs, matched-quality/gradient contracts, actual diagnostics, and small falsification experiments. The mathematical guarantees require their stated regularity, event-topology, and coverage assumptions. They are not claims of scientific priority or general speedup over STGS.

## Run the mathematical diagnostics

Tested in this environment with Python 3.13.5 and NumPy 2.3.5. The programs require NumPy and use float64. They do not require a GPU, PyTorch, downloaded datasets, or network access.

```bash
python checks.py --out results_reproduced.json
python temporal_operator.py --out temporal_results_reproduced.json
```

Both programs raise an assertion failure if their numerical checks fail. Random seeds are fixed in the sources. Small floating-point differences across platforms and NumPy builds are expected.

### `checks.py`

1. Two moving/deforming overlapping boxes: exact interval color and a hand-written reverse pass for 62 parameters. All parameters are checked using central finite differences at three step sizes.
2. Independent midpoint volume integration without sorting endpoints, using 131,072 depth samples.
3. A moving and mass-conservingly dilating sphere: compare a bilinear ray-time operator against exact point-ray colors and all six analytic parameter derivatives at 16,384 queries.
4. An interpolation transpose identity check.
5. A forward-perfect, backward-wrong cache counterexample and a colored-layer order counterexample.
6. Explicit operation counts separating fixed node work from growing dense query work.

The selected 289-node sphere example has maximum image error 4.62e-5 and maximum row-Jacobian L2 error 2.83e-4 on the sampled test grid. It has 56.7 times fewer direct node evaluations than the 16,384-query reference, NOT a measured 56.7-times speedup. It is a two-dimensional patch strictly away from silhouettes, NOT a full novel-view reconstruction test.

### `temporal_operator.py`

Constructs one ray through 24 independently translating, overlapping constant-density slabs, with event order fixed on the physical interval [-0.5, 0.5]. Their 47 ray intervals induce a 48-term exponential color law. It tests degree-1 through degree-4 polynomial temporal operators and their moment-based reverse passes on 2,048 time queries.

A quadratic uses three coefficients per RGB channel. Its maximum image error is 5.91e-8 and maximum coefficient-space row-Jacobian L2 error is 2.38e-5. The analytic truncation bounds are 9.50e-6 and 9.25e-4 respectively, excluding floating-point roundoff. A separate finite-difference test covers 192 exponential weights and optical-depth slopes, with relative reverse error 2.97e-11.

These are derivatives with respect to intermediate exponential coefficients and slopes. The test does not verify the full pullback from those intermediates to slab-motion parameters, changing event topology, finite-pixel boundaries, or camera interpolation. No end-to-end timing speedup is asserted.

## Compile the document

With a TeX Live installation containing the packages named in the preamble:

```bash
pdflatex -interaction=nonstopmode -halt-on-error research.tex
pdflatex -interaction=nonstopmode -halt-on-error research.tex
```

The source uses NewTX fonts through the local TeX installation; no font files are distributed in this package.

## What remains untested

No matched GPU STGS benchmark, external-dataset reconstruction, training convergence comparison, full 5D ray-time coverage test, filtered-silhouette gradient validation, or large independently moving scene experiment was performed. All are explicit falsification targets in the document. In particular, a fixed nonzero direct-fallback fraction reintroduces linear expensive scene work; rebuilding and derivative certification can erase any advantage.

Dense image output is always Omega(TP). Arbitrary dense upstream image gradients also require Omega(TP) reads. Streaming and reuse do not remove those costs.
