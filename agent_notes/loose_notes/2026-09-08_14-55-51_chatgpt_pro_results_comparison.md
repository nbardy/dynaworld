# ChatGPT Pro results retrieved, verified, and compared

The user asked to download both finished Pro responses, assess their math and
code against the existing work, and decide whether to finish World Tubes or
incorporate a different representation.

Canonical session result:

[`research_notes/external_reviews/2026-09-08_chatgpt_pro/COMPARISON.md`](../../research_notes/external_reviews/2026-09-08_chatgpt_pro/COMPARISON.md).

## Actions and evidence

- Retrieved the actual PDFs and source ZIPs through the existing in-app ChatGPT
  tabs: “Explore Dynamic Scene Rendering” and “Dynamic Rasterizer Design.”
  PDF lengths are 22 and 20 pages. Both tasks were complete and remain open.
- In-app browser content export was unsupported. Documented download events
  and actual link clicks worked; the files were copied from Downloads into the
  dated external-review directory. Archive paths were checked for traversal,
  symlinks, and excessive uncompressed size before extraction.
- Preserved the supplied sources, results, archives, and PDFs. PDF/archive
  hashes and all 16 supplied source-manifest hashes verified. Browser response
  claims were treated as untrusted until source/code review.
- Hubble reviewed the open mathematical derivations and existing WorldFoam
  overlap. Volta reviewed quadric lowering, filtered conic gradients, and
  numerical-check coverage. Both were read-only; no nested agents or GPU work.
- Ran all four downloaded scripts sequentially after static inspection, with
  single-thread BLAS and 60-second process timeouts. All exited zero. Full
  commands, outputs, logs, environment, hashes, and timing provenance are under
  `research_notes/external_reviews/2026-09-08_chatgpt_pro/verification/`.
  These are CPU mathematical diagnostics, not accepted paper benchmark rows.
- Bundled Python lacked SciPy; the existing repo venv already had NumPy 2.4.4
  and SciPy 1.17.1. No dependencies were installed. No Torch/GPU initialization
  or native builds were needed.
- Rendered and inspected the central temporal theorem (open PDF page 17) and
  filtered conic derivation (gauge PDF pages 8-9). Preserved original PDFs.
- Checked primary sources for EVER, 2006 GPU quadric ray casting, 2018 edge
  sampling derivatives, and DiffTetVR. This is targeted prior-art checking,
  not exhaustive novelty clearance.

## Assessment and revision of the earlier model

The responses are useful, appropriately scoped research studies. The strongest
open result constructs Taylor coefficients directly from the cumulative
optical-depth slopes of affine-length, fixed-order words and bounds both value
and gradient truncation. It requires planar faces, fixed shape/density/color,
linear relative translations and fixed ray direction. A translating sphere
already violates affine chord length. Its temporal test slows motion until
all crossings disappear and validates intermediate weights/slopes, not the
complete geometry-to-compiled-response pullback.

The important backtrack is that **WorldFoam already has completed-transfer
compilation, shared node adjoints, derivative-sensitive rank conditions, and a
constant-state exact word reverse**. Pro's contribution is a possible specialized
coefficient constructor/certificate inside that existing lane. It does not
justify another temporal-operator framework. Compare it against the existing
affine-log/Chebyshev code if this follow-on lane becomes active.

The strongest gauge-directed result is exact one-ellipse/box filtered coverage
and a compact conic-moment adjoint. Congruence and simple-root derivatives are
sound, but neither the full rational motion compiler nor multi-object filtered
visibility is implemented. The ten-entry camera-relative quadric carries
geometry only: a textured sphere can rotate without changing it. Attached
appearance requires extra state and derivative paths.

Recommendation: finish World Tubes after applicable correctness repairs and
the existing empirical gates. Use camera-program compilation, coordinate
Jacobians, Gaussian marginalization, certified reuse domains, and the shared
compiler adjoint as the explanation. Keep gauge invariance as an optional
coordinate-consistency identity. Dense output/residual work remains Omega(FP);
only expensive reusable work can become independent of requested frame density
under fixed physical complexity and accuracy.

No production source, canonical manuscript, baseline row, TODO acceptance state,
or GPU resource gate changed during this review. The repo had substantial
pre-existing/concurrent edits, which were preserved. The comparison links to
the previous correctness and scaling audits instead of duplicating or silently
rewriting their history. No new GPU run was launched.

## Validation

The comparison's local links, original PDF/archive hashes, source manifests,
and all four rerun exit codes were checked successfully. The numerical results
closely reproduce the supplied diagnostics, with expected finite-difference
roundoff differences. The final comparison includes the exact scope limitations,
mathematical assumptions, remaining failures, and falsifiable follow-on tests.
