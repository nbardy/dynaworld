# Ordered Ray Transfer Paper And Shader Update

Date: 2026-07-27 KST

## Context

The user approved the World Tubes / retained-transfer paper split and asked to
update the papers and fork any required shaders.

The checkout already contained newer uncommitted work:

```text
strict SPD(4) World Tubes source/reference
WorldFoam M0--M5 finite-element material reference
parameterized tiny-Metal material evaluator and VJP
accepted 11-record CPU/Metal foundation artifact
four-pass implementation and independent terminology audits
```

Those files were treated as live user work and preserved.

## Terminology Backtrack

The prior intake used "gauge-invariant ray holonomy renderer" as the Paper-B
headline. Independent review identified a real collision:

```text
open ray:
    ordered parallel transport / product integral

closed cell-complex loop:
    holonomy
```

The paper now uses:

```text
Gauge-Invariant Ordered Ray Transfer for Moving Cameras
```

and preserves "ray holonomy" only as informal geometric intuition. This keeps
the strong framing without overloading the existing closed-loop diagnostic.

## Paper A Update

Updated:

```text
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER.tex
```

Added:

```text
explicit retained-fiber contribution boundary
thin-event optical-transfer commutator
extended-profile limitation of opacity/color/representative-depth summaries
confidence-band overlap implication
precise sibling split between baseline-compatible STAR and retained transfer
```

Regenerated the LaTeX from Markdown with the command in `REPRODUCE.md`.

## Paper B Update

Updated:

```text
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
```

Changes:

```text
new ordered-ray-transfer subtitle
camera-program versus gauge distinction
path-ordered optical generator in the abstract
coordinate-invariance statement with physical-length Jacobian
open-ray transport versus loop-holonomy terminology
M0--M5 controlled material-shader contribution
accepted CPU/Metal parity metrics and artifact path
explicit missing production/native-4D/training claims
```

## Shader Decision

No fused-slab renderer clone was created. The required fork already existed as:

```text
research_experiments/world_foam_lane2/finite_element_material_transfer.metal
research_experiments/world_foam_lane2/finite_element_material_metal.py
```

This is the scoped shader fork selected by the four-pass audit: fork the
segment material evaluator, not geometry, owner/event tape, scan, or renderer.
M0--M5 share one `(tau,beta,m,density_bounds,branch_status)` ABI and one VJP.

The current shader SHA-256:

```text
1826fe6c7cd5416d6e8295dedd701d77ee0cc642ea45dcb6f06df67c237cb778
```

matches the accepted artifact:

```text
artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_metal_20260727.json
```

Accepted bounded metrics:

```text
CPU independent-quadrature max abs error  5.96e-15
CPU explicit-VJP normalized error         5.55e-17
Metal forward normalized error            7.51e-8
Metal VJP normalized error                5.96e-8
invalid rows                              0
```

## Verification

Focused CPU gate:

```text
45 passed, 2 skipped in 2.11 s
```

Covered:

```text
strict SPD(4) World Tubes reference/adapter
WorldFoam M0--M5 material transfer
WorldFoam cell-path optical-transfer fixture
```

The two skips are opt-in Metal tests. The already accepted tiny-Metal artifact
was not regenerated because its stored source hash exactly matches the current
shader. No training or publication-scale MPS run was launched.

`git diff --check` passed for the manuscript sources and generated LaTeX.

## Remaining Promotion Gate

Do not fork the production renderer yet. Integrate one material law into the
owner-run fused loss only after it beats M0 and the affine-color control at
matched bytes or matched quality. A production retained-fiber STAR fallback
and compact native-4D WorldFoam field/compiler remain separate future gates.
