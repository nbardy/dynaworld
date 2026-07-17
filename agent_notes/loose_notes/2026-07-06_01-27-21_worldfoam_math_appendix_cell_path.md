# WorldFoam Math Appendix And Cell-Path Paper Gate

Date: 2026-07-06

## Context

The user attached a polished math-appendix style dump for the WorldFoam paper
lane and asked whether the new parts should be added to the paper, proofs, or
replace the approach. The conclusion was: add the best math as an appendix and
paper/proof pointers, but do not replace the current evidence-facing approach.

## What Changed

Added:

```text
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
```

Updated:

```text
research_notes/worldfoam_paper/README.md
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
research_notes/worldfoam_paper/WORLDFOAM_HANDOFF.md
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
research_notes/README.md
TODO/README.md
EXPERIMENTS.md
PROJECT_INDEX.md
```

## Promoted Math

The appendix promotes this identity:

```text
WorldFoam is gauge-covariant optical transfer factored through a compiled
cell-path atlas.
```

Stable paper objects:

```text
optical matter measures
ray-fiber lambda/eta pullback
gauge-invariant optical one-form
visibility monoid
optical transfer matrix / product integral
splats as atomic optical-transfer events
cell-path word rasterization
compiled atlas K_Gamma = {C_l, w_l, Phi_l, S_l, P_l, E_l}
same-representation replay theorem
monoid/prefix-suffix VJP
constant-density owner-run derivatives
commutator visibility theorem
```

## Gated Branches

These remain behind finite-difference or correlation tests:

```text
segment Hessian / second-order optical-depth structure
Magnus / commutator compression
optical-depth polynomial basis
interface flux
radical-face crossing derivatives
sphere endpoint derivatives
flux witness score
gauge-covariant feature transfer
universal ray-space transfer
```

## Next Concrete Work

The next paper-math implementation gate is:

```text
cell-path optical-transfer fixture
```

Minimum contents:

```text
constant-density owner-run word
monoid scan
same-representation replay equivalence
finite differences for DeltaTau, sigma, color, and run length
```

Do not jump to boundary flux, flux witness scores, or Magnus compression before
the cell-path replay and VJP fixture pass.

## Verification

Ran:

```text
git diff --check
code-fence parity checks on edited Markdown files
```

Both passed.
