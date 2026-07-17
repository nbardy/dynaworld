# WorldFoam Cell-Path Code Plan

Date: 2026-07-06 01:32 KST.

## Context

The WorldFoam paper/math pass had promoted the visibility monoid, cell-path
atlas, same-representation replay theorem, and monoid VJP as the strong core.
The docs said the next gate was a cell-path optical-transfer fixture, but the
file/function/test plan was still implicit.

The user asked whether we know what code to implement, how to extend it, and
whether it is planned enough to make the theory testable.

## Decision

Write a concrete implementation spec before touching renderer code:

```text
research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md
```

The spec defines the first testable code as a pure CPU/Torch fixture, not a
Metal or real-scene integration:

```text
research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py
research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py
```

It names the transfer element API, algebra contract, replay equivalence test,
VJP formulas, finite-difference thresholds, commutator probe, summary JSON
schema, pytest command, and the promotion ladder.

## Current Model

The next useful code is not another shader micro-variant. It is a small
exactness gate:

1. Prove `(beta, m)` transfer composition matches alpha/transmittance semantics.
2. Prove a compiled cell-path word matches same-representation per-frame replay.
3. Prove fixed-topology prefix/suffix VJP matches finite differences for
   `beta`, `m`, `DeltaTau`, `sigma`, run length, and color.
4. Connect the commutator theorem to a two-layer swap fixture.
5. Only then promote boundary flux, witness scores, Magnus compression,
   feature-gauge transfer, or hot renderer integration.

## Files Updated

Added:

```text
research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md
agent_notes/loose_notes/2026-07-06_01-32-58_worldfoam_cell_path_code_plan.md
```

Updated routing/index docs:

```text
research_notes/worldfoam_paper/README.md
research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
research_notes/worldfoam_paper/WORLDFOAM_HANDOFF.md
research_notes/README.md
TODO/README.md
EXPERIMENTS.md
PROJECT_INDEX.md
```

## Falsification

If the fixture fails alpha equivalence, the optical-transfer algebra is not
wired to baseline semantics. If replay equivalence fails, the compiler changes
representation semantics rather than amortizing work. If finite differences
fail, the appendix VJP should stay out of claims until fixed.
