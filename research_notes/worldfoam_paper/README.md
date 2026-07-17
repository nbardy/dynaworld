# WorldFoam Paper Packet

Status: second-paper lane, opened 2026-07-05.

This folder separates the **WorldFoam** paper from the **World Tubes in Gauged
Camera Space** paper.

The split is deliberate:

- **World Tubes** is the baseline-compatible camera-path compiler for dynamic
  Gaussian primitives. It preserves Gaussian-splat semantics, compiles
  support/visibility/adjoints through a known camera program, and compares most
  directly against per-frame replay of the same representation.
- **WorldFoam** is the lifted opacity/transmittance paper. It treats matter as
  bounded world cells or foam regions, pulls those cells through the camera-ray
  bundle, and renders by cumulative optical depth along ray fibers. It is a
  cleaner visibility model, but it is less mature as a quality/SOTA claim.

Use this folder when the question is:

```text
Can we replace discrete splat sorting with a camera-gauged ray-fiber opacity
field whose dominant cell/intersection work is reused across time?
```

Use `research_notes/gauged_uvt_trace_atlas/paper/` when the question is:

```text
Can dynamic Gaussian splats be compiled into reusable world-tube traces while
remaining compatible with baseline Gaussian-splat alpha compositing?
```

## Files

- `WORLDFOAM_HANDOFF.md` - dense clean-thread handoff: what was tried, current
  solution, goals, differences from World Tubes, known issues, and next work.
- `WORLD_FOAM_PAPER_DRAFT.md` - arXiv-style draft skeleton for the second paper.
- `WORLD_FOAM_EXPERIMENT_PLAN.md` - baselines, ablations, charts, acceptance
  gates, and honest claim boundaries.
- `WORLD_FOAM_MATH_APPENDIX.md` - polished math appendix scaffold: optical
  measures, ray-fiber pullback, visibility monoid, product integral, cell-path
  rasterization, replay theorem, VJP, commutator, and validation gates.
- `WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md` - current best plan after the
  reformulation pass: promote visibility monoid, optical-transfer event
  elements, commutator theorem, monoid VJP, and event closure; keep Magnus and
  boundary flux behind tests.
- `experiment_designs/cell_path_optical_transfer_fixture.md` - code-level plan
  for the first falsifiable fixture: exact optical-transfer monoid scan,
  same-representation cell-path replay, finite-difference VJP, commutator
  probe, artifact schema, and pytest command.
- `scientist_notes/` - triaged external-model/scientist dumps. Use this for
  incoming scratchpads so good ideas become searchable notes instead of chat
  sediment. The current reformulation triage is
  `scientist_notes/2026-07-05_optical_transfer_reformulation_intake.md`.
- `proofs/depth_fiber_operator_ordering.md` - cleaned theorem scaffold for the
  depth-fiber/operator-ordering story: gauge-invariant traces, non-commutation,
  lifted opacity, transmittance prefixes, same-representation replay, and
  fixed-topology VJP boundaries.

Related cross-track note:

```text
../gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
```

## Expansion Map

The folder is intentionally small but ready to expand:

```text
scientist_notes/      incoming external-model dumps and triage
proofs/               gauge invariance, replay equivalence, error bounds, scaling
experiment_designs/   active experiment specs before code
figures/              figure and table plans
paper/                eventual LaTeX/arXiv manuscript sources
```

Create optional subfolders only when they get their first real file. `proofs/`
now exists because the depth-fiber/operator-ordering theorem scaffold is a real
paper object, and `experiment_designs/` now exists because the cell-path
optical-transfer fixture has a concrete code/test plan.

## Current Claim Boundary

WorldFoam currently has real local evidence:

```text
Metal Gate4/native-cutwalk speed gates
sublinear-ish frame-count timing in focused microgates
trainable one-step real32 loader smoke
bounded-cell / Cech-AABB / raytrace implementation artifacts
```

It does **not** yet have the full evidence needed for a strong quality paper:

```text
public dynamic-scene benchmark quality parity
official CUDA/Warp parity fixture acceptance
DeepView-quality acceptance above the current low-PSNR/low-SSIM rows
clean evidence that foam transmittance beats STAR/GS on real RGB quality
```

So the right near-term paper shape is either:

```text
theory + prototype + speed/scale paper
```

or:

```text
full rendering paper after the quality/parity gates clear
```

Do not blend the stronger World Tubes compiled-adjoint evidence into WorldFoam
as if it proves WorldFoam quality.
