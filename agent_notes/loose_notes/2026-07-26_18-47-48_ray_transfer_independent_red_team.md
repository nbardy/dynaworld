# Ray Transfer Independent Red-Team

Date: 2026-07-26 18:47:48 +0900

Status: documentation-only research audit

## Trigger

Review the proposed "gauge-invariant ray polynomial rendering" or
"gauge-invariant ray holonomy renderer for moving cameras" against the
existing STAR UVT work, then decide:

```text
own paper versus STAR integration
defensible novelty
falsification gates
claims that must remain out
publication sequence
```

## Material Read

Primary proposal:

```text
/Users/nicholasbardy/.codex/attachments/
c3f6b522-fd32-4797-941d-8fc2ed5722e2/pasted-text.txt
```

Related proposal chain:

```text
d3e27c47-2005-4080-9515-c4eb7668a86e/pasted-text.txt
ed37275e-39f1-43bb-b4a7-4f692d8ceb02/pasted-text.txt
59da4296-f678-4d80-a5cb-8ec2526c3360/pasted-text.txt
```

Canonical repository material:

```text
research_notes/renderer_lane_taxonomy.md
research_notes/gauged_uvt_trace_atlas/
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
research_notes/spacetime_gaussian_representation/
PROJECT_INDEX.md
BASELINES.md
```

Implementation and artifact surfaces:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/
torch_gsplat_bridge_star_uvt/projective_trace.py

src/train/star_uvt_projective_interval_backend.py
src/train/star_uvt_feature_overfit_trainer.py

research_experiments/world_foam_lane2/
cell_path_optical_transfer_fixture.py

outputs/benchmarks/
2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json
2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json
2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json
2026-07-08_worldfoam_cell_path_optical_transfer_summary.json
```

Concurrent documentation was discovered during the pass:

```text
research_notes/worldfoam_paper/scientist_notes/
2026-07-26_gauge_invariant_ray_holonomy_intake_and_paper_split.md

agent_notes/loose_notes/
2026-07-26_18-39-35_ray_holonomy_paper_split_review.md
```

Those files were treated as user/concurrent work and were not rewritten.

## Chronology

1. Read the full 1,165-line proposal and extracted the transfer operator,
   convex-potential atom, compiler, derivative, and complexity claims.
2. Compared the proposal against World Tubes' early depth pushforward and
   actual projective interval implementation.
3. Compared it against the WorldFoam proof scaffold and math appendix.
4. Found exact equation-level duplication: WorldFoam already has the same
   optical generator, product integral, gauge one-form, visibility monoid,
   commutator, prefix/suffix VJP, replay theorem, and event-scaling boundary.
5. Inspected the WorldFoam optical-transfer fixture and tests to verify that
   part of this algebra is already executable evidence, not only prose.
6. Inspected STAR's projective trace implementation and saved artifact metrics
   to separate implemented evidence from proposed work.
7. Performed a narrow primary-source sanity check on standard volume rendering,
   deterministic interval integration, and learned convex primitives.
8. Found the concurrent normalized intake, then wrote independent supplements
   instead of modifying it.

## Main Corrections

### The transfer half is WorldFoam

The proposal's:

```text
A = [ -sigma I   j ]
    [ 0          0 ]

P exp integral A
```

is already section E of the WorldFoam math appendix.

The overlap counterexample is already Proposition 3 / the commutator theorem.
The Duhamel derivative is another expression of the same forward/reverse
transfer derivative already represented by prefix/suffix VJP.

### "Holonomy" is not the right headline

The proposed object is open-ray transport or a product integral. Holonomy
normally concerns closed-loop transport. The proposal proves invariance under
ray-coordinate change, not invariance under a general internal feature-basis
gauge. An open-path connection transforms by endpoint basis factors.

### Camera time and ray depth are separate

Depth, inverse depth, and log depth can be fiber coordinates. A half-angle
orbit parameter normally reparameterizes the camera-time base. Exposure-time
changes require their own measure/Jacobian and cannot be justified by the
ray-depth change-of-variables proof.

### The atom is the actual new hypothesis

The self-normalized strongly convex atom has valuable theorems:

```text
unique ridge
compact convex slices
regular boundary
one support interval per straight ray
exact single-atom optical depth
```

Its compiler value remains unproved because:

```text
the minimizer/minimum are generally implicit algebraic time functions
sensor-time coefficient rank is not guaranteed low
full colored transfer remains numerical
orientation is not unique at repeated Hessian eigenvalues
active overlap and root/event cost are unmeasured
training-time recompilation is unmeasured
```

## Decision

```text
Paper A:
    finish World Tubes / STAR

Paper B:
    WorldFoam retained-fiber transfer only after decisive correctness,
    native systems, training, and public breadth gates

convex atom:
    optional producer until a matched capacity/compiler test passes
```

The proposal is an elegant WorldFoam extension hypothesis, not a new paper
result today.

## Durable Outputs

Added:

```text
research_notes/worldfoam_paper/scientist_notes/
2026-07-26_ray_transfer_lineage_and_novelty_audit.md

research_notes/worldfoam_paper/scientist_notes/
2026-07-26_ray_transfer_falsification_and_publication_strategy.md
```

The first records the exact repository lineage, implementation evidence,
terminology corrections, atom caveats, and narrow novelty ledger.

The second pre-registers quantitative semantic, primitive, overlap, native
systems, training, public-evidence, and external-novelty gates, plus hard kill
criteria and publication sequencing.

## Validation

No code or benchmark was changed. No training, rendering, or GPU experiment
was launched.

