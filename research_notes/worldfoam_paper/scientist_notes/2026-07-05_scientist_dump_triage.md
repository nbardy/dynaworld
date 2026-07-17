# Scientist Dump Triage: WorldFoam

Date: 2026-07-05

Source:

```text
/Users/nicholasbardy/.codex/attachments/dc3af06b-dd69-492c-93a1-14696f5788f7/pasted-text.txt
```

Purpose: extract durable research value from the scientist scratchpad, decide
what notes to take, identify which parts are already covered by the current
WorldFoam folder, and turn the dump into an actionable research plan.

## 0. Verdict

The dump is good. It is not just rephrasing the handoff; it sharpens the paper
positioning in several important ways.

Best contributions:

```text
1. WorldFoam should be framed as a camera-gauged transmittance compiler, not
   "PowerFoam on Metal" or "World Tubes with cells."

2. The central distinction from World Tubes is retained fiber:
   World Tubes pushes toward UVT footprints and then certifies order;
   WorldFoam keeps z and makes visibility a prefix optical-depth integral.

3. Do not force a Schur-complement story onto foam.
   Foam's beauty is not Gaussian marginalization; it is bounded-cell event
   structure plus transmittance.

4. Do not store dense sigma(u,v,t,z).
   The atlas must store sparse cell-ray events, intervals, basis coefficients,
   and prefix summaries.

5. Do not sell foam as merely a better sorter.
   Sorting can be an ablation/fallback, but the main semantics should be
   transmittance.

6. Same-representation replay is the proof baseline.
   External STAR/GS comparisons are contextual; per-frame WorldFoam replay is
   the theorem-level comparator.

7. The first paper should be theory/prototype/speed-scale unless quality gates
   clear.
```

The strongest one-sentence version from the dump is worth keeping almost
verbatim:

```text
World Tubes compiles dynamic Gaussians into reusable UVT traces and then
certifies visibility; WorldFoam keeps the depth fiber alive, compiles bounded
world cells into sigma(u,v,t,z), and lets transmittance prefixes solve
visibility as optical depth instead of sort order.
```

## 1. What Is Already Covered

The current folder already has the right spine:

```text
research_notes/worldfoam_paper/README.md
research_notes/worldfoam_paper/WORLDFOAM_HANDOFF.md
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md
```

Already covered:

```text
bounded cells pulled through camera gauges
sigma(u,v,t,z)
Beer-Lambert / transmittance rendering
foam atlas object
World Tubes vs WorldFoam split
speed evidence vs quality gap
same-representation replay requirement
synthetic exactness suite
gradient correctness requirement
topology/witness/holonomy caveats
official parity caveat
```

The dump is still useful because it organizes the argument more sharply around
rollbacks and reviewer attacks.

## 2. Notes To Take Directly

### N1. Primary reframe

Take:

```text
WorldFoam is a camera-gauged lifted opacity/transmittance compiler.
Its object is sigma(u,v,t,z) plus transmittance prefix, not a projected splat,
not a marginal UVT Gaussian, and not merely a bounded Voronoi cell.
```

Where to use:

```text
paper abstract
paper introduction
README one-screen summary
talk/deck first slide
```

### N2. Core thesis

Take:

```text
For known camera programs, WorldFoam compiles bounded world-cell opacity into a
sensor-time-depth atlas. Rendering frames, finite exposures, and rolling-shutter
images becomes evaluation of transmittance prefixes along shared ray-fiber event
intervals rather than independent per-frame traversal/sorting.
```

Important nuance:

```text
Output samples remain output samples.
The sublinear claim is about dominant intersection/visibility/backward metadata,
not a magical sublinear bound in pixels.
```

### N3. Rollback: do not Gaussianize foam

Take strongly.

Bad idea:

```text
Approximate every bounded cell as a Gaussian and reuse World Tubes Schur math.
```

Why bad:

```text
It destroys the finite support, face events, adjacency, and traversal structure
that make foam interesting.
```

Decision:

```text
Let World Tubes own Gaussian marginalization.
Let WorldFoam own retained-fiber transmittance.
```

### N4. Rollback: do not store dense UVTZ

Take strongly.

Bad idea:

```text
Dense 4D sigma(u,v,t,z) grid.
```

Why bad:

```text
It throws away event sparsity and turns the paper into generic volume rendering
with a memory problem.
```

Decision:

```text
Store sparse cell-ray events, endpoint records, layer summaries, compact basis
coefficients, and prefix state.
```

### N5. Rollback: do not sell foam as a better sorter

Take strongly.

WorldFoam should not be positioned as:

```text
Power cells + improved primitive sorting
```

Main semantics:

```text
optical-depth prefix
```

Sorting remains:

```text
compatibility mode
fallback
ablation
bridge to World Tubes/GS
```

### N6. Alpha semantics bridge

Take.

World Tubes:

```text
Can claim baseline-GS equivalence if support/order certificates match the
baseline renderer.
```

WorldFoam:

```text
Changes semantics. Alpha is induced by integrated density:
alpha(y) = 1 - exp(- integral sigma(y,z) dz)
```

Therefore:

```text
First comparison is compiled WorldFoam vs per-frame WorldFoam replay.
External STAR/GS comparisons are contextual.
```

### N7. Adjoint structure

Take.

Forward:

```text
I(y) = integral T(y,z) sigma(y,z) c(y,z) dz
```

Variation:

```text
delta I(y)
  = integral T sigma delta c dz
  + integral T (c(z) - I_behind(y,z)) delta sigma(z) dz
```

Implication:

```text
Backward needs front transmittance prefix and behind-radiance suffix.
Gradient is not purely local in z because density changes affect everything
behind it through transmittance.
```

This should drive the implementation contract.

### N8. Tape versus recompute discipline

Take.

The dump gives a useful rule:

```text
Do not add another tape variant unless it has both:
    memory contract
    adjoint contract
```

This is exactly right. The old shader lane already risks infinite shader
gardening.

### N9. Event complexity as the real asymptotic

Take.

Define:

```text
E_cell = certified cell-ray intersection events over camera program
E_depth = depth-layer/prefix partition events
E_topo = topology refresh/split/fallback events
```

Then:

```text
payload_atlas = O(E_cell + E_depth + E_topo + basis_payload)
payload_replay ~= O(F * visible_cell_intersections_per_frame)
```

This is the right asymptotic story.

### N10. Where sublinear dies

Take. This should become a chart family.

WorldFoam loses if:

```text
cell-camera intersections churn every frame
near-camera cells explode support
wide FOV creates too many gauge domains
training topology changes force constant refresh
transmittance requires too many depth layers
fallback dominates
CPU prep dominates
memory grows nearly per-frame
```

The "death curve" is important. A paper that shows failure boundaries will feel
more credible.

### N11. Theorem set

Take as a paper scaffold.

Likely theorem/proposition set:

```text
1. Gauge invariance of pulled-back opacity with fiber Jacobian.
2. Compiled transmittance equivalence to per-frame WorldFoam replay when
   intervals/bases are exact.
3. Approximation error bound from optical-depth error + radiance basis error +
   interval/support error.
4. Event-based scaling proposition.
```

### N12. Reviewer attack map

Take. These are likely real reviewer objections:

```text
"Isn't this just volume rendering?"
"Isn't quality worse than Gaussian splatting?"
"Where is sublinear if pixels are linear?"
"Topology is nondifferentiable."
```

Keep the prepared answers, but make them evidence-backed.

### N13. Figure plan

Take and refine.

Strongest figure sequence:

```text
1. World Tubes vs WorldFoam conceptual split.
2. Camera-ray bundle and gauge coordinate z.
3. Bounded power cell pulled into ray-fiber intervals.
4. Transmittance prefix along z.
5. Foam atlas data structure.
6. Same-representation replay ladder.
7. Synthetic exactness results.
8. Crossing translucent slabs stress test.
9. Frame-count scaling curve.
10. Event-density death curve.
11. Gradient finite-difference checks.
12. Quality gap / scoped limitations.
```

### N14. Algorithm skeletons

Take and move into the paper draft once experiments start.

Algorithms:

```text
Algorithm 1: Compile Foam Atlas
Algorithm 2: Evaluate Frame / Exposure
Algorithm 3: Direct VJP
```

The pseudocode is aligned with our current theory.

## 3. What Not To Take Literally

### D1. Do not overclaim "sublinear train time"

The dump says measured train time can be sublinear when bottlenecks dominate.
That is fair, but paper language must remain precise:

```text
dominant compiled bottlenecks can scale with event complexity
end-to-end time may be sublinear in tested regimes
output samples are still linear
```

### D2. Do not make Beer-Lambert the novelty

The dump correctly warns:

```text
the rendering equation is standard
```

The novelty must be:

```text
bounded-cell camera-path compilation
sparse ray-fiber event atlas
reusable transmittance/adjoint prefixes
known-camera program amortization
```

### D3. Do not promote topology regularizers yet

Witness/holonomy ideas are promising but unproven.

Current status:

```text
diagnostics first
regularizers only if diagnostics predict heldout-free failure
```

### D4. Do not make public RGB quality the first proof

The dump's experiment order is right:

```text
analytic sphere
dense raymarch reference
per-frame replay
compiled equivalence
gradient finite differences
crossing translucent slabs
fast orbit event-density curve
real loaded-frame speed scaling
public data later
```

Follow that order.

## 4. Updates We Should Make From This Dump

### U1. Strengthen the README

The WorldFoam folder should explicitly say it can expand into:

```text
scientist_notes/
proofs/
experiment_designs/
figures/
paper/
```

Current action taken in this pass:

```text
created scientist_notes/
added this triage note
updated README with the scientist_notes route
```

### U2. Add theorem/proof TODOs

Add future files:

```text
research_notes/worldfoam_paper/proofs/gauge_invariance.md
research_notes/worldfoam_paper/proofs/transmittance_equivalence.md
research_notes/worldfoam_paper/proofs/error_bound.md
research_notes/worldfoam_paper/proofs/event_scaling.md
```

Do not create empty stubs until someone is ready to fill them.

### U3. Add experiment design TODOs

Add future files:

```text
research_notes/worldfoam_paper/experiment_designs/synthetic_exactness_suite.md
research_notes/worldfoam_paper/experiment_designs/same_representation_replay.md
research_notes/worldfoam_paper/experiment_designs/translucent_crossing_stress.md
research_notes/worldfoam_paper/experiment_designs/gradient_correctness_contract.md
research_notes/worldfoam_paper/experiment_designs/real32_clean_speed_gate.md
```

The current `WORLD_FOAM_EXPERIMENT_PLAN.md` is enough until these become active.

### U4. Add figure/table plan

Add future file:

```text
research_notes/worldfoam_paper/figures/figure_plan.md
```

Use the scientist dump's 12-figure sequence as the starting point.

## 5. What Is Good For The Paper Right Now

Best paper language:

```text
WorldFoam keeps the depth fiber alive.
```

Best mathematical contrast:

```text
World Tubes:
    Schur-complement marginalization to UVT footprint + visibility certificate.

WorldFoam:
    retained ray-depth fiber + transmittance prefix.
```

Best novelty boundary:

```text
Beer-Lambert is not new.
Camera-compiled sparse bounded-cell transmittance atlases are the contribution.
```

Best proof baseline:

```text
compiled WorldFoam atlas vs per-frame WorldFoam replay
```

Best near-term experiment:

```text
constant-density sphere + crossing translucent slabs + gradient finite
differences
```

Best scaling metric:

```text
event complexity growth vs frame-count growth
```

Best honesty sentence:

```text
Current evidence supports a theory/prototype/speed-scale paper, not yet a SOTA
dynamic novel-view rendering claim.
```

## 6. Organized Folder Status

Yes, we now have an organized WorldFoam research folder:

```text
research_notes/worldfoam_paper/
```

Current contents:

```text
README.md
WORLDFOAM_HANDOFF.md
WORLD_FOAM_PAPER_DRAFT.md
WORLD_FOAM_EXPERIMENT_PLAN.md
scientist_notes/2026-07-05_scientist_dump_triage.md
```

Recommended expansion:

```text
scientist_notes/      incoming external-model dumps and triage
proofs/               gauge invariance, equivalence, error bounds, scaling
experiment_designs/   active experiment specs before code
figures/              figure/table plans
paper/                eventual LaTeX or arXiv manuscript sources
```

Do not over-structure yet. Create the subfolders only when they get their first
real file. The current folder is organized enough to expand without scattering
the paper lane across the repo.

## 7. Next Actions

Immediate:

```text
1. Leave current paper draft and handoff intact.
2. Use this triage note as the filter for future scientist dumps.
3. Next implementation goal should be synthetic exactness + same-representation
   replay, not another broad shader variant.
```

Next document additions when needed:

```text
proofs/gauge_invariance.md
experiment_designs/synthetic_exactness_suite.md
experiment_designs/same_representation_replay.md
figures/figure_plan.md
```

Next code/experiment work:

```text
analytic constant-density sphere
dense raymarch reference
per-frame WorldFoam replay
compiled atlas equivalence
prefix/suffix gradient finite differences
crossing translucent slabs
fast orbit event-density curve
clean real32 speed gate only after representation gates
```
