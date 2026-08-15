# WorldFoam tests, ablations, memory, and spatial tractability

## Context

The user correctly challenged two conflations during the paper-completion work:

1. source/unit/runtime tests are not paper ablations; and
2. a small logical state formula is not a measured native memory result.

This note records the evidence boundary after re-running only allocation-free
plans and source/import verifiers.  No MPS workload, cache conversion, native
build, or publication row ran in this pass.

## Current evidence ledger

Observed facts:

- G6 native training-memory ablation: `0/21` measured rows.
- G4 public heldout-quality ablation: `0/36` measured rows.
- G6 dry planning produces 12 shared-adjoint rows, 9 same-representation
  sequential controls, and 3 restart processes: 21 evidence rows in 24 fresh
  processes.
- The G6 dry plan reports `dry_run_is_evidence=false`,
  `paper_claim_permitted=false`, and the single implementation blocker
  `native_extension_older_than_bound_native_sources`.
- Native source contains `133/133` schemas/implementations.  The installed
  CPython 3.11 binary exposes `103/133`; its missing set is exactly the 30
  post-103 kinetic operations.
- G4 planning now reports three top-level code/runtime blockers:
  missing runtime capability, missing certified spatial compile reuse, and the
  stale native extension.  Scene rows additionally require mapped public
  caches; Cook Spinach and Cut Roasted Beef also require their initializer
  assets.

Completed real ablations that must not be erased by the two missing matrices:

- adaptive M3/M5 material selection: 72 fits, 36 selection rows, seeds
  `17/29/43`, pure-family selection/oracle agreement `1.0`, zero selection
  regret, adaptive/best-fixed mean loss ratio `0.313405`;
- WorldFoam synthetic ordered-transfer G0/G3: 8 scenes x 7 cameras, 224
  convergence rows, 168 comparator rows, and 56 adaptive rows; and
- accepted World Tubes theorem and variable-camera closure/death evidence.

These are genuine bounded/synthetic ablations.  They do not substitute for G6
allocator evidence or G4 public quality.

There are also three completed 600-step Coffee Martini progressive-512 rows
from 2026-07-22, but `BASELINES.md` correctly classifies them as historical
schema-v1 diagnostics rather than accepted paper evidence.  Mean heldout World
Tubes PSNR was only about `5.9153 dB` (dynamic 3DGS was about `4.91 dB` in the
inspected seed-17 row), and the later schema-v2 audit revoked their promotion.
They prove that a full schedule ran; they do not prove publication quality or
replace the current ablation matrix.

## Memory statement

### Observed analytic/source accounting

At `S=1024` sites, the current P0 material plus affine kinetic geometry state
has:

```text
combined live logical state                 = 114,688 B
raw restart checkpoint                      =  81,920 B
conservative live + checkpoint clone bound  = 278,528 B
```

The live state is `112 B/site`; the checkpoint is `80 B/site`.  None of these
terms contains requested frame count `F`.

### What remains unobserved

The formulas exclude:

- MPS allocator and driver residency;
- Metal private/register/spill and command-buffer state;
- native compiler and lowering scratch;
- Python object/allocator and process-group RSS;
- mapped-file residency, OS readahead, and decoder scratch; and
- transient checkpoint/update clones outside the explicit logical contract.

Therefore the only correct current flag is:

```text
native_memory_fit = false
```

This means “not yet measured,” not “known to exceed the limit.”  G6 promotion
requires every frozen row to remain below `<2 GiB` sampled MPS working set and
`<4 GiB` sampled process-group RSS, with the cross-`F` work/memory invariants
and restart parity green.

The 8-GiB available-host-RAM launch guard is incident headroom.  It is not a
representation-size estimate and gives no basis for a 32-GB requirement.

## New G4 failure mode: bounded memory, unbounded work

The current exact active-track compiler admits up to all 1,024 sites, validates
the complete 300-frame camera record, and cold-compiles one continuous program
for each unique `(view,pixel)` after each full-geometry update.  It retains no
cross-pixel template or spatial candidate cache.

For the frozen 300-step, four-image, `384x512` all-pixel schedule, the exact
deterministic scheduler counts are:

| Seed | Cold track compiles | Spatial bundles |
|---:|---:|---:|
| 17 | 115,015,680 | 898,560 |
| 29 | 112,852,992 | 881,664 |
| 43 | 113,442,816 | 886,272 |

The two-distinct-view-every-step upper bound is `117,964,800` cold compiles.
The same-representation framewise control requires exactly `1,843,200` native
step calls for every seed.

Inference:

- the bounded spatial-bundle lifetime can keep peak memory small;
- it does not make the current full-pixel public schedule computationally
  feasible; and
- a one-pixel native smoke establishes wiring only, never full-schedule
  tractability.

The G4 worker and overall runner now fail closed on this distinction.

## Initialization and dynamic-quality risk

The current WorldFoam point-cloud initializer explicitly sets all site
velocities to zero and seeds time-constant P0 material.  Full-geometry SGD can
in principle learn motion, but its every-step cold recompile is exactly the
tractability failure above.  Consequently the current G4 source is not yet a
credible dynamic public-quality protocol merely because its optimizer exposes
velocity gradients.

Cheap falsification before any 36-row spend:

1. freeze a small public selected-ray schedule;
2. compare zero-velocity initialization against a content-addressed
   multi-time/motion initializer;
3. report loss decrease, velocity update norm, event/topology churn, compile
   time, and heldout temporal error; and
4. reject the quality lane if motion remains effectively static or structural
   compilation dominates the row.

## Remedy branches

### Branch A: certified cross-pixel/spatial reuse

Hypothesis:
    A regular-triangulation/cell-adjacency structure plus conservative
    screen-time candidate bounds can reduce exact track compilation from all
    sites and allow neighboring rays to share structural work.

What would falsify it:
    candidate sets remain close to all 1,024 sites, certification misses owner
    events, or rebuild cost after geometry updates still dominates.

Cheap test:
    on a 64x64 public crop, compare exhaustive active-owner programs against
    adjacency-pruned programs for every ray/time sample; require zero missed
    owners/events and report candidate-count and compile-time quantiles.

If supported:
    implement the spatial compiler as the full-pixel G4-v1 promotion path.

### Branch B: matched selected-ray G4-v2

Hypothesis:
    A frozen, deterministic selected-ray training budget shared by all four
    routes is enough to measure quality fairly while full heldout evaluation
    remains streamed and exhaustive.

What would falsify it:
    WorldFoam alone receives fewer target pixels, Gaussian routes use a
    different loss schedule, or the selected-ray budget is too small for any
    route to cross its quality floor.

Cheap test:
    one scene/seed budget curve with identical ray indices at several fixed
    target-pixel budgets; report both target and rasterized pixels.

If supported:
    freeze a new v2 contract and preserve the all-pixel v1 as a separate
    stress/scaling target.  Never silently mutate v1.

### Branch C: topology-cache phases plus sparse geometry updates

Hypothesis:
    Most optimization can update material on a cached structural generation,
    with explicitly scheduled geometry updates and full invalidation/recompile.

What would falsify it:
    heldout quality needs near-every-step geometry updates or cached topology
    becomes invalid before the scheduled refresh.

Cheap test:
    fixed selected-ray budget, geometry refresh periods `1/10/50/never`, with
    event-certificate failures, compile share of wall time, and quality.

If supported:
    present refresh cadence as a named ablation and report it honestly; do not
    call it identical to every-step full geometry.

## Immediate decisions

1. Rebuild and attest the 133-op extension only on a quiet host.
2. Run G6 first; it directly answers the user's memory question and does not
   depend on solving full-pixel G4 spatial reuse.
3. Keep G4-v1 fail closed.
4. Choose between Branch A and a separately frozen Branch-B protocol using a
   bounded public crop/budget experiment, not taste.
5. Improve the dynamic initializer before spending on full G4.
6. Continue to use source/unit/runtime tests as preflight, while reporting only
   accepted measured rows as ablations.

## Host constraint at this audit

The local data volume had about `6.7 GiB` free and the VM compressor remained
under severe pressure.  That is why this pass did not rebuild the extension,
decode approximately `1.59 GB` of mapped public cache payloads, or launch MPS.
The decision is incident prevention, not evidence that WorldFoam needs 32 GB.
