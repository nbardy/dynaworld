# WorldFoam G4 executor and tractability closeout

## Scope

This session completed the source integration for the two WorldFoam rows in
the frozen G4 public-quality ablation:

- `worldfoam_native4d`: bounded compiled shared-adjoint training;
- `worldfoam_framewise_replay`: the same representation, compiler, material,
  geometry optimizer, targets, rays, and checkpoint with sequential per-frame
  reverse scheduling.

Tests and the one-pixel runtime smoke are preflight only.  The deliverable is
the real ablation matrix; no smoke is paper evidence.

## Source completed

- Sealed split-specific public target/ray inputs and deterministic 1,024-site
  point-cloud initialization.
- Real MPS/native shared and framewise executor sessions with full-geometry
  gradients, CPU optimizer promotion, raw-only checkpoints, bounded heldout
  prediction, and sampled RSS/MPS peak accounting.
- Empty-store next-generation promotion after every geometry update, so stale
  compiled programs are never reused without a certificate.
- A real-native public one-pixel train/backward/update/checkpoint/heldout smoke
  seam.  It cannot emit paper evidence.
- Exact row-ray binding for heldout chunks; the executor now verifies the row
  rays against the sealed calibrated provider before rendering.
- Heldout requests are rejected before allocation unless camera, frame,
  dimensions, pixel interval, and chunk size match the sealed work plan.  The
  CPU prediction limit is derived from the independent `32,768`-pixel plan cap,
  not from the caller's requested size.  Every prediction receipt is chained.
- The frozen stage `lr_multiplier=0.35` now scales material and geometry SGD
  for both WorldFoam routes, is bound into each promotion receipt/step digest,
  and is preserved in checkpoint provenance.  The earlier source silently used
  multiplier `1.0`, unlike the Gaussian routes, so it was not yet a fair row.
- Training/checkpoint MPS/time measurement begins exactly at optimizer step 0;
  shared row-level accounting separately measures setup and the complete row
  through heldout evaluation.  Session shutdown quarantines accelerator roots
  and raises if completion cannot be proven rather than swallowing the fence.
- Runtime capability now requires the exact post-103 prediction schema and a
  sealed evidence file bound to source commit/source hashes, dataset/protocol,
  native-library hash, prediction-schema hash, and per-route smoke receipts.
- Shared fail-closed tractability accounting used by row preflight and executor
  capability.

## Deterministic full-schedule census

The full `384x512`, `K=4`, 300-step schedule contains `235,929,600` sampled
observations.  Because the current compiler cold-compiles each unique
`(view,pixel)` track after every geometry update and the sampler sometimes has
only one distinct view at an epoch boundary, realized counts depend on seed:

| Seed | Cold track compiles | 128-track spatial bundles |
| ---: | ---: | ---: |
| 17 | 115,015,680 | 898,560 |
| 29 | 112,852,992 | 881,664 |
| 43 | 113,442,816 | 886,272 |

The exact two-distinct-view-per-step upper bound is `117,964,800` cold track
compiles and `921,600` bundles.  The framewise control makes exactly
`1,843,200` bounded native step calls per seed.  At seed 17, the current
factory also implies `34,504,704,000` complete camera-record validations and a
`117,776,056,320` total admitted-site-reference upper bound before deeper
active-owner work.  Compiler-bound chart rows/native blocks are explicitly
labelled total upper bounds, not observed counts or maximum-live allocations.

The factory deliberately retains no compiled-program cache and has no
certified neighboring-track/spatial candidate or topology-template reuse.
Consequently the implementation is bounded in logical peak residency but the
full v1 public schedule is not tractable.  Both row preflight and executor
capability now fail closed on this exact census even if a one-pixel smoke later
passes.

## Memory truth

`native_memory_fit=false` remains the only honest status.  The representation
and live logical state are frame-independent and small, but no real G4 row has
measured native scratch, allocator behavior, compiler/Python working sets,
process RSS, or MPS driver peak.  No MPS, decode, native rebuild, or heavy run
was performed in this session.

## Runtime/assets blockers

1. The installed native extension is stale relative to its source.
2. A real-native runtime smoke must run on a quiet host after rebuild.
3. G4 v1 stays blocked on the full-schedule tractability census above.
4. The mapped public train/heldout cache capability and PyAV conversion
   capability are not present for the current Coffee Martini dry plan.
5. Cook Spinach and Cut Roasted Beef initialization assets still need their
   configured deterministic build/seal path if absent or unsealed.
6. No G4 paper row or memory acceptance artifact exists yet.

## Bounded v2 remedies, not implemented here

Choose one explicitly rather than silently weakening v1:

1. certify cross-pixel/spatial candidate or topology-template reuse with exact
   stale-geometry invalidation and a measured compiler census; or
2. refreeze one matched selected-ray training protocol for all four routes,
   keeping targets, rays, optimizer steps, and final evaluation identical.

Do not reduce only WorldFoam pixels, freeze geometry without relabelling the
experiment, or promote the one-pixel smoke as an ablation.

## CPU-only verification

- All new/edited WorldFoam integration modules passed `py_compile`.
- All modules imported successfully without accelerator allocation.
- The three seeded dry plans reproduced the exact counts above.
- A CPU behavioral check verified that material, position, velocity, and
  weight updates at multiplier `0.35` equal `0.35` times the base update.
- Seed-17 row preflight exposed the tractability receipt and blocker.
- Executor capability remained `source_only=true` with two independent
  blockers: stale native ABI and full-schedule tractability.
