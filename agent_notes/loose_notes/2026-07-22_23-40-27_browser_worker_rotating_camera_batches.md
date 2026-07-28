# Browser Worker Rotating Camera Batches

## Scope

Implemented explicit rotating K-camera membership in the worker-owned WebGPU trainer without changing
the SPA, dataset loader, calibration contract, or Python trainer hierarchy.

## Contract

- Camera roles define training membership. If roles are absent, the legacy first-`trainViewCount`
  convention is the fallback.
- `camerasPerStep` defaults to 4 when more than 4 train cameras exist, otherwise all train cameras.
- Step `s` begins at `(s * K) % trainViewCount` and takes K consecutive entries circularly.
- The Coffee Martini train17/holdout1 bundle therefore covers all 17 training cameras within five
  K=4 optimizer steps. `cam06`, at `heldoutViewIndex`, never enters a training membership set.
- At K >= train count, legacy contiguous datasets retain the original all-camera shader sampling
  branch. Interleaved role layouts use explicit indices so heldout exclusion wins over compatibility.

This is not same-time K-camera batching. Every sampled ray still selects its frame independently.
Motion/static samples are grouped into per-camera GPU ranges once at initialization. Every active
camera receives deterministic ray slots and focused lookup is O(1); if a camera has no sample in that
focus class, the ray remains uniform on the same active camera. That preserves the motion-focused
sampler without inventing a synchronized patch-loss contract.

## Rendering And Status

- Worker render options now accept `viewIndices`.
- The role-aware default triptych is first train, middle train, heldout. The Coffee Martini 18-camera
  bundle resolves to `[0, 8, 17]`, rather than the historical `[0, 1, 2]` fallback.
- Shared status adds cameras per step, rotation start, and total train-view count.
- The ready event reports train indices, render indices, and `sameTimeGrouped: false`.
- Training remains nonblocking: no optimizer, metric, validation, or render readback was added to the
  submission pump.

## Verification

- `node --test web/dynaworld_browser_trainer/tests/*.test.mjs`: 16/16 passed.
- `git diff --check` passed for the owned files.
- Final isolated-browser smoke on the 18-camera bundle reached step 108 with SharedArrayBuffer status,
  OffscreenCanvas rendering, train/heldout validation, and no console or WebGPU warnings.
