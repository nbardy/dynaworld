# Browser 30K / 384x288 Execution Board

Date: 2026-07-31

## Objective

Move the calibrated browser prototype toward a real 384x288, 30K-splat
training milestone without creating a parallel data/calibration contract or
weakening correctness. The work has four coupled resource goals:

1. remove duplicate host frame banks;
2. stop dense optimizer traffic over provably dormant reserve slots;
3. reduce cold projection/VJP storage traffic with measured storage-only FP16;
4. prove a complete 30K/384x288 camera-time cycle with finite loss and zero
   dropped tile references.

This is browser systems work. It does not promote trajectory-gated dynamic
3DGS to native 4DGS, World Tubes, or a paper baseline.

## Current Evidence

- Quiet-host 30K/96x72 staged training: 410-411 steps/s.
- Quiet-host 8K/192x144 exact separable SSIM: about 838 steps/s.
- Fast 30K/96x72 allocation: 37.91 MiB.
- Current reliable all-active ceiling: 30K at 96x72.
- Internal ID ceiling: 32,768; the full 32K camera-time cycle is invalid due
  to cumulative tile overflow.
- Current canonical browser bundle: train17/holdout1, 16 frames, 96x72, 4,096
  checked-in external seeds.

## Progress Snapshot

- `c4c27ac` implements exact dormant-prefix clear and Adam dispatch. Active
  slots retain dense Adam semantics; partial activation boundaries explicitly
  zero child moments, statistics, and gradients. Direct and staged runtime
  smokes crossed an activation event, and the full browser suite passed.
- `c90c086` shares the immutable Float32 frame/background roots across main,
  training, and validation workers through `SharedArrayBuffer`, with a
  structured-clone fallback. The current 32.27 MiB target bank avoids about
  64.55 MiB of duplicate backing.
- The headless harness now derives a byte-level requested resource plan before
  launching Chrome. The model exactly tracks the saved 30K/96 allocations plus
  the tested 16-byte sparse-prefix config delta.
- A no-browser 30K/384 staged preflight resolved 205.6 MiB of WebGPU buffers,
  a 108 MiB largest checkpoint binding, and 1.74 GiB recommended available
  memory. The request fits the portable 128 MiB per-binding floor.
- That run was correctly rejected for timing: Apple GPU utilization was 36%,
  and 10.15 GiB of used swap equaled 39.4% of physical memory. No performance
  result from that host state is promotable.

## Resource Derivation For 30K / 384x288

Assume 8x8 tiles, tile capacity 4,096, packed-FP16 checkpoints, and the current
compact staged path.

```text
pixels                 = 384 * 288 = 110,592
tiles                  = 48 * 36 = 1,728
pair slots             = 1,728 * 4,096 = 7,077,888
pair storage           ~= 54 MiB
checkpoint stride      = 32
checkpoint storage     = 110,592 * 128 * 8 bytes = 108 MiB
image/SSIM workspace   ~= 22.4 MiB
30K capacity state     ~= 19.5 MiB
one RGBA32F page       = 1.6875 MiB
estimated GPU buffers  ~= 206 MiB total
largest one binding    = 108 MiB < 128 MiB adapter binding limit
```

The total appears allocatable, but a successful allocation is not a valid
training result. The run must cover all 272 train-camera/time pairs and retain
zero cumulative overflow.

The real decoded Float32 frame bank is a separate host problem:

```text
18 cameras * 16 frames * 384 * 288 * RGBA32F = 486 MiB
three structured-clone owners                         ~= 1.46 GiB
one shared RGBA8 owner                                = 121.5 MiB
```

## Task Board

### T1: Shared Read-Only Dataset Storage

Owner: delegated worker.

Tasks:

- define an explicit worker-wire dataset representation;
- share large immutable typed arrays through `SharedArrayBuffer`;
- preserve a structured-clone fallback when cross-origin isolation is absent;
- ensure mutable optimizer/validation state never aliases accidentally;
- expose source bytes, unique backing bytes, and sharing mode in readiness
  telemetry;
- test main -> training -> validation identity and fallback behavior.

Acceptance:

- numerical dataset contract unchanged;
- one physical backing for shared frame data;
- full browser tests pass;
- worker smoke passes under the isolated server.

### T2: Dormant-Prefix Sparse Clear And Adam

Owner: delegated worker.

Tasks:

- derive active prefix exactly from the fixed fill schedule;
- dispatch gradient clear and reduce/update only over active slots;
- initialize newly activated gradient, moments, and statistics before first
  use;
- retain dense Adam semantics for every active slot;
- leave visibility-sparse Adam out of this first patch;
- report `active_update_slots / capacity`.

Acceptance:

- dense and sparse paths agree before, during, and after activation boundaries;
- all-active throughput does not regress materially;
- 4K-active/30K-reserve throughput improves;
- one-step parity and full browser tests pass.

### T3: Storage-Only FP16 Compact Projection Packet

Owner: parent after delegated design review.

Tasks:

- retain FP32 arithmetic and projected-gradient atomics;
- select fields by measured numerical sensitivity;
- prefer packed `u32` storage when native `f16` struct alignment does not
  actually reduce bytes;
- expose FP32 and packed-FP16 packet controls in the kernel lab;
- include packet bytes in memory telemetry;
- require finite-difference and active Apple WebGPU parity before timing.

Acceptance:

- RGB/objective parity remains inside the existing fixture tolerances or a
  separately justified storage tolerance;
- 9/9 active gradient families remain accepted;
- complete-step throughput wins under reversed-start quiet-host pairs;
- no convergence regression in a matched short trace.

### T4: Canonical Real 384x288 Bundle

Owner: delegated worker.

Tasks:

- reuse `multicam_video_data.py` and the existing browser exporter;
- preserve train17/holdout1 and exact 16-time semantics;
- scale intrinsics through the canonical loader/export path;
- retain fail-closed seed provenance;
- write real 384x288 target atlases, not an upscaled 96x72 benchmark tensor;
- record expected output bytes and reject unsafe host conditions before decode.

Acceptance:

- exporter-focused tests pass;
- output bundle reloads with exact dimensions, roles, times, and provenance;
- no heldout camera enters initialization;
- generated assets are clearly separated from the legacy untracked bundle.

### T5: 30K / 384x288 Systems Gate

Owner: parent.

Tasks:

- run allocation/one-step smoke first;
- capture actual memory plan and adapter limits;
- run parity on the selected precision path;
- run the complete 272-pair cycle with cumulative overflow telemetry;
- compare preview disabled versus enabled separately;
- use strict host contention policy and do not benchmark while memory pressure
  or unrelated Apple GPU work is high.

Acceptance:

- finite loss;
- zero current and cumulative tile overflow;
- no WebGPU validation errors;
- no storage buffer exceeds its reported limit;
- at least two stable timed rounds plus reversed-start repeat;
- artifact records whether topology maintenance, preview, and validation are
  included.

### T6: Quality And Capacity Follow-Up

Tasks:

- compare fixed-wall-time train and heldout metrics against 96x72;
- run at least two seeds and one second scene before a general claim;
- report low-pass versus high-frequency residuals;
- distinguish 30K capacity from 30K unique SfM points;
- treat 30K/384x288 as a systems milestone: it is only 0.27 splats per output
  pixel, versus 1.19 for 8K/96x72.

## Parallelization Boundaries

- Worker dataset files and exporter files can move concurrently.
- Sparse optimizer and FP16 packet work both touch
  `trainerWebGpu3dTiled.js`; only one edits that file at a time.
- GPU benchmarks run serially on a quiet host. Concurrent browser or Metal
  timing jobs invalidate each other.
- The 384x288 export and benchmark must not run while the workstation is under
  the current heavy compression/memory pressure.

## Stop Conditions

Stop and preserve a diagnostic artifact when:

- host preflight fails;
- the exporter would create another ambiguous or heldout-contaminated bundle;
- any active gradient family fails parity;
- cumulative overflow is nonzero;
- an apparent speedup depends on initial variant order;
- a precision lane wins throughput but produces a material matched-loss or
  short-convergence regression.
