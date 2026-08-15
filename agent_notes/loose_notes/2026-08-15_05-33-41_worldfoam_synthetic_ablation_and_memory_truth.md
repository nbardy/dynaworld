# WorldFoam synthetic ablation and memory truth

Date: 2026-08-15 KST

## Why this work happened

The user correctly objected that passing tests are not the requested paper
ablations and asked whether the new memory-light WorldFoam had actually been
shown to fit in memory. The answer remained no: the tests only guard the
producer/verifier, while the measured native artifact was absent.

## Host state and execution boundary

The current Mac was still unsafe for native publication work:

- only about `9.3 GiB` disk free (`98%` full);
- `15.94/17 GiB` swap used;
- severe compression and unrelated workloads.

No native rebuild, Metal training, or broad MPS job was launched. This was not
a generic 32-GB requirement. It was an incident-calibrated stop on this host.

## WorldFoam memory-ablation audit

The full training-memory contract is source-complete:

- `S=1024` sites, `P=512` selected tracks;
- staged `F=8` versus fused `F=8` parity;
- fused `F=8/64/300`, three repeats;
- same-representation sequential replay `F=8/64/300`, three repeats;
- three restart processes;
- 21 measured rows / 24 fresh processes total;
- material and geometry updates, checkpoint/restart, RSS/MPS/bridge peaks,
  direct selected-pixel targets, and fail-closed source/native/hardware hashes.

The two focused producer/verifier files pass `36` tests. A dry plan emits zero
evidence and reports exactly one blocker:

```text
native_extension_older_than_bound_native_sources
```

The current fused-slab source verifier is green. The stale binary registers
`103/133` schemas and is missing 30 new schemas. Logical state remains only
accounting (`112 B/site`, or `114,688 B` at 1024 sites); it is not a measured
runtime peak. The next G6 operation on a quiet eligible Mac is therefore:

1. rebuild `world_foam_lane2_fused_slab_v0` with Python 3.11;
2. pass source and import/ABI verifiers with all 133 schemas;
3. rerun the dry plan with zero blockers;
4. execute the guarded 24-process sequence;
5. independently verify the final 21-row measured artifact.

There is no current CUDA/B200 implementation of this Metal ABI. Porting before
the paper would be a new backend, not a drop-in execution move.

## New real paper ablation: WorldFoam G0/G3

Implemented:

- `research_experiments/world_foam_lane2/worldfoam_synthetic_visibility_suite.py`
- `research_experiments/world_foam_lane2/verify_worldfoam_synthetic_visibility_suite.py`
- `research_experiments/world_foam_lane2/test_worldfoam_synthetic_visibility_suite.py`

The suite is a deterministic float64 CPU ray-section experiment, not a unit
test proxy. It crosses all eight scenes and all seven camera programs in the
paper plan. It evaluates:

- dense/analytic ordered Beer--Lambert transfer;
- WorldFoam depth-layer approximations at `16/32/64/128` layers;
- a 32/64 estimator with 128-layer fallback;
- representative-depth sorted component transfer;
- depth-marginal transfer;
- transmittance, temporal flicker, and representative order-flip diagnostics;
- ordinary-depth versus log-depth gauge integration with and without the
  physical Jacobian.

The final artifact is:

```text
outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/summary.json
```

It contains 224 layer rows, 168 comparator rows, and 56 adaptive rows. The
independent verifier accepts it. All six acceptance gates pass.

Key results:

- independent analytic sphere RGB max error: `4.36e-4`;
- physical-Jacobian gauge error: `3.33e-7`;
- error without the Jacobian: `0.305335` (`916,927x` larger);
- 128-layer fifth-percentile context PSNR: `37.9252 dB`;
- crossing mean-MSE improvement versus representative-depth sort: `82.25x`;
- crossing mean-MSE improvement versus depth marginalization: `528.95x`.

Three deterministic publication SVGs are hash-bound beside the JSON:

- depth-layer convergence;
- adaptive fallback by camera program/speed;
- crossing-scene temporal error.

They were rendered locally and visually corrected for legend placement,
nonoverlapping ticks, portable colored line segments, titles, and labels.

## Claim boundary

This closes a meaningful Paper-B gap: synthetic ordered-transfer exactness,
depth-layer convergence, crossing superiority, and the necessity of the gauge
Jacobian. It does not prove:

- native runtime speed;
- native allocator or peak-memory scaling;
- end-to-end kinetic compiler acceptance;
- public-data trained quality.

Those exclusions are encoded in the artifact and independently verified.

## Paper/project synchronization

Updated:

- `EXPERIMENTS.md`;
- `TODO/README.md`;
- `PROJECT_INDEX.md`;
- `research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md`.

Parallel work added a concise Paper-B manuscript/BibTeX and two result-free
concept figures, and began a fail-closed Paper-B evidence bundle plus direct
integration of this G0/G3 result. Paper A separately gained a strict concise
venue-package verifier. Its remaining numerical queue is still the frozen
same-world sweep, a clean variable-camera rerun, and seven schema-v2 public
contexts. The existing variable-camera curve is numerically strong but invalid
for publication because it was generated from dirty source.

