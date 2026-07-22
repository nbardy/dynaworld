# World Tubes submission completion audit

## Context

The sole project goal is to finish **World Tubes in Gauged Camera Space** as a
renderer/compiler paper, implemented by projective STAR UVT. The workstation
has suffered severe unified-memory pressure during the fixed-512 matrix. This
audit therefore used only repository text and existing JSON artifacts. It did
not import Torch, load a dataset, run a verifier, or execute a renderer.

The conceptual hierarchy is fixed:

```text
gauged camera-ray / depth-fiber mathematics = paper method and correctness
World Tubes                            = compiled sensor-time traces
projective STAR UVT                    = primary implementation
per-frame STAR replay                  = causal systems baseline
WorldFoam                              = retained-depth challenger
dynamic 3DGS                           = conventional baseline
```

Calling projective STAR UVT the primary implementation does not demote the
gauge mathematics to an optimization. Large-motion overlap/order crossings are
precisely where the lifted depth/order field and visibility-stratified gauge
domains change the result relative to a raw overlapping interval.

## Evidence standard

Statuses below mean:

- **proved**: a current accepted artifact directly covers the requirement;
- **implemented, unverified**: source/config exists but the latest change was
  not executed after the memory incident;
- **partial**: some required rows exist and missing rows are enumerated;
- **missing**: no accepted artifact currently proves the requirement.

File existence, an old smoke, or a plausible command is not treated as proof
of publication-scale behavior.

## Requirement-by-requirement result

| Requirement | Status | Authoritative evidence | Remaining work |
|---|---|---|---|
| 1. Land the runner cleanly | **implemented, unverified after containment edits** | Main commits `c0b4438`, `0ad9300`; STAR commits `e2af20d`, `64a4e0a`; prior CPU gates are recorded in the process-isolation note | On an approved machine, verify the clean current commits through the actual call graph. Do not use the pre-containment test result to certify `0ad9300/64a4e0a`. |
| 2. Evidence schema | **proved for accepted rows** | Three progressive `run_summary.json` files contain schema-v1 quality, cost, timing, memory, checkpoint, and diagnostic fields; matrix aggregation accepted all nine lane rows | Confirm the same schema on every new control/breadth row. |
| 3. Matrix orchestration/aggregation | **proved at accepted-existing scale** | `run_unified_paper_matrix.py`; accepted JSON/CSV/Markdown/LaTeX/SVG bundle; partial summary explicitly lists absent runs | Full-matrix execution remains incomplete. |
| 4a. Coffee progressive seeds 17/29/43 | **proved** | Three clean-source summaries under `outputs/benchmarks/2026-07-22_world_tubes_submission_matrix_clean_v1/`; all use repo `fbf95dc...`, STAR `0781fbb...` | None for this row. |
| 4b. Pixel-matched fixed seeds 17/29/43 | **missing** | `existing_evidence_summary.json` lists all three keys as missing | Run elsewhere or after an explicitly approved bounded profile proves safety. |
| 4c. Global shuffle seed 17 | **missing** | Same partial summary lists the key as missing | Same execution boundary. |
| 4d. Deterministic STAR audit | **missing result; frozen workload now present** | Full public matrix labels a 64-wide all-300-frame `deterministic_quality` row separately | Produce a correctness/timing artifact; never substitute it for throughput. |
| 5. Per-frame replay vs compiled atlas, `F=4..128` | **proved** | `2026-07-22_world_tubes_same_representation_scaling_f4_128_cap256/summary.json`; consumed by the verified theorem table | Preserve this as the central causal result; do not replace it with cross-representation quality. |
| 6. Theorem table / orbit claim | **proved at narrowed scope** | Theorem table status `complete`, 11 rows, all sources verified, `full_orbit_multigauge_claim=false` | Do not claim complete `360/720` transitions. |
| 7a. Additional Coffee triplets | **configured, results missing** | Two protocols, two manifest rows, and a six-run matrix exist | Six accepted three-lane rows. |
| 7b. Two additional Neural3D scenes | **configured, results missing** | `cook_spinach` and `cut_roasted_beef` protocols/manifests and six-run matrix exist | Acquire/validate data on execution host and produce six accepted rows. |
| 7c. Controlled D-NeRF | **configured, result missing** | Bouncing-balls posed-frame manifest/protocol and one-run matrix exist; one-frame chart fallback is explicit | Produce one separately labelled result without making a sublinear camera-chart claim. |
| 8. Paper packaging | **partial** | Accepted rows are in `BASELINES.md`; generated theorem/scaling/public tables exist; Markdown and LaTeX manuscript exist; protocols/manifests are checked in | Missing results must populate the manuscript; final LaTeX/PDF compilation and figure audit are not proved. |

## Unified public workload

`src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc`
now freezes the whole public workload as 21 independent run keys:

```text
3  Coffee progressive
3  Coffee pixel-matched fixed
1  Coffee global-shuffle
6  alternate Coffee camera triplets
6  two additional Neural3D scenes
1  controlled D-NeRF sequence
1  deterministic correctness/timing audit
--
21 total
```

Three are currently accepted, so 18 results remain. Each result contains three
representation lanes, but the deterministic row is a kernel-policy audit and
must stay separate from the 512-wide quality aggregate.

## Current claims that are defensible

1. Gauge value and gradient invariance hold in the certified fixtures.
2. A raw overlapping projective interval can fail at a depth-order crossing;
   visibility stratification repairs the tested case.
3. Finite-exposure, rolling-shutter, fallback, forward, and gradient parity
   pass their bounded synthetic fixtures.
4. For the accepted identical-representation `F=4..128` experiment, compiled
   projective STAR UVT has fixed payload growth and lower final compile,
   forward, and backward cost than per-frame replay.
5. On one Coffee Martini split and the progressive protocol, World Tubes has
   the best mean PSNR/L1; dynamic 3DGS has the best SSIM/LPIPS. Absolute quality
   is low, so this is evidence, not a broad superiority claim.

## Claims that remain indefensible

- submission-ready public-data breadth;
- superiority across scenes or camera splits;
- safety of another publication-scale run on the incident workstation;
- full `360/720` multi-gauge orbit support;
- native-resolution readiness;
- deterministic throughput parity;
- full external SOTA parity;
- a claim that host-resident targets alone bound total unified memory.

## Memory branch and falsification

**Current belief:** the killed run was caused by the combined eager data,
model/optimizer, renderer scratch, and allocator-residency envelope on unified
memory, not by a single tensor alone.

**Evidence:** the accepted rows reported multi-gigabyte driver residency, the
fixed run triggered compression/swap/kernel pressure, and the containment code
only removes known target/ray/evaluation multipliers.

**Could be wrong if:** a single-lane micro-profile on the current commits shows
a bounded peak well below both physical-memory and swap-pressure thresholds,
or reveals one specific leak independent of workload size.

**Only acceptable local falsifier after explicit approval:** one lane, tiny
dimensions, hard timeout, no parallel children, no W&B media, and external
resident-memory observation. Success would authorize a slightly larger
profile, not a paper row. Failure stops local escalation.

## Decision

The paper is **not complete**. The implementation, schema, causal scaling
experiment, theorem table, and partial Coffee row are real. The exact remaining
scientific dependency is 18 accepted workload results plus final manuscript
integration. The preferred execution path is a sufficiently provisioned
separate machine. No cleanup deletion or new research direction should preempt
that chain.
