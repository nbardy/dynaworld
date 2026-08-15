# World Tubes paper closure audit

Date: 2026-07-28 KST

## Request

Review the July 27 SPD(4), Beer--Lambert, retained-fiber, and WorldFoam
material-selection work against the actual World Tubes paper goal. Identify
what remains, what should not become a blocker, and avoid any device work while
the incident host is under severe memory pressure.

## Host safety

No training, Torch, or Metal process was launched. At audit time the 24 GiB
host had only tens of MiB of free VM pages, more than 8 GiB in the compressor,
and a multi-GiB Node process. Publication MPS work remains unauthorized on
this host.

## Findings

- The gauged camera-ray formulation remains the method, not a cosmetic STAR
  optimization. The manuscript keeps the ray-fiber pushforward, conditional
  depth, and visibility-stratified repair for large-motion order crossings as
  central contributions.
- The accepted `F={4,8,16,32,64,128}` same-representation result remains the
  strongest causal systems result.
- The strict-SPD(4), projected Beer--Lambert, and retained-fiber work is valid
  bounded integration evidence. It does not yet support exact nonlinear or
  projective retained-depth transfer, adaptive quadrature, or selective
  dense-scene fallback.
- The WorldFoam M3/M5 result is a second-paper material-chart result. Its
  complementary-basis conclusion does not block World Tubes and correctly
  keeps native-4D WorldFoam integration gated.
- The manuscript had accidentally promoted the retained-fiber extension into
  an unfinished main contribution and manually inserted three retained rows
  into the generated theorem table. The retained path is now labelled a
  bounded extension, and its metrics remain in the dedicated SPD(4) ablation.
- The actual submission blockers are evidence closure: frozen identical-world
  replay versus compiled evaluation, fixed pixel-matched seeds 17/29/43,
  global-shuffle seed 17, public camera/scene breadth, and final scholarly
  packaging. The current Pandoc TeX is not yet a venue manuscript and contains
  a "References To Cite" list rather than integrated citations.

## Safe evidence packaging completed

Added
`research_experiments/spd4_world_tubes/summarize_bounded_training.py` and its
behavioral verifier test. The standard-library-only tool packages the three
accepted bounded reports, checks their exact protocol/cost/overflow/media
contracts, and records SHA-256 hashes. It generated:

```text
artifacts/spd4_bounded_16f_40step/summary.json
artifacts/spd4_bounded_16f_40step/summary.md
```

The aggregate verifies and records the provenance boundary honestly:
the runs came from an uncommitted working tree, so they are bounded engineering
evidence rather than clean-source publication evidence.

## Frozen identical-world executor implemented

The central prospective causal protocol is now executable without adding a
second trainer:

```text
research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py
```

The existing STAR multicamera trainer now supports an opt-in post-training
comparison that:

1. snapshots and SHA-256 hashes the final learned World Tubes state;
2. reprojects, bins, and renders one STAR frame at a time on the heldout camera;
3. projects the identical world once and lowers one event-stratified projective
   interval atlas;
4. evaluates the same heldout targets and robust-L1 loss;
5. compares images, losses, world-parameter VJPs, payload, timing, interval
   complexity, and fallback;
6. preserves negative results with explicit acceptance checks.

The unified runner threads this mode through command generation, dry-run
manifests, stale-report identity, report validation, W&B IDs/metrics, and final
run summaries. The lane-isolated executor avoids retraining WorldFoam and
dynamic 3DGS for this same-representation experiment.

Only syntax, document generation, and static diff checks were run in this
session. The behavior tests and MPS executor remain unrun because the host
became unsafe again: an unrelated Node process grew to roughly 7 GiB RSS and
used several CPU cores. No device allocation was attempted.

A follow-up static audit tightened checkpoint provenance before execution.
The verifier now recomputes SHA-256 from the checkpoint in bounded 1 MiB
chunks, requires the recorded byte count to match the file, and rejects a
mutated checkpoint. Checkpoint creation uses the same bounded streaming hash
rather than materializing the file as another byte buffer.

## Stop boundary

Do not spend World Tubes submission time on adaptive M3/M5 selection, native-4D
WorldFoam, browser trainers, V-JEPA/world-token work, new shader families,
full-orbit naming/theory, native resolution, or retained-projective completion.
Preserve their artifacts, but route new execution only into the frozen paper
controls and breadth matrix on an adequate clean host.

## Expansion Pass 2: Evidence semantics and bounded execution

### Backtrack: the 21-row matrix is not compiler evidence

Status:
    Prior wording weakened and corrected.

Observed fact:
    The World Tubes lane in the unified matrix uses selected-time STAR
    rendering. It does not evaluate the compiled projective interval atlas.

Replacement model:

```text
21-row unified matrix
    -> representation quality, cost, stored-state context

frozen identical-world route
    -> causal replay-versus-compiled public evidence
```

Three of 21 matrix protocols are accepted, producing nine representation rows.
Eighteen matrix protocols remain missing. The frozen causal run is outside that
count and is also missing, so the runtime debt is 19 jobs.

### Frozen route hardening

The source-only implementation was tightened after a static red-team:

- targets remain host-resident and move to MPS only in bounded chunks;
- one full atlas is compiled, then compact differentiable frame slices are
  evaluated without retaining all output frames;
- replay and compiled robust-L1 losses use the same global normalization;
- image parity is recomputed chunkwise outside timed forward measurements;
- missing gradients no longer become a vacuous zero-versus-zero pass;
- world-state digests bind the checkpoint and both routes;
- checkpoint and report hashes are streamed rather than read as whole files;
- reused child artifacts require matching command, protocol hash, source
  start/finish, and report hash;
- the shared atlas-slice helper remaps trace IDs, forbids empty active
  intervals, and has a forward/VJP equivalence test.

Current belief:
    The code spine is close to runtime-ready.

Confidence:
    Medium, because only `py_compile` and static diff checks have run.

Could be wrong if:
    The first CPU behavior gate finds schema drift, or the approved MPS run
    exposes native lifetime/allocator behavior not visible in source review.

### Falsification sequence

1. On a quiet host, run the focused CPU tests without importing the production
   workload.
2. Run a tiny frozen comparison and require nonzero matching VJPs plus bounded
   residency.
3. Run all 300 frames from one clean checkpoint.
4. Only after acceptance, run the public same-checkpoint frame-count sweep.
5. Then execute the 18 missing selected-time matrix protocols.

No MPS, Torch import, or pytest was started during this pass because the
incident workstation remained outside the approved resource envelope.

## Expansion Pass 3: July 28 meta-review reconciliation

The full 2,463-line `research_notes/meta_review_jul_28th.md` was read against
the current paper, runner, STAR submodule, and retained-transfer work. The
review sharpens the paper but does not justify another architecture cycle.
The longer topic-by-topic reconciliation is in
`2026-07-28_16-50-53_meta_review_project_integration.md`.

### Status matrix

| Status | Decision |
| --- | --- |
| Already implemented | World/compiler/evaluator/adjoint separation; standard SPD(4) novelty boundary; gauged ray pushforward and conditional depth; bounded projective interval charts; support/order/denominator events; visibility-stratified repair; Metal forward and fixed-topology VJP; exposure/rolling-shutter fixtures; bounded synthetic `F=4..128` scaling. |
| Partial, describe narrowly | Continuous camera compilation is bounded rather than generic `SE(3)` splines; the adjoint holds topology fixed; retained transfer is static affine; structural refresh lacks trust-region/local-rebuild evidence. |
| Submission P0 | Clean behavior gates; all-frame frozen identical-world public result; public same-checkpoint frame sweep; 18 missing selected-time rows; authoritative aggregate; tables/figures/demo/citations/venue TeX/rendered PDF. |
| Post-paper | Event-boundary estimator; structural trust regions; Type-II composite-transfer patches; full `360/720` transitions; projective retained fiber; adaptive quadrature; adaptive M3/M5; native-4D WorldFoam; BSVA/pentatope work. |
| Reject | Rename the paper to generic STAR; claim SPD(4) as novel; replace World Tubes with WorldFoam; import the review's unimplemented generic transfer/boundary-adjoint abstract. |

### Ordered Ray Transfer boundary

Holonomy remains geometric inspiration for closed-loop transport. An open
camera ray uses ordered parallel transport / a path-ordered optical-transfer
product, so the paper extension remains **World Tubes + Ordered Ray Transfer**.
Keep the noncommutation identity
`alpha_i alpha_j (c_i - c_j)`, the proof that mean depth plus total opacity is
insufficient for thick colored overlap, and conditional-depth order
certification. The implemented evidence is bounded:

- the hybrid currently certifies strict depth-band separation, not the general
  color-commutation residual;
- the 16-atom static-affine fixture routes `10/64` tiles to retained transfer
  and matches recorded all-retained heldout metrics;
- the 199-atom fixture routes `64/64` and is a negative selectivity control;
- native VJP correctness is a separate gate, not direct full-image/VJP parity
  for the hybrid table.

No ordered-transfer experiment is submission-critical. WorldFoam remains the
broader retained-depth/cellular representation rather than a renamed World
Tubes backend.

### Falsification criteria adopted now

The public causal run must bind one checkpoint, camera program, targets,
evaluator contract, parameter names, native extension, clean source, and
nonzero world VJPs. It must report image/loss/per-parameter gradient parity,
fallback counts with an auditable denominator, and route timing with consistent
total/per-frame arithmetic. Requested frame density must change evaluation
sampling without silently changing the camera-program descriptor.

Tensor-only payload excludes topology, allocator overhead, and transient
working memory; it is not a storage claim. Topology-inclusive bytes,
route-scoped peak memory, structural refresh cost, generic boundary
derivatives, and full-orbit behavior remain unclaimed until measured.

### Host-safety consequence

The live paper gate is centralized and checks memory, swap, disk, and
CPU-normalized load before MPS execution, with a fresh snapshot before each
expensive child lane. This closes the direct unified-runner path that could
otherwise reproduce the incident after only static acknowledgements. No
runtime gate was exercised in this pass.
