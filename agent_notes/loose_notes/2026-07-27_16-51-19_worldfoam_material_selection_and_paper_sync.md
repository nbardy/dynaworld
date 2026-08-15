# WorldFoam material selection and paper synchronization

Date: 2026-07-27 KST

## Request

The user asked to pursue the full follow-up from the camera-program/compiler
synthesis: preserve interesting information, compare it with the current
notes and STAR/World Tubes paper, update shaders/papers, and run the justified
training gates.

## Initial audit

The repository was already ahead of the previous synthesis note:

- native `mu4 + SPD(4)` capacity/compiler work existed;
- a trainable static-camera full-SPD producer and paper-runner axis existed;
- four 4-frame/2-step Coffee Martini smokes already existed;
- one parameterized M0--M5 CPU/Metal material evaluator existed;
- the World Tubes and WorldFoam manuscripts already contained part of the new
  representation framing.

The literal instruction to “fork all shaders” was corrected to six modes
behind one shared material ABI. Cloning the roughly 94-kernel renderer would
duplicate topology, scan, and adjoint implementations and invalidate the
fairness contract. The accepted structure is one forward entrypoint, one VJP
entrypoint, and a material-mode selector.

The host was not safe for publication-scale MPS:

- 24 GiB physical memory;
- fixed/progressive paper-row estimates above the 60% safety limit;
- prior unified-memory compression/swap incident;
- unrelated high CPU load;
- dirty parent and nested STAR revisions.

Therefore no full 512-wide publication row was launched. Existing two-step
MPS smokes were not repeated because their source had not yet changed and they
already establish dispatch/memory mechanics only.

## Correctness baseline

The combined existing foundation suite passed:

```text
125 passed, 2 skipped in 114.30s
```

It covered SPD(4), trainable SPD(4), M0--M5 material transfer, STAR projective
producer/visibility, cell-path transfer, and unified paper-ablation tests.

## Material identifiability correction

The first proposed fit reused complete constant-color segments. That is
insufficient: for

```text
tau = L integral sigma(xi) dxi
beta = exp(-tau)
m = (1-beta)c
```

the observation depends only on total optical depth. Density shape is
unidentifiable.

The replacement fixture shares one global material field across partial
chords and exactly restricts each M0--M5 law into the existing segment
evaluator. Direct Bernstein modes use exact subdivision; log modes substitute
the affine chord coordinate into their negative-log polynomial.

The initial Adam-only short gate correctly separated positive P2, but failed
the predeclared log-P2 `100x` threshold because M5 coefficients span a larger
scale:

```text
M0 log-target loss: 8.12e-3
M1 log-target loss: 8.10e-3
M5 after 500 Adam steps at lr 0.04: 1.25e-3
```

This was optimization conditioning, not representational failure. Applying the
same deterministic strong-Wolfe L-BFGS polish to every mode recovered the
exact log-P2 target:

```text
M5 loss: 4.58e-15
recovered controls: [12.0000087, -12.0000087, 3.0000020]
```

The gate was then tightened again:

- twelve training chords;
- eight disjoint held-out chords;
- independent target integration rather than the fitted evaluator:
  closed form for positive P2 and composite Simpson for convex log-P2;
- all six M0--M5 modes;
- seeds 17/29/43;
- canonical held-out `(beta,m)` loss;
- source hashes;
- static serialized material sizes;
- an independent artifact verifier.

## Three-seed result

Artifact:

```text
artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json
```

Verifier:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run python -m \
  research_experiments.world_foam_lane2.verify_finite_element_material_fit
```

Result:

```text
verified=true
rows=36
seeds=3
```

Median held-out losses:

```text
positive-P2 target:
  M0 6.8437e-3
  M1 6.7683e-3
  M2 6.1889e-3
  M3 5.2579e-17
  M4 6.3151e-3
  M5 8.7967e-5

convex-log-P2 target:
  M0 8.1040e-3
  M1 8.0999e-3
  M2 8.0315e-3
  M3 1.3282e-3
  M4 8.0498e-3
  M5 6.1884e-15
```

M3 and M5 each use six float32 material scalars (24 bytes). Each wins its own
generating family by more than 100x. The correct result is complementary
bases, not a universal winner:

```text
winner=null
eligible_for_native_4d_integration=false
```

The next gate is adaptive per-cell M3/M5 selection or real held-out material
evidence. Native-4D integration was intentionally not started.

## Derivative and branch coverage

The fixed-segment tests now include:

- independent central finite differences for explicit VJPs in every mode;
- a nonzero tiny-optical-depth record;
- opt-in Metal tiny-tau parity;
- opt-in shader-produced invalid-row rejection.

CPU result:

```text
34 passed, 3 Metal-only skipped
max finite-difference VJP normalized error: 6.86e-10
max integral error: 5.96e-15
small_tau_series branch count: 1
```

No MPS work was launched by the root task or its material-test subagent. During
the session, a concurrent writer updated the accepted Metal artifact to the
expanded 12-record fixture. The artifact matches the current shader hash,
counts one tiny-tau row, and reports the unchanged `7.51e-8` forward /
`5.96e-8` VJP normalized errors. Treat that saved artifact as confirmed
workspace evidence, but do not attribute its MPS launch to this task.

## Paper and project synchronization

World Tubes manuscript changes:

- affine/local qualification of Gaussian closure;
- explicit world/compiler/evaluator/adjoint decomposition;
- SPD(4) conditional-Gaussian proposition with a non-novelty disclaimer;
- Native 4DGS context;
- honest 18-versus-14 scalar capacity disclosure;
- explicit compiled-adjoint chain;
- camera pushforward diagnostics and affine/projective closure-death curves;
- prospective frozen-world replay-versus-compiled protocol.

WorldFoam manuscript/plan changes:

- synchronized “Gauge-Invariant Ordered Ray Transfer” title;
- numerical microkernel demoted to validation infrastructure;
- fixed-segment reference/shader/parity steps marked complete without
  overclaiming broader gates;
- finite-element prior art added;
- material-value gate now recorded with the no-universal-winner result;
- native-4D work remains gated.

`TODO/README.md`, `PROJECT_INDEX.md`, `EXPERIMENTS.md`,
`research_notes/README.md`, and `agent_notes/key_learnings.md` were updated.
The durable interpretation is:

```text
research_notes/worldfoam_material_basis_selection_gate.md
```

The project-status audit also corrected the Coffee Martini queue: progressive
seeds 17/29/43 are accepted; fixed seeds 17/29/43 and global-shuffle seed 17
remain.

## Concurrent duplicate fit harness

While this session was active, a second untracked positive-P2-only pilot
appeared:

```text
research_experiments/world_foam_lane2/fit_finite_element_materials.py
research_experiments/world_foam_lane2/test_material_value_fit.py
artifacts/foundation_gates/worldfoam_material_value_fit_cpu.json
```

It is useful corroborating evidence for M3 on a positive-P2 target, but it has
no disjoint held-out split and no convex-log-P2 countertarget. It was preserved
because it may belong to another concurrent user/task. The canonical
two-family selection result is the dated 36-row artifact and verifier above.

## Beer--Lambert status

A concurrent uncommitted STAR Beer--Lambert patch already existed in the
nested fast-mac tree. Audit found that its core alpha/VJP formulas were
structurally sound, but runner-derived configs initially dropped `alpha_mode`
and projective trace support/reference paths still assumed peak alpha. The
completed patch now restricts support to
`static_view + full_spd4 + fast_exploration/direct_atomic+index_add`,
propagates semantics through configs/reports/W&B identity, uses the exact
Beer cutoff in the tile-load proxy, and fails closed for projective atlas
paths. A deterministic full-SPD CPU SGD step has finite nonzero optical-
thickness/color gradients and decreases image loss. The opt-in Metal behavior
test was not run by this task.

## Remaining order

1. In an approved quiet window, run only the opt-in Beer and material Metal behavior
   gates; do not run the paper matrix on this host.
2. Add adaptive/real-data material selection.
3. Only a selected material law/selector enters the RGB-only owner-run
   integration, then native-4D compiler work.
