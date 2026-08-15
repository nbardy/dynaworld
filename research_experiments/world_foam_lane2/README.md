# World Foam Lane 2 Research Gates

Isolated references and Metal/MPS smokes for the World Foam / beam traversal
lane. The new op is suffixed and does not change trainer/default routing; the
memory-light completion work is registered in the project TODO index.

## Native rebuild and attestation status (2026-08-15)

The correct extension is
`third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0`.
Current source contains exactly 133 schemas and 133 matching implementations;
the retained CPython 3.11 binary contains the older 103-schema inventory. A
forced clean rebuild can therefore expose the full source ABI, but it has not
been run on the resource-constrained host. The source-only verifier passes and
the import verifier fails closed on the exact 30 missing registrations, stale
mtime, and absent build receipt.

The rebuild and receipt commands are recorded in the variant README. The
receipt binds the exact CPython 3.11 Darwin binary, architecture, Torch and
compiler identities, native/runtime/Python source hashes, and every dispatcher
name/signature. Passing that receipt will prove build/registration integrity,
not Metal parity or measured memory fit.

## G6 clean-host execution bundle (2026-08-15)

The canonical G6 launcher is
`run_worldfoam_g6_clean_host_bundle.py`; its exact contract and commands are in
[`WORLDFOAM_G6_CLEAN_HOST_RUNBOOK.md`](WORLDFOAM_G6_CLEAN_HOST_RUNBOOK.md).
The default invocation is an allocation-free source/config plan: no
subprocess, Torch/native import, build, MPS dispatch, output, or evidence row.
`--execute` is the real ablation. It host-guards and force-rebuilds the exact
133-schema fused-slab extension with the selected `.venv/bin/python`, writes
and re-verifies its build receipt, requires both native G6 ABI seals, then runs
12 primary rows, 9 controls, and 3 auxiliary restart processes. It writes a
bundle receipt only after the independent 21-row verifier accepts.

The launcher and producer deliberately preserve the virtual-environment
Python path instead of resolving its symlink to the Homebrew base executable;
the latter loses the venv Torch installation. The producer also installs the
bound fused-slab variant path before importing native ops. These are runtime
deployment requirements, not paper measurements.

This remains `0/21` until `--execute` completes on a safe Mac. A B200 cannot
validate the current Metal/MPS memory claim without a separately implemented
CUDA backend and acceptance contract.

### Staged-sparse D2H receipt semantics

The staged sparse geometry bridge already has a complete synchronous copy
contract; it does not need a fabricated second epoch. It fences the native
producer once before reading `grad_node_physical_length_f32`. Each bounded row
is then copied with `.to(device="cpu", dtype=torch.float64)` without
`non_blocking=True` and immediately read by CPU finite/reduction operations.
The returned CPU tensor therefore proves that row's D2H copy completed before
the next statement. `native_length_bar_row_copy_count` records those copies;
the consumed producer-fence receipt binds the source lifetime across them.

`geometry_d2h_completion_fence_count` names the separate fused-union-v2 device
scatter/D2H settlement and correctly remains zero for staged sparse. A new
post-copy epoch becomes necessary only if the bridge later switches to
nonblocking copies or another asynchronous CPU-consumption path.

## Memory-light shared adjoint CPU/source path (2026-08-03)

The CPU references now separate three fixed-topology contracts:

1. exact chunked owner-word replay with a constant-state two-pass VJP; and
2. exact sparse track-boundary incidence reduction with one Mobius/boundary
   VJP per referenced incidence; and
3. an affine-log total-transfer atlas whose expensive word scan/VJP runs at
   `J` compiler nodes while requested samples only evaluate and reduce a small
   temporal basis.

The relevant files are `compiled_transfer_adjoint.py`,
`exact_sparse_incidence_oracle.py`, `transfer_lie_chart.py`,
`compiled_lie_world_adjoint.py`, `compact_lie_schedule.py`,
`native_track_adapter.py`, `material_parameterization.py`,
`material_training_step.py`,
`piecewise_topology_staged_adjoint.py`,
`native_piecewise_topology_adapter.py`, `host_memory_contract.py`, and
`compiled_route_cost_gate.py`. The scientist review's one strong formulation
newly derived in this project, the translated optical-depth measure, is checked
by
`optical_depth_translated_measure_oracle.py`. It is an integrated proof/tangent
object, not runtime state; the canonical runtime remains the compact
`(beta,m)` affine quotient plus owner word. The external literature-novelty
claim remains open. The geometry frontends are
`sparse_power_word_compiler.py` for fixed 4D sites and
`kinetic_power_word_compiler.py` for direct affine kinetic 3D sites, with
corresponding `test_*.py` suites. Continuous kinetic proof references are
`kinetic_owner_chart_compiler.py`, `kinetic_owner_chart_oracle.py`, and
`kinetic_active_owner_chart_compiler.py`. The transfer/reverse seam is
`kinetic_chart_transfer_bridge.py`, `kinetic_multichart_transfer_program.py`,
`kinetic_continuous_transfer_acceptance.py`,
`kinetic_stable_stratum_vjp.py`, and
`kinetic_multichart_stable_stratum_vjp.py`. The frame-independent native seam
is specified by `kinetic_native_topology_lowering.py`,
`kinetic_native_precompiled_length_oracle.py`, and
`kinetic_native_precompiled_length_adapter.py`. Bounded multi-row packing and
the fake-native warm lifecycle live in `kinetic_native_equal_rank_lowering.py`
and `kinetic_native_equal_rank_runtime_adapter.py`; the separate fenced
length-bar-to-world reduction is
`kinetic_native_equal_rank_geometry_reduction.py`. The bounded frame-free CPU
program/sampler store is `kinetic_compiled_cpu_artifact_store.py`; dense
observations can be replayed without retaining frames, targets, or rays through
`../../src/train/paper_kinetic_replayable_observations.py`. The source-only
fixed-site material coordinator in
`../../src/train/paper_kinetic_fixed_site_material_step.py` now composes these
two pieces directly; the legacy coordinator remains separate. The narrow
exact event-free update certificate is `kinetic_geometry_trust_region.py`, while the
restricted separated-singleton multichart reference is
`kinetic_simple_root_reisolation.py`. The
step-scoped block-major material bridge and its direct-autograd/invariance gate
are `kinetic_ragged_paper_step_cpu_fake_native.py` and
`test_kinetic_ragged_paper_step_cpu_fake_native.py`. The source
row-ragged sample contract is covered by
`test_kinetic_ragged_lie_sample_source_contract.py`. Paper-observation joining,
heterogeneous-block union-local bar assembly, and global-denominator
coordination live in `../../src/train/paper_kinetic_ragged_sample_plan.py`,
`../../src/train/paper_kinetic_union_local_bar_assembly.py`, and
`../../src/train/paper_ragged_material_bar_coordinator.py`, again with
behavioral test companions.

The reference uses sparse active track-boundary incidences and caller-supplied
sparse power-cell pairs; it does not allocate dense track-by-boundary
coefficients or construct all-pairs boundaries. The staged wrapper reduces
arbitrary `K`-frame target blocks into one fixed node-cotangent accumulator,
then runs one world/boundary finalize. A compact `B_p` seam reconstructs words
from flat CSR, derives active faces from the same gathered 4D sites, lowers the
face VJP to site/weight bars once, and scatters site/density/color gradients.
Version signatures and prepared-token identity reject stale world, topology,
or cross-block results. Caller-owned global site bars now accumulate exact
spatial blocks without reallocating. The procedural memory fixture exposes a
sealed direct selected-pixel source, eliminating the source-audited `5.41 TiB`
full-frame decode amplification its fallback would have caused. Real compressed
or public targets still need a tiled or mmap-backed independently decodable
backend; selected MP4 frame seeking/full-frame decode does not close the v3
paper-memory contract. Rectangular multi-view observations factor exactly into
`(view,pixel) x time`. A backpressured source adapter now
consumes those blocks through the native token lifecycle, and an owner-only P0
material-training session retains only lightweight compact topology, compact
spec schedules, owner bindings, and a policy-bounded entry/byte LRU of sealed
native topology tokens across steps. The cache may retain or evict any block
under its caps; one active token is separately preflight-bounded. It retains zero compiled CPU
atlases per spatial block and performs zero per-step CPU atlas compiles. Its
material-only reverse skips geometry bars, and its hot sample blocks stage
targets only after one bounded exact fixed-camera reference-row check. That
means live scratch is `max_b`-bounded, but persistent topology/schedule/
binding/token bytes must still be summed over blocks; they are not covered by
a max-only memory formula. A CPU adapter now groups arbitrary paper samples by
view without Cartesian padding, and a source-only native reducer consumes
row-selected track-chart samples without a global time refinement. A bounded
actual-rank lowerer, union-local heterogeneous-block assembler, and outer
multi-view material-bar coordinator now close the CPU/source coverage and
global-denominator lifecycle. The block-major CPU/fake-native bridge now keeps
each spatial bundle live across all `K` chunks, accumulates their residuals
into bounded node cotangents, runs one material-only word VJP per active native
block, union-scatters once, and releases bundles sequentially. It allocates no
`[J,W]` geometry bar. `K=1/4` matches a direct-autograd oracle, `F=5/41` leaves
compiled-word invocations and retained runtime bytes invariant, and a
two-bundle fixture proves a max-over-bundles rather than sum-over-bundles live
node-state peak. The session itself remains a hand-built
fixed-topology fixture. A bounded zero-velocity point-cloud initializer, exact
static-camera program factory, CPU-only fixed-site material/manual-SGD state,
raw-only checkpoint, and caller-owned one-step authorization coordinator now
exist as runtime-unverified source. Production-scale compilation, credible
dynamic initialization, forward-only evaluation, extension runtime, and a
distinct unified paper-runner lane remain open.

`verify_worldfoam_memory_scaling_acceptance.py` and its checked-in
`worldfoam_memory_scaling_acceptance_v3.json` contract define the measured
promotion gate for the fixed-site material-only end state. Fresh-process
`F=8/64/300` rows mean denser requested samples over one fixed physical interval
and fixed compiled world, not longer represented duration. Structural
signatures and word-VJP work must stay fixed; only streamed sample work and
small identity/time metadata may grow. Schema v3 seals the AST-resolved
transitive local-Python import closure plus declared native sources. Driver
capability schema 3 claims only MPS and the direct selected-pixel contract;
runtime measurements come from receipts. The producer is written to apply and
bind a per-process MPS allocator limit with effective bound `<=2 GiB`. A
separate parent polls process-group RSS at a configured 0.25-second interval and
terminates after observing more than 4 GiB, so that watchdog is sampled rather
than an exact allocator-hard RSS cap. The public MPS current/driver sampler's
configured and reported interval must equal 5.0 ms; its maxima remain lower
bounds. Raw MPS-limit and per-trial watchdog receipts are hash-bound to the
artifact and child execution evidence. The 8-GiB host-availability threshold is
incident launch headroom, not a 32-GB representation requirement. Attestation
covers exactly node forward, loss-only sample accumulation, and material-only
word VJP, recording observable execution width, maximum threads, and static
threadgroup bytes. That query does not prove kernel execution; Metal
private/register/spill bytes remain unobservable and are neither estimated nor
certified. The opt-in producer is bound to the Metal fused-slab ABI and has a
checked-in real coordinator driver/config:
`worldfoam_memory_scaling_mps_trial_driver.py` and
`worldfoam_memory_scaling_mps_trial_v1.json`. The driver binds observed native
calls and direct selected-pixel receipts to the sealed coordinator execution.
No row has been executed, and the changed verifier, producer, driver, native
attestation, tests, and extension remain unrun/unbuilt. A CUDA result requires a
separately bound CUDA native port/producer rather than relabelling this MPS
contract.

Public target streaming now also has a source-only cache seam in
`src/train/powerfoam_training_data.py`. `MappedRgb8PowerFoamTargetSource`
stores each camera as raw uint8 `[H,W,F,3]`, matching block-major
`(view,pixel) x time` replay. Each selected-pixel call preflights bounded
logical scratch, maps one payload transiently, preserves arbitrary order and
duplicates, copies a standalone CPU float32 `[N,3]`, and closes the mapping
before returning. The strict manifest is content-hashed through each opened
payload at construction and requires caller-supplied per-payload mapping and
total verification-I/O caps. Its
receipt reports per-read and cumulative requested-page bounds, but those are
not OS residency/readahead measurements; the cold full-payload verification
scan and host/system memory pressure must be measured separately from process
RSS in the public companion gate.
The offline converter's disk preflight counts two payload-sized temporary
files per active view: the completed/current payload overlaps either the raw
frame-major spool or the independent cache-verification spool. Its global peak
counts all completed payloads plus one such active spool.
Normal full-frame evaluation delegates to the existing path/MP4 source. This
is a standalone source primitive, not a wired unified/per-frame-trainer lane;
that trainer still requests full frames. The class, receipt propagation, and
static tests have not run, no cache has been generated, and the public dataset
converter, populated binding, companion gate, and fixed-site trainer integration
remain open. `worldfoam_target_dataset_binding.py` now provides the strict
source-only `target_dataset_binding/v1` validator and exact cache-file rehash
contract. It checks equality of declared raw/cache decoded hashes but does not
decode either representation or recompute that equality; the bounded converter
and public companion must do so. It is also unrun and is not public-data
evidence by itself.

It deliberately fails any full-geometry certification claim. Full native
kinetic-geometry scaling, including bounded `[J,W]` length bars and streamed
request-local site/trajectory/weight/ray reductions, remains a separate
required promotion gate.

Structural reuse is deliberately narrower than material reuse. The current
production policy is to reuse a sealed compiled artifact across material-only
updates and to run a fresh structural compile/recertification after every site,
weight-trajectory, or camera-ray update. `kinetic_simple_root_reisolation.py`
certifies a restricted whole-registry homotopy; it is not an output-sensitive
program repairer and does not rebuild payloads, ranks, or native dispatch.

The next geometry integration must extend the existing native step executor,
which already seals node forward, every sample launch, the accumulated node
bar, and exactly one reverse per active block. A standalone finalizer that
accepts an arbitrary node bar plus caller-reported sample counts is not a
coverage proof. The executor should expose mutually exclusive
`material_only` and `full_geometry` finalizers; dense cached replay should use
the latter and immediately reduce its physical-length bar through the fenced
geometry bridge. Do not grow a parallel whole-step coordinator for this.
That executor-mode source change now exists, including tests for the unchanged
material path and full-mode frame-density invariance, but has not been run
after the change because the host remained saturated. The reduced
full-geometry finalizer and its tests now require the executor-sealed block
execution, including actual sample coverage, node-bar identity, and loss-scalar
identity; they no longer accept the old free-standing coverage proof. This
remains source-only until the focused quiet-host gate passes. A mutually
exclusive executor-bound full-geometry request path now exists in source, but
it is outside the fixed-site material coordinator and remains unrun,
native-unbuilt, allocator-unmeasured, and absent from an end-to-end trainer.

The geometry audit narrowed the original world claim. In a fixed world-coordinate
gauge, a fixed shared-SPD(4) power world slices exactly to a common translation
of fixed anisotropic 3D sites with affine relative weights; every candidate
face has a constant spatial normal. In the executable `M=I` case, even the
common site translation vanishes. A time-dependent global scene gauge can
freeze one rotating normal, so the two-site rotating-face fixture is a
fixed-gauge separation test. One shared gauge cannot generally freeze several
independently rotating faces. The selected general model therefore uses the
camera/scene gauge for shared bulk motion and the exact CPU
`kinetic_power_word_compiler.py` frontend for direct affine kinetic residuals:

```text
p_i(t) = p_i0 + t v_i
w_i(t) = w_i0 + w_i1 t + w_i2 t^2.
```

For affine rays it derives exact binary64-rational `A_ij(t),B_ij(t)` of degree
at most two and adjacent concurrence of degree at most four, reuses the exact
fixed-time sparse lower envelope, demonstrates a fixed-gauge rotating face,
and reports parameter bytes invariant from one to one million requested
frames. Exact rational square-free/Sturm isolation through quartics and a
guarded finite-cut concurrence wrapper are implemented at CPU scope. An
exhaustive `O(S^3)` continuous reference compiler now isolates/group events,
filters them by exact one-sided owner words, emits half-open charts, and agrees
with an independent global-product/Sturm oracle. An active-owner compiler now
matches those routes on supported strata while deriving predicates from
witnessed owners/cuts. It reports `O(U S R_max)` predicate construction over
unique owner words plus `O(W (S log S + S R_max))` cumulative
root-complement/certification work; this is not a flat `O(SR)` theorem. A CPU
multi-chart bridge evaluates exact ordered P0 transfer at fixed `J_c` nodes,
dispatches binary samples right-continuously, certifies the actual barycentric
primal/material actions, reduces residuals to `O(sum J_c)` node cotangents, and
runs one frozen-program stable-stratum VJP for site positions/velocities,
quadratic weights, affine rays, density, and RGB. Supported persistent/full-
fiber semantics beyond fail-closed behavior, bounded-cell sphere/vacuum events,
dataset-bound program generation, warm/output-sensitive affected-chart repair,
total recompilation derivatives, and trainer wiring remain open. The exact CPU
reference already proves whole-direction persistence and endpoint re-isolation
for separated singleton simple roots, with fixed-seed differential agreement
against fresh compilation and fail-closed unsupported strata. A
provenance-sealed actual-rank batch lowerer, independent Lie oracle,
fake-native CPU lifecycle adapter, source-only precompiled-length forward/VJP,
and CPU node-length geometry VJP close the narrow numeric seam. The exact
directional trust certificate covers only one strict event-free single chart;
the extension has not been rebuilt or runtime-verified.

It includes the ordinary-depth fiber Jacobian `||d(t)||`, analytic
fixed-topology boundary/ray and boundary-to-site/weight VJPs, and affine
depth-coordinate rescaling parity. General log-depth charts remain open.

Historical four-file core CPU gate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compiled_transfer_adjoint.py \
  research_experiments/world_foam_lane2/test_exact_sparse_incidence_oracle.py \
  research_experiments/world_foam_lane2/test_transfer_lie_chart.py \
  research_experiments/world_foam_lane2/test_compiled_lie_world_adjoint.py -q
```

The original four-file gate was `38 passed`; the staged/compact, fixed-duration,
and continuous-certificate suites extend that evidence separately. The
float64 checks now have outward-rounded continuous P0 transfer/first-derivative
and all-competitor owner certificates, with an optional boundary-to-site error
bound. The dense continuous Lie jet remains the tiny-fixture oracle; strict
production callers can explicitly select `track_local_sparse`, which streams
one track's referenced boundaries/incidences/sites plus 12 ray coefficients,
retains only aggregate bounds and a canonical global-id digest, and rejects a
track before constructing quadratic dual state if its configured local
dimension cap is exceeded. Small shared-world tests match the dense oracle
exactly, unrelated global resources do not increase local dual dimension, and
the exact `kappa=0` identity chart uses its removable Taylor jet. The dense
oracle is not a production path: its pointer-slot floor is
`max(16D^2, 64 P J_max D)` bytes and exceeds `768 GiB` already at
`P=8192,J=16` even under impossible zero-boundary/zero-incidence assumptions,
before Python interval and `Fraction` objects. Fixed-time
owner discovery uses an exact `O(S log S)` line-envelope hull. Exact near/far
and triple-concurrence event polynomials isolate rational and irrational roots,
and supplied fixed-4D piecewise charts stream with exact binary-sample guard
dispatch. Exact irrational native endpoints remain non-paper. The new direct
kinetic 3D frontend raises adjacent concurrence to quartic. Its guarded CPU
predicate isolation, exhaustive continuous chart compiler, independent oracle,
active-owner closure, multi-chart material-transfer bridge, continuous
material-action certificate, frozen-program geometry/material VJP, and
single-ray native-shaped node-length lowering are now implemented. Source-only
native Lie-node forward/VJP and row-ragged sample ABIs are wired but unbuilt.
Dataset-bound program generation, bounded-cell event coverage, warm/output-
sensitive affected-source repair, and derivatives through event/chart/rank
choices remain open. Restricted separated-singleton event re-isolation and
bounded equal-rank lowering plus the ragged CPU/source coordination seam are
green. A hard fixture demonstrates why joint primal/tangent rank is
necessary: its primal transfer is near floating-point exact from `J=2`, while
maximum world-VJP error falls only from about `1.40e-2` at `J=2` to `1.27e-8`
at `J=32`. Its rank-16 interval certificate also exceeds the current bounded
work budget, which is an explicit certificate-cost/rank-death result. See
`../../agent_notes/loose_notes/2026-08-03_03-35-19_worldfoam_memory_light_shared_adjoint.md`
and `../../TODO/worldfoam_memory_light_native4d.md` for proof boundaries and
the production Metal/trainer sequence.

The source tree has suffixed, non-promoted Metal bridges for exact
constant-state replay, staged sparse-Mobius replay, and the compiled Lie path
(`J`-node word compile, `K x J` sample reduction, `J`-node reverse, sparse
incidence finalization). Earlier source-verification gates passed before the
latest 2026-08-04 integration edits, but the current tree has not been rerun and
the extension has not been rebuilt or run on MPS. The lifecycle binds topology/world/chart/`K`/
gradient versions, certificate generations, global `P_global x F_global x 3`
normalization, exact half-open partitions, and resident-site scatter. The
source block adapter, owner-only mutable-material binding, constrained
softplus/sigmoid optimizer step, and right-continuous piecewise-topology
adapter previously passed their focused source/CPU gates. Its loss-only sample
ABI reduces directly into loss and node
cotangents without allocating or writing a discarded prediction tensor. The
material binding selects a reverse ABI without geometry bars and a target-only
staging route; strict frozen evaluation remains separately certified.
Native sample-time state is now block-local: a prepared token owns no global
`[F]` or chart-local `[F_c]` clone, and each launch receives only its live
CPU-float64 `[K]` time block. This removes an avoidable frame-sized source
allocation; it does not remove the unavoidable `O(PF)` target/output stream or
the current `O(N_B FJ)` spatial-block-first weight construction.
The current combined 2026-08-03 CPU/source verification is `152 passed, 11
source-verifier subtests passed` across every `test_kinetic*.py`, the ragged
sample/union/coordinator/staging tests, and the native source verifier. The
focused union-local/sample/coordinator gate is `15 passed`. The independent
oracle exposed and helped
repair a Sturm sign-normalization defect, now covered by a rootless `x^2+1`
regression. Tiny-optical-depth and transmittance-underflow regressions also
require direct `kappa=sum(tau)` accumulation rather than `-log(product beta)`.
Runtime
parity, exact irrational native endpoints,
projective event compilation, unified-runner wiring, and measured allocator/
bandwidth scaling remain open.

The source memory audit gives about `4 MiB` for node state plus node bars at
`B_p=8192,K=8,J=16`. One float32 material target block is `0.75 MiB`; the
loss-only ABI adds neither prediction nor explicit sample rays. Target-only
staging saves `1.5 MiB` at that block size. Forward media/evaluation may
explicitly add a separate `0.75 MiB` prediction block and retain bounded rays.
There is no intrinsic 32-GB requirement. Fit-derived second-form barycentric
weights provide `O(FJ)` construction per spatial block for the actual rounded
nodes, hence `O(N_B FJ)` in the current material step; exact nodes are one-hot,
while explicitly counted exceptional rows cost `O(F_fallback J^2)` through the
dense oracle or fail closed. Compact interpolation is `31 passed`; the staged adjoint is
`13 passed, 6 subtests passed`. The low-run two-run fixture still routes to
exact replay even with linear weights, so this is not yet a speed claim. The
native source remains unbuilt and unmeasured, and the unified runner still has
no `worldfoam_native4d` lane.

## STAR UVT vs Gate4 WorldFoam Scale Gate (2026-05-18)

The current paired small-MPS speed gate (resolution/frame schedule matched,
but not representation/capacity/quality matched) is:

```text
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_scale_32px_224t_vs_12site_2_4_8_16.json
```

It compares STAR UVT `direct_atomic/index_add` at 32px/224 tubes against the
fixed WorldFoam Gate4 `fused_mse_rgb_only` artifact at 32px/12 sites for
2/4/8/16 frames. Result: `status ok`, `failures []`.

Warm-step medians:

```text
frame  STAR total  WF total  STAR/WF total  STAR backward  WF backward  STAR/WF backward
2      5.719 ms    1.522 ms  3.758x         2.937 ms       1.212 ms     2.424x
4      6.464 ms    1.705 ms  3.792x         3.609 ms       1.387 ms     2.602x
8      4.834 ms    1.893 ms  2.554x         2.476 ms       1.559 ms     1.589x
16     7.604 ms    2.224 ms  3.420x         4.101 ms       1.877 ms     2.185x
```

First-to-last scale: STAR total/backward medians are `1.329x`/`1.396x`;
WorldFoam total/backward medians are `1.461x`/`1.549x`; WorldFoam mixed tape
storage is `0.992x` while explicit ray storage is still `8.000x`.

Scope boundary: this says the fixed WorldFoam fused warm step is locally
competitive on a tiny 32px timing surface. It does not prove WorldFoam is
system-level competitive with STAR UVT because WorldFoam still pays
frame-scaling explicit-ray storage and tape-build wall time outside the fused
kernel.

The less-tiny 64px follow-up is:

```text
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_scale_64px_896t_vs_12site_2_4_8_16.json
```

It compares STAR UVT `direct_atomic/index_add` at 64px/896 tubes against the
fixed WorldFoam Gate4 `fused_mse_rgb_only` artifact at 64px/12 sites for
2/4/8/16 frames. Result: `status ok`, `failures []`.

Warm-step medians:

```text
frame  STAR total  WF total  STAR/WF total  STAR backward  WF backward  STAR/WF backward
2      11.560 ms   3.608 ms  3.204x         5.131 ms       2.847 ms     1.802x
4      13.998 ms   2.226 ms  6.288x         6.687 ms       1.791 ms     3.735x
8      8.805 ms    2.630 ms  3.348x         4.983 ms       2.279 ms     2.186x
16     9.044 ms    3.534 ms  2.559x         5.953 ms       3.151 ms     1.889x
```

First-to-last scale: STAR total/backward medians are `0.782x`/`1.160x`;
WorldFoam total/backward medians are `0.979x`/`1.107x`; WorldFoam mixed tape
storage is `0.997x` while explicit ray storage is still `8.000x`.

The first 64px/24-site WorldFoam fused-MSE attempt failed before train/eval
because one candidate row had `222` candidates and the old Metal local boundary
cap was `128`. A follow-up initially raised the Python/array cap to `256`, but
that artifact is superseded: the Metal insertion helper still stopped at the
old 128-depth cap, so rows above 128 were silently truncated. The corrected
high-cap fix adds a 256-cap helper and routes only the high-cap affine forward
eval and fused-MSE kernels through it:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_highcap_insert_fix_parity_mps.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_capcheck_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_capcheck_render64_site24_2_4_8_16_verifier.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_repeat20_render64_site24_2_4_8_16_verifier.json
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_highcap_insertfix_scale_64px_896t_vs_24site_2_4_8_16.json
```

The focused Metal regression now exercises a 140-candidate row and proves both
affine replay RGB and fused-MSE loss use candidates beyond 128. The corrected
1-step capcheck verifier is `status ok`; max train candidate row is still `222`,
accepted under the fused-MSE cap `256`. The corrected repeat20 artifact is
`status ok` for training/quality, but its verifier intentionally reports
`status failed` on broad timing scale: total median scale is `3.945x` and
backward median scale is `4.260x` for an `8x` frame-count increase. Mixed tape
storage is still near-flat (`0.992x`) while explicit ray storage remains
`8.000x`.

The corrected higher-capacity STAR comparison is split rather than a clean win:

```text
frame  STAR total  WF total   STAR/WF total  STAR backward  WF backward  STAR/WF backward
2      4.686 ms    3.572 ms   1.312x         2.598 ms       3.211 ms     0.809x
4      5.479 ms    4.725 ms   1.160x         3.297 ms       4.360 ms     0.756x
8      6.504 ms    6.819 ms   0.954x         4.114 ms       6.441 ms     0.639x
16     8.268 ms    14.091 ms  0.587x         5.633 ms       13.678 ms    0.412x
```

First-to-last scale: STAR total/backward medians are `1.764x`/`2.168x`;
WorldFoam total/backward medians are `3.945x`/`4.260x`. Practical read:
WorldFoam's corrected high-cap tape is still storage-sublinear, but the fused
kernel compute is not sublinear enough; per-frame sample replay dominates once
all candidates are actually inserted. The next WorldFoam work is reducing or
amortizing high-cap per-sample candidate replay, not lifting caps or citing the
superseded truncating artifacts.

Two post-fix shader forks were tried on the same corrected 64px/24-site gate.
The append-then-shell-sort depth-order fork preserved quality but slowed every
frame count, so it was reverted:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_shellsort_parity_mps.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_shellsort_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_shellsort_repeat20_render64_site24_2_4_8_16_verifier.json
```

```text
frame  insert-fix total  shellsort total  insert-fix backward  shellsort backward
2      3.572 ms          4.004 ms         3.211 ms             3.649 ms
4      4.725 ms          5.380 ms         4.360 ms             5.026 ms
8      6.819 ms          8.508 ms         6.441 ms             8.134 ms
16     14.091 ms         15.310 ms        13.678 ms            14.950 ms
```

The local-tape-footprint fork is the current keeper. It removes stored
`segment_alpha`, `weights`, and `segment_rgb` arrays from the high-cap fused
RGB-MSE VJP kernel and recomputes them during the reverse pass. Parity stayed
exact and quality stayed unchanged:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_localtape_parity_mps.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_localtape_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_localtape_repeat20_render64_site24_2_4_8_16_verifier.json
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_localtape_scale_64px_896t_vs_24site_2_4_8_16.json
```

```text
frame  insert-fix total  local-tape total  speedup  insert-fix backward  local-tape backward  speedup
2      3.572 ms          3.400 ms          1.05x    3.211 ms             3.019 ms             1.06x
4      4.725 ms          4.511 ms          1.05x    4.360 ms             4.120 ms             1.06x
8      6.819 ms          6.493 ms          1.05x    6.441 ms             6.137 ms             1.05x
16     14.091 ms         11.549 ms         1.22x    13.678 ms            11.070 ms            1.24x
```

The local-tape verifier still fails the broad scale gate: total median scale is
`3.397x` and backward median scale is `3.667x` for an `8x` frame-count increase.
The refreshed STAR comparison shows WorldFoam now wins/ties total step time
through 8f but still loses 16f and scales worse:

```text
frame  STAR total  WF local-tape total  STAR/WF total  STAR backward  WF local-tape backward  STAR/WF backward
2      4.684 ms    3.400 ms             1.378x         2.591 ms       3.019 ms                0.858x
4      5.507 ms    4.511 ms             1.221x         3.301 ms       4.120 ms                0.801x
8      6.577 ms    6.493 ms             1.013x         4.117 ms       6.137 ms                0.671x
16     8.220 ms    11.549 ms            0.712x         5.528 ms       11.070 ms               0.499x
```

First-to-last scale: STAR total/backward medians are `1.755x`/`2.134x`;
WorldFoam local-tape total/backward medians are `3.397x`/`3.667x`. Practical
read: local-tape fixed a register/local-memory cliff, not the replay-growth
problem. The next fork should reduce real per-frame candidate/segment replay
work, for example by adding an owner-run or interval tape that can skip repeated
same-owner segments.

The next keeper is an inline owner-run reverse-tape merge in the high-cap fused
RGB-MSE kernel. It keeps the per-interval forward accumulation, but collapses
adjacent same-owner intervals into one reverse-tape entry before the reverse
pass and atomic gradient writes:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_ownerrun_final_parity_mps.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerrun_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerrun_repeat20_render64_site24_2_4_8_16_verifier.json
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_ownerrun_scale_64px_896t_vs_24site_2_4_8_16.json
```

```text
frame  local-tape total  owner-run total  speedup  local-tape backward  owner-run backward  speedup
2      3.400 ms          2.724 ms         1.25x    3.019 ms             2.396 ms            1.26x
4      4.511 ms          3.233 ms         1.40x    4.120 ms             2.915 ms            1.41x
8      6.493 ms          6.032 ms         1.08x    6.137 ms             5.627 ms            1.09x
16     11.549 ms         6.610 ms         1.75x    11.070 ms            6.205 ms            1.78x
```

Quality remains unchanged (`train PSNR 13.732/13.753/13.661/13.735`,
`heldout PSNR 14.170/13.992/14.220/14.232`) and the parity probe reports
`max grad diff 0.0`. The verifier still reports `status failed`, but the miss is
now small: total median scale is `2.427x` against the `2.000x` threshold and
backward median scale is `2.590x` against the `2.500x` threshold. This is a
real replay-reduction win, not just a storage win.

The matched STAR comparison should be read carefully because the STAR rerun in
the owner-run artifact was noisy at 16f. Against the stable STAR medians from
the immediately preceding local-tape comparison, owner-run WorldFoam is faster
on total step time at all four frame counts (`STAR/WF total
1.72/1.70/1.09/1.24x`) while STAR still has better backward at 8f/16f. Practical
read: WorldFoam is now locally competitive on this 64px/24-site total-step
surface, but the formal WorldFoam scale verifier is not closed.

A more aggressive forward-merge variant that delayed same-owner forward
accumulation also passed parity but was negative at 16f:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_ownerrun_forwardmerge_parity_mps.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerrun_forwardmerge_repeat20_render64_site24_2_4_8_16.json
```

Forward-merge totals were `2.878/3.197/4.061/10.909 ms`, so the shader was
reverted back to reverse-only owner-run.

An in-kernel boundary-pair owner-update fork was also tried and reverted. It
used candidate boundary ids plus boundary site pairs to call
`wf2_realray_owner_at` only for the first segment, then toggled ownership across
crossed boundaries. A tiny MPS boundary regression passed, but the real
64px/24-site gate was slower than the current keeper:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerupdate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerupdate_repeat20_render64_site24_2_4_8_16_verifier.json
```

Owner-update totals were `3.798/5.100/7.883/16.696 ms` and backward medians
were `3.447/4.793/7.535/15.868 ms`, with verifier scale
`4.396x` total / `4.603x` backward. Practical read: boundary-pair toggling
increases local state/register pressure enough to lose; the next serious fork
is still the larger precomputed owner-run/site-pair path that removes
candidate-depth replay and owner scans from the warm fused-MSE kernel, not
another in-kernel ownership variant.

## Gate4 Affine Moving-Camera Tape Bridge (2026-05-18)

The current moving first-person-camera bridge has a reusable Python tape object:

```text
research_experiments/world_foam_lane2/gate4_affine_slab_tape.py
```

The MPS fused affine real-ray smoke now builds through `Gate4AffineSlabTape`
instead of carrying a private duplicate CSR builder. The focused Gate4 verifier
guards the render32/site12 artifact, including owner-update scope, mixed
num32/den16 error, no missing sample events, boundary-test scaling, and pure
coeff16 rejection:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix_verifier.json
```

Current verified artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix_verifier.json
```

Verifier result:

- status: `ok`
- mixed num32/den16 max error: `0.00016689300537109375`
- pure coeff16 rejected: max error `0.02310839295387268`, outside tolerance
- no missing sample events at 2/4/8/16 frames
- compiled boundary-test ratio: `0.5 / 0.25 / 0.125 / 0.0625`
- explicit ray storage scales `8.0x` from 2 to 16 frames
- mixed tape storage scales `0.967x` from 2 to 16 frames
- owner-update checks are explicitly marked unchecked when
  `--include-ownerupdate` is not present, so the artifact no longer records
  owner-update acceptance by omission

Scope boundary: this proves the moving-camera affine slab tape representation
and MPS bridge at render32/site12 with VJP checks. It is still not a full
trainer, PSNR/capacity, or STAR-UVT competitiveness claim.

The owner-update variant is now fixed for Gate4 affine slab tapes. The earlier
toggle-owner implementation was invalid for this tape because the slab CSR
contains extra pair-boundary candidates that are not guaranteed to be
lower-envelope owner transitions. The fixed Metal path uses the same segment
midpoint owner selection as the mixed forward/VJP path, keeping the
owner-update entrypoint correct under extra candidate boundaries:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --tile-h 1 \
  --tile-w 1 \
  --include-vjp \
  --include-ownerupdate \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_render32_site12_2_4_8_16_midowner.json
```

Owner-update verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_render32_site12_2_4_8_16_midowner.json \
  --require-ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_render32_site12_2_4_8_16_midowner_verifier.json
```

Owner-update verifier result:

- status: `ok`
- owner-update forward checked: `true`
- owner-update VJP checked: `true`
- owner-update forward max error: `0.00016689300537109375`
- owner-update VJP max relative delta versus reduce:
  `7.990560968252593e-6`
- mixed tape storage scale 2->16: `0.967x`
- explicit ray storage scale 2->16: `8.0x`
- compiled boundary-test ratio: `0.5 / 0.25 / 0.125 / 0.0625`

Scope boundary: this fixes the Gate4 owner-update shader correctness gate for
the affine slab tape. It does not recover the original toggle-owner speed idea;
that shortcut is only valid for a true owner-transition tape, not a tape with
extra candidate pair boundaries.

A stronger owner-update gate now requires the nonzero RGBA/depth VJP seed
artifact rather than accepting an RGB-only adjoint by accident:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --tile-h 1 \
  --tile-w 1 \
  --include-vjp \
  --include-ownerupdate \
  --vjp-seed-mode rgba-depth \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_rgba_depth_render32_site12_2_4_8_16.json
```

Strict RGBA/depth owner-update verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_rgba_depth_render32_site12_2_4_8_16.json \
  --require-ownerupdate \
  --require-vjp-seed-mode rgba-depth \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_rgba_depth_render32_site12_2_4_8_16_verifier.json
```

RGBA/depth owner-update verifier result:

- status: `ok`
- VJP seed mode: `rgba-depth`
- gradient scope: `mixed_num32_den16_site_rgba_vjp_rgba-depth_seed`
- mixed max error: `0.00016689300537109375`
- owner-update VJP max relative delta versus reduce:
  `6.516429842513915e-6`
- RGB-only sidecar expected divergence under nonzero alpha/depth adjoints:
  `true`
- mixed tape storage scale 2->16: `0.967x`
- explicit ray storage scale 2->16: `8.0x`

This is the owner-update correctness gate to cite when the question is whether
the shader handles non-RGB loss adjoints. The RGB-only owner-update verifier
above remains useful history, but this stricter gate is the safer current
acceptance artifact.

The same tape object is now exercised inside the frozen-geometry site-RGBA
train/eval harness:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 5 \
  --warmup-steps 1 \
  --vjp-mode direct_atomic_grad_only \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_render32_site12_2_4_8_16.json
```

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_render32_site12_2_4_8_16.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_render32_site12_2_4_8_16_verifier.json
```

Current train/eval verifier result:

- status: `ok`
- train PSNR at 2/4/8/16 frames:
  `11.794 / 11.879 / 12.020 / 12.103`
- heldout PSNR at 2/4/8/16 frames:
  `12.038 / 13.058 / 13.130 / 13.274`
- total mean time at 2/4/8/16 frames:
  `9.986 / 21.291 / 10.460 / 6.961 ms`
- backward mean time at 2/4/8/16 frames:
  `3.935 / 8.667 / 5.753 / 2.966 ms`
- total mean scale 2->16: `0.697x` for an `8x` frame-count increase
- backward mean scale 2->16: `0.754x`
- train mixed tape storage scale 2->16: `0.992x`
- heldout mixed tape storage scale 2->16: `1.013x`
- explicit ray storage scale 2->16: `8.0x` for both train and heldout
- compiled boundary-test ratio: `0.5 / 0.25 / 0.125 / 0.0625`

Timing caveat: the 4-frame row is visibly spiky, so the useful conclusion is
that the patched Gate4 tape path passes a scoped optimizer-loop smoke with
sublinear end-to-end scaling and flat tape storage. This is still not a stable
benchmark, full-geometry-gradient proof, full trainer proof, or STAR-UVT
quality/capacity claim.

A stronger repeat20 pass adds median timing to the same scoped Gate4
train/eval path:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 20 \
  --warmup-steps 5 \
  --vjp-mode direct_atomic_rgb_only \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_rgbonly_repeat20_render32_site12_2_4_8_16.json
```

Strict median verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_rgbonly_repeat20_render32_site12_2_4_8_16.json \
  --vjp-mode direct_atomic_rgb_only \
  --require-median-timing \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_rgbonly_repeat20_render32_site12_2_4_8_16_verifier.json
```

Repeat20 median verifier result:

- status: `ok`
- train PSNR at 2/4/8/16 frames:
  `13.845 / 13.869 / 13.918 / 13.998`
- heldout PSNR at 2/4/8/16 frames:
  `14.288 / 14.504 / 14.536 / 14.592`
- total median time at 2/4/8/16 frames:
  `65.028 / 78.987 / 78.954 / 79.031 ms`
- backward median time at 2/4/8/16 frames:
  `31.097 / 32.186 / 33.283 / 36.573 ms`
- total median scale 2->16: `1.215x` for an `8x` frame-count increase
- backward median scale 2->16: `1.176x`
- train/heldout mixed tape storage scale 2->16: `0.992x / 1.013x`
- explicit ray storage scale 2->16: `8.0x`
- shared max-frame data load: `30.457 s`; per-row load/slice after sharing is
  `<= 0.003 s`
- per-row total wall after shared load at 2/4/8/16 frames:
  `3.610 / 3.007 / 3.382 / 3.903 s`

This supersedes the noisy 5-step timing read for this narrow gate. It also
separates video/data loading from shader/tape runtime: the current artifact is
still sublinear in the optimizer loop and flat in tape storage, but process wall
time is dominated by the one 16-frame multicam load. It is still a
frozen-geometry site-RGBA optimizer-loop result, not a full trainer or STAR-UVT
competitiveness claim. The RGB-only speed gate uses `direct_atomic_rgb_only`
because the same-process mode comparison showed lower total/backward medians
than the full RGBA/depth `direct_atomic_grad_only` path for the default RGB MSE
objective.

Same-process VJP mode comparison:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/compare_fused_slab_vjp_modes_mps.py \
  --frame-counts 2,16 \
  --vjp-modes direct_atomic_grad_only,direct_atomic_rgb_only,direct_atomic_track \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 8 \
  --warmup-steps 3 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_vjp_mode_compare_direct_render32_site12_2_16.json
```

Mode-comparison result:

- `direct_atomic_grad_only`: total median `67.478 -> 86.495 ms`, backward
  median `31.555 -> 40.613 ms`
- `direct_atomic_rgb_only`: total median `63.997 -> 78.474 ms`, backward median
  `28.598 -> 33.170 ms`
- `direct_atomic_track`: total median `78.992 -> 80.475 ms`, backward median
  `36.164 -> 40.451 ms`
- matched train/heldout PSNR across modes to the displayed precision

The fixed owner-update shader is now also routed through the frozen-geometry
train/eval autograd path as `direct_atomic_grad_only_ownerupdate`. This uses
owner-update forward plus owner-update grad-only VJP inside `loss.backward()`:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 5 \
  --warmup-steps 1 \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16.json
```

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16.json \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16_verifier.json
```

Owner-update train/eval verifier result:

- status: `ok`
- train PSNR at 2/4/8/16 frames:
  `11.794 / 11.879 / 12.020 / 12.103`
- heldout PSNR at 2/4/8/16 frames:
  `12.038 / 13.058 / 13.130 / 13.274`
- total mean time at 2/4/8/16 frames:
  `64.501 / 92.000 / 84.809 / 108.908 ms`
- backward mean time at 2/4/8/16 frames:
  `29.684 / 40.167 / 33.580 / 52.514 ms`
- total mean scale 2->16: `1.688x` for an `8x` frame-count increase
- backward mean scale 2->16: `1.769x`
- train/heldout mixed tape storage scale 2->16: `0.992x / 1.013x`
- explicit ray storage scale 2->16: `8.0x`

Interpretation: owner-update is now optimizer-loop correct, but it is not the
speed path. Recomputing midpoint owners makes it much slower than the normal
`direct_atomic_grad_only` repeat20 path. Cite this as a correctness gate, not
as the competitive runtime result.

The owner-update optimizer-loop path also has a stricter RGBA/depth-adjoint
gate. The train/eval harness can add tiny alpha/depth auxiliary losses so
`loss.backward()` sends nonzero adjoints through the alpha and depth outputs,
not only RGB:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 5 \
  --warmup-steps 1 \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --alpha-aux-weight 0.01 \
  --depth-aux-weight 0.01 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16.json
```

Strict RGBA/depth optimizer-loop verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16.json \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --require-alpha-depth-aux-loss \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16_verifier.json
```

RGBA/depth optimizer-loop verifier result:

- status: `ok`
- alpha/depth aux weights: `0.01 / 0.01`
- alpha output gradient abs-sum at 2/4/8/16 frames:
  `0.0199991 / 0.0199991 / 0.0199991 / 0.0199991`
- depth output gradient abs-sum at 2/4/8/16 frames:
  `0.0001728 / 0.0001624 / 0.0001567 / 0.0001569`
- train PSNR at 2/4/8/16 frames:
  `11.794 / 11.879 / 12.020 / 12.103`
- heldout PSNR at 2/4/8/16 frames:
  `12.038 / 13.058 / 13.130 / 13.274`
- total mean scale 2->16: `1.126x` for an `8x` frame-count increase
- backward mean scale 2->16: `0.949x`
- train/heldout mixed tape storage scale 2->16: `0.992x / 1.013x`
- explicit ray storage scale 2->16: `8.0x`

This is the current strongest owner-update correctness gate: moving-camera
optimizer-loop, owner-update forward/backward, and nonzero RGB/alpha/depth
adjoints. It remains a correctness gate, not the speed path.

## Current Fused Slab Mixed Status (2026-05-15)

The current fork for the moving first-person-camera World Foam lane is:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
```

The current measured RGB-only Gate4 train/eval speed path is
`direct_atomic_rgb_only`; the full RGBA/depth correctness path remains
`direct_atomic_grad_only_ownerupdate` with alpha/depth aux seeds. The older
2026-05-15 summary below is historical for the pre-Gate4 fused slab lane, not
the current moving-camera RGB speed winner.

Canonical status summary:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
```

Verifier for the summary's scope boundary:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Verifier for the saved mixed-mode scaling artifacts, the render32 framegroup16
loss-reduction guardrail, and the repeated-frame 16/32/64/128 framegroup16
compare-harness speed-scale smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_fused_slab_mixed_scaling_verifier_with_framegroup_lossreduce.json
```

Focused robust timing classifier for framegroup16 artifacts. Use the strict
form for clean speed-scale promotion artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_framegroup16_timing_robust_lossreduce_accepted.json
```

When a full mixed sweep is visibly contaminated by MPS spikes, the classifier
can pair it with a separate warm max-frame confirmation. That mode may prove
the promoted path is not regressed, but it deliberately does not mark the full
matrix as a clean speed-scale proof:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_current_revalidated_warm3_steps5_render32_site12_16_32_64_128.json \
  --confirm-artifact research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_current_revalidated_128only_warm10_steps20_render32_site12.json \
  --allow-confirmed-outliers \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_framegroup16_timing_robust_current_revalidated_confirmed_outlier.json
```

Same-process speed-scale compare smoke for the current framegroup16 fused-MSE
path:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 16,32,64,128 \
  --render-size 32 \
  --site-count 12 \
  --steps 8 \
  --warmup-steps 3 \
  --optimizer-mode manual-vjp \
  --include-delta-framegroup16-fused-mse \
  --repeat-loaded-frames \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render32_site12_warm3_steps8_16_32_64_128.json
```

Real-loaded 16/32 compare for the same path, using the generated real 32-frame
multicam fixture and no repeated-frame expansion:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --config research_experiments/world_foam_lane2/results/fixed_step_speed_compare_inputs/128px_32f_config.json \
  --frame-counts 16,32 \
  --render-size 32 \
  --site-count 12 \
  --steps 8 \
  --warmup-steps 3 \
  --optimizer-mode manual-vjp \
  --include-delta-framegroup16-fused-mse \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_real32_render32_site12_warm3_steps8_16_32.json
```

Current summary artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
research_experiments/world_foam_lane2/results/2026-05-16_fused_slab_mixed_scaling_verifier_with_framegroup_lossreduce.json
research_experiments/world_foam_lane2/results/2026-05-16_framegroup16_timing_robust_lossreduce_accepted.json
research_experiments/world_foam_lane2/results/2026-05-16_framegroup16_timing_robust_current_revalidated_confirmed_outlier.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render32_site12_warm3_steps8_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_run_vs_delta_framegroup16_compare_real32_render32_site12_warm3_steps8_16_32.json
```

As of the 2026-05-16 framegroup pass, the canonical status summary consumes
the framegroup16 loss-reduction verifier and the status verifier enforces the
same scope boundary: the result is a guarded render32/site12 microbench
frame-scaling win, not a full-trainer or quality/capacity claim.

The historical summary reports:

- winner: `direct_atomic_grad_only`
- total step scaling: `7.17 ms` at 2 frames to `9.32 ms` at 16 frames
- total scale 2->16: `1.30x` for an `8x` frame-count increase
- render scale 2->16: `1.06x`
- backward scale 2->16: `1.74x`
- 16-frame heldout PSNR: `13.27`
- max matched-frame PSNR spread across modes: about `2.1e-6`
- isolated owner-run RGB autograd train/eval: `5.40 ms` at 2 frames to
  `6.04 ms` at 16 frames with matched fused-parameter PSNR
- owner-run 16-frame heldout PSNR: `13.27`
- owner-run 16-frame total step ratio versus the fused winner: `0.648x`
- owner-run 16-frame train storage ratio versus full segment tape: `0.056x`
- endpoint-run continuous-depth RGB autograd train/eval: `7.84 ms` at 16
  frames with matched fused-parameter PSNR
- endpoint-run 16-frame train storage ratio versus full segment tape: `0.111x`
- endpoint-run Metal VJP relative error versus torch autograd: `1.02e-4`
- endpoint-record edit RGB-only VJP sidecar: relative error `2.86e-6`
  versus the full edit VJP with zero alpha/depth adjoints
- latest RGB-only paired repeat: endpoint-record edit is still slower than
  endpoint-run at 16f (`1.33x` total-step ratio), while keeping edit storage
  at `0.0261x` full segment CSR
- track-loop forward sidecar: correct versus endpoint-run (`8.94e-7` max abs
  error) but not a speed win at 16f (`2.15 ms` versus endpoint-run `1.18 ms`)
- block4 anchored forward sidecar: correct versus endpoint-run (`8.94e-7` max
  abs error), faster than original edit replay at 16f (`1.72 ms` versus
  `2.91 ms`), but slightly slower than endpoint-run in the refreshed raw probe
  (`1.69 ms`)
- block4 RGB-only VJP sidecar: relative error `2.93e-6` versus full VJP with
  zero alpha/depth adjoints and `2.86e-6` versus the existing edit RGB-only VJP;
  16f timing is `2.84 ms` versus `3.30 ms` for edit RGB-only VJP
- block4 endpoint-record edit RGB autograd train/eval with the dedicated
  block4 VJP: storage remains compact at `0.0438x` full segment CSR, and total
  step still scales sublinearly (`3.01x` for an `8x` frame-count increase), but
  the 16f step is not speed competitive in the corrected rerun (`75.18 ms`)
- coefficient-cached block edit sidecar: correct forward replay (`1.37e-6` max
  abs error), correct RGB-only VJP (`3.63e-6` relative error versus full VJP
  with zero alpha/depth), and a green warmed 2/4/8/16 autograd smoke
- refreshed sequential 16f render16 coefficient timings after the near/far
  contract fix: endpoint forward `6.26 ms`, block4 forward `4.74 ms`,
  coefficient forward `6.00 ms`, original edit forward `6.31 ms`, and
  coefficient RGB-only VJP `3.06 ms`
- promoted paired 16f render32 train/eval in one process: the 5-step 2/4/8/16
  smoke has endpoint-run `13.04 ms`, raw edit `11.28 ms`, block4 `9.09 ms`,
  and block-coeff `8.06 ms`; the stronger 20-step 16f repeat has endpoint-run
  `9.42 ms`, raw edit `11.31 ms`, block4 `9.19 ms`, and block-coeff `7.48 ms`
- practical read from paired repeats: raw edit is compact but speed-noisy,
  block4 is near/below endpoint-run, and block-coeff is the current fastest
  16f sidecar despite heavier storage
- coefficient storage remains the main drawback: above endpoint CSR
  (`~1.6x`-`1.7x` depending on artifact), though only about `0.18x` full
  segment CSR
- framegroup16 loss-reduced render32/site12 guardrail: total step scales
  `1.20x`, backward scales `1.21x`, and selected storage scales `1.04x` from
  16 to 128 frames; mixed-sweep 128f total max is `4.108 ms`, and the 128-only
  rerun2 median/max total is `3.840 / 7.735 ms`
- compare-harness framegroup16 speed-scale smoke: with render32/site12, 8
  measured steps, and repeated loaded frames for the 32/64/128f rows, the
  framegroup16 fused-MSE mode is `1.472 / 3.363 / 5.188 / 4.300 ms` at
  16/32/64/128 frames versus endpoint-run `5.817 / 5.617 / 9.924 / 15.020 ms`;
  total ratios versus endpoint-run are `0.253 / 0.599 / 0.523 / 0.286`, heldout
  PSNR stays matched within `0.0029 dB`, and total-step scale is `2.92x` for an
  `8x` frame-count increase, but this is still a smoke-scale harness check, not
  a stable benchmark
- real-loaded framegroup16 compare: with the generated 128px 32-frame multicam
  fixture and no repeated rows, endpoint-run total is `4.923 / 7.091 ms` at
  16/32f, while framegroup16 fused-MSE is `2.812 / 3.166 ms`; total ratios are
  `0.571 / 0.447`, backward ratios are `0.901 / 0.641`, and storage versus
  endpoint-run is `0.764x / 0.385x`. This is now real-frame sublinear at this
  narrow shader scope: framegroup total/backward/storage scale is
  `1.126x`/`1.106x`/`1.008x` for a `2x` frame-count increase.
- mixed-scaling verifier now fails if that compare smoke loses the framegroup
  all-frame speed ratio (`>0.75x` endpoint-run total at any checked frame),
  loses the 16f backward ratio (`>0.95x` endpoint-run backward), drifts in
  16f PSNR (`>1e-3`) or all-frame PSNR (`>5e-3`), loses compact selected
  storage (`>0.15x` full at 16f), changes away from render32/site12, scales
  beyond total/backward/storage guards (`3.25x`/`3.75x`/`1.10x`), or drops the
  repeated-fixture/not-stable-benchmark scope boundary
- the status verifier also fails if the real-loaded 16/32 guard is turned into
  a repeated-frame artifact, loses its speed/storage wins, drops the measured
  real-frame sublinear claim, or widens it into full-trainer/STAR-competitive
  scope

It also includes a tiny STAR-UVT direct-atomic 32px speed reference:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fixed_step_speed_compare_star_directatomic_20step_32px_2_4_8_16.json
```

That reference uses the same `20` measured steps and `5` warmup steps as the
promoted coefficient-cached sweep. It reports `32.83 ms` mean STAR step time at
16 frames, versus the coefficient-cached World Foam sidecar's `6.84 ms` 16f
total step and an earlier active-internal fused World Foam winner's `9.32 ms`.
Treat this only as a small speed reference, not a matched quality/capacity
comparison: current World Foam is fixed-geometry/site-RGBA with 12 sites, while
STAR UVT uses its world-tube model. In that tiny reference, STAR mean step time
scales `1.23x` from 2 to 16 frames while frame count scales `8x`; that is
runtime-sublinear evidence, still not a matched quality/capacity result.

### Chunked Delta-Replace Framegroup16 Fused MSE (2026-05-16)

The current best flat-storage World Foam shader fork is:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse
```

It keeps the i16x3 delta-replace record tape but changes the execution shape:
one Metal threadgroup handles a 16-frame chunk for one track, builds the selected
replacement rows in threadgroup memory, then each local frame thread runs the
RGB MSE/VJP path. The host launcher chunks longer sequences, so the same mode
runs the 16/32/64/128 speed sweep.

Primary artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunked_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
```

Result:

- status: `ok`
- total ms for 16/32/64/128 frames: `3.057 / 3.972 / 3.341 / 6.691`
- fused backward ms: `2.300 / 2.728 / 2.494 / 5.802`
- selected tape storage bytes: `49936 / 49902 / 49902 / 49916`
- storage scale 16->128: `0.9996x` for an `8x` frame-count increase
- total-step scale 16->128: `2.19x` for an `8x` frame-count increase
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

Same-setting controls:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_fused_mse_repeat_loaded_warm5_steps12_render16_2_4_8_16_control.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_edit_blockcoeff16_fused_mse_repeat_loaded_warm5_steps12_render16_2_4_8_16_control.json
```

On the 2/4/8/16 frame control, chunked framegroup total ms was
`1.650 / 3.243 / 2.315 / 1.847`, versus old i16x3
`3.709 / 2.947 / 2.491 / 3.758` and block-coeff16
`2.268 / 2.074 / 3.229 / 2.536`. This makes framegroup16 the first path in this
lane that keeps delta-replace's flat storage while also being runtime-competitive
with the formerly fastest block-coeff16 sidecar at 16 frames.

A follow-up shader-only "select-start" shortcut was measured and rejected:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_selectstart_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
```

It improved 16f (`2.473 ms`) but worsened the longer rows
(`4.247 / 4.279 / 7.578 ms` at 32/64/128). The live kernel is restored to the
better chunked-base replay path in that artifact. Reducing 128f overhead likely
needs an explicit chunk-start table or a deeper row-layout change, not just a
per-chunk scan inside the current metadata.

An explicit chunk-start table was then wired into the live framegroup16 op and
counted into selected storage:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunkstarts_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
```

Result:

- total ms for 16/32/64/128 frames: `2.470 / 3.553 / 3.909 / 6.481`
- fused backward ms: `1.968 / 2.954 / 3.408 / 5.615`
- selected tape storage bytes: `54032 / 56046 / 60142 / 68348`
- storage scale 16->128: `1.26x` for an `8x` frame-count increase
- total-step scale 16->128: `2.62x` for an `8x` frame-count increase
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

Interpretation: this is a partial 128f-overhead win, not a clean replacement
for the previous row. It improves 16f, 32f, and 128f wall time, but the 64f row
is slower and the chunk offset table moves selected storage from effectively
flat to still-sublinear. Use it with the base chunked artifact when judging
whether explicit indexing is worth the storage tradeoff.

The live indexed path was then narrowed from int32 chunk offsets to int16 chunk
offsets, with a hard `change_count <= 32767` guard:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunkstarts_i16_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_chunkstarts_i16_spot_rerun_warm5_steps12_render16_16_64_128.json
```

Full sweep result:

- total ms for 16/32/64/128 frames: `4.500 / 3.270 / 6.266 / 3.939`
- fused backward ms: `3.389 / 2.735 / 4.981 / 3.217`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- storage scale 16->128: `1.14x` for an `8x` frame-count increase
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

Spot rerun result for 16/64/128 frames:

- total ms: `2.932 / 4.731 / 4.853`
- fused backward ms: `2.247 / 3.591 / 3.925`
- selected tape storage bytes: `51984 / 55022 / 59132`

Interpretation: int16 offsets halve the chunk-start metadata cost and move the
indexed path closer to flat storage (`1.14x` instead of `1.26x`), while the 128f
row remains materially better than the old base chunked row. Wall-clock scaling
is still MPS-noisy and the 64f row is not fixed, so the honest claim is
sublinear representation plus a partial 128f kernel win, not clean STAR-UVT
style runtime sublinearity.

The next shader fork changed the framegroup kernel from per-frame threadgroup
row materialization to per-frame row references: local frame 0 now records each
frame lane's selected base/change row, and the frame lane reads the selected
i16x3 row directly during RGB MSE/VJP. This removes the duplicated
`frames_in_chunk * row_count` threadgroup row copy while keeping the same int16
chunk-start storage:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_i16_fused_mse_repeat_loaded_warm5_steps12_render16_site4_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_i16_spot_rerun_warm5_steps12_render16_site4_16_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_i16_128only_warm10_steps20_render16_site4.json
```

Comparable site4 full sweep:

- total ms for 16/32/64/128 frames: `2.598 / 2.287 / 3.009 / 3.816`
- fused backward ms: `2.013 / 1.854 / 2.556 / 3.111`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- total-step scale 16->128: `1.47x` for an `8x` frame-count increase
- storage scale 16->128: `1.14x`
- heldout PSNR: `14.6148 / 14.5789 / 14.5713 / 14.5684`

The spot reruns keep the caveat alive:

- 16/64/128 total ms: `3.645 / 3.616 / 8.143`
- 16/64/128 fused backward ms: `2.711 / 2.518 / 6.766`
- 128-only warm10/steps20 total/backward ms: `9.788 / 8.385`

Interpretation: row references fix the obvious 64f failure in the clean full
sweep and produce the best 16/32/64 row set so far, but they do not make 128f
stable. Treat this as a useful shader fork and a likely ingredient for a
hybrid path, not as the final STAR-clean runtime result. The old materialized
chunk-start path was slower at 64f but more stable around 128f, so the next
useful fork is a thresholded materialized-vs-rowref dispatch or a separate
128f-specific row-cache path.

I then tried that hybrid dispatch: `frame_count >= 128` used a separate
materialized chunk-start kernel, while lower frame counts kept the lighter
row-reference kernel. A 128-frame parity test covered the fallback path.

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_hybrid_i16_fused_mse_repeat_loaded_warm5_steps12_render16_site4_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_hybrid_i16_128only_warm10_steps20_render16_site4.json
```

Hybrid full sweep:

- total ms for 16/32/64/128 frames: `3.800 / 4.326 / 4.445 / 6.208`
- fused backward ms: `2.806 / 3.234 / 3.617 / 5.380`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- total-step scale 16->128: `1.63x` for an `8x` frame-count increase
- storage scale 16->128: `1.14x`
- heldout PSNR: `14.6148 / 14.5790 / 14.5713 / 14.5684`

Hybrid 128-only warm10/steps20 confirmation:

- total/backward ms: `7.232 / 6.233`
- selected tape storage bytes: `59132`

Interpretation: hybrid dispatch is a worst-case stabilizer, not a win over the
best row-reference sweep. It improved the slow row-reference 128f confirmations
(`8.143` and `9.788 ms`) but remained slower than the best earlier 128f rows,
and the 16/32/64 rows in this run were noisier than the clean rowref sweep. It
was then superseded by the small-site row-reference reduction below; do not
claim this materialized fallback as the fixed 128f path.

An 8-frame high-frame chunk-span fork was also tested and rejected:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_hybrid_i16_chunk8_128only_warm10_steps20_render16_site4.json
```

It reported 128-only total/backward `7.524 / 6.867 ms` and increased selected
storage to `67324` bytes, so the live code was restored to 16-frame chunks.

The current live framegroup op now uses the row-reference kernel for all frame
counts and adds a small-site threadgroup gradient reduction for `site_count <=
16`. Each 16-frame per-track threadgroup accumulates site-RGBA gradients in
threadgroup memory and emits one global atomic add per site instead of one per
local frame segment. This targets the 128f atomic hot spot without changing the
delta-replace i16x3 storage layout.

Current winning artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_fused_mse_repeat_loaded_warm5_steps12_render16_site4_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_128only_warm10_steps20_render16_site4.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_fused_mse_repeat_loaded_warm5_steps12_render16_site12_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_reducecap_fallback_fused_mse_repeat_loaded_warm5_steps12_render16_site20_16_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_rowref_smallsite_reduce_128only_warm10_steps20_render32_site12.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_128only_warm10_steps20_render32_site12.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_128only_rerun2_warm10_steps20_render32_site12.json
```

Comparable site4 full sweep:

- total ms for 16/32/64/128 frames: `2.674 / 2.312 / 2.382 / 2.210`
- fused backward ms: `2.002 / 1.828 / 1.896 / 1.801`
- selected tape storage bytes: `51984 / 52974 / 55022 / 59132`
- total-step scale 16->128: `0.83x` for an `8x` frame-count increase
- backward scale 16->128: `0.90x`
- storage scale 16->128: `1.14x`
- heldout PSNR: `14.6148 / 14.5790 / 14.5713 / 14.5684`

128-only warm10/steps20 confirmation:

- total/backward ms: `1.674 / 1.341`
- selected tape storage bytes: `59132`
- heldout PSNR: `15.5822`

Interpretation: this is the first fork in this sublane that makes the 128f
runtime confirmation look fixed, not merely sublinear on a single clean sweep.
Coverage was then broadened with a 128f multi-track parity test that forces
cross-threadgroup accumulation into shared site IDs, plus a 128f `site_count=20`
parity test that bypasses the small-site reduction cap and exercises the global
atomic fallback. Focused replay is now 16 tests OK, and the full lane is 46
tests OK at this intermediate small-site-reduction point. After the later
loss-reduction and status-summary guardrails, the full lane is 51 tests OK.

Site-count follow-ups:

- site12 full sweep total ms for 16/32/64/128 frames:
  `2.992 / 4.512 / 2.434 / 2.229`
- site12 fused backward ms: `2.188 / 3.311 / 2.007 / 1.900`
- site12 selected storage bytes: `327556 / 331118 / 333748 / 338408`
- site12 total-step scale 16->128: `0.74x`; storage scale: `1.03x`
- site20 above-cap fallback total ms for 16/128 frames: `2.531 / 2.478`
- site20 above-cap fallback fused backward ms: `2.135 / 2.142`
- site20 above-cap fallback selected storage bytes: `851092 / 867222`
- site20 total-step scale 16->128: `0.98x`; storage scale: `1.02x`

Render32 site12 confirmation:

- render32 site12 total ms for 16/32/64/128 frames:
  `2.638 / 3.296 / 3.281 / 6.178`
- render32 site12 median total ms: `2.376 / 3.243 / 3.061 / 4.389`
- render32 site12 fused backward ms:
  `2.182 / 2.588 / 2.928 / 5.620`
- render32 site12 median fused backward ms:
  `1.901 / 2.545 / 2.719 / 4.049`
- render32 site12 selected storage bytes:
  `1322952 / 1339766 / 1353624 / 1373646`
- render32 site12 total-step scale 16->128: `2.34x` for an `8x`
  frame-count increase; storage scale: `1.04x`
- render32 site12 heldout PSNR:
  `14.6291 / 14.6161 / 14.6010 / 14.6233`
- 128-only warm10/steps20 confirmation total/backward ms: `4.488 / 3.711`
  by mean and `3.803 / 3.170` by median; heldout PSNR: `14.6857`

Loss-reduced render32 site12 follow-up:

- change: the live row-reference kernel now reduces per-frame `sample_loss`
  inside the 16-frame threadgroup and emits one global loss atomic per group.
  This shares the same barrier used by the small-site gradient reduction.
- loss-reduced total ms for 16/32/64/128 frames:
  `3.046 / 3.701 / 3.590 / 4.459`
- loss-reduced median total ms: `2.717 / 3.448 / 3.143 / 4.164`
- loss-reduced fused backward ms:
  `2.510 / 3.269 / 3.032 / 3.857`
- loss-reduced median fused backward ms:
  `2.314 / 3.115 / 2.475 / 3.546`
- loss-reduced selected storage bytes:
  `1322952 / 1335670 / 1345432 / 1357262`
- loss-reduced total-step scale 16->128: `1.46x` for an `8x`
  frame-count increase; backward scale: `1.54x`; storage scale: `1.03x`
- loss-reduced 128f full-sweep total max: `7.207 ms`, still below the widened
  MPS outlier guard and far below the earlier `22.776 ms` outlier
- loss-reduced heldout PSNR:
  `14.6291 / 14.6161 / 14.6010 / 14.6233`
- loss-reduced 128-only warm10/steps20 confirmation total/backward ms:
  `5.462 / 4.709` by mean and `5.295 / 4.509` by median; total max:
  `9.811 ms`
- loss-reduced 128-only rerun2 total/backward ms: `2.973 / 2.679` by mean
  and `2.808 / 2.509` by median; total max: `5.258 ms`

The scope is still fixed-geometry, RGB-only site-RGBA, MPS smoke; it does not
prove full World Foam trainer quality or STAR-UVT capacity parity. Render16 is
now clearly sublinear in practice on this microbench. Render32 is now
sublinear in the full frame-scale sweep, and the loss-reduced kernel removes
the prior mixed-sweep 128f outlier. The caveat is that the separate 128-only
confirmation is still noisy: the first loss-reduced 128-only repeat was slower
by median than the earlier row-reference kernel, while rerun2 recovered almost
the same median and improved max. Treat this as a practical frame-scaling fix,
not a full-trainer or universal runtime claim.

Robust timing reclassification after the later owner-reduce rollback confirms
the distinction: the accepted loss-reduced artifact is a clean speed-scale
artifact (`status=ok`, total median scale `1.53x`, backward median scale
`1.53x`, storage scale `1.03x` for `8x` frames). The later current-revalidated
mixed sweep is not a clean proof (`clean_speedscale_artifact=false`) because
32f and 128f were contaminated by MPS spikes; paired with the clean warm10
128-only confirmation it is only `status=confirmed_outlier` with substituted
128f total/backward median scales `1.37x / 1.38x`. Treat
`confirmed_outlier` as "promoted path not regressed," not as promotion evidence
for a new shader fork.

Scope boundary: this is still fixed-geometry RGB-only site-RGBA training on the
tiny render16 MPS smoke. The result is a real World Foam replay-kernel win, but
not a full-trainer or quality-capacity claim.

Rejected shortcuts are also captured in the summary:

- owner-update VJP/forward: the old toggle-owner artifact was rejected because
  forward RGB error reached `0.424` and VJP relative delta reached `2.44e-4`;
  this is superseded for Gate4 affine slab tapes by the 2026-05-18 midpoint
  owner-selection fix documented above
- ordered append: rejected by the depth-order probe because `803444` adjacent
  inversions were observed across `2251403` adjacent pairs

Scope boundary: this is a verified current World Foam shader gate with isolated
practical RGB train/eval paths, not a claim that World Foam is STAR-UVT
competitive. The remaining gap is structural: exact frame-local segment
topology still grows with frame count. The endpoint-run path is compact and
density-independent only under the continuous-absorption depth semantic, which
is not a drop-in replacement for the current segment-mid depth tape. The exact
owner+boundary-cut record delta probe verifies an exact replay representation,
but the compactness result is negative rather than STAR-like.

### Segment Tape Probe (2026-05-15)

The first structural segment-tape math probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_fused_slab_segment_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --timing-iters 3 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json
```

Current result:

- status: `ok`
- max forward error versus current mixed shader: `1.67e-4`
- max Metal-tape VJP relative error versus current `direct_atomic_grad_only`
  winner: `8.55e-6`
- max Metal track-accumulating tape VJP relative error versus current winner:
  `6.04e-6`
- isolated 16-frame Metal tape timings: `1.53 ms` forward, `8.46 ms`
  sample-atomic grad-only VJP, and `4.35 ms` track-accumulating grad-only VJP
- total segment scale from 2 to 16 frames: `8.06x` for an `8x` frame-count
  increase
- 16-frame compact CSR segment-tape storage: about `15.4 MB`, or `13.3x`
  the current mixed CSR plus affine-ray storage

Interpretation: the fixed-geometry segment tape is mathematically compatible
with the current World Foam forward/VJP contract and removes per-step depth
sort plus owner lookup inside the new compact Metal replay shader. The naive
per-sample segment tape is still not STAR-UVT-style structurally sublinear by
itself, and it has not been integrated into the training path.

The simple owner-topology sharing probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_segment_topology_sharing_probe_render32_pertrack_2_4_8_16.json
```

It is an informational negative/weak structural result. At 16 frames, no track
keeps one owner sequence across all frames, per-track unique topology rows are
still `0.908x` of the full sample count, and the frame-to-frame topology
transition rate is `0.905`. That means a simple "store one owner sequence per
track" tape does not recover the STAR-UVT-style structure on the moving-camera
probe.

The delta-tape probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_segment_delta_tape_probe_render32_2_4_8_16.json
```

It splits the topology question more finely. Coarse "row changed?" events are
not useful enough: they scale `13.57x` from 2 to 16 frames. But owner edit
operations scale only `1.30x`, and the 16-frame owner-delta storage estimate is
`0.325x` of the full compact segment CSR. This is the first promising
STAR-port-shaped signal, but it is not exact replay yet because unchanged owner
topology still needs a compact representation for frame-varying segment
`length`/`mid` values.

The boundary-order delta probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_segment_boundary_delta_tape_probe_render32_2_4_8_16.json
```

It tests the exact-geometry side of that problem. Boundary ids plus the existing
rational depth coefficients can recover segment `length`/`mid` without storing
those values per frame. Boundary edit operations scale `6.21x` from 2 to 16
frames, below the `8x` frame-count scale, and the 16-frame replacement boundary
order estimate is `0.346x` of full segment CSR. This is closer to exact replay
than owner-only deltas, but raw all-boundary order is noisy and owner assignment
still needs either a compact exact encoding or a cheap update rule.

The exact owner+boundary-cut record delta probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_segment_record_delta_tape_probe_render32_2_4_8_16.json
```

It joins the two halves of the delta idea: each record stores
`(owner, left_cut_id, right_cut_id)`, so boundary ids recover segment
`length`/`mid` through the existing rational depth coefficients, while owners
match the segment tape exactly. At 16 frames, counts and owners match the full
segment tape, but the replacement record stream is `1.015x` full segment CSR
and the edit-op record stream is still `0.909x` full segment CSR. Exact record
edit ops scale `7.82x` for an `8x` frame-count increase, but the record count
itself scales `8.06x`.

Interpretation: exact owner+cut-id replay works as math, but it is not a
compact STAR-style tape on the moving-camera probe. The remaining structural
question is not whether boundary ids can recover geometry; they can. The
remaining question is how to encode exact owner+boundary record evolution
without storing almost the full frame-local record stream.

The same-owner run probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_segment_owner_run_tape_probe_render32_2_4_8_16.json
```

It is the strongest practical compression result in this batch. It merges
adjacent segment-tape rows that have the same owner and feeds the compressed
tape directly into the existing compact segment-tape Metal kernels. At 16
frames, it cuts `1272086` full segments to `129395` owner runs (`0.102x`) and
storage to `0.109x` of the full compact tape. Forward RGB/alpha/depth match the
full tape at current density within `5e-7`, and RGB-only VJP matches with
relative error `6.95e-6`. The 16-frame RGB-only VJP timing drops from
`16.48 ms` to `1.51 ms` in the isolated probe.

Scope boundary: owner-run depth uses a current-density effective midpoint, and
threshold truncation is also current-density dependent. This is a strong
RGB-training candidate, not yet a final density-independent geometry tape or a
full RGBA/depth-gradient replacement.

The owner-run boundary endpoint probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_owner_run_boundary_tape_probe_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_owner_run_boundary_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_boundary_tape_probe_render32_2_4_8_16.json
```

Current result:

- status: `informational`
- owner-run endpoint records match the current owner-run tape counts and owners
- endpoint ids recover run length with max absolute error about `2.2e-16`
- endpoint-only continuous density depth does not match the current segment-mid
  depth tape; max depth error versus the current owner-run tape is `0.412`
- endpoint alpha remains effectively exact; max alpha error at 16 frames is
  about `3.7e-10`
- 16-frame owner-run endpoint storage is `0.056x` of full segment tape
- owner-run endpoint record count scales `9.89x` from 2 to 16 frames for an
  `8x` frame-count increase

Interpretation: replacing per-run `length`/`mid` floats with
`(owner, left_cut_id, right_cut_id)` is a useful geometry-coefficient step:
boundary ids plus ray coefficients recover run length exactly, and RGB/alpha
can be replayed from endpoint length. It does not fix the structural count
scaling. Endpoint-only continuous absorption depth is a plausible semantic, but
it does not reproduce the current segment-mid depth after same-owner internal
cuts are discarded; exact current-depth replay needs internal moments/cuts or
an explicit depth-semantic change.

The owner-run internal-cut probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_owner_run_internal_tape_probe_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_owner_run_internal_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_internal_tape_probe_render32_2_4_8_16.json
```

Current result:

- status: `informational`
- active internal cuts exactly match current-density RGB/alpha/depth
- active internal cuts are compact at 16 frames: nested CSR is `0.148x` of full
  segment CSR
- active internal cuts are not density independent: at half density the 16-frame
  max alpha error is `0.00999` and max depth error is `0.0105`
- all internal cuts preserve density-independent replay by keeping every
  segment, but 16-frame nested CSR is `0.738x` of full segment CSR
- all-owner-run endpoint storage at 16 frames is compact at `0.111x` of full
  segment CSR, but only if depth semantics change to continuous absorption
  within a same-owner run
- active internal segment count scales `8.73x` and all internal segment count
  scales `8.03x` from 2 to 16 frames, both worse than the `8x` frame scale

Interpretation: internal cuts prove the missing depth information is recoverable,
but the exact density-independent version gives back most of the STAR-style
compression. Owner-run endpoints are compact if depth changes to continuous
absorption within a same-owner run, but that is an explicit semantic change
from the current segment-mid tape. A fixed-density active-cut shader could be a
practical RGB/depth probe, but it would not be the clean STAR UVT analogue.

The endpoint-run continuous-depth Metal probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_tape_probe_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_endpoint_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --timing-iters 3 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_tape_probe_render32_2_4_8_16.json
```

Current result:

- status: `informational`
- Metal forward matches torch continuous endpoint replay; max error `5.96e-8`
- Metal VJP matches torch autograd; max relative error `1.02e-4`
- 16f endpoint storage ratio versus full segment CSR: `0.111x`
- 16f endpoint runs: `134747` versus `1301934` full segments
- max endpoint runs per sample: `7`
- endpoint run count scales `8.61x` from 2 to 16 frames for an `8x` frame
  increase

Interpretation: this is the compact density-independent endpoint shader path if
we accept continuous absorption depth inside same-owner runs. It fixes the
representation problem under that semantic, but it still is not structurally
STAR-like because run count grows about with frame count.

The endpoint-record delta probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_tape_probe_render32_2_4_8_16.json
```

It converts the all-run endpoint tape into discrete
`(owner, left_cut_id, right_cut_id)` records and checks the frame-to-frame edit
stream. The records match the continuous endpoint-run tape counts and owners.
At 16 frames, the full endpoint-record CSR is `0.111x` full segment CSR, and
the endpoint record count still scales `8.61x` for an `8x` frame-count
increase. The important new signal is the delta stream: endpoint-record edit
ops scale only `1.87x`, edit-op delta storage scales about `1.55x`, and the
16-frame edit-op stream is about `0.026x` full segment CSR (`0.23x` of full
endpoint-record CSR).

The owner+cut-id replacement-row replay shader is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replay_render32_2_4_8_16.json
```

This is the first real Metal replay shader for the endpoint-record delta lane.
It stores first-frame endpoint rows plus changed `(owner, left_cut_id,
right_cut_id)` rows, recovers endpoint depths from boundary ids and moving rays
inside the shader, and matches endpoint-run replay:

- max forward error versus endpoint-run replay: `8.94e-7`
- max VJP relative error versus endpoint-run replay: `2.82e-6`
- replacement-row storage scale from 2 to 16 frames: `1.87x` for an `8x`
  frame-count increase
- 16f replacement-row storage: `0.235x` full endpoint CSR, `0.0261x` full
  segment CSR

Interpretation: this ships the first endpoint-record delta replay shader and
keeps the STAR-port-shaped storage signal real. It is still replacement-row
replay, not the smaller edit-op stream, and it is not integrated into the main
trainer.

The owner+cut-id edit-op replay shader is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_render32_2_4_8_16.json
```

The current cut-depth-cache rerun is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_cutcache_render32_2_4_8_16.json
```

It replays endpoint rows from first-frame records plus insert/delete/replace
ops instead of full replacement rows. It also recovers endpoint depths from
boundary ids and moving rays inside the shader and matches endpoint-run replay:

- max forward error versus endpoint-run replay: `8.94e-7`
- max VJP relative error versus endpoint-run replay: `6.91e-6`
- edit-op count scale from 2 to 16 frames: `1.87x` for an `8x` frame-count
  increase
- edit-op storage scale from 2 to 16 frames: `1.53x`
- 16f edit-op storage: `0.235x` full endpoint CSR, `0.0261x` full segment CSR
- 16f timing: `3.29 ms` edit forward versus `2.01 ms` endpoint forward, and
  `3.00 ms` edit VJP versus `2.64 ms` endpoint VJP

Interpretation: the compact endpoint-record representation is sublinear in
practice, not just in theory. The current edit-op shader is still a storage
win rather than a speed win; it reconstructs rows in-kernel, and the paired
train/eval check below is still slower than endpoint-run.

The matched-parameter endpoint-record edit RGB autograd train/eval probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --tape-mode endpoint-record-edit \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Current result:

- status: `ok`
- total step scale 2f->16f: `0.70x` for `8x` frames on this smoke-scale run
- 16f total step: `6.18 ms`
- 16f render: `2.59 ms`
- 16f backward: `2.76 ms`
- 16f heldout PSNR: `13.273873589186554`
- edit-op count scale 2f->16f: `1.87x`
- selected edit storage scale 2f->16f: `1.53x`
- 16f selected storage ratio versus full segment tape: `0.0261x`
- 16f selected storage ratio versus endpoint-run CSR: `0.235x`

Interpretation: the endpoint-record edit shader now runs through a real
fixed-geometry site-RGBA autograd train/eval path with matched PSNR and compact
storage. The timing should be treated as smoke-scale and noisy; this is still
not main-trainer integration or a STAR-UVT competitive quality/capacity claim.

A paired same-process endpoint-run versus endpoint-record edit train/eval
comparison is also saved:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_cutcache_render32_2_4_8_16.json
```

At 16 frames in that sidecar run, endpoint-record edit keeps matched PSNR and
cuts storage (`0.0261x` full segment storage versus endpoint-run `0.111x`), but
is slower than endpoint-run (`7.93 ms` versus `7.47 ms`, `1.06x`). Treat this
as the current speed sanity check: edit-op replay is a storage win that still
needs replay optimization before a runtime-competitive claim.

The RGB-only endpoint-record edit VJP sidecar is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_rgbonly_render32_2_4_8_16.json
```

It adds a dedicated RGB-loss VJP kernel for the edit-op path. The sidecar
matches the full edit VJP with zero alpha/depth adjoints:

- max RGB-only VJP relative error versus full zero-alpha/depth VJP: `2.86e-6`
- max full VJP relative error versus endpoint-run replay: `5.00e-6`
- 16f edit storage: `0.235x` endpoint CSR, `0.0261x` full segment CSR
- 16f isolated timing: endpoint VJP `2.16 ms`, full edit VJP `2.63 ms`,
  RGB-only edit VJP `2.85 ms`

The paired RGB-only train/eval repeat is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_rgbonly_repeat12_render32_2_4_8_16.json
```

Latest 12-step repeat at 16f:

- endpoint-run total/render/backward `5.07 ms` / `1.57 ms` / `2.77 ms`
- endpoint-record-edit total/render/backward `6.74 ms` / `2.35 ms` /
  `3.56 ms`
- edit/endpoint total ratio `1.33x`
- heldout PSNR matches within `5e-7`

Interpretation update: RGB-only VJP is correct and scoped, but it did not turn
the edit-op path into a stable runtime win. Short smoke timings have flipped
sign across reruns, so the status summary keeps this as a correctness/storage
sidecar rather than a speed or STAR-UVT competitiveness claim.

The manual-VJP paired repeat is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_manualvjp_repeat12_render32_2_4_8_16.json
```

Manual VJP removes most autograd-wrapper ambiguity. At 16f:

- endpoint-run total/render/backward `4.57 ms` / `1.40 ms` / `2.46 ms`
- endpoint-record-edit total/render/backward `5.32 ms` / `2.26 ms` /
  `2.43 ms`
- edit/endpoint total ratio `1.16x`
- edit/endpoint backward ratio `0.99x`
- edit/endpoint render ratio `1.61x`

Interpretation update: the VJP side is no longer the main gap in manual mode.
The remaining speed problem is forward row reconstruction/replay, not just
autograd dispatch or RGB-only gradient math.

The track-loop endpoint-record edit forward sidecar is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_trackloop_render32_2_4_8_16.json
```

It adds a forward-only Metal kernel that launches one thread per track and
walks frames in order while applying edit ops incrementally. It is numerically
correct:

- max track-loop forward absolute error versus endpoint-run: `8.94e-7`
- edit ops/storage remain sublinear from 2f to 16f: `1.87x` ops and `1.53x`
  storage for `8x` frames
- 16f storage remains `0.235x` endpoint CSR and `0.0261x` full segment CSR

But it is not the speed fix. At 16f, endpoint-run forward is `1.18 ms`, the
existing edit replay forward is `1.96 ms`, and track-loop forward is
`2.15 ms`. The likely issue is reduced GPU parallelism: it amortizes row edit
application, but launches far fewer independent threads. The status summary
therefore records this as a correct rejected forward-optimization sidecar, not
a STAR-UVT competitive result.

The block4 anchored endpoint-record edit forward sidecar is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block4_vjp_render32_2_4_8_16.json
```

It stores endpoint-record anchor rows every four frames and replays only edit
ops inside the current block while keeping the one-thread-per-sample launch
shape. In the refreshed raw probe it is a correct compact replay layout and
beats the original edit replay, but it is near rather than faster than the
endpoint-run raw forward:

- max block4 forward absolute error versus endpoint-run: `8.94e-7`
- max block4 RGB-only VJP relative error versus full VJP with zero alpha/depth:
  `2.93e-6`
- max block4 RGB-only VJP relative error versus edit RGB-only VJP: `2.86e-6`
- 16f block4 forward: `1.72 ms`
- 16f endpoint-run forward: `1.69 ms`
- 16f original edit forward: `2.91 ms`
- 16f track-loop forward: `2.68 ms`
- 16f framegroup16 forward: `6.18 ms`
- 16f block4 RGB-only VJP: `2.84 ms`
- 16f edit RGB-only VJP: `3.30 ms`
- 16f block4 storage: `0.395x` endpoint CSR and `0.0438x` full segment CSR

Interpretation: block anchoring ports the STAR-style "move time work out of the
sample hot path" idea better than track-loop or threadgroup frame materializing,
and the dedicated RGB-only VJP validates the backward math. The raw forward is
not currently a clear endpoint-run speed win.

The block-anchor shader is now parameterized by `--edit-block-size` for
follow-up sweeps. The op names still say `block4` for historical continuity,
but the Python/C++ checks now accept any positive block size and pass it through
to the existing Metal config. Cheap render16/2f MPS smokes verify correctness
for both `--edit-block-size 2` and `--edit-block-size 8`:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block2_vjp_smoke_render16_2f.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block8_vjp_smoke_render16_2f.json
```

Those single-frame-count smokes are correctness gates only; they are not speed
or sublinear-scale claims. A reduced render16 2/4/8 sweep now exists for
block sizes 2/4/8:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block2_vjp_smoke_render16_2_4_8.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block4_vjp_smoke_render16_2_4_8.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block8_vjp_smoke_render16_2_4_8.json
```

All three pass forward/VJP correctness and sublinear storage checks over 2/4/8,
but none is a speed win in that one-iteration smoke. At 8f, block2/block4/block8
forward timings are `10.67/11.72/9.11 ms` versus endpoint-run
`4.06/4.27/5.64 ms` in their respective runs. Treat this as a negative
block-size speed screen, not a final benchmark.

The next sidecar caches per-track/per-boundary depth coefficients and keeps the
same block edit topology stream:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block_coeff_smoke_render16_2_4_8.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block_coeff_render16_16f.json
```

It is numerically correct versus endpoint-run. After the near/far contract fix
and a sequential 16f render16 rerun, coefficient-cached forward is `6.00 ms`
versus endpoint-run `6.26 ms`, block4 boundary replay `4.74 ms`, and original
edit replay `6.31 ms`. The coefficient RGB-only VJP is also numerically correct
(`3.75e-6` relative error versus full VJP with zero alpha/depth, `1.93e-6`
versus edit RGB-only VJP, and `2.61e-6` versus block4 RGB-only VJP), with saved
16f VJP timing `3.06 ms`. The tradeoff is storage: block+coefficient storage
is `1.68x` endpoint CSR at 16f, though still `0.185x` full segment CSR. This is
a forward speed-positive, storage-heavy sidecar, not a block4 win or
STAR-competitive result.

The promoted coefficient-cached RGB train/eval scale sweep is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_repeat20_render32_2_4_8_16.json
```

It uses the coefficient-cached forward and RGB-only VJP through the autograd
wrapper. The 2/4/8/16 render32 run uses `20` measured steps and `5` warmup
steps. It is green: gradients are nonzero, parameters update, losses decrease,
and heldout PSNR records across frame counts. Measured total-step timings are
`12.23/8.15/6.39/6.84 ms`; the 2f-to-16f measured total scale is `0.56x` for
an `8x` frame-count increase. Render scales `0.51x`, backward scales `0.66x`,
and selected coefficient tape storage scales `1.17x`. The 16f
total/render/backward timings are `6.84/1.84/3.89 ms`, 16f heldout PSNR is
`14.5922`, and 16f selected tape storage is `0.181x` full segment CSR.

Important storage nuance: the fixed coefficient table is larger than full CSR
at tiny 2f (`1.243x` full), then amortizes below full by 4f and reaches
`0.181x` at 16f. Treat the saved artifact as trainability and warmed scaling
evidence for the coefficient sidecar; it is still an MPS smoke-scale run, not a
full stable benchmark, main-trainer integration, geometry-gradient path, or
STAR-UVT competitive quality/capacity claim.

The paired same-process endpoint-run/edit/block4/block-coeff train/eval smoke
is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_warm5_render32_2_4_8_16.json
```

At 16 frames, all four modes match heldout PSNR at `13.2739`. Total/render/
backward timings are:

- endpoint-run: `13.04 / 3.79 / 5.91 ms`
- raw endpoint-record edit: `11.28 / 3.54 / 4.90 ms`
- block4 endpoint-record edit: `9.09 / 2.43 / 4.97 ms`
- block-coeff endpoint-record edit: `8.06 / 2.47 / 3.63 ms`

Interpretation: in this promoted paired current-process smoke, raw edit is
faster than endpoint-run (`0.865x` total), block4 is faster (`0.698x` total),
and block-coeff is fastest (`0.618x` versus endpoint-run and `0.886x` versus
block4). That is the best practical speed sign so far for the coefficient
path in the 5-step sweep, but the 20-step repeat below is the stronger 16f
stability check. It is still not the clean STAR UVT story: block-coeff storage
is `0.181x` full segment CSR and still above endpoint CSR, the runs are
smoke-scale MPS results, and none of this is integrated into the main trainer or
matched to STAR quality/capacity.

A focused 16f-only render32 repeat with `20` measured steps and `5` warmup steps
checks the sign with a longer sample:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_16f.json
```

That repeat keeps heldout PSNR matched across modes (`14.5922`), block-coeff
fastest, and block4 slightly faster than endpoint-run, but raw edit flips back
to slower than endpoint-run:

- endpoint-run: `9.42 / 2.91 / 4.57 ms`
- raw endpoint-record edit: `11.31 / 4.20 / 5.62 ms`
- block4 endpoint-record edit: `9.19 / 2.69 / 4.66 ms`
- block-coeff endpoint-record edit: `7.48 / 2.20 / 3.78 ms`

Interpretation: block anchoring is the robust practical speed improvement. Raw
edit remains a compact storage path with noisy speed sign, while block-coeff is
the current best speed-positive sidecar at 16f despite heavier storage
(`0.794x` endpoint-run total, `0.814x` block4 total). This repeat is not
frame-count scaling evidence by itself; keep the 20-step render32 2/4/8/16
coefficient-cached sweep as the standalone coefficient path scaling artifact.

A longer same-process paired 2/4/8/16 repeat now exists:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_2_4_8_16.json
```

This artifact is negative/informational. It exited with status `failed` because
the f32 block-coeff path failed the 16f speed gate. At 16 frames, PSNR still
matches across modes (`14.5922`), but total/render/backward timings are:

- endpoint-run: `43.11 / 15.82 / 18.23 ms`
- raw endpoint-record edit: `25.44 / 10.71 / 11.29 ms`
- block4 endpoint-record edit: `11.08 / 3.04 / 5.76 ms`
- block-coeff endpoint-record edit: `71.50 / 17.85 / 27.85 ms`

Interpretation: the long paired 2/4/8/16 run makes the practical claim more
conservative, not stronger. Block4 was the clear 16f winner in this artifact,
raw edit beat endpoint-run at 16f, and block-coeff regressed to `1.66x`
endpoint-run and `6.45x` block4. Keep the standalone block-coeff sweep as
evidence that the coefficient path can scale, but do not promote block-coeff as
a stable paired speed win until this failure is understood.

A f16 coefficient-cache sidecar also exists:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff16_manualvjp_smoke_render32_16f.json
```

It runs a 16f render32 manual-VJP paired smoke with endpoint-run, raw edit,
block4, f32 block-coeff, and f16 block-coeff. PSNR matches within the f16
tolerance, and f16 coefficient storage is below the f32 coefficient table. The
original paired artifact reported the f16 selected storage close to endpoint-run
storage (`0.111x` versus `0.181x` full CSR for f32), which exposed a harness
accounting bug: the coeff16 selected-storage branch was unreachable.

A follow-up storage-accounting smoke after the fix is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff16_manualvjp_storagefix_smoke_render16_16f.json
```

That real manual-VJP train/eval path now reports selected f16 storage as
`0.1137x` full CSR, endpoint-run storage as `0.1103x`, and block4 storage as
`0.0423x`; the selected storage is no longer accidentally counted as
endpoint-run storage. The speed result is still negative:
block-coeff16 is `7.67 ms` total, slower than endpoint-run (`6.40 ms`) and
slower than f32 block-coeff (`4.55 ms`). Treat it as a recorded negative
storage experiment, not a promoted speed path.

The cached-clear fused-MSE path has a stricter real-loaded 32-frame smoke:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_blockcoeff_rgb_cachedclear_fused_mse_real32_warm3_steps8_render16_16_32.json
```

Those runs use the existing real 32-frame multicam config and do not pass
`--repeat-loaded-frames`; synthetic moving rays remain active. The standalone
fused-MSE run reports `3.455 ms` at loaded 16f and `2.619 ms` at loaded 32f
(`0.758x` total-step scale for `2x` frames). In the paired current-process
render16/site4 compare, 16f total times are endpoint-run `4.298 ms`,
block-coeff-rgb `3.443 ms`, and block-coeff-fused-mse `1.927 ms`; 32f total
times are endpoint-run `3.936 ms`, block-coeff-rgb `3.703 ms`, and
block-coeff-fused-mse `2.570 ms`. Heldout PSNR matches across the paired modes.

Interpretation: this is now real-frame practical sublinear/speed evidence for
the fused-loss hot loop, not only a repeated-loaded artifact. It still is not a
structural STAR-style topology proof: endpoint-run selected segments scale
`1.993x` for `2x` frames, and the result remains fixed-geometry/site-RGBA,
render16/site4, smoke-scale MPS evidence rather than main-trainer or matched
STAR UVT quality/capacity integration.

The raw endpoint-record edit fused-MSE fork keeps the compact edit tape and
fuses RGB loss+VJP directly over that representation:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_rawfused_blockcoeff_fused_mse_real32_warm3_steps8_render16_16_32.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

On the real-loaded 16/32-frame gate, standalone raw-edit fused-MSE reports
`2.228 ms` at 16f and `2.206 ms` at 32f (`0.990x` total-step scale for `2x`
frames). Backward/fused-loss VJP scales `1.019x`, selected edit storage scales
`1.009x`, and edit-op count scales `1.014x`. In the paired current-process
compare, 16f totals are endpoint-run `4.590 ms`, raw edit `4.728 ms`,
raw-edit-fused-MSE `2.600 ms`, block-coeff-rgb `4.261 ms`, and
block-coeff-fused-MSE `1.943 ms`; 32f totals are endpoint-run `3.755 ms`,
raw edit `4.317 ms`, raw-edit-fused-MSE `2.519 ms`, block-coeff-rgb
`3.703 ms`, and block-coeff-fused-MSE `2.412 ms`. Heldout PSNR matches across
the paired modes.

The repeated-loaded 16/32/64/128 smoke reports raw-edit fused-MSE totals
`3.134/2.628/3.018/3.956 ms`: `1.262x` total-step scale and `1.563x`
fused/backward scale for an `8x` frame-count increase. The storage side is the
cleaner STAR-shaped result: selected edit storage is effectively flat
(`0.999x`) and edit ops are effectively flat (`0.996x`). Interpretation: this
fork finally makes the compact raw edit representation speed-positive while
preserving the sublinear storage signal. It still does not beat block-coeff
fused at 16f, and it is still a fixed-geometry/site-RGBA smoke-scale result
rather than main-trainer STAR-UVT competitiveness.

For longer paired comparisons, pass `--partial-out-json`. The compare harness
now writes a top-level partial after each completed mode and also gives each
mode its own row-level partial file, so a render32 run is no longer progress
blind:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --steps 5 \
  --warmup-steps 1 \
  --include-block4 \
  --include-block-coeff \
  --edit-block-size 4 \
  --partial-out-json research_experiments/world_foam_lane2/results/paired.partial.json \
  --out-json research_experiments/world_foam_lane2/results/paired.json
```

The block4 fixed-geometry RGB train/eval artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block4_rgb_train_eval_autograd_block4vjp_repeat12_render32_2_4_8_16.json
```

It uses block4 forward replay and the dedicated block4 RGB-only VJP inside the
autograd wrapper. In the corrected standalone 2/4/8/16 repeat, total step scales
`3.01x`, render `4.11x`, and backward `2.73x` for an `8x` frame-count increase;
16f total/render/backward are `75.18/30.63/32.92 ms`. Storage remains compact at
`0.395x` endpoint CSR and `0.0438x` full segment CSR, and final heldout PSNR is
`14.34`.

Interpretation: this moves block4 from forward-only to isolated fixed-geometry
RGB train/eval with its own VJP. The result is runtime-sublinear across frame
count, but it is not speed competitive in this corrected rerun. It still is not
a main-trainer integration, geometry-gradient path, or matched STAR-UVT
competitive claim.

The matched-parameter endpoint-run RGB autograd train/eval probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --tape-mode endpoint-run \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Current result:

- status: `ok`
- total step scale 2f->16f: `1.26x` for `8x` frames
- 16f total step: `7.84 ms`
- 16f render: `2.29 ms`
- 16f backward: `4.22 ms`
- 16f heldout PSNR: `13.273873589186554`
- 16f selected storage ratio versus full segment tape: `0.111x`
- 16f selected segment ratio versus full segment tape: `0.103x`

Endpoint-run train/eval is faster than the current fused winner (`7.84 ms` vs
`9.32 ms`) and active-internal (`7.84 ms` vs `8.67 ms`) while matching PSNR, but
it remains slower than the current-density owner-run shortcut (`6.04 ms`). It
is the compact density-independent semantic-change path, not a current-depth
drop-in and not a STAR-style structural win.

The matched-parameter active-internal RGB autograd train/eval probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_active_internal_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --tape-mode active-internal \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_active_internal_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Current result:

- status: `ok`
- total step scale 2f->16f: `1.60x` for `8x` frames
- 16f total step: `8.67 ms`
- 16f render: `2.38 ms`
- 16f backward: `5.18 ms`
- 16f heldout PSNR: `13.273993928035445`
- 16f selected storage ratio versus full segment tape: `0.170x`
- 16f selected segment ratio versus full segment tape: `0.163x`

Compared with the current fused `direct_atomic_grad_only` winner, active-internal
train/eval is slightly faster at 16f (`8.67 ms` vs `9.32 ms`) with matched PSNR.
Compared with owner-run train/eval, it is slower (`8.67 ms` vs `6.04 ms`) but
keeps exact current-density segment-mid depth semantics. Its selected segment
count still scales `8.73x` for `8x` frames, so this is practical evidence, not
structural STAR-style sublinearity.

The matched-parameter full segment-tape RGB autograd train/eval probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_full_segment_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --tape-mode full \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_full_segment_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Current result:

- status: `ok`
- total step scale 2f->16f: `1.82x` for `8x` frames
- 16f total step: `9.79 ms`
- 16f render: `2.74 ms`
- 16f backward: `5.70 ms`
- 16f heldout PSNR: `13.273979487197515`
- 16f selected storage ratio versus full segment tape: `1.0x`

This is the exact fixed-geometry density-independent replay cost baseline. It
matches PSNR but is slower than owner-run and active-internal, slightly slower
than the current fused winner at 16f, and is not compact by definition.

The matched-parameter owner-run RGB autograd train/eval probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_owner_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --near 0.1 \
  --far 6.0 \
  --density 10.0 \
  --invalid-epsilon 1.0e-6 \
  --transmittance-threshold 1.0e-4 \
  --optimizer-mode autograd \
  --segment-tape-vjp-mode direct_atomic_grad_only \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Current result:

- status: `ok`
- optimizer mode: `autograd`
- total step scaling: `5.40 ms` at 2 frames to `6.04 ms` at 16 frames
- total scale 2->16: `1.12x` for an `8x` frame-count increase
- render scale 2->16: `0.89x`
- backward scale 2->16: `1.40x`
- 16-frame heldout PSNR: `13.274`, matching the fused train/eval winner within
  verifier tolerance
- 16-frame train owner-run segments: `62968` versus `1301934` full segments
  (`0.048x`)
- 16-frame train owner-run storage: `0.056x` of full segment tape

Interpretation: this is the first practical evidence that the owner-run tape can
carry the RGB/site-RGBA objective through a normal PyTorch autograd/Adam path
across the 2/4/8/16 moving-camera sweep while improving the saved 16-frame
total step time. It is still an isolated owner-run script, not integrated into
the main fused-slab trainer, and it does not prove density-independent depth or
geometry gradients.

The segment-tape autograd smoke is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_segment_tape_autograd_smoke_render16_2f.json
```

Run it from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/smoke_segment_tape_autograd_mps.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_segment_tape_autograd_smoke_render16_2f.json
```

Current result:

- status: `ok`
- modes checked: `direct_atomic_grad_only`, `direct_atomic_track`
- max autograd-gradient relative error versus explicit Metal VJP: about
  `4.0e-7`
- owner-run segments: `1477` versus `38728` full segments (`0.038x`)

Interpretation: the compact segment-tape replay now has a normal PyTorch
autograd wrapper for frozen-geometry site-RGBA training. This reduces the
integration gap versus the manual owner-run optimizer, but it is still not full
trainer integration or a geometry-gradient proof.

The main gate artifact is `gate0_beam_toy.py`: it matches the handoff's
shared-metric power-cell formulation and measures whether a screen-time slab can
reuse boundary events across multiple frames.

`beam_events_reference.py` is an auxiliary moving-disk oracle for simple sorted
enter/exit event traces. It is useful for sanity-checking traversal/event-stream
plumbing, but it is not the World Foam power-cell model.

## Fixed-Step Speed Compare

`fixed_step_speed_compare.py` runs the cleaner fixed-step speed baseline for
STAR-UVT, free dynamic GSplats, and World Foam. It generates per-case
manifests/configs so requested frame counts are exact, then reports measured
optimizer-step timings for the default matrix `128x8,128x16,128x32,256x32`.

From the repo root:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --steps 8 \
  --warmup-steps 2 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/fixed_step_speed_compare_default.json
```

Scope boundary: STAR-UVT and dynamic GSplats default to one train camera's full
sequence per optimizer step. World Foam still renders all train-camera rays and
is fixed geometry/site-RGBA only, so this is speed accounting, not full trainer
parity.

## Main Power-Cell Sweep

From the repo root:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/gate0_beam_toy.py --json
python3 dynaworld/research_experiments/world_foam_lane2/gate0_event_sharing_benchmark.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_event_sharing.json
```

The default scene uses sites in `(x, z, t)`, orthographic camera rays
`X(s,t) = (u + v_camera * t, s, t)`, and shared-metric power-cell boundaries.
It compares per-frame sliced boundary events against one native screen-time
beam slab over frame counts `2,4,8,16`.

Key fields:

- `per_frame_event_sum`: summed event count if every frame is traversed
  independently.
- `beam_slab_event_sum`: event count for the shared screen-time slab.
- `event_sharing_ratio`: `beam_slab_event_sum / per_frame_event_sum`.
- `missing_sample_events`: sample events not covered by the slab candidates;
  must stay zero for the toy to be useful.
- `growth.sublinear_event_growth`: whether beam events grew slower than
  per-frame events across the frame-count sweep.

`gate0_event_sharing_benchmark.py` wraps the same toy in a benchmark-shaped JSON
payload with explicit placeholders for the later STAR-UVT and dynamic-splat
matched rows. By default it runs two camera-velocity sweeps, `0.35` and `0.7`,
and requires zero missing sample events plus sublinear beam-event growth in both.
It also reports `backward_status:
event_replay_accounting_only_no_gradients`; this is only a check that the same
slab event list can be reused in a backward replay accounting model, not a
gradient implementation.

Current Gate 0 status:

- CPU event sharing passes on the toy sweeps: both camera velocities keep
  `missing_sample_events=0` and sublinear beam-event growth.
- The 16-frame summary row has `per_frame_event_sum=2266`,
  `beam_slab_event_sum=149`, and `event_sharing_ratio=0.06575463371579876`.
- `gate0_event_sharing_benchmark.py` backward rows remain accounting-only. The
  separate Gate 0.5 reference below adds CPU signal-gradient evidence, but there
  is still no Metal backward pass or trainable renderer.

## Gate 0.5 Shared Forward/Backward Reference

`gate0_shared_forward_backward.py` adds a CPU-only reference for one narrow
backward claim: site-signal gradients can reuse the shared forward segmentation
tape instead of re-enumerating all power boundaries per frame.

From the repo root:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/gate0_shared_forward_backward.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_shared_forward_backward.json
python3 -m unittest discover -s dynaworld/research_experiments/world_foam_lane2
```

Current Gate 0.5 status:

- direct per-frame output and shared-slab output match exactly on the toy:
  `max_output_abs_error=0.0`.
- direct signal gradients and shared replay signal gradients match exactly:
  `signal_gradient_max_abs_error=0.0`.
- finite-difference check matches the shared analytic gradient:
  `finite_difference_max_abs_error=1.3797318842989625e-10`.
- 16-frame forward+backward boundary-scan ratio is `0.03125`: direct reference
  scans `2720` forward boundaries and `2720` backward boundaries, while shared
  replay scans `170` forward boundaries and `0` backward boundaries.

Scope boundary: this is only a CPU reference for site-signal gradients through
fixed segments. It is not a Metal backward pass, image renderer, site-position
gradient, weight gradient, topology/sorting-gradient treatment, or heldout
quality metric.

## Gate 0.6 MPS Shared Replay Op

The isolated Metal variant now has a narrow Torch/MPS shared-replay op that
replays the Gate 0.5 shared slab candidates on device and emits forward scalar
signals plus per-site signal-gradient samples.

Build and smoke it from the repo root:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_shared_replay_mps.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_6_mps_shared_replay_smoke.json
```

Current saved result:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_6_mps_shared_replay_smoke.json
```

The 16-frame row reports:

- `max_output_abs_error=1.1920928955078125e-07`;
- `signal_gradient_max_abs_error=7.62939453125e-06`;
- `loss_abs_error=2.5603027253850996e-06`;
- `mps_shared_replay_wall_clock_ms=1.0247604001051513` over 20 timed
  launches;
- direct forward+backward boundary scans `2720 + 2720`;
- shared forward+backward boundary scans `170 + 0`;
- shared forward+backward scan ratio `0.03125`.

Scope boundary: this is real MPS execution for fixed-segment site-signal replay,
but it is still not a renderer, compositor, trainable position-gradient path,
topology-gradient treatment, or heldout-quality result.

## Gate 0.7 MPS RGB Strip Smoke

`tools/smoke_rgb_strip_mps.py` in the isolated Metal variant adds the first
image-shaped MPS proof for this lane. It uses one shared-RGB replay op and
writes a 16-frame toy strip.

Build and smoke it from the repo root:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_rgb_strip_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip.ppm
```

Current saved artifacts:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip_smoke.json
dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip.ppm
```

The current JSON reports:

- `strip_image_shape=[16,17,3]`;
- `max_rgb_abs_error=4.76837158203125e-07`;
- `color_gradient_max_abs_error=4.57763671875e-05`;
- `loss_abs_error=1.0043974384643661e-05`;
- `mps_rgb_strip_wall_clock_ms=0.8528229001967702` over 20 timed iterations;
- `shared_forward_backward_boundary_scan_ratio=0.03125`;
- all pixels finite and RGB output nonconstant.

Scope boundary: this is a toy RGB strip/shared-replay smoke. The separate Gate
0.8 path adds forward alpha/depth compositing, but there is still no trainable
geometry-gradient path or heldout-quality comparison.

## Gate 0.8 MPS Alpha/Depth Composite Strip Smoke

`tools/smoke_composite_strip_mps.py` in the isolated Metal variant adds the
first forward-only compositor proof for this lane. It uses one shared
RGBA-depth replay op and writes a 16-frame toy strip with RGB, accumulated
alpha, and alpha-weighted expected depth.

Build and smoke it from the repo root:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_composite_strip_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip.ppm
```

Current saved artifacts:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip_smoke.json
dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip.ppm
```

The current JSON reports:

- `rgb_shape=[16,17,3]`;
- `alpha_shape=[16,17]`;
- `depth_shape=[16,17]`;
- `max_rgb_abs_error=1.7881393432617188e-07`;
- `max_alpha_abs_error=1.7881393432617188e-07`;
- `max_depth_abs_error=3.5762786865234375e-07`;
- `mps_composite_wall_clock_ms=0.8724833500309614` over 20 timed iterations;
- `shared_forward_boundary_scan_ratio=0.0625`;
- alpha is finite, nonconstant, and inside `[0,1]`; depth is finite and inside
  the configured near/far range.

Scope boundary: this is a toy forward compositor. It is not a trainable
geometry-gradient path, density-gradient path, real-video renderer, or
heldout-quality comparison.

## Gate 0.9 MPS Fixed-Segment Composite VJP Smoke

`tools/smoke_composite_vjp_mps.py` in the isolated Metal variant adds the first
backward proof for the toy compositor. It keeps the segment tape fixed and
differentiates only site RGBA values.

Build and smoke it from the repo root:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_composite_vjp_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_9_mps_composite_vjp_smoke.json
```

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_9_mps_composite_vjp_smoke.json
```

The current JSON reports:

- `max_rgb_abs_error=1.1920928955078125e-07`;
- `max_alpha_abs_error=1.7881393432617188e-07`;
- `max_depth_abs_error=3.5762786865234375e-07`;
- `max_rgba_gradient_abs_error=1.9073486328125e-06`;
- `finite_difference_max_abs_error=0.0003147125244140625`;
- `loss_abs_error=3.4033663922627966e-06`;
- `mps_composite_vjp_wall_clock_ms=1.3421896001091227` over 20 timed
  iterations.

Scope boundary: this is a fixed-segment site-RGBA VJP. Boundary cuts, segment
ownership, sorting, topology, site positions, site weights, camera projection,
and real-video image formation remain non-differentiated.

## Gate 0.95 MPS Slab-Indexed Candidate Mask Smoke

`tools/smoke_composite_vjp_slab_mask_mps.py` keeps the Gate 0.9 fixed-segment
RGBA VJP but changes the candidate-mask ABI from one mask per beam to row-major
`[beam, slab]`. The legacy length-`beam_count` shape remains the
`time_slabs=1` case.

Build and smoke it from the repo root:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_composite_vjp_slab_mask_mps.py \
  --timing-iters 20 \
  --time-slabs 1,2,4 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_95_mps_composite_vjp_slab_mask_smoke.json
```

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_95_mps_composite_vjp_slab_mask_smoke.json
```

The current JSON reports:

- `time_slabs=1`: `total_candidates=149`, scan ratio `0.0625`,
  `mps_composite_vjp_wall_clock_ms=0.6196104499395005`;
- `time_slabs=2`: `total_candidates=292`, scan ratio `0.125`,
  `mps_composite_vjp_wall_clock_ms=0.975545800247346`;
- `time_slabs=4`: `total_candidates=576`, scan ratio `0.25`,
  `mps_composite_vjp_wall_clock_ms=1.1342895497364225`;
- all rows have `max_rgba_gradient_abs_error=1.9073486328125e-06`,
  `finite_difference_max_abs_error=0.0003147125244140625`, and
  `segment_overflow_count=0`.

Scope boundary: this removes only the single-slab mask limitation. It is still
toy strip output with int32 bitmask candidates, no CSR storage, no full-frame
`(u,v,t)` rendering, no geometry/topology gradients, and no real-video heldout
metric.

## Paired Report Wrapper

The repo-level benchmark wrapper normalizes World Foam Gate 0 rows with optional
STAR-UVT and dynamic-splat rows:

```bash
python3 dynaworld/src/benchmarks/world_foam_gate0_paired_benchmark.py \
  --star-comparison-json \
  dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/smoke_tile_load_reg_mps_64_4f/comparison_report.json \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_paired_with_star_dynamic_smoke.json
```

That report is a routing and normalization artifact. It does not make World
Foam a comparable image-quality renderer yet; `comparison_unit` stays explicit
for each row.

The current paired smoke artifact is:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_paired_with_star_dynamic_smoke.json
```

It contains:

- World Foam Gate 0 rows for per-frame and shared-beam power-boundary events.
- World Foam Gate 0.5 CPU site-signal gradient reference rows.
- World Foam Gate 0.6 MPS shared-replay rows when
  `gate0_6_mps_shared_replay_smoke.json` exists.
- World Foam Gate 0.7 MPS RGB strip rows when
  `gate0_7_mps_rgb_strip_smoke.json` exists.
- World Foam Gate 0.8 MPS alpha/depth composite rows when
  `gate0_8_mps_composite_strip_smoke.json` exists.
- World Foam Gate 0.9 MPS fixed-segment compositor VJP rows when
  `gate0_9_mps_composite_vjp_smoke.json` exists.
- World Foam Gate 0.95 MPS slab-indexed fixed-segment compositor VJP rows when
  `gate0_95_mps_composite_vjp_slab_mask_smoke.json` exists.
- World Foam Gate 1 image-shaped toy full-frame VJP rows when
  `gate1_mps_full_frame_vjp_smoke.json` exists.
- World Foam Gate 1B/1C/2/2B/2C/2D real-ray rows when the corresponding
  real-ray reference, MPS replay, shared-forward, materialized-VJP, and
  reduced-VJP JSON artifacts exist.
- World Foam Gate 2F CSR/tiled candidate-storage rows when
  `gate2f_mps_shared_realray_csr_candidate_storage_smoke.json` exists.
- World Foam Gate 2G CSR/tiled frame-scaling rows when
  `gate2g_mps_shared_realray_csr_scaling_smoke.json` exists.
- World Foam Gate 2E teacher-target autograd-overfit rows when
  `gate2e_mps_shared_realray_autograd_overfit_smoke.json` exists.
- World Foam Gate 3 frozen-geometry real-target training-smoke rows when
  `gate3_mps_shared_realray_real_target_train_smoke.json` exists.
- World Foam Gate 3 CSR 256px/16f quality rows when
  `gate3_mps_shared_realray_csr_quality_256px_16f.json` exists.
- A STAR-UVT smoke row sourced from
  `variants/star_uvt_v0/.../smoke_tile_load_reg_mps_64_4f/comparison_report.json`.
- A dynamic-splat smoke row sourced from the same STAR comparison report's
  `free_dynamic_splats` section.

The STAR and dynamic rows are smoke/routing context, not a quality or speed win
claim for World Foam.

The stronger local heldout pilot artifact is:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_9_paired_with_star_dynamic_heldout_pilot.json
```

It cites the 256px/16-frame STAR-UVT versus dynamic-splat comparison report and
preserves STAR heldout PSNR, selected-checkpoint heldout PSNR, dynamic-splat
heldout PSNR, render timings, the Gate 2F/2G CSR candidate-storage smokes, and
the Gate 3 CSR 256px/16f World Foam fixed-geometry heldout metric row. The
World Foam quality row is comparable as a same-split metric artifact, but it is
still not a full trainer or geometry/topology-gradient implementation.

## Gate 1 Real-Data Feeder Smoke

`gate1_realdata_feeder_smoke.py` proves the lane can load the same DeepView
multicam data and camera rays used by the PowerFoam and STAR comparison
harnesses before a full-frame World Foam shader exists.

Run from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/research_experiments/world_foam_lane2/gate1_realdata_feeder_smoke.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate1_realdata_feeder_smoke.json
```

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate1_realdata_feeder_smoke.json
```

The current JSON reports:

- sample: `deepview_03_Dog_camera_0001_to_camera_0040`;
- train views: `camera_0001`, `camera_0015`;
- heldout view: `camera_0040`;
- pose source: `deepview_models_relative_opencv_fisheye`;
- train targets shape: `[4,3,32,32]`;
- train sample rays shape: `[4,32,32,6]`;
- heldout targets shape: `[2,3,32,32]`;
- heldout sample rays shape: `[2,32,32,6]`;
- all target/ray tensors are finite, ray directions are nonzero, and train
  view rays differ.

Scope boundary: this is a feeder smoke only. The JSON intentionally says
`world_foam_renderer_status=not_connected_full_frame_shader_missing_u_v_t_image_op`.

## Gate 1A Toy Full-Frame-Shaped MPS VJP Smoke

`tools/smoke_full_frame_vjp_mps.py` in the isolated Metal variant proves the
existing fixed-segment RGBA/depth VJP can run over an image-shaped batch. It
flattens `H*W` pixels into beam rows, uses slab-indexed candidate masks, and
reshapes outputs back to full-frame sequences.

Run from the repo root after building the local extension:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_full_frame_vjp_mps.py \
  --height 8 --width 9 --frames 8 --time-slabs 2 --timing-iters 10 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_full_frame_vjp_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_full_frame_vjp.ppm
```

Current saved artifacts:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_full_frame_vjp_smoke.json
dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_full_frame_vjp.ppm
```

The current JSON reports:

- RGB shape: `[8,8,9,3]`;
- alpha/depth shapes: `[8,8,9]`;
- pixel-ray count: `576`;
- candidate mask shape: `[72,2]`;
- total candidates: `1212`;
- `max_rgb_abs_error=1.7881393432617188e-07`;
- `max_alpha_abs_error=1.7881393432617188e-07`;
- `max_depth_abs_error=3.5762786865234375e-07`;
- `max_rgba_gradient_abs_error=7.62939453125e-06`;
- `finite_difference_max_abs_error=0.001068115234375`;
- `mps_full_frame_vjp_wall_clock_ms=2.1561333000136074`;
- shared forward boundary-scan ratio `0.25`.

Scope boundary: this is a toy full-frame image shape only. It reuses the
existing `u/t` replay op with synthetic row variation and explicitly reports
`world_foam_renderer_status=toy_full_frame_image_shape_only_existing_u_t_replay_op_no_true_u_v_t_camera_rays`.
It is not true `(u,v,t)` camera-ray rendering, does not consume the real-data
feeder rays, and is not a heldout-quality metric.

## Gate 1B CPU Real-Ray Per-Sample Reference

`gate1_realray_per_sample_reference.py` consumes the real train and heldout
camera rays from `load_powerfoam_training_data`, initializes a deterministic
12-site 4D power-cell scene from train rays/colors, and renders CPU
RGB/alpha/depth outputs by scanning all 4D power boundaries per ray.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate1_realray_per_sample_reference.json
```

The current JSON reports train shape `[4,3,32,32]`, heldout shape
`[2,3,32,32]`, `site_count=12`, `boundary_count=66`, train linear boundary
scans `270336`, heldout linear boundary scans `135168`, train target PSNR
`12.085450317610754`, and heldout target PSNR `11.926894222356951`.

Scope boundary: this is a CPU forward-only real-camera-ray reference with
`quality_claim=false`. It is linear per sample and has no Metal, no temporal
sharing, no backward pass, and no training.

## Gate 1C MPS Real-Ray Forward Smoke

`tools/smoke_realray_replay_mps.py` in the isolated Metal variant adds the first
true camera-ray MPS forward op. The op consumes flattened `[origin, direction]`
rays, per-ray normalized time, `[x,y,z,t,weight]` sites, and
`[nx,ny,nz,nt,b]` 4D power boundaries.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_realray_replay_smoke.json
```

The current JSON reports:

- train RGB shape `[4,3,32,32]`, heldout RGB shape `[2,3,32,32]`;
- train max RGB/alpha/depth errors versus CPU:
  `3.5762786865234375e-07`, `4.172325134277344e-07`,
  `3.5762786865234375e-07`;
- heldout max RGB/alpha/depth errors versus CPU:
  `2.384185791015625e-07`, `3.5762786865234375e-07`,
  `3.5762786865234375e-07`;
- train MPS replay wall clock `1.6542249999474734` ms over 10 launches;
- heldout MPS replay wall clock `1.208595799835166` ms over 10 launches.

Scope boundary: this is the true-ray Metal forward baseline, not the World Foam
sharing result. It is still linear per sample, forward-only, and untrained.

## Gate 2 CPU Real-Ray Event Sharing

`gate2_realray_event_sharing.py` tests whether the true camera rays from Gate 1
can share 4D power-boundary candidate events across time slabs before a shared
real-ray Metal compositor exists.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2_realray_event_sharing.json
```

The current JSON reports:

- frame counts `2,4,8` at render size `16`;
- train views `camera_0001` and `camera_0015`;
- real rays are static within each view over time;
- 8-frame per-frame event sum `149341`;
- 8-frame shared slab event sum `21041`;
- 8-frame event-sharing ratio `0.1408923202603438`;
- 8-frame shared forward boundary-scan ratio `0.125`;
- zero missing sample events in every row;
- per-frame event growth `4.017999354283255x` versus shared event growth
  `0.9842821724283108x` from 2 to 8 frames.

Scope boundary: this is real-ray CPU candidate-event sharing only. Gate 2B adds
the first shared real-ray Metal forward compositor; this CPU gate is still not
backward sharing and not training.

## Gate 2B MPS Shared Real-Ray Forward Smoke

`tools/smoke_shared_realray_replay_mps.py` in the isolated Metal variant adds
the first true camera-ray MPS compositor that replays shared time-slab
candidates. It keeps one bitset candidate row per `(pixel track, time slab)`,
with enough int32-backed words to cover the current 66 real-ray boundaries.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2_mps_shared_realray_forward_smoke.json
```

The current JSON reports:

- train RGB shape `[4,3,32,32]`, heldout RGB shape `[2,3,32,32]`;
- candidate mask shapes train `[2048,3]`, heldout `[1024,3]`;
- train max RGB/alpha/depth errors versus direct CPU:
  `3.5762786865234375e-07`, `4.172325134277344e-07`,
  `3.5762786865234375e-07`;
- heldout max RGB/alpha/depth errors versus direct CPU:
  `2.384185791015625e-07`, `3.5762786865234375e-07`,
  `3.5762786865234375e-07`;
- train direct/shared boundary scans `270336` / `135168`;
- heldout direct/shared boundary scans `135168` / `67584`;
- shared forward boundary-scan ratio `0.5`;
- zero missing sample events for train and heldout candidates.

Scope boundary: this is real-ray MPS shared forward only. It has no real-ray
VJP/backward pass, no trainer, no CSR candidate storage, and no heldout-quality
claim.

## Gate 2C MPS Shared Real-Ray VJP Smoke

`tools/smoke_shared_realray_vjp_mps.py` in the isolated Metal variant adds the
first true camera-ray MPS backward proof for this lane. It replays the same
time-slab shared candidate bitsets as Gate 2B and emits per-ray per-site
RGBA/density gradient samples through fixed segments.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2c_mps_shared_realray_vjp_smoke.json
```

The current JSON reports:

- train RGB shape `[4,3,16,16]`, heldout RGB shape `[2,3,16,16]`;
- train gradient shape `[512,2,12,4]`, heldout gradient shape `[256,2,12,4]`;
- train max RGB/alpha/depth errors versus CPU:
  `2.980232238769531e-07`, `4.172325134277344e-07`,
  `3.5762786865234375e-07`;
- heldout max RGB/alpha/depth errors versus CPU:
  `1.7881393432617188e-07`, `2.980232238769531e-07`,
  `2.384185791015625e-07`;
- train max RGBA-gradient error `4.842877388000488e-08`;
- heldout max RGBA-gradient error `4.470348358154297e-08`;
- train loss absolute error `1.52587890625e-05`;
- shared forward boundary-scan ratio `0.5`;
- zero missing sample events for train and heldout candidates.

Scope boundary: this is fixed-segment site RGBA/density VJP only. It has no
site-position, weight, ray, camera, geometry, topology, or sorting gradients,
no autograd wrapper, no trainer, no CSR candidate storage, and no
heldout-quality claim.

## Gate 2D MPS Shared Real-Ray Reduced VJP Smoke

`tools/smoke_shared_realray_vjp_reduce_mps.py` in the isolated Metal variant
adds a reduced fixed-segment VJP boundary for true camera rays. The reduced op
returns site gradients shaped `[S,4]` and does not allocate the Gate 2C
`[K,T,S,4]` sample-gradient tensor. Internally it uses chunked partial
reductions shaped `[chunk_count,S,4]`; the smoke still uses Gate 2C as a
verifier, not as part of the reduced op.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2d_mps_shared_realray_reduced_vjp_smoke.json
```

The current JSON reports:

- train RGB shape `[4,3,16,16]`, heldout RGB shape `[2,3,16,16]`;
- train and heldout reduced gradient shape `[12,4]`;
- train partial gradient shape `[256,12,4]`, heldout partial gradient shape
  `[128,12,4]`;
- Gate 2C oracle sample-gradient shapes train `[512,2,12,4]`, heldout
  `[256,2,12,4]`;
- partial gradient float count is `0.25x` the Gate 2C oracle gradient-float
  count for both train and heldout;
- train max RGB/alpha/depth errors versus CPU:
  `2.980232238769531e-07`, `4.172325134277344e-07`,
  `3.5762786865234375e-07`;
- heldout max RGB/alpha/depth errors versus CPU:
  `1.7881393432617188e-07`, `2.980232238769531e-07`,
  `2.384185791015625e-07`;
- train max reduced RGBA-gradient error `3.0517578125e-05`;
- heldout max reduced RGBA-gradient error `7.62939453125e-06`;
- train reduced-vs-unreduced-MPS-sum error `3.0517578125e-05`;
- heldout reduced-vs-unreduced-MPS-sum error `7.62939453125e-06`;
- frozen-geometry autograd wrapper `shared_realray_rgba_depth_autograd`
  exposes `loss.backward()` for site RGBA/density only;
- train and heldout max autograd RGBA-gradient errors `0.0`;
- train and heldout autograd loss absolute errors `0.0`;
- train reduced VJP wall time `4.916041599062737 ms`;
- heldout reduced VJP wall time `3.7805999992997386 ms`;
- train and heldout single-call autograd backward wall times
  `46.593958999437746 ms` and `12.067124996974599 ms`;
- shared forward boundary-scan ratio `0.5`;
- zero missing sample events for train and heldout candidates.

Scope boundary: this is a frozen-geometry trainer-shaped gradient boundary, not
a fast trainer path. The current implementation avoids float atomics with
deterministic chunk partials and removes the full sample-gradient tensor. The
autograd wrapper only differentiates site RGBA/density; it still has no
geometry/topology gradients, trainer, parameter update, or heldout-quality
claim. Gate 2E and Gate 3 add parameter-update smokes on top of this boundary;
Gate 2F adds a CSR candidate-storage smoke, and the remaining performance step
is a larger scale/memory gate, not a quality promotion.

## Gate 2F MPS CSR Candidate Storage Smoke

`tools/smoke_shared_realray_csr_candidate_storage_mps.py` in the isolated Metal
variant replaces the Gate 2D bitset candidate input with CSR candidate rows.
It tests two layouts:

- per-track CSR: exact candidate sets per `(pixel track, time slab)`;
- tiled CSR: one candidate row per spatial tile and time slab, used as a
  superset storage layout.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2f_mps_shared_realray_csr_candidate_storage_smoke.json
```

The current JSON reports:

- train RGB shape `[4,3,16,16]`, heldout RGB shape `[2,3,16,16]`;
- per-track CSR exactly matches bitset candidate sets;
- tiled CSR is a candidate superset but still matches bitset RGB/alpha/depth
  outputs and reduced site-RGBA/density gradients with max error `0.0`;
- tiled CSR storage is below the bitset reference: train
  `3956 / 6144 = 0.6438802083333334x`, heldout
  `1788 / 3072 = 0.58203125x`;
- train direct/shared boundary scans remain `67584 / 33792`;
- heldout direct/shared boundary scans remain `33792 / 16896`;
- all CSR offsets are monotonic, final offsets match index counts, indices are
  in bounds, missing sample events are zero, and outputs are finite.

Scope boundary: this is a candidate-storage and parity smoke for the shared
real-ray reduced VJP path. It does not make a large-scale memory claim, does
not add geometry/topology gradients, and does not upgrade the tiny real-target
training smoke into a fair heldout baseline.

## Gate 2G MPS CSR Frame-Scaling Smoke

`tools/smoke_shared_realray_csr_scaling_mps.py` extends Gate 2F from one
2-frame parity row to a small MPS frame-count sweep. It reuses the same shared
real-ray reduced VJP path and tiled CSR candidate rows, then reports storage,
candidate-iteration, and boundary-scan growth over frame counts `2,4,8`.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2g_mps_shared_realray_csr_scaling_smoke.json
```

The current JSON reports:

- render size `32`, frame counts `[2,4,8]`, one time slab, 12 fixed 4D sites,
  and 66 fixed boundaries;
- tiled CSR matches the bitset reduced-VJP oracle with max output/gradient
  error `0.0` across train and heldout rows;
- from 2 to 8 frames, direct boundary scans grow `4.0x` while shared
  candidate-build scans grow `1.0x`;
- tiled CSR candidate iterations grow `3.9218009478672986x` on train and
  `3.915492957746479x` on heldout, both below direct scan growth;
- at 8 frames, tiled CSR storage remains below bitset storage: train
  `14944 / 24576 = 0.6080729166666666x`, heldout
  `6944 / 12288 = 0.5651041666666666x`;
- missing sample events are zero and CSR rows are valid for every row.

Scope boundary: this is fixed-geometry scaling evidence for shared candidate
storage and reduced site-RGBA/density gradients. It is not a production
trainer, not a geometry/topology gradient path, and not a 256px/16f heldout
quality comparison against STAR-UVT or dynamic splats.

Target-resolution follow-up artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2g_mps_shared_realray_csr_scaling_256px_8_16f.json
```

This reruns the same tiled CSR reduced-VJP scaling check on the STAR/dynamic
comparator split at 256px over frame counts `8,16`. It reports direct boundary
scan growth `2.0x`, shared boundary-scan growth `1.0x`, tiled candidate
iteration growth `1.9840766633277915x` on train and `1.982501348355007x` on
heldout, and max tiled CSR versus bitset MPS output/gradient error `0.0`.

## Gate 2E MPS Frozen-Geometry Autograd Overfit Smoke

`tools/smoke_shared_realray_autograd_overfit_mps.py` in the isolated Metal
variant proves that the Gate 2D autograd wrapper can drive an actual
site-RGBA/density parameter update. It keeps the same real train camera rays,
fixed 4D sites, fixed boundaries, and bitset candidates, then renders a
teacher target from the original site RGBA/density values and optimizes a
perturbed copy back toward that teacher.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2e_mps_shared_realray_autograd_overfit_smoke.json
```

The current JSON reports:

- train RGB shape `[4,3,16,16]`;
- loss drops from `0.004183291457593441` to
  `0.0001105795890907757`, a `0.02643363251442915x` ratio;
- first-step site-RGBA gradient abs sum `0.09477987885475159`;
- max site-RGBA parameter update `0.9012540578842163`;
- mean absolute site-RGBA error to teacher drops from
  `0.26774919033050537` to `0.16736829280853271`;
- shared forward boundary-scan ratio `0.5`;
- zero missing sample events.

Scope boundary: this is a teacher-target parameter-update smoke on real camera
rays. It is not real-target training, not a full trainer, not a geometry or
topology gradient path, and not a quality claim. Gate 3 is the corresponding
real-target smoke.

## Gate 3 MPS Frozen-Geometry Real-Target Training Smoke

`tools/smoke_shared_realray_real_target_train_mps.py` in the isolated Metal
variant optimizes site RGBA/density against actual train RGB targets through
the same shared real-ray MPS autograd path. Geometry, boundaries, site
positions, weights, ray geometry, sorting, and ownership remain fixed.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_real_target_train_smoke.json
```

The current JSON reports:

- train RGB shape `[4,3,16,16]`;
- train RGB MSE drops from `0.04258020222187042` to
  `0.025183267891407013`, a `0.5914313830682598x` ratio;
- train PSNR improves from `13.707922803417098` to
  `15.988879146126294`;
- first-step site-RGBA gradient abs sum `0.1965487003326416`;
- max site-RGBA parameter update `1.1732829809188843`;
- shared forward boundary-scan ratio `0.5`;
- zero missing sample events.

Scope boundary: this is the first real-target training smoke for the lane, but
only at 16px/2f with frozen geometry and 12 sites. It is not a full trainer, a
heldout-quality claim, or an apples-to-apples comparison against STAR-UVT or
dynamic splats.

## Gate 3 CSR 256px/16f Fixed-Geometry Train/Eval

`tools/train_eval_shared_realray_csr_mps.py` adds a heldout-quality artifact on
the STAR/dynamic comparator split. It uses tiled CSR candidate rows and the
CSR reduced-VJP path as a frozen-geometry autograd wrapper for site
RGBA/density only.

Current saved artifacts:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_csr_quality_256px_16f.json
dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_csr_quality_256px_16f_train.ppm
dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_csr_quality_256px_16f_heldout.ppm
```

The current JSON reports:

- train cameras `camera_0006` and `camera_0014`, heldout `camera_0005`;
- render size `256`, frame count `16`, one time slab, 12 fixed 4D sites, and
  66 fixed boundaries;
- 5 optimizer steps with train PSNR improving from `10.504453575214932` to
  `12.552081185301754`;
- heldout PSNR `12.703601126620978`;
- shared forward boundary-scan ratio `0.0625` for train and heldout;
- tiled CSR storage below bitset storage: train `0.5828577677408854x`,
  heldout `0.5909423828125x`;
- zero missing sample events and valid CSR rows.

Scope boundary: this is the first same-split World Foam heldout metric row
against the STAR/dynamic comparison split. It remains frozen-geometry
site-RGBA/density training only, with no geometry/topology gradients and no
full production trainer.

## Gate 4 Moving-Ray Slab Compiler

`gate4_moving_ray_slab_compiler.py` is the first STAR-style port back into
World Foam: compile each pixel track into one affine ray-time slab before
replay. For a 4D power boundary and affine ray track,

```text
o(t) = o0 + odot * t
d(t) = d0 + ddot * t
s(t) = -(n . o(t) + nt * t + b) / (n . d(t))
```

so each boundary's depth over a slab is a rational one-dimensional interval.
The compiler uses that interval to build one candidate set per
`(view, pixel, time_slab)` instead of re-testing every boundary at every frame.

Current saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate4_moving_ray_slab_compiler_affine_motion_smoke.json
dynaworld/research_experiments/world_foam_lane2/results/gate4_moving_ray_slab_compiler_affine_motion_timed_2_4_8_16.json
```

Reproduce from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 12 \
  --time-slabs 1 \
  --origin-velocity-x 0.08 \
  --origin-velocity-z 0.02 \
  --direction-velocity-x 0.02 \
  --out-json research_experiments/world_foam_lane2/results/gate4_moving_ray_slab_compiler_affine_motion_timed_2_4_8_16.json
```

The current JSON reports:

- status `ok` with moving ray tracks present;
- zero missing sample events across `2,4,8,16` frames;
- affine ray fit residuals at float tolerance;
- direct boundary tests grow `8.0x` from 2 to 16 frames;
- compiled boundary tests grow `1.0x` over the same frame range;
- compiled boundary-test ratio improves from `0.5` at 2 frames to `0.0625`
  at 16 frames;
- CPU candidate-tape compile time stays roughly flat:
  `0.055s -> 0.054s -> 0.065s -> 0.060s`;
- candidate replay iterations still grow `7.77x`, so replay/compositing
  remains frame-scaled.

Scope boundary: this is a CPU compiler/accounting gate only. It ports the
compact world-tube idea into World Foam's real-ray candidate generation, but it
does not dispatch Metal, composite images, implement backward, or prove
quality. The next useful gate is a Metal CSR compositor that consumes this
compiled tape without re-expanding candidate work into per-frame boundary
tests.

## Gate 0 Metal/MPS Count Smoke

The isolated Metal variant now has a count-only Torch/MPS bridge source at:

```text
dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/
```

Build and smoke it from the repo root:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_power_boundary_mps.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_mps_power_boundary_smoke.json
```

Current saved smoke:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_mps_power_boundary_smoke.json
```

It matches the CPU slab-count fixture for both default velocities:

- `camera_velocity_x=0.35`: MPS `149`, CPU expected `149`.
- `camera_velocity_x=0.7`: MPS `151`, CPU expected `151`.

This is real GPU execution for the power-boundary event count only. It is not a
renderer, tile compositor, or Metal backward pass.

## Endpoint Record Block-Coeff16 Fused-MSE Sidecar

The half-coefficient block path now has a fused RGB MSE + RGB-only site-RGBA
VJP shader:

```text
endpoint-record-edit-block-coeff16-fused-mse
```

It reuses the existing block edit anchors and `coeff_f16` boundary-depth
coefficients, then computes the loss and site gradient in one Metal pass.

Focused parity and lane gates:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

After adding the packed-record, separated-int16-record, and interleaved-int16
record variants, the local gates still pass: focused replay `11` tests OK, full
lane suite `41` tests OK.

Real 16/32 fixture:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

Results:

- 16f total/backward: `2.07 / 1.68 ms`; selected storage `0.268x` full;
  heldout PSNR `14.7128`.
- 32f total/backward: `1.89 / 1.50 ms`; selected storage `0.203x` full;
  heldout PSNR `14.7955`.
- 2x frame-count scale: total `0.916x`, backward `0.896x`.

Repeated-frame 16/32/64/128 speed-scale gate:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Results:

- total ms: `2.752 / 2.476 / 2.241 / 2.759`
- backward/fused ms: `2.290 / 1.997 / 1.871 / 2.362`
- 8x frame-count scale: total `1.003x`, backward `1.031x`
- selected storage versus full: `0.257x / 0.210x / 0.185x / 0.173x`
- selected storage scale: `5.38x` for `8x` frames

Interpretation: this is the cleanest practical WorldFoam runtime curve so far:
near-flat fused loss/VJP across 16f to 128f with matched heldout PSNR. It is not
a pure STAR-like storage curve. Half coefficients only slightly reduce the
block-coeff storage ratio because block anchor/edit metadata now dominates; the
128f ratio improves from the fp32 block-coeff fused path's `0.179x` full to
`0.173x` full. The raw edit fused-MSE path is still the cleaner storage story
(`0.012x` full at 128f), but its hot shader scales worse (`3.96 ms` at 128f).

Two metadata-compression forks were then tried:

```text
endpoint-record-edit-block-coeff16-packed-fused-mse
endpoint-record-edit-block-coeff16-i16-fused-mse
endpoint-record-edit-block-coeff16-i16x3-fused-mse
```

The packed-record fork stores `(owner,left,right)` as one int32 per anchor/op
record. It is a storage win but a runtime negative on the real 16/32 gate:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_packed_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

- 16f total/backward/storage/PSNR: `7.238 ms / 5.015 ms / 0.1666x` full /
  `14.7128`
- 32f total/backward/storage/PSNR: `4.768 ms / 3.399 ms / 0.1157x` full /
  `14.7955`

The int16-record fork stores owner/left/right as three int16 arrays. It avoids
the bitfield decode cost and gives a better storage/runtime tradeoff than the
packed fork:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Real 16/32 result:

- 16f total/backward/storage/PSNR: `1.415 ms / 1.089 ms / 0.1919x` full /
  `14.7128`
- 32f total/backward/storage/PSNR: `2.673 ms / 2.288 ms / 0.1375x` full /
  `14.7955`
- strict backward sublinearity fails on this real 16/32 gate: backward scale is
  `2.10x` for `2x` frames, although total step scale is `1.89x`

Repeated-loaded 16/32/64/128 result:

- total ms: `2.976 / 2.433 / 4.224 / 2.859`
- backward/fused ms: `2.236 / 2.026 / 3.341 / 2.147`
- 8x frame-count scale: total `0.961x`, backward `0.960x`
- selected storage versus full: `0.1805x / 0.1391x / 0.1174x / 0.1068x`
- selected storage scale: `4.73x` for `8x` frames

The interleaved-int16 fork stores `(owner,left,right)` as one three-short
record stream for anchors and ops. It reduces Python-side stream fanout versus
the separated-int16 path without the packed int32 bitfield decode:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16x3_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16x3_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Real 16/32 result:

- 16f total/backward/storage/PSNR: `2.156 ms / 1.752 ms / 0.1919x` full /
  `14.7128`
- 32f total/backward/storage/PSNR: `3.229 ms / 2.262 ms / 0.1375x` full /
  `14.7955`
- 2x frame-count scale: total `1.50x`, backward `1.29x`

Repeated-loaded 16/32/64/128 result:

- total ms: `2.950 / 3.990 / 5.320 / 4.985`
- backward/fused ms: `2.150 / 2.980 / 3.896 / 4.317`
- 8x frame-count scale: total `1.69x`, backward `2.01x`
- selected storage versus full: `0.1805x / 0.1391x / 0.1174x / 0.1068x`
- selected storage scale: `4.73x` for `8x` frames

Practical read: WorldFoam is sublinear in theory once the tape stops
materializing per-frame segment arrays, and it is sublinear in practice on the
fused repeated-frame scale gate. It is not yet STAR-clean on real frame-count
changes: real 16/32 can still expose per-frame replay, edit metadata, and MPS
timing costs. STAR UVT is cleaner because the temporal object is naturally
fixed-size and basis-like; WorldFoam's exact owner/cut topology is branchy and
metadata-heavy unless more of the replay is fused or made topology-static.
Among the current forks, unpacked coeff16 is the smoothest runtime curve,
separated-int16 is the best storage/runtime compromise, packed int32 is a
runtime negative, and interleaved-int16 is useful evidence but not the current
winner.

The raw endpoint-record edit path now has its own coeff16 fused-MSE fork:

```text
endpoint-record-edit-coeff16-fused-mse
```

This keeps the raw edit tape's nearly flat owner/cut edit storage, but replaces
per-sample boundary/ray cut-depth solves with the same per-track half-precision
linear depth coefficients used by the block-coeff path.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_coeff16_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_coeff16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Real 16/32 result:

- 16f total/backward/storage/PSNR: `2.574 ms / 1.564 ms / 0.1565x` full /
  `14.7128`
- 32f total/backward/storage/PSNR: `3.002 ms / 2.066 ms / 0.0789x` full /
  `14.7955`
- 2x frame-count scale: total `1.17x`, backward `1.32x`, selected storage
  `1.006x`

Repeated-loaded 16/32/64/128 result:

- total ms: `2.530 / 10.603 / 4.169 / 3.413`
- backward/fused ms: `1.948 / 6.647 / 3.280 / 2.714`
- 8x frame-count scale: total `1.35x`, backward `1.39x`
- selected storage bytes: `75420 / 75288 / 75300 / 75348`
- selected storage versus full: `0.1458x / 0.0728x / 0.0364x / 0.0182x`
- selected storage scale: `0.999x` for `8x` frames

Interpretation: this is the strongest WorldFoam storage result so far while
also improving the real 16/32 raw-edit runtime. It is not yet the smoothest
runtime result; the 32f repeat-loaded row is noisy and later 32-only reruns also
slowed a known block-coeff control, so use the warm3 16/32/64/128 artifact as
evidence for sublinear overall scaling but not as proof of stable per-frame
latency. The practical frontier is now split: raw-edit coeff16 gives the
STAR-shaped storage curve, while block-coeff16 still gives the smoothest
runtime curve.

### Endpoint-Record Delta-Replace + Coeff16 Fused MSE

This fork replaces raw edit-op replay with full replacement rows per changed
frame, then uses coeff16 cut-depths and fused RGB MSE/VJP:

```text
endpoint-record-delta-replace-coeff16-fused-mse
endpoint-record-delta-replace-coeff16-i16x3-fused-mse
endpoint-record-delta-replace-coeff16-i16x4-fused-mse
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replace_coeff16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replace_coeff16_fused_mse_repeat_loaded_rerun_warm3_steps8_render16_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replace_coeff16_i16x3_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x4_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x4_scalar_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_control.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_binarysearch_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_control.json
```

Rerun repeated-loaded 16/32/64/128 result:

- total ms: `3.398 / 5.903 / 6.409 / 5.634`
- backward/fused ms: `2.541 / 4.393 / 4.679 / 4.637`
- selected storage bytes: `66580 / 66528 / 66528 / 66548`
- selected storage versus full: `0.1287x / 0.0644x / 0.0322x / 0.0161x`
- selected storage scale: `0.9995x` for `8x` frames
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`

The compact i16x3 record fork stores owner/left/right replacement rows as
interleaved int16 triples. Its repeated-loaded 16/32/64/128 result:

- total ms: `4.136 / 90.967 / 4.975 / 11.760`
- backward/fused ms: `2.660 / 72.771 / 3.670 / 9.015`
- selected storage bytes: `49936 / 49902 / 49902 / 49916`
- selected storage versus full: `0.0965x / 0.0483x / 0.0241x / 0.0121x`
- selected storage scale: `0.9996x` for `8x` frames
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`

The padded i16x4 fork keeps owner/left/right records at 8 bytes each to test
whether aligned record stride improves the compact path. It did not. The first
short4-load sweep reported:

- total ms: `29.012 / 12.648 / 5.052 / 11.262`
- backward/fused ms: `23.284 / 10.114 / 3.596 / 8.209`
- selected storage bytes: `55484 / 55444 / 55444 / 55460`
- selected storage versus full: `0.1073x / 0.0536x / 0.0268x / 0.0134x`

After revising the kernel to scalar loads with a padded 4-short stride, the
16/32 control remained negative:

- total ms: `23.145 / 31.418`
- backward/fused ms: `18.352 / 24.052`
- selected storage bytes: `55484 / 55444`

A binary-search replacement for the per-sample "last change <= frame" scan was
also rejected and reverted. The i16x3 16/32 control with that helper failed the
sublinear gate:

- total ms: `21.351 / 47.938`
- backward/fused ms: `13.947 / 35.825`
- selected storage bytes: `49936 / 49902`

Interpretation: storage is the best STAR-shaped WorldFoam result so far, but
runtime is still not the winner. Replacement rows avoid applying edit scripts
in the shader and improve storage over raw-edit coeff16, but the fused kernel
still loses to block-coeff16 on smooth runtime and generally loses to raw-edit
coeff16 except for that mode's noisy 32f row. The i16x3 fork improves storage
again, but its 32f row exploded in the full sweep. A later 32f-only control
reduced that to `6.832 ms` total / `4.762 ms` fused-backward, so the giant row
was mostly timing noise, but not enough to make the path a runtime winner. The
i16x4 and binary-search attempts are explicit negative results. Keep this
family as a storage win and a runtime mixed/negative, not as the current winner.

### Gate4 Endpoint-Record Fast Path

After the owner-run reverse-tape keeper and the in-kernel owner-update
negative, the next STAR-style port was to build endpoint replacement rows from
the Gate4 affine slab tape and feed the packed framegroup16 coeff16 fused-MSE
shader directly.

New artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_pairupdate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_pairupdate_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecdepth_ownerscan_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecdepth_ownerscan_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecowner_batchsort_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecowner_batchsort_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_chunkbatch128_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_chunkbatch128_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_skipvalidate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_skipvalidate_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_directdelta_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_directdelta_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_prealloc_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_prealloc_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_toporeuse_directdelta_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_ownerother_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_ownerother_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_directdelta_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_tensormerge_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeowner_directdelta_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeowner_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeowner_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_directdelta_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativecutprep_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativecutprep_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativecutprep_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_packed_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x3_fg16_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x3_ownerreduce_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16cols_fg16_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x4_fg16_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x3_fg64_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_cli_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_cli_repeat20_render64_site24_2_4_8_16.verify.json
agent_notes/loose_notes/2026-05-18_18-56-39_gate4_endpoint_record_fastpath.md
```

First verified 64px/24-site, real `2/4/8/16f`, warm5/steps20:

- total median ms: `2.871 / 3.424 / 2.303 / 2.985`
- backward median ms: `2.485 / 2.990 / 1.990 / 2.665`
- total median scale `2f -> 16f`: `1.040x`
- backward median scale `2f -> 16f`: `1.072x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The follow-up vectorized-depth/owner-changing scan compiler keeps the same
quality and storage scale while reducing setup:

- total median ms: `2.062 / 2.058 / 2.405 / 2.871`
- backward median ms: `1.747 / 1.779 / 2.082 / 2.563`
- total median scale `2f -> 16f`: `1.392x`
- backward median scale `2f -> 16f`: `1.467x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`
- 16f endpoint sequence build:
  train `63.17s -> 17.61s`, heldout `22.61s -> 7.63s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The latest host-compiler pass vectorizes initial owner selection and batches
per-track frame sorts. It is another setup improvement, not a new semantic
claim:

- total median ms: `2.225 / 2.148 / 2.422 / 2.929`
- backward median ms: `1.889 / 1.835 / 2.099 / 2.605`
- total median scale `2f -> 16f`: `1.316x`
- backward median scale `2f -> 16f`: `1.379x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`
- 16f endpoint sequence build:
  train `63.17s -> 16.46s`, heldout `22.61s -> 7.43s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The chunked candidate-sort pass batches single-slab rows across 128 tracks.
This is a marginal compiler pass, not a semantic change and not the native
compiler fix:

- total median ms: `2.070 / 2.190 / 2.442 / 2.928`
- backward median ms: `1.758 / 1.868 / 2.106 / 2.607`
- total median scale `2f -> 16f`: `1.414x`
- backward median scale `2f -> 16f`: `1.483x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`
- endpoint sequence build:
  train `4.46 / 6.25 / 9.63 / 16.49s`, heldout
  `1.90 / 2.66 / 4.13 / 7.31s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The validation-skipped fast path removes the per-sample debug event replay from
the benchmark setup while keeping full validation as the default in the tape
builder and test gate. `train_eval_owner_run_tape.py --endpoint-record-source
gate4-affine` now records `sample_validation=skip` and
`missing_sample_events_authoritative=false` in the Gate4 metadata:

- total median ms: `2.005 / 2.188 / 2.195 / 3.789`
- backward median ms: `1.702 / 1.875 / 1.916 / 3.348`
- total median scale `2f -> 16f`: `1.890x`
- backward median scale `2f -> 16f`: `1.966x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`
- endpoint sequence build:
  train `3.18 / 3.89 / 4.61 / 6.59s`, heldout
  `1.32 / 1.54 / 1.96 / 2.64s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The vectorized coefficient pass computes the full `[track, boundary, 4]`
affine depth coefficient table once and reuses it for slab event selection,
candidate ordering, row-local candidate coefficients, and the shader-facing
coefficient tensor:

- total median ms: `2.133 / 2.151 / 2.354 / 3.019`
- backward median ms: `1.825 / 1.837 / 2.041 / 2.674`
- total median scale `2f -> 16f`: `1.415x`
- backward median scale `2f -> 16f`: `1.465x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`
- endpoint sequence build:
  train `2.14 / 2.70 / 3.55 / 5.52s`, heldout
  `0.84 / 1.03 / 1.57 / 2.10s`
- coefficient build:
  train `0.027 / 0.023 / 0.025 / 0.026s`, heldout
  `0.010 / 0.012 / 0.012 / 0.011s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The direct delta-replace pass emits the packed delta arrays from the Gate4 row
replay without materializing `Gate4EndpointRunRecord` sequences or running the
edit/delta packers in the selected packed-delta path:

- total median ms: `2.303 / 3.594 / 2.255 / 2.967`
- backward median ms: `1.955 / 3.269 / 1.971 / 2.625`
- total median scale `2f -> 16f`: `1.288x`
- backward median scale `2f -> 16f`: `1.342x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`
- endpoint direct-delta build:
  train `2.09 / 2.65 / 3.56 / 5.17s`, heldout
  `0.85 / 1.07 / 1.49 / 2.03s`
- edit pack and delta pack:
  all `0.0s` in the selected packed-delta path
- `verify_framegroup16_timing_robust.py`: `status=ok`

The cut-array preallocation cleanup is safe and verified, but it is not a clear
successor to the direct-delta artifact:

- total median ms: `2.315 / 2.579 / 2.366 / 2.952`
- backward median ms: `1.982 / 2.270 / 2.048 / 2.624`
- total median scale `2f -> 16f`: `1.275x`
- backward median scale `2f -> 16f`: `1.324x`
- selected storage scale `2f -> 16f`: `1.040x`
- endpoint direct-delta build:
  train `2.14 / 2.65 / 3.42 / 5.45s`, heldout
  `0.85 / 1.11 / 1.33 / 1.95s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The topology-reuse fork was reverted after a negative 16f spot. Reusing
adjacent-frame owner-run records after Python cut-id/start/owner checks kept
exactness, but measured `3.814 ms` total, `3.459 ms` backward, `5.40s` train
endpoint build, and `2.21s` heldout endpoint build at 16f. The overhead is not
worth it on the moving-camera gate.

The current Python-side keeper precomputes an owner-membership table,
`boundary_other_by_owner[owner, boundary_id]`, so the hot owner-run walk checks
one table instead of gathering left/right site arrays for every boundary
search:

- total median ms: `2.127 / 3.592 / 2.368 / 2.916`
- backward median ms: `1.795 / 3.272 / 2.036 / 2.599`
- total median scale `2f -> 16f`: `1.371x`
- backward median scale `2f -> 16f`: `1.448x`
- selected storage scale `2f -> 16f`: `1.040x`
- endpoint direct-delta build:
  train `2.09 / 2.53 / 3.23 / 4.85s`, heldout
  `0.83 / 1.05 / 1.32 / 1.90s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

The native chunk packer is the current Gate4 endpoint-record keeper. It leaves
sorted cut-array construction in Python, then calls
`world_foam_lane2_fused_slab_v0.gate4_delta_replace_from_cuts_cpu` to replay
owner runs and emit the packed base/change delta arrays for each chunk:

- total median ms: `2.258 / 2.154 / 2.464 / 2.966`
- backward median ms: `1.935 / 1.833 / 2.144 / 2.640`
- total median scale `2f -> 16f`: `1.314x`
- backward median scale `2f -> 16f`: `1.364x`
- selected storage scale `2f -> 16f`: `1.040x`
- endpoint direct-delta build:
  train `1.97 / 2.20 / 2.63 / 3.48s`, heldout
  `0.83 / 0.89 / 1.06 / 1.53s`
- `verify_framegroup16_timing_robust.py`: `status=ok`

Two follow-up cleanup forks were negative and reverted. A tensor-merge fork
kept native chunk outputs as tensors and concatenated once, but the 16f spot
regressed to `3.118 ms` total, `2.704 ms` backward, and
`4.01s/2.12s` train/heldout endpoint build. A native first-owner-selection fork
computed initial owners in C++ before calling the same native row packer; it
passed exactness and the robust verifier, but the full sweep was worse
(`2.117 / 2.433 / 8.227 / 3.185 ms` total, `1.504x` total scale, 16f train
setup `4.22s`). Both were removed from the active path.

A native sorted-row packer also failed promotion. It improved 16f setup to
`3.15s` train and `1.28s` heldout, but changed endpoint segment counts on the
real gate (`222276` selected/full train segments instead of `222501`) and
regressed warm timing to `6.179 ms` total / `4.132 ms` backward. The op and
dispatch were removed after rebuild and exactness recheck. The follow-up guard
is now in `test_highcap_single_slab_sorted_rows_match_cut_array_delta_records`:
it uses a high-cap 24-site / 16-frame moving-camera fixture, checks direct
delta tensors against the cut-array record path, and confirms that the naive
sorted-row/no-dedupe reconstruction would fail there.

A corrected exact native sorted-row op now exists but is not promoted. It is
behind `GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA = False`, and the
extension-imported high-cap unittest temporarily enables it and checks exact
delta tensor parity. On the real 16f fixture, exact sorted and cut-array native
paths produce bit-identical delta tensors, and setup improves (`2.84-2.98s`
train, `1.15-1.17s` heldout). However, the timed 16f spots regressed warm
total/backward to `8.248/7.847 ms` and `7.840/7.319 ms`, while the restored
default cut-array spot is `2.990/2.621 ms`. Keep the native cut-array packer as
the active default until the sorted-path warm timing interaction is isolated.
A same-process probe that keeps both cut-array and sorted MPS tapes resident
shows all selected-device tensors equal, but both paths run slow there
(`7.913 ms` cut-first, `7.545 ms` sorted-second, `6.769 ms` cut-third median
VJP). A naive `torch.mps.empty_cache()` before the train loop also slows the
default cut-array path to `8.010/7.554 ms` total/backward. Treat the remaining
sorted blocker as an MPS residency/allocation interaction, not a record
semantic mismatch.
The clean-process single-tape probe is also lifetime-order sensitive: allocating
target/site MPS tensors before tape preparation makes cut-only slow
(`7.554 ms`), while matching the trainer's tape-first order drops cut-only to
`3.751 ms`. Sorted-only remains slow in that order (`7.834 ms`); `gc.collect()`
and device tensor cloning do not fix it, and a sync-before-target allocation
only moves both paths into a slower `4.5-4.8 ms` band. Use the probe for
diagnosis only; full train/eval artifacts remain the promotion gate.

The explicit CLI train/eval gate confirms the corrected sorted op should stay
unpromoted. `train_eval_owner_run_tape.py --experimental-native-sorted-delta`
ran the real `2/4/8/16f` 64px/24-site ladder with unchanged heldout PSNR
(`14.170 / 13.993 / 14.221 / 14.232`) and unchanged selected storage scale
(`1.040x`). Endpoint build improved slightly at 16f (`3.30s` train plus
`1.33s` heldout; total prepare `3.55s/1.38s`), but warm timing lost badly to
the native cut-array keeper:

- sorted CLI total median ms: `3.507 / 4.371 / 4.826 / 7.447`
- sorted CLI backward median ms: `3.007 / 3.681 / 4.263 / 6.866`
- robust verifier: `status=failed`
- robust failures: total mean/median scale `2.044x/2.124x`, backward
  mean/median scale `2.143x/2.284x`, all above the `2.0x` promotion cap

This is still sublinear versus the raw `8x` frame increase, but it is not the
project's accepted speed-scale result and is not competitive with the current
native cut-array keeper (`2.966 ms` total, `2.640 ms` backward at 16f, verifier
`status=ok`). Keep the flag for diagnostics only.

A native cut-prep fork also failed promotion. The new
`gate4_cut_arrays_from_sorted_cpu` helper computes `cut_depths`, `cut_ids`,
`cut_offsets`, `start_segments`, and `initial_owner` from each sorted chunk in
C++, then still calls the promoted `gate4_delta_replace_from_cuts_cpu` final
packer. This removes the Python per-frame cut-row assembly without using the
slow sorted final packer, and it passes the high-cap cut-vs-packed parity test.
The real ladder is nevertheless a timing negative:

- cut-prep total median ms: `2.463 / 2.579 / 5.248 / 5.438`
- cut-prep backward median ms: `2.025 / 2.190 / 4.460 / 4.472`
- 16f endpoint build: `3.00s` train plus `1.15s` heldout
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`
- robust verifier: `status=failed`
- robust failures: total median scale `2.208x`, backward mean/median scale
  `2.064x/2.208x`

Read: the Python cut assembly is real setup work, but eliminating just that
piece is not enough. The extra native prep/tensor allocation still perturbs the
warm MPS step, so this flag is diagnostic-only.

The existing device-side delta representations were also screened on the same
64px/24-site/16f Gate4 path. The sweep is noisy because the fresh packed row in
that process measured slower than the keeper artifact, so it is a ranking
screen rather than promotion evidence. It still gives a clear ordering:

- packed framegroup16: `4.097/3.558 ms` total/backward median
- i16x3 framegroup16: `4.513/4.102 ms`
- i16x3 owner-reduce framegroup16: `4.645/4.130 ms`
- i16cols framegroup16: `5.387/4.416 ms`
- i16x4 framegroup16: `8.409/6.838 ms`
- i16x3 framegroup64: `11.022/10.443 ms`

All rows kept heldout PSNR at `14.232`, but none beat packed. Keep
`endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse` as the
device-side default for this Gate4 path; the next real fork needs a different
representation, not another i16 layout variant.

I also tried a minimal selected-device fork for packed framegroup16 fused-MSE.
The hypothesis was that the warm packed kernel should not keep unused
unpacked endpoint records, boundary tensors, or rays resident on MPS; final
PSNR rendering can build the full replay device lazily after timed training.
Correctness held, but timing did not:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_minimalpacked_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_default_after_minimalpacked_spot_render64_site24_16.json
```

- minimal selected-device 16f total/backward median: `8.641/8.003 ms`
- same-session default 16f total/backward median: `8.006/7.456 ms`
- both heldout PSNRs: `14.232`

This was a noisy slow session, so it is not a new default-regression claim
against the clean keeper artifact. It does show that simply removing unused
MPS tensors from the selected packed loss device is not a win; the next fork
needs to change the endpoint representation or shader work, not just tensor
residency.

The existing materialized i16x3 framegroup16 Metal kernel is now wired into the
harness as
`endpoint-record-delta-replace-coeff16-i16x3-framegroup16-materialized-fused-mse`.
This path changes warm shader work by materializing each chunk's endpoint rows
into threadgroup memory before per-frame RGB/VJP. It is valid and frame-scale
sublinear, but too slow to promote:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_i16x3_materialized_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_i16x3_materialized_repeat20_render64_site24_2_4_8_16.verify.json
```

- robust verifier: `status=ok`
- materialized total median ms: `4.991 / 5.132 / 9.428 / 8.388`
- materialized backward median ms: `4.561 / 4.772 / 8.716 / 7.450`
- mean scale 2f -> 16f: total `1.424x`, backward `1.418x`
- selected storage scale: `1.054x`
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`

This is useful as a scale-positive shader diagnostic, not a keeper. The clean
native cut-array direct-delta artifact is still much faster at 16f
(`2.966/2.640 ms` total/backward median) with better scale
(`1.207x/1.241x` mean total/backward).

A packed materialized follow-up also failed promotion. The new
`endpoint-record-delta-replace-coeff16-packed-framegroup16-materialized-fused-mse`
mode keeps the compact int32 row format, materializes each 16-frame chunk's
selected rows into one threadgroup int array, and unpacks from that array in
the RGB/VJP loop. It rebuilds and exports as
`endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only`,
and the Gate4 high-cap unittest still passes. The full ladder artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_materialized_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_materialized_repeat20_render64_site24_2_4_8_16.verify.json
```

- robust verifier: `status=ok`
- packed-materialized total median ms: `4.518 / 6.777 / 7.503 / 5.757`
- packed-materialized backward median ms: `4.130 / 5.919 / 7.074 / 5.209`
- mean scale 2f -> 16f: total `0.909x`, backward `1.049x`
- median scale 2f -> 16f: total `1.274x`, backward `1.261x`
- selected storage scale: `1.040x`
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`

This narrows the materialization diagnosis: packed materialization is much
better than i16x3 materialization, but it is still slower than the cut-array
keeper (`2.966/2.640 ms` at 16f). Threadgroup row materialization is therefore
not the missing STAR-clean trick for this Gate4 endpoint-record path.

A high-site threadgroup gradient-reduction fork was also negative. Raising the
framegroup reduction cap from 16 to 32 sites would cover the 24-site gate but
Metal refused the packed framegroup16 pipeline because threadgroup memory grew
to `34048` bytes over the `32768` byte cap. Narrowing to exactly 24 sites
launched, but the 16f spot regressed to `8.134/7.710 ms` total/backward with
unchanged PSNR. The cap is back to the keeper value `16`; the lesson is that
the extra threadgroup memory pressure is worse than the current global grad
atomics at this scale.

A small-run16 replay-cap specialization was also wired as
`endpoint-record-delta-replace-coeff16-packed-framegroup16-smallrun16-fused-mse`.
It keeps the packed int32 endpoint rows and 32-frame chunking but compiles the
warm VJP kernel with local replay arrays capped at 16 segments. The wrapper and
torch op export, the Gate4 unittest passes, and a 16f repeat can be
quality-correct, but the full ladder failed the robust timing verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_smallrun16_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_smallrun16_repeat20_render64_site24_2_4_8_16.verify.json
```

- verifier: `status=failed`
- smallrun16 total median ms: `4.474 / 6.558 / 4.832 / 13.474`
- smallrun16 backward median ms: `3.978 / 4.684 / 3.878 / 12.213`
- scale 2f -> 16f: total mean/median `3.010x / 3.012x`, backward
  mean/median `2.729x / 3.070x`
- selected storage scale: `1.040x`
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`

So shrinking thread-private replay arrays is not a free STAR-cleanup here. It
can produce an isolated warmed 16f spot around `4.270/3.818 ms`, but the ladder
is noisy and regresses the robust scale gate. Keep it as a guarded diagnostic
mode only; do not auto-select or promote it over the packed keeper.

A min-state recompute shader was the next bounded fork. The new
`endpoint-record-delta-replace-coeff16-packed-framegroup16-recompute-fused-mse`
mode keeps the packed framegroup16 endpoint rows, but drops the
`segment_trans`, `segment_alpha`, `weights`, and `segment_rgb` private replay
arrays from the Metal VJP kernel. The reverse pass reloads site RGB/density and
recomputes transmittance terms from `owner`, `length`, and `trans_before`.

The extension rebuilds, the wrapper/torch op export, and the Gate4 unittest
still passes. Timing rejected the fork before a full ladder:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_recompute_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_recompute_repeat2_spot_render64_site24_16.json
```

- first 16f spot: `11.795/9.314 ms` total/backward median
- warmed repeat: `7.377/6.689 ms` total/backward median
- heldout PSNR unchanged: `14.232`

The clean keeper artifact remains `2.966/2.640 ms` at 16f, so recomputing
reverse replay terms is not worth the private-array reduction at this scale.
Keep this only as an explicit diagnostic; do not auto-select or promote it.

This is now a real STAR-shaped warm-kernel result for WorldFoam endpoint
records. It is not a full-pipeline win yet: with
`--endpoint-record-source gate4-affine`, the harness intentionally skips
baseline full segment-tape construction, and the benchmark path now skips
full per-sample candidate validation after unit-test exactness coverage. The
remaining endpoint-row materialization is much lower but still scales with
frame count (`3.48s` train plus `1.53s` heldout at 16f in the latest native
chunk pass). Next work should remove the remaining Python cut-row assembly
itself or change the representation so those rows are never materialized in
Python; owner-run row walking has already moved native for the single-time-slab
path, while tensor-only merge cleanup and native first-owner selection did not
pay for themselves. A future native cut-row attempt should first pass the new
high-cap sorted-vs-cut parity fixture before timing, and should not promote the
current exact sorted op without solving the warm-step regression.

## Auxiliary Moving-Disk Oracle

From the repo root:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/beam_events_reference.py
python3 dynaworld/research_experiments/world_foam_lane2/beam_events_reference.py --json
python3 dynaworld/research_experiments/world_foam_lane2/beam_events_reference.py --self-test
python3 -m unittest discover -s dynaworld/research_experiments/world_foam_lane2
```

Optional time samples:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/beam_events_reference.py --times 0,0.25,0.5,0.75,1 --json
```

## Auxiliary Evidence Produced

The compact table reports one row per `(time, beam)`:

```text
time   beam        events  hits  max_depth  coverage
0.0    centerline       4     2          1  ...
```

The JSON output is the fuller artifact. It contains:

- `total_event_count`: total sorted enter/exit events over the sampled grid.
- `total_hit_count`: total disk intervals hit by all beam/time samples.
- `global_max_depth`: maximum number of overlapping supports along any beam.
- `traces`: per-beam/time intervals and sorted events.

The default pinned scene currently self-tests to:

```text
total_event_count = 34
total_hit_count = 17
global_max_depth = 3
```

## Promotion Rule

Do not add trainer integration until the power-cell sweep keeps zero missing
sample events, shows a beam-event ratio below the summed per-frame traversal
cost on a moving-camera toy sequence, and has a Metal parity harness. Gate 0.5
allows a narrow CPU color/signal-gradient reference, but it does not clear the
Metal backward or image-quality gates.
