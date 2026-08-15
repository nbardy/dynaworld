# WorldFoam scientist feedback and fixed-site source closure

Date: 2026-08-04

## Context

The external scientist compared the repository work with five proposed
mathematical formulations. Their central judgment was useful: the repository
has made more progress as an implementation/research program, while the
translated optical-depth-measure formulation is the strongest additional
mathematical lens. This session audited that judgment against the live tree and
continued the narrow production path without running Python, Torch imports,
tests, builds, Metal, MPS, or CUDA. The host still had too little free VM for a
safe runtime gate.

The primary project remains the World Tubes paper. This WorldFoam work is the
memory-light second-paper lane and must not delay the World Tubes evidence
queue.

## Current model

Current belief: the canonical WorldFoam architecture remains

```text
direct kinetic sites and camera program
  -> exact event-stratified owner charts
  -> ordered P0 transfer words
  -> bounded J-node temporal surrogate
  -> streamed sample residuals into node cotangents
  -> one material-word VJP per active compiled block
```

Confidence: high for the algebra and CPU/source material-only architecture;
low for native allocator behavior, publication-scale compile cost, and trained
quality because the current continuation is source-only and unrun.

The safe update policy is already decided:

- material-only changes may reuse a sealed structural program;
- every geometry, weight-trajectory, or camera-ray change recompiles and
  recertifies the full structural program;
- the simple-root continuation routine remains a whole-registry correctness
  oracle, not an output-sensitive program patcher.

Output-sensitive warm repair is therefore optional future optimization, not a
mathematical blocker.

## Expansion Pass 1: proof object versus runtime object

For an ordered P0 word, let segment `r` have optical-depth width

```text
Delta kappa_r = density_r * physical_length_r >= 0
```

and color `c_r`. Its cumulative optical-depth boundaries are

```text
K_0 = 0
K_r = sum_(q <= r) Delta kappa_q.
```

The order-explicit proof object is

```text
(kappa, nu),
kappa = K_R,
nu = sum_r c_r 1_[K_(r-1), K_r] du.
```

Concatenating front word `A` with rear word `B` translates the support of the
rear measure:

```text
(kappa_A, nu_A) odot (kappa_B, nu_B)
  = (kappa_A + kappa_B,
     nu_A + shift(kappa_A)_# nu_B).
```

The practical affine transfer is its Laplace image:

```text
beta = exp(-kappa)
m    = integral exp(-u) dnu(u)
I    = m + beta I_bg.
```

This map is a monoid homomorphism into the four-scalar affine transfer
`(beta,m)`. The runtime should store `(beta,m)` and the compact owner word; it
should not materialize `nu`. This is not a foam Schur complement. Eliminating
the ordered depth coordinate would erase the differently colored overlap/order
phenomenon WorldFoam is intended to represent.

The distributional tangent has smooth color-density terms and boundary atoms.
Those atoms explain how moving segment boundaries can have nonzero tangent
effect even when a zero-width segment is the identity in the primal. This is a
proof/tangent interpretation, not by itself a completed geometry/ray tangent
certificate for the compiled temporal surrogate.

## Expansion Pass 2: what is new, useful, and actionable

The five proposed formulations should not become five implementation lanes.
Their repository value is asymmetric:

| Formulation | Current judgment | Action |
| --- | --- | --- |
| Parametric tropical ray-time complex | Rigorous formalization of the kinetic lower-envelope/event compiler already selected | Use for event-completeness and seam proofs; do not fork the compiler |
| Translated optical-depth measure | Strongest repository-derived mathematical addition | Put in the WorldFoam proof spine and tangent/certificate work; keep runtime on compact `(beta,m)` |
| Constructible transfer sheaf | Useful language for chart gluing and root continuation, not a distinct algorithm | Use only where it shortens a real seam theorem; do not market “sheaves” as the method |
| Persistent affine-monoid circuit | Real product-tree option for very long words or sparse local edits | Park until profiling shows ordered-word length/update cost is the bottleneck; it does not remove temporal sampling by itself |
| Similarity-gauge residual foam | Plausible optional preconditioner with an exact isotropic-power-diagram preservation theorem | Park until it measurably lowers event/run/rank counts without breaking the low-degree camera/event contract |

Observed fact: the repository now contains exact kinetic event predicates,
ordered affine-transfer algebra, stable-chart reverse components, a
translated-measure proof oracle, and memory-bounded material replay pieces.

Inference: this is our own repository-derived synthesis and contains real new
structure relative to the earlier codebase. It is not yet evidence of
literature novelty. The correct paper wording is “we derive” or
“repository-derived formulation,” followed by an explicit comparison to prior
volume rendering, kinetic power diagrams, differentiable rendering, and
ordered-transfer work once the literature audit is complete.

Decision: do not derive a sixth formulation now. The active work is systems
closure of one falsifiable fixed-site material path, followed by quiet-host
runtime evidence. New mathematics is justified only when a concrete
correctness, approximation, or measured scaling failure cannot be closed by
the current formulation.

## Scientist-feedback audit

### Supported

- The translated optical-depth measure is the strongest repository-derived
  mathematical addition.
- The canonical implementation architecture should not be replaced by the
  other proposed formulations.
- The currently verified block-major native-shaped evidence is material-only.
- A product tree, similarity-gauge preconditioner, sheaf terminology, or a
  sixth formulation should not become a new implementation lane now.

### Corrected or stale

- Do not call the translated-measure formulation literature-novel until a
  formal novelty review is complete. The proof oracle explicitly makes no
  literature-novelty claim.
- Weighted-total-variation certification and training-safe opacity-tail
  truncation are not landed results. The current oracle covers associativity,
  the Laplace homomorphism, noncommutativity, distributional tangents, P0 VJP
  parity, and one-sided zero-width limits.
- Full geometry is no longer absent from source: an executor-bound request and
  reduction candidate exists. It remains absent from green native evidence and
  end-to-end training.
- Deciding local repair versus full recompilation is no longer the immediate
  mathematical task. Full recompilation is the production rule until measured
  cost justifies reopening repair.

### Exact evidence wording

For a fixed world, camera program, physical interval, tolerance, chart
partition, rank selection, and fallback policy, the verified CPU/fake-native
material path streams requested samples into `O(sum J_c)` node cotangents and
performs one material-word VJP per active compiled block. Expensive
ordered-word work is independent of requested frame density; target, output,
interpolation, and residual work remains linear and rank-weighted in requested
observations.

A source-only full-geometry request path exists, but it has not passed the
latest CPU/fake-native gate, rebuilt-native parity, allocator measurement, or
dataset-bound end-to-end training.

## Source work closed in this session

### Frame-independent world initializer

File:

```text
src/train/paper_kinetic_world_initializer.py
```

The initializer produces CPU float64 direct kinetic sites from an explicitly
transformed point-cloud asset and a separate CPU float32 physical P0 material
seed. It has no target, video, camera, sample, or requested-frame parameter
axis. The current production seed is deliberately narrow: zero velocity,
shared degree-at-most-two weight coefficients, and P0 material.

Cold admission now requires explicit source-byte and source-point-count caps
before the allocating point loader runs. This session also replaced SHA-ranked
selection's `O(source_point_count)` Python tuple list with an exact bounded
heap retaining at most `site_count` entries. The selected canonical
`(sha256_rank,index)` order is unchanged; only cold scratch changed.

### Static-camera exact track factory

File:

```text
src/train/paper_kinetic_active_track_program_factory.py
```

The factory scans the full calibrated camera record sequence and accepts only
content-identical static records. It constructs a constant ray from the
provider's calibrated endpoint witness, compiles exact active owner charts
without requested-frame topology sampling, and uses one fixed externally
provenanced P0 node count. Moving, projective, gauged, and endpoint-fitted
camera programs fail closed.

This session added an explicit pre-compile site-count cap and post-compile
chart/run caps. The site cap is an actual admission bound. Chart/run caps are
still post-compile guards; exact compiler scratch and wall time remain
unmeasured and therefore cannot yet support a publication-scale compiler
claim.

### Fixed-site material state and checkpoint

File:

```text
src/train/paper_kinetic_fixed_site_material_state.py
```

The material-only lifecycle stores raw sigmoid RGB, raw thresholded-softplus
density, a physical `[R,G,B,density]` snapshot, and preallocated raw gradient/
candidate buffers. Persistent storage is 12 float32 scalars per site
(`48 B/site`) and is independent of requested frame count. The stateless manual
SGD checkpoint stores only four float32 raw scalars per site (`16 B/site`) plus
scalar provenance; it stores no frame, sample, target, prediction, autograd, or
optimizer-history tensor.

The first lifecycle now fails closed on non-CPU devices until the optimizer
owns an accelerator completion-fence/quarantine contract. State construction
and restore require an explicit `48 B/site` logical-tensor admission budget;
checkpoint payload ingest requires the expected world site count plus an
explicit `16 B/site` logical-tensor budget before it clones caller-owned
tensors. These are logical admission bounds, not measured allocator peaks.

The sigmoid/thresholded-softplus decode and exact manual VJP no longer have two
implementations. They share the stateless canonical helper in
`research_experiments/world_foam_lane2/material_parameterization.py`, while
optimizer/checkpoint provenance stays in the fixed-site lifecycle. The old
`1e-30` inverse-softplus clamp was removed: tiny positive density gaps now
round-trip when representable and fail closed when a finite raw value cannot
be represented, instead of silently changing the physical seed.

Authorization is revalidated before scratch or parameter mutation. The current
state accepts material-only bars, rejects every geometry/ray authorization,
binds a cold-certified world snapshot plus its exact live physical material
snapshot/generation, and consumes an authorization once. The updater separately
checks the accumulator's world and site-content digests so a same-sized foreign
world cannot supply bars. Receipt construction/sealing is inside the mutation
transaction; any post-mutation failure poisons the state instead of reporting a
clean failed step after parameters advanced. Checkpoints bind exact raw numeric
content on CPU and require the same world snapshot on restore; live state uses
a version-chain plus tensor identity/storage/version signatures to avoid a
device-to-host content hash on every step.

### Caller-owned fixed-site material authorization coordinator

File:

```text
src/train/paper_kinetic_fixed_site_material_step.py
```

The source-integrated coordinator now joins the replayable dense observation
source, byte-bounded compiled-artifact LRU, direct artifact compiler, canonical
view-major/contiguous-pixel partition, built-in bounded target decoder, dense
native request transaction, exact request-delta commits, full-manifest replay
seal, and optimizer authorization. It returns only the still-live
authorization plus the exact accumulator and replay receipt required to
revalidate it. It performs zero parameter mutations; the separate CPU-only
fixed-site updater consumes that capability. Its ledger therefore counts
`authorized_step_count`, not completed optimizer steps, and reports
`authorization_only_external_optimizer_apply_required`. A future trainer must
wrap authorization, material apply, and capability release as one transaction.
On checkpoint restore, a genuinely fresh coordinator can bootstrap only from a
sealed same-world/same-device material state. It restores the authorization
count, prior logical-step identity, and consumed parent material generation;
the current restored material generation remains eligible for the next step.

The step policy preflights world-site, fixed-state, checkpoint, whole-step
accumulator, track, artifact, decode, request, and sample-launch bounds.
Unsupported asynchronous device types fail closed. CPU is synchronous; MPS is
admitted only with the canonical completion fence. Fresh whole-step
accumulator allocation is treated as device progress and fenced before any
post-allocation invariant; a failed initialization fence quarantines the
exact roots without attempting tensor cleanup. Any later failure after partial
or device progress poisons the caller-owned state and retains the source,
session, accumulator, exception context, and other unsafe lifetime roots until
process restart. A static red-team found and repaired an outer-poison path
that had attempted `zero_()` after an inner failed cleanup fence; quarantined
tensors are now preserved rather than touched again.

The behavior test composes the real sealed result into the material updater,
but the test has not run in this session. Therefore this is source-integrated
code, not a green runtime/trainer claim.

## Assumptions and boundaries

- scalar extinction and RGB emission;
- P0 material on each ordered run;
- primary-ray emission-absorption;
- fixed compiled topology for the material update;
- static calibrated cameras in the first production track factory;
- one-sided semantics at birth/death seams;
- full structural recompilation after any geometry or camera-ray change;
- no claim that source tensor accounting equals allocator peak;
- no claim that zero-velocity initialization is a credible dynamic quality
  model.

## Backtracks and risks

1. `tensor-free` does not imply memory-safe. Python objects can still scale with
   source points, frames, tracks, or blocks. The SHA-rank initializer was one
   concrete example and is now bounded in selected-site count.
2. A post-hoc artifact-byte check does not bound compiler scratch. Production
   compilation now has a site admission cap, but stronger internal event/root
   budgets and measured compile peaks are still needed.
3. F-independent persistent state does not imply F-independent total work.
   Camera metadata validation, target decode, interpolation, residuals, and
   output writes remain linear in requested observations.
4. The current zero-velocity fixed-site material trainer can prove lifecycle,
   memory, gradient, loss-decrease, and restart behavior. It cannot prove
   dynamic-scene quality.
5. Static-camera compilation deliberately preserves correctness by rejecting
   the moving/gauged camera case. The retained gauged camera mathematics is not
   deleted; it remains the later projective compiler route and is central to
   World Tubes.
6. Site count and tensor shape are not world identity. The material boundary
   must bind both world generation and exact site-content generation, including
   checkpoint restore.
7. A successful parameter mutation is not complete until its receipt is sealed.
   Failures in post-mutation accounting must poison the mutable state.

## Falsification and promotion gates

Cheap quiet-host source/CPU gate:

1. import the canonical source path with only `PYTHONPATH=src/train`;
2. run the initializer/factory/material/coordinator focused tests;
3. prove identical selected sites and structural program for sparse versus
   dense requested sample schedules over the same camera path;
4. run two material steps, verify finite gradients and decreasing loss;
5. checkpoint after step one, restore, repeat step two, and require bitwise or
   declared-tolerance agreement;
6. compare `F=1` and `F=300` persistent parameter/checkpoint bytes;
7. force site, artifact, target, and failure-quarantine budgets and require
   fail-before-unsafe-release behavior.

Native promotion gate:

1. rebuild and attest the exact native schemas;
2. compare CPU/fake-native and native forward/material bars;
3. measure allocator current/peak and command-buffer queue depth for
   `F=16,64,256` at fixed physical interval and fixed compiler state;
4. require reverse interaction peak ratio `F=256/F=16 <= 1.10` while reporting
   the still-linear sample/target/output bytes separately;
5. run same-world replay versus compiled timing and quality parity before any
   public training claim.

If native peak grows with `F` after fixing world/camera/interval/rank/fallback,
the queue/fence contract is false. If event count, chart count, rank, or
fallback grows with sample density, the strong compiler theorem has been
misapplied. If realistic scenes require enough cells or rebuilds to erase the
memory/bandwidth advantage, narrow or stop the systems claim.

## Open work

- run the focused coordinator-to-CPU-updater gate, including two finite
  updates, loss decrease, checkpoint after step one, restore, and step two;
- run the deferred quiet-host CPU gates;
- measure exact-compiler scratch and add internal event/root budgets if the
  site cap is insufficient;
- rebuild/native parity and allocator telemetry;
- streamed forward-only evaluation and capped media;
- moving/gauged camera compiler and credible dynamic initialization;
- unified-runner registration only after the above evidence;
- eventual cleanup: mark superseded sparse/fake-native coordinators
  reference-only and remove duplicate assemblers only after canonical tests
  migrate.

## Verification performed in this session

Only bounded shell inspection and static whitespace checks were allowed. No
Python interpreter, import, pytest, build, native extension, MPS, Metal, CUDA,
dataset decode, or training workload ran.

## Expansion Pass 3: dtype-correct memory and time-scaling theorem

### Question being answered

The static PowerFoam/"Brilliant Foam" lineage is compact. The key question is
whether WorldFoam can preserve that compact-cell advantage while representing
motion and moving depth order, and whether its expensive raster backward can
be moved off the requested-frame axis in the same sense as World Tubes.

The answer is conditional but decisive:

- the current kinetic ordered-transfer mathematics is sufficient for the
  desired **memory shape** and for frame-independent expensive word work on a
  fixed physical interval;
- it cannot and should not make target I/O, interpolation, residual reduction,
  or output writes sublinear in the number of requested observations;
- the current public `MetalPowerFoamVideo` trainer does not have this shape,
  because it stores model parameters with an explicit frame axis; and
- the cold compiler peak, rebuilt-native allocator peak, and full-geometry
  runtime remain unproved. Those are real unfinished systems gates, not a
  reason to discard the transfer algebra.

### Symbols sharpened for implementation accounting

Use separate names for quantities that older notes sometimes overloaded:

```text
S              global kinetic site count
L_w            weight-trajectory coefficient count per site, currently 1..3
P_r            retained affine camera-ray-program row count
F              requested time samples over one fixed physical interval
B_p            maximum simultaneously active spatial-track request size
K              maximum simultaneously active temporal block size
q              one native equal-rank block
Q_q            track-chart row count in block q
J_q            temporal transfer-node count in block q
W_q            flattened ordered-word entry count in block q
S_q            compact referenced-site count in block q
U              union referenced-site count for one spatial request
N              observations in one sample launch, N <= B_p K
E              continuous topology/order event count
A              adjacency/face-incidence count where needed
```

`Q_q` is deliberately not called `R_q`. A native row is one `(track,chart)`;
`W_q` is the number of ordered cell-run entries across those rows. Confusing
rows with runs makes both node-state and word-replay formulas wrong.

### Static PowerFoam versus the current dynamic wrapper versus WorldFoam

The copied static PowerFoam raster contract stores float32

```text
points[S,3], radii[S], densities[S], features[S,C]
```

plus int32 neighbor CSR. Its raw model payload is therefore

```text
M_powerfoam_static_model = 4 S (5 + C) bytes,
M_powerfoam_static_csr   = 4 (A + S + 1) bytes.
```

For P0 RGB, `C=3`, so the raw cell table is exactly `32 S` bytes before
gradients, optimizer history, decoded aliases, raster scratch, and allocator
overhead. This is genuinely compact. It is also one static world; it does not
by itself encode arbitrary motion or reuse a backward across time.

The current paper-baseline class `MetalPowerFoamVideo` constructs
`raw_xy`, `raw_z`, `raw_radii`, `raw_densities`, and `raw_features` with a
leading `frame_count` axis. Even in its simplest P0 RGB mode its raw model lower
bound is

```text
M_current_per_frame_P0_raw = 4 F S (2 + 1 + 1 + 1 + 3)
                           = 32 F S bytes.
```

Its richer surface/texel/SV modes and persistent initialization/EMA buffers
add more. This explains the historical 28.6-million-parameter, roughly
116.7-MB-checkpoint progressive row. It is a valid per-frame baseline, not the
memory-light WorldFoam implementation.

The direct-kinetic P0 WorldFoam source stores CPU float64
`positions0[S,3]`, `velocities[S,3]`, and
`weight_coefficients[S,L_w]`, plus a physical float32
`site_rgba[S,4]`. The exact initialized-world payload is

```text
M_kinetic_geometry = 8 S (6 + L_w),
M_physical_P0      = 16 S,
M_initialized_world = 8 S (8 + L_w) bytes.
```

At the current maximum `L_w=3`, this is `88 S` bytes. It is about `2.75x` the
raw static P0 cell table per site, but it is independent of `F`. That constant
factor buys affine 3D motion and quadratic power-weight motion rather than a
per-frame cell copy.

The landed material-only training state additionally owns raw RGB/density and
their preallocated raw bars. Including its physical material snapshot, it is
exactly

```text
M_material_live = 48 S bytes,
M_material_checkpoint = 16 S bytes.
```

Consequently, current direct-kinetic geometry plus the complete fixed-site
material training state is

```text
M_geometry_plus_material_session
  = 8 S (6 + L_w) + 48 S
  = 8 S (12 + L_w) bytes,
```

or `120 S` bytes at `L_w=3`, before compiled programs. Ignoring topology and
optimizer differences, that crosses below the simplest `32 F S` per-frame raw
table once `F >= 4`. The comparison is illustrative, not a benchmark: the two
renderers store different structural state and use different dtypes.

### Correction to `host_memory_contract.py`

The current properties

```text
global_source_parameter_tensor_bytes = 36 S + 48 P
global_gradient_tensor_bytes         = 36 S
```

are legacy fixed-4D/float32 accounting. `36 S` corresponds to nine float32
site/material scalars and `48 P` to twelve float32 affine-ray scalars. Those
numbers do not describe the current direct-kinetic source, whose
`AffineKineticPowerSites` and camera-ray programs are CPU float64.

For the current direct-kinetic source, the exact replacements are:

```text
geometry source                    8 S (6 + L_w)
physical P0 material              16 S
retained affine-ray programs       96 P_r

source + material + stored rays
  = 8 S (8 + L_w) + 96 P_r bytes.
```

At `L_w=3`, that is `88 S + 96 P_r`, not `36 S + 48 P`.
If rays are generated procedurally from calibrated cameras and a pixel lattice,
the persistent `96 P_r` term can become camera metadata plus bounded
`O(B_p)` row staging; that reduction is an implementation option, not current
source evidence.

The exact whole-step gradient accumulator is lane-dependent:

```text
material-only:
  M_step_bar = 16 S + 4

full geometry:
  M_step_bar = 16 S
             + 8 S (6 + L_w)
             + 96 P_bar
             + 4 bytes,
```

where `P_bar` is the exact selected `(view,pixel)` affine-ray bar catalog. At
`L_w=3`, the full bar is `88 S + 96 P_bar + 4`. Therefore the old `36 S`
gradient property is also stale for the direct-kinetic path. This audit does
not patch that utility because this session is restricted to proof/source
inspection; future code should either parameterize it by representation/dtype
or stop exposing those legacy properties as general WorldFoam totals.

### Exact persistent compiled-program payloads

For one materialized equal-rank source block, the implementation stores:

```text
int64: source sites + global rows + track ids
int32: chart ids + CSR offsets + owners + config
bool:  right-closure bit per row
f64:   t_min, t_max, near, far per row
f32:   physical lengths [J_q,W_q].
```

Expanding `_expected_block_tensor_bytes` gives the exact tensor payload

```text
M_source_payload,q
  = 8 S_q + 57 Q_q + (4 + 4 J_q) W_q + 20 bytes.
```

The source lane's conservative runtime-copy allowance for that block is

```text
M_runtime_copy_upper,q
  = 8 S_q + 4 Q_q + (4 + 4 J_q) W_q + 24 bytes.
```

Some launch tensors may alias rather than copy, so the second formula is an
admission upper bound, not a claim that every byte is duplicated.

The CPU artifact store separately charges all reachable CPU tensor bytes and
canonical descriptor/key bytes. There is no honest closed coefficient formula
for its Python/metadata content. The correct bound is the caller's explicit
byte-cap `C_artifact`, plus an independently bounded native-token cache
`C_native`; neither target observations nor device runtime objects are admitted
to the CPU artifact store.

Thus the persistent structural state is better stated as

```text
M_structural_persistent
  <= C_artifact + C_native + M_descriptor_scalar_metadata,
```

with output complexity determined by track-local charts, words, events, and
ranks, not by requested frame samples. It may still be large in spatial
resolution or physical event complexity.

### Exact successful-step live tensors

For all native blocks `q` belonging to one spatial request, the landed
material-only block-major schedule retains a compact material snapshot and
node state/bar for every active block until all its `K` chunks have reduced.
The source's exact active-state admission formula is

```text
M_active_material
  = sum_q [16 S_q + 32 Q_q J_q]
    + 16 U
    + 16 max_q S_q
    + 4 (# active blocks)
    + 16 bytes.
```

The terms are, respectively:

```text
compact material per block               16 S_q
node transfer plus node cotangent         32 Q_q J_q
union-local material bar                  16 U
one sequential compact material bar       16 max S_q
per-block scalar losses + total/bg         4 n_q + 16.
```

Material-only reverse allocates no `[J_q,W_q]` geometry-length bar. Full
geometry adds at most one sequential native length bar

```text
M_length_bar_peak = max_q 4 J_q W_q bytes,
```

plus request-local CPU float64 geometry/ray bars. The current request-local
geometry bridge uses full site arrays rather than union-local site arrays:

```text
M_request_geometry_bars
  = 8 S (6 + L_w) + 96 P_request bytes.
```

That is independent of `F`, but it is a real constant-factor and bandwidth
cost. A future compact/union-local geometry reduction could lower it without
changing the math.

For one ragged sample launch of `N` observations at rank `J_q`, the sealed
source block retains exactly

```text
M_sample_block = 4 N J_q + 24 N bytes,
```

for float32 weights, RGB targets, int32 row ids, and int64 provenance ids. The
native prepare boundary adds a transient `8 N` CPU row conversion, `4 N`
device row copy, and 20 config bytes, so the named public tensors at that seam
are

```text
M_named_sample_peak = 4 N J_q + 36 N + 20 bytes.
```

This is not the whole-path peak. `_materialize_dense_sample_block` also builds
float64 `[N,J_q]` weights before the float32 copy; Python groupings, allocator
lifetimes, and device command buffers are unmeasured. The proof must not turn a
known lower boundary into an allocator upper bound.

The built-in target decoder's exact public-tensor bridge peak is

```text
M_decode_peak
  = max(
      12 N_chunk + 12 H W + 28 N_frame_max,
      12 N_chunk * (2 if target_device != CPU else 1)
    ).
```

Only one full decoded frame is present. The second branch counts simultaneous
CPU/device chunk copies. PIL/NumPy and allocator internals remain outside it.

A useful whole-step envelope is therefore

```text
M_success_peak
  <= M_world_and_optimizer
   + M_structural_persistent
   + M_step_bar
   + max_request(
       M_active_material
       + [M_length_bar_peak + M_request_geometry_bars if full geometry]
       + M_named_sample_peak
       + M_decode_peak
       + unmeasured bounded implementation overhead
     ).
```

No sum over temporal chunks appears in the live scratch term. That fact relies
on the one-in-flight completion-fence policy; source lifetimes alone do not
prove it for an asynchronous allocator.

### Forward and backward work theorem

For a warm compiled block, exact word evaluation at temporal nodes costs

```text
T_node_forward = Theta(sum_q J_q W_q).
```

Streamed sample evaluation and RGB loss for launches `l` cost

```text
T_sample_forward = Theta(sum_l N_l J_(q(l)) + sum_l N_l).
```

The loss-only path need not allocate a prediction tensor. Evaluation/media may
add `12 N_l` bytes of bounded prediction output, or `12 P F` if a caller
incorrectly retains the whole rendered video.

Backward first reduces sample residuals into the already resident node bars:

```text
T_sample_to_node = Theta(sum_l N_l J_(q(l))).
```

After complete coverage of a spatial bundle, each active ordered word is
replayed once at its nodes:

```text
T_word_VJP = Theta(sum_q J_q W_q).
```

The two-pass prefix-only word VJP uses final transfer plus one prefix state, so
its thread-local interaction scratch is `Theta(1)` in word length. Material
bars are scattered into bounded compact/union/global buffers. Full geometry
also reduces one sequential `[J_q,W_q]` length bar through the frozen kinetic
chain. It adds geometry work, but still no requested-frame tape.

The warm material step therefore has the desired split:

```text
heavy ordered-world work
  = Theta(sum_q J_q W_q) forward
    + Theta(sum_q J_q W_q) reverse,

cheap requested-sample work
  = Theta(sum_l N_l J_(q(l)) + P F).
```

The exact-stream fallback instead costs `Theta(sum_l N_l r_l)` but retains
constant local reverse state. It is the correct route for shallow words whose
`J`-node compile does not break even.

Cold structural compilation is separate:

```text
T_compile
  = O(U_owner S R_max)
    + O(W_closure (S log S + S R_max))
```

for the landed active compiler's observed-word predicate construction and
monotone closure. This cost is independent of requested frame density, but it
is not yet a final output-sensitive event-queue theorem. More importantly,
its peak scratch is not measured or conservatively preflighted inside the
compiler. The exhaustive `O(S^3)` compiler and dense continuous-dual
certificate are references only and must never enter production-scale training.

### Precise frame-density proof

Fix all of the following:

```text
world parameters and temporal basis,
continuous camera program,
physical time interval,
error tolerance,
continuous event/chart partition,
owner words and fallback decisions,
certified ranks J_q,
spatial block policy B_p,
temporal live-block policy K.
```

Now vary only the number `F` of requested samples in that same interval.
Continuous compilation does not inspect the requested sample grid, so

```text
E, Q_q, W_q, J_q,
M_initialized_world,
M_structural_persistent,
M_step_bar,
M_active_material,
node-forward count,
material-word-VJP count
```

are invariant in `F`. The number of sample launches and the total
sample-to-node interactions grow with `F`, but each launch remains bounded by
`N <= B_p K`. Therefore expensive reverse interaction peak and world-word work
are independent of requested frame density, while total work still obeys the
unavoidable lower bound

```text
Omega(P F)
```

for reading/comparing or emitting `P F` RGB observations.

A stochastic trainer may hold the number of sampled observations per optimizer
step fixed, making the entire **step** independent of dataset `F`. That changes
the statistical schedule, not the lower bound: one exact full-coverage epoch or
evaluation still performs `Omega(PF)` observation work.

The current replay source also retains `24(F+V)` bytes of cheap scalar frame/
view metadata. Strict end-to-end constant memory in `F` would require a
streamed metadata iterator. This does not reintroduce an `F x word` tape, but
it must be reported rather than described as zero frame dependence.

This theorem does **not** hold merely because a configuration calls a longer
video "more frames." If increasing `F` also extends the physical duration,
then the trajectory basis size, event count, chart count, or approximation rank
may grow with duration. No representation can encode arbitrary new motion with
constant state. The strong claim is sampling-density scaling on a fixed
physical program. Longer-duration scaling is conditional on physical motion
complexity `L_w,E,J,W = o(F)` or must use bounded temporal windows.

### Does the mathematics need to be redone?

Not for the current memory target. The translated optical-depth measure and its
affine `(beta,m)` image provide exactly what the renderer needs:

1. an associative, order-explicit transfer object;
2. a four-scalar runtime quotient at each compiler node;
3. a constant-local-state exact word VJP; and
4. boundary-measure language for the missing geometry tangent certificate.

Changing the transfer representation cannot remove the `Omega(PF)` sample
term, and marginalizing depth would destroy the differently colored overlap
signal. The remaining mathematical gap is narrower: certify the uniform
geometry/ray tangent error of the actual affine-Lie barycentric surrogate and
close exact algebraic seam dispatch. Those strengthen the derivative theorem;
they do not change its asymptotic storage form.

Reopen formulation work only if one of these falsifiers occurs:

1. `J` or `E` grows when only the sample grid is densified. First check for a
   sample-driven compiler/rank bug; a continuous compiler should not do this.
2. Real fixed-interval charts require such high `J` that exact replay always
   wins. Then test a better nonlinear local transfer chart or a product tree,
   not depth marginalization.
3. Physical duration/event complexity grows linearly with video length. Then
   use windowing, periodic/low-rank motion, or admit linear scene complexity;
   arbitrary dynamics have no constant-state theorem.
4. Full geometry tangent error fails despite primal parity. Then repair the
   tangent certificate/surrogate, not the ordered monoid.
5. The target physics adds scattering, matrix-valued attenuation, or spatially
   varying within-cell material. Then the four-scalar affine statistic itself
   must be enlarged.
6. Compiler peak violates its budget. Then stream active predicates and
   artifacts blockwise; this is a compiler algorithm problem unless event
   output itself is intrinsically too large.

Two numerical edge cases also remain implementation gates rather than new
representation requirements. At tiny optical depth, `(1-exp(-kappa))` and the
affine-Lie encode/decode must use their analytic limits. At very large optical
depth, float32 `beta=exp(-kappa)` may underflow and suffix influence becomes
numerically zero; cancellation and gradient accuracy must be tested against a
float64/log-domain oracle. Neither case changes the memory theorem, but either
can invalidate a quality or gradient claim if hidden behind finite-value checks.

### Exact acceptance invariants for the runtime gate

For fixed world/camera/interval/tolerance and `F in {16,64,256}`, a promotable
runtime report must show:

```text
identical world and compiler provenance,
identical E, Q_q, W_q, J_q and fallback decisions,
identical persistent world/optimizer bytes,
identical artifact and cached-topology logical bytes,
identical whole-step accumulator bytes,
identical node-forward and word-VJP counts/interactions,
zero resident F*S, F*W, P*F target/ray, or frame-by-word tape,
sample/target blocks bounded by B_p*K and the formulas above,
one in-flight sample block at every measured allocator boundary,
reverse-interaction allocator peak ratio F=256/F=16 <= 1.10.
```

It must separately report the quantities that are expected to grow:

```text
sample count and launch count,
sample-to-node interactions,
target bytes decoded/transferred over the whole step,
output/residual writes,
the current 24(F+V) host metadata,
wall time for the linear camera/sample slice.
```

A stable overall allocator peak without those category checks is insufficient:
framework caching can hide an `F`-dependent tensor. Conversely, linear total
wall time is not a failure if node forward, word VJP, and reverse interaction
memory remain invariant and only the explicitly allowed sample slice grows.

### Revised confidence

Current belief: WorldFoam can be compact in the intended sense and does not
need new transfer mathematics. Confidence is high for the algebraic and
source-lifetime theorem, medium for the corrected logical-byte accounting, and
low for native allocator/compiler peak until the quiet-host gates run.

The most important backtrack is that the legacy `36 S + 48 P` and `36 S`
figures must not be cited for direct kinetic WorldFoam. Correcting those numbers
increases the constant factor, but it does not change the frame-independence
result or create any intrinsic 32-GB requirement.

## Expansion Pass 4: translated-measure incorporation and evidence boundary

The scientist's optical-depth translated-measure formulation is **already
incorporated**. It is not a missing replacement for the current runtime and it
should not become another implementation branch or public method name. The
durable derivation is theorem `T0` and Section 6.1 of
`research_notes/worldfoam_paper/WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md`;
the paper-facing named theorem and its certificate corollaries are in Section
3.4 of
`research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md`; the implementation
contract is closed in `TODO/worldfoam_memory_light_native4d.md`; and
`research_experiments/world_foam_lane2/optical_depth_translated_measure_oracle.py`
is the deliberately proof-only CPU certificate.

For a front-to-back P0 word with segment optical depths `tau_r`, cumulative
depths `K_0=0`, `K_r=sum_{q<=r} tau_q`, and colors `c_r`, define

```text
kappa = K_R,
d nu(u) = c_r du,  u in [K_{r-1}, K_r).
```

For a front word `A` and rear word `B`, concatenation is the shifted-measure
product

```text
(kappa_A,nu_A) odot (kappa_B,nu_B)
  = (kappa_A+kappa_B,
     nu_A + shift(kappa_A)_# nu_B).
```

Translations compose, hence `odot` is associative. Its Laplace image

```text
L(kappa,nu) = (beta,m)
beta = exp(-kappa)
m = integral exp(-u) d nu(u)
```

is a monoid homomorphism into affine optical transfer:

```text
L(A odot B)
  = (beta_A beta_B, m_A + beta_A m_B)
  = L(A) star L(B).
```

For a differentiable fixed owner word, the exact distributional tangent is

```text
dot nu =
    sum_r dot(c_r) 1_[K_{r-1},K_r) du
  + sum_{r<R} (c_r-c_{r+1}) dot(K_r) delta_{K_r}
  + c_R dot(kappa) delta_kappa,

dot m = integral exp(-u) d(dot nu)(u).
```

This theorem does three useful things: it makes order explicit without trying
to marginalize it away, proves that compact `(beta,m)` composition is the
correct quotient for rendered action, and exposes density/geometry tangents as
moving optical-depth boundary masses. It does **not** certify derivatives
through owner-word births, deaths, chart changes, or rank/dispatch choices, and
it does not by itself bound the temporal interpolation error of the kinetic
surrogate.

There is no runtime or memory change. Native execution should continue to use

```text
kinetic charts -> ordered owner words -> J-node affine/Lie approximation
               -> streamed node cotangents -> one word VJP.
```

`nu` is a proof object only. The executor stores compact owner-word data and
node `(beta,m)`/Lie coordinates; it must not discretize or retain `nu`, create a
per-frame depth tape, or add an `F`-scaled payload. Consequently none of the
Pass-3 byte or time formulas changes.

The evidence boundary must remain explicit:

1. The translated-measure CPU oracle proves the algebra, homomorphism,
   commutator behavior, distributional tangent, and fixed-P0 VJP on bounded
   fixtures. It is not a renderer or performance result.
2. The integrated block-major CPU/fake-native path currently proves streamed
   accumulation followed by exactly one **material-only** ordered-word VJP per
   active compiled block, including material-gradient parity and
   frame-density-invariant compiled-word counts.
3. A source-level full-geometry candidate has the intended one full word VJP,
   bounded `[J,W]` physical-length bar, immediate kinetic reduction, receipts,
   and fences. Its latest integration has not been run, and the installed
   native extension predates the required schemas.
4. Therefore full kinetic-geometry native reverse, real accelerator fencing,
   allocator peak, and continuous geometry/ray tangent certification remain
   separate acceptance gates. Material-VJP evidence must not be cited as if it
   closed them.

No new formulation is required for the memory-light target. The next useful
work is implementation and evidence: rebuild in a quiet environment, verify
the native material path, then separately run the full kinetic-geometry reverse
and allocator/scaling gates. If those fail, diagnose the concrete failing
boundary before reopening the transfer mathematics.

## Expansion Pass 5: hidden-retention closure and executable evidence boundary

The scientist feedback changed no runtime architecture. It did sharpen the
division of labor:

```text
(kappa,nu) translated optical-depth measure  proof/tangent object
(beta,m) affine transfer                     native runtime quotient
kinetic charts + owner words + J nodes       temporal compiler/runtime
streamed node cotangents + one word VJP      expensive reverse schedule
```

The translated-measure theorem is now explicitly described as a
repository-derived proposed contribution. It is new relative to the older
repository scaffold, but no literature-novelty claim is closed. Beer--Lambert,
optical measures, affine alpha composition, product integrals, and prefix VJPs
remain prior foundations. The other four proposed formulations remain compiler
theorem language, an optional similarity preconditioner, or deferred data
structures rather than replacement executors.

### Source-only memory defects closed in this pass

1. **Fixed-camera ray bars.** The full-geometry source candidate now defaults
   to fixed cameras and omits both the request-local and whole-step
   float64 `[P_bar,12]` ray cotangent. At `2704x2028`, that tensor alone would
   be `526,436,352` bytes (`502.05 MiB`) per selected view. Camera calibration
   remains an explicit opt-in and is not part of the default paper trainer.
2. **Interpolation scratch.** Float64 barycentric evaluation is split into
   bounded row subchunks. For `N` materialized samples, node rank `J`, and
   subchunk size `K_sub`, the source-visible upper bound is

   ```text
   max(
     N(8J+12) + 4096+512J+8J^2+K_sub(1024+512J) + 16K_sub,
     N(16J+32)
   ).
   ```

   Public native preparation adds `4N+20` bytes after removing a redundant
   CPU int64 row clone. This is logical tensor accounting, not allocator/RSS
   measurement.
3. **Cold initializer retention.** `PaperKineticLazyProgramBundleProvider`
   no longer owns the point-cloud initializer after it clones the live world.
   It retains only the initializer provenance/generation receipt already bound
   into the provider/world digest. This removes a hidden second geometry
   template plus physical RGBA seed: `8S(6+L_w)+16S`, or `88S` bytes for the
   degree-2 ABI. The caller must still create the material state and release
   its own initializer and `PaperKineticP0MaterialInitialization` references.
4. **Exact selected-path accounting.** For `L_w=3`, steady shared geometry plus
   the five-tensor fixed-site material state is `72S+48S=120S` bytes. During a
   live material-only step, the physical RGBA bar and scalar loss make the base
   `136S+4` bytes. Optimizer moments, compiled artifacts, target decoding,
   interpolation, native scratch, allocator slabs, and driver memory are
   separate categories.
5. **Row-ragged launch shape.** The selected executor uses flattened
   `N <= B_p K`, not a legacy `[K,J]` weight table. The retained launch block is
   `4NJ+24N`; coexisting public preparation gives
   `4NJ+28N+20`. The host audit now reports the older compact P0 binding as a
   legacy per-block row and marks its all-spatial-block residency unbounded.
   The direct-kinetic program path instead reports the explicit bounded CPU
   artifact-store byte budget.
6. **Mode-exact deltas.** A full-geometry request delta must contain all three
   site-geometry tensors, and a ray tensor iff camera-ray optimization is
   enabled. Matching only the total tensor count could admit a missing site bar
   plus a stray ray bar; the seal now checks the exact field pattern.

All six edits are source-only and unrun on the saturated host.

### Scaling evidence was red-teamed, not merely schema-expanded

The first `F=8/64/300` verifier was insufficient because a hand-authored JSON
could assert every measurement flag and because it conflated the target-chunk
cardinality `N_target` with the smaller native-launch cardinality `N_launch`.
The corrected contract separates both dimensions and checks the coordinator's
emitted policies. It also requires all intermediate frame rows, so a memory
spike at `F=64` cannot hide behind equal `F=8` and `F=300` endpoints.

A new MPS/Metal-bound opt-in producer launches every frame/repeat trial in a
serial fresh subprocess, nonce-binds the receipt, hashes the source/config/driver/hardware
and compiled extension, attests the kinetic ABI before the driver runs, and
passes the exact native module identity into the driver. It rejects fake-native
and non-production-coordinator reports before invoking the verifier.

This is still not evidence. No checked-in real trial driver currently connects
the producer to a rebuilt native coordinator, and no trial ran. The producer
is a fail-closed execution scaffold; the missing driver, rebuilt extension,
allocator probes, and resulting rows are the next runtime work.

### Current proof/evidence frontier

What is now defensible:

- for a fixed physical interval, fixed certified charts/ranks/words, and
  bounded `B_p,K`, expensive node forward and ordered-word reverse state are
  independent of requested frame density;
- unavoidable camera/target/sample interpolation and residual work remains
  linear in requested observations;
- the current integrated end-to-end evidence is material-only CPU/fake-native;
- a source-level full-geometry candidate has bounded local bars and one word
  VJP per active block, but it is not native/trainer evidence.

What remains open:

1. write the checked-in real-native trial driver around the production
   coordinator and allocator/target-loader probes;
2. rebuild and attest the native extension on a quiet Mac; a CUDA result first
   needs its own native port and bound producer;
3. run the `F=8/64/300`, three-repeat material-only gate;
4. run the distinct full-geometry parity/scaling gate;
5. certify uniform geometry/ray tangent interpolation and seam dispatch;
6. wire streamed evaluation and only then register `worldfoam_native4d` in the
   unified paper runner.

Reopen mathematics only if those measurements show chart/event/rank growth
under frame-density refinement, intolerable rank at fixed tolerance, or a real
full-geometry tangent failure. A failed allocator budget by itself is a
streaming/lifetime problem, not evidence that the translated-measure/affine
transfer formulation is wrong.

## Expansion Pass 6: scientist verdict accepted; v3 producer authority closed in source

The external scientist's final scorecard does not change the selected runtime.
It makes the research accounting more precise:

```text
kinetic event charts
  -> ordered owner words
  -> J-node temporal approximation
  -> streamed node cotangents
  -> one material word VJP per active block
```

The five-formulation exercise produced one strong formulation newly derived in
this project: the translated optical-depth measure `(kappa,nu)` with compact
Laplace quotient `(beta,m)`. It is already integrated in the paper draft,
theorem ledger, TODO, and proof-only CPU oracle. It is useful because it proves
ordered composition, exposes moving optical-depth boundaries as tangent atomic
mass, and explains why the four-scalar affine transfer is sufficient for the
declared rendered action. It is not a sixth executor and should not be added to
native state. External literature novelty is still unreviewed and must not be
claimed from the internal derivation alone.

The other proposed formulations are narrower: kinetic lower-envelope and
constructible-chart language formalize the existing compiler; a similarity
gauge is an optional preconditioner that must earn its extra camera complexity;
and product trees or full ray--time arrangements remain deferred until measured
cost or a proved local-repair theorem requires them. The production geometry
update rule therefore remains conservative: material changes may reuse sealed
structure, while every geometry, weight-trajectory, or camera-ray change gets a
fresh complete structural compile and recertification. Simple-root continuation
is still an oracle/certificate, not a local program patcher.

### Source-only scaling closure added after Pass 5

Pass 5's statement that the real driver was missing is superseded. A checked-in
MPS coordinator driver and config now exist for the material-only schema-v3
matrix. No row has run and the native extension remains unbuilt.

1. The procedural acceptance source implements direct selected-pixel reads,
   preserves request order and duplicates, preflights its source-visible budget,
   and reports zero full-frame materializations. This removes an audited
   `5.41 TiB` transient full-frame-decode amplification at `F=300`; the payload
   actually needed is about `506 MiB` total and at most `49,152` bytes per
   selected RGB chunk. This is an avoided source-level counterfactual, not a
   measured allocator result. Public compressed data still needs independently
   decodable tiled or mmap-backed targets.
2. Each `F=8/64/300` row builds its own provider/cache identity but represents
   the same static camera, world, physical interval, rank, and compiler policy.
   Provider, camera-grid, and artifact-cache keys remain intentionally
   F-specific; only semantic world/camera/interval/program/lowering signatures
   are compared across rows. The former ambiguous provider/world digest field
   is now named `provider_world_generation_digest` so it cannot masquerade as
   the semantic world-content digest.
3. The producer source manifest now walks the transitive local Python import
   closure without importing runtime code, including relative imports and
   ancestor package initializers. It also seals the native package, C++/
   Objective-C++/Metal sources, and native `setup.py` build source. The manifest
   digest, native binary digest, driver/config digests, and hardware identity are
   independently bound.
4. Driver capability schema 3 deliberately claims only the MPS backend and the
   `PowerFoamSelectedPixelRead/v1` direct-pixel contract. The dead v2 allocator/
   compiler measurement strata were removed; real measurements must arrive as
   producer-owned receipts.
5. The producer is written to apply a per-process MPS allocator limit whose
   effective bound is at most `2 GiB`, retain the raw limit receipt, and bind its
   digest in every trial. The public MPS current/driver counters use a configured
   5.0-ms sampler; their maxima remain lower bounds, never exact peaks. A lock
   now serializes background and completion-fence samples so maxima/counts cannot
   be lost to a writer race.
6. A separate parent polls process-group RSS at a configured 0.25-second
   interval and terminates after a sampled value above `4 GiB`. This is a
   sampled watchdog, not an exact allocator-hard RSS cap. It now fails closed if
   a live process group cannot be observed with positive RSS. Verification
   requires positive RSS, at least two samples, a feasible sample-count/elapsed-
   time relation, clean exit, empty process group, and no watchdog termination.
   The raw watchdog receipt is hashed together with the unique child execution
   evidence and bound into the normalized trial.
7. The selected material-path resource attestation is exactly three Metal
   functions: node forward, loss-only sample accumulation, and material-only
   word VJP. The prior documentation phrase naming full geometry VJP was wrong
   and is corrected. The query records only observable maximum threads,
   execution width, and static threadgroup memory; it neither proves execution
   nor exposes registers, private memory, or spills.

The 8-GiB available-host launch guard is incident headroom. It is not a 32-GB
representation requirement. The actual acceptance allocator limit is at most
2 GiB on MPS, with the separate sampled 4-GiB process-group watchdog described
above.

### Evidence boundary and next action

This pass ran no Python, imports, tests, builds, Metal/MPS, CUDA, dataset,
training, or native workload. All changes are source-only. Static JSON,
whitespace, source-reference, and conflict-marker checks are the only allowed
local validation.

The next approved-runtime sequence is:

1. rebuild the Metal extension from the sealed sources and run focused source/
   ABI behavior gates on a quiet host;
2. execute three fresh-process repeats at `F=8/64/300` and require the v3
   verifier to pass without relaxing the 2-GiB allocator or 4-GiB sampled-RSS
   policies;
3. add a tiled/mmap real-target backend and repeat on public data;
4. run a separate full-geometry native parity and memory/scaling gate; and
5. only after those results, register `worldfoam_native4d` in the unified paper
   runner or reopen the mathematics.

Reopen the transfer formulation only if fixed-world/frame-density refinement
forces chart/event/rank growth, the required `J` is uneconomic at fixed error,
or the complete geometry tangent fails its physical reference. An allocator or
decoder failure alone remains a systems/lifetime defect.

## Expansion Pass 7: public-target identity closure and the exact remaining trainer boundary

The external scientist's final verdict is accepted without changing the
executor. The most useful combination remains:

```text
project-derived mathematical layer
  translated optical-depth measure (kappa,nu)
    -> proves ordered concatenation, boundary tangents, and the Laplace quotient

canonical runtime layer
  kinetic event charts
    -> ordered owner words
    -> J-node transfer approximation
    -> streamed node cotangents
    -> one word VJP per active compiled block
```

The proof object must not become a second stored representation. The runtime
continues to store compact owner words and `(beta,m)`/affine-Lie node state. The
five-formulation exercise produced one strong formulation new to this project,
not five replacement algorithms. External literature novelty is still open.

### Public selected-target source: what is now actually implemented

`src/train/powerfoam_training_data.py` now contains a standalone
`MappedRgb8PowerFoamTargetSource` with this data layout:

```text
one payload per camera
shape on disk: [height,width,stored_frame,RGB]
dtype on disk: uint8
selected result: contiguous CPU float32 [N,3]
mapping lifetime: at most one camera payload at a time
```

The source preserves arbitrary request order and duplicates, maps only during
one selected-pixel call, copies the selected bytes, closes the mapping before
normalization/return, and returns the existing sealed selected-pixel receipt.
The source-visible logical-tensor bound is conservatively `70N` bytes:

```text
12N  output float32 RGB
40N  five int64 index families
 6N  two possibly overlapping uint8 RGB selections
12N  normalized float32 selection
----
70N
```

The mapping itself is not called tensor memory. Its receipt separately records:

```text
maximum mapped address-space bytes
maximum requested unique pages
total requested unique pages within the read
page size
maximum and total requested-page byte upper bounds
mapping closure before return
```

Request accounting accumulates these per chunk, and the fixed-step accounting
emits both peaks and cumulative requested pages/bytes. This closes the earlier
source-to-step receipt drop. It does **not** measure OS residency, readahead, or
page-cache pressure.

The cache manifest is now fail-closed at source level:

1. at most `1 MiB` of manifest JSON is read;
2. duplicate JSON keys and unexpected object keys fail;
3. view records must be sorted and use canonical relative paths beneath the
   manifest directory;
4. each declared payload must fit a caller-supplied mapped-address-space cap,
   and the complete selected payload set must fit a separate construction-
   verification I/O cap;
5. every payload is hashed through the same opened file descriptor between
   `fstat` checks;
6. the device/inode/size/mtime/ctime signature is checked before and after each
   mapping.

The old `decoded_f32_sha256` cache-manifest field was removed. It was declared
but not independently proved by the runtime loader and therefore looked more
authoritative than it was. The cache manifest now binds cache bytes only.
Raw-decoder-versus-cache decoded equality belongs to a separate dataset binding.

### Exact scale of the target issue

For the current `384 x 384`, `F=300` mechanical matrix:

```text
one camera RGB8 payload
  = 384 * 384 * 300 * 3
  = 132,710,400 bytes
  = 126.56 MiB

four-camera total raw target cache
  = 530,841,600 bytes
  = 506.25 MiB

maximum selected float32 RGB chunk
  = 4,096 * 3 * 4
  = 49,152 bytes
```

This is why the earlier `5.41 TiB` decoder estimate was a software amplification
bug rather than an intrinsic WorldFoam requirement. The old compatibility path
repeated full-frame decoding inside pixel-major chunks. The selected-pixel path
does useful work linear in requested observations and keeps the live decoded
target chunk bounded.

This still does not prove native-resolution safety. At `2704 x 2028`, one
300-frame RGB8 camera payload is about `4.60 GiB`; a full construction hash scan
and sequential page touches can create substantial system page-cache pressure
even though process tensor memory stays small. A native-resolution backend must
therefore add bounded tiles/windows or explicit cache-pressure control rather
than citing mmap address space as a memory result.

### Dataset binding: source schema exists; public evidence does not

`research_experiments/world_foam_lane2/worldfoam_target_dataset_binding.py`
now provides a strict stdlib-only `target_dataset_binding/v1` validator. It
binds:

```text
dataset id and train split
converter source identity
raw dataset-manifest and per-view input identities
matching declared raw-decoded and cache-decoded float32 hashes per view
strict mapped-cache manifest and payload identities
stored frame grid and RGB conversion/hash contract
static per-view intrinsics K[V,3,3]
dynamic extrinsics w2c[V,F,4,4]
frame times, lens models, optional distortions, pose source
exact logical frame maps for the required F rows
one canonical binding SHA-256
```

It can rehash the cache manifest and payload files actually consumed by
training, rejects path escape and duplicate JSON keys, and byte-caps both the
binding and mapped manifest. It intentionally does not decode raw videos or
recompute either decoded identity. Equality at this layer means only that the
two declared decoded hashes match. The future converter/companion must derive
both hashes independently from the raw decoder and cache bytes before treating
that declaration as evidence.

No converter, populated binding, generated cache, companion verifier, or
trainer selection of the mapped source exists yet. The base procedural
schema-v3 acceptance contract remains unchanged. The future public companion
must first require a complete v3 pass, then additionally bind the dataset/cache
identity and require all selected reads to be mapped, closed, page-accounted,
and free of full-frame materialization.

### Full geometry is a different gate from material-only scaling

The current fixed-site paper coordinator still invokes the material-only mode.
Its strongest integrated claim remains:

```text
one material word VJP per active compiled block
after complete request replay
with bounded target/sample chunks
```

A mutually exclusive full-geometry executor/request path exists in source. It
returns bounded `[J,W]` physical-length cotangents, immediately reduces them
through the kinetic stable-stratum bridge, fences block/request commits, and
produces request-local site/trajectory/weight/ray/material deltas. That source
has not run after integration and the installed native extension predates its
schemas.

The conservative first geometry trainer should use fixed cameras. Otherwise a
global float64 affine-ray cotangent has shape `[P,12]`; at `2704 x 2028` this is
`526,436,352` bytes (`502.05 MiB`) per selected view before any optimizer state.
That cost is avoidable for the paper's first geometry gate and is not evidence
against the world-side formulation.

No new Schur-like derivation is needed for this bridge. The missing work is:

1. expose full geometry through the production coordinator/trainer;
2. add a geometry updater and checkpoint contract;
3. fully recompile and recertify after every geometry/weight/camera change;
4. run a two-step loss-decrease/parity gate with mandatory recompilation;
5. measure native allocator and host/system peaks; and
6. only then test whether output-sensitive structural repair is worth building.

The stronger theorem still open is a uniform continuous geometry/ray tangent
bound across certified charts and seams. That theorem matters for a broad paper
claim, but it does not block the conservative recompile-every-update trainer.

### Branches and falsification tests

#### Branch A: current architecture is sufficient

Hypothesis:
    Fixed-world compiled rank/event structure remains bounded while requested
    frame density rises, and material/full-geometry native peaks remain under
    their explicit caps.

Cheap falsification:
    Run fresh-process `F=8/64/300` rows, then a separate fixed-camera
    full-geometry two-step gate. Require invariant compiled work, exact coverage,
    one reverse per active block, and measured allocator/RSS/system receipts.

If supported:
    Stop deriving representations. Finish public cache conversion, companion
    evidence, runner integration, and the paper table.

#### Branch B: rank/event complexity grows with sample density

Hypothesis:
    The fixed physical interval hides unresolved topology or tangent variation,
    so certification forces `J`, chart count, or fallback rows to grow as `F`
    becomes denser.

Cheap falsification:
    Hold world/cameras/interval/tolerances fixed and plot `sum J_c`, event count,
    owner-word count, fallback count, and maximum primal/tangent residual versus
    requested `F`.

If supported:
    Reopen the compiler/certificate mathematics at the measured failure. Do not
    blame target streaming or invent another material monoid.

#### Branch C: tensor memory is bounded but file-backed system pressure is not

Hypothesis:
    Full payload verification and page-cache/readahead consume host memory that
    process RSS and logical tensor receipts miss.

Cheap falsification:
    Run the mapped public companion under a parent that records system available
    memory/pressure across construction and the whole step, plus cumulative
    requested pages. Compare mmap against a bounded-window/tiled reader.

If supported:
    Replace full-file mapping/hash behavior with a content-addressed block/Merkle
    or bounded-window reader. The ordered-transfer math remains unchanged.

#### Branch D: full geometry fails despite material success

Hypothesis:
    Length/site/trajectory reduction, seam convention, or optimizer update is
    wrong even though material-only gradients and memory scaling pass.

Cheap falsification:
    On one event-free fixed-camera fixture, compare the executor-bound full
    geometry delta to the independent CPU stable-stratum reference, apply one
    update, force a complete recompile, and require the second loss to decrease.

If supported:
    Fix the exact geometry bridge or narrow the claim. Do not cite the material
    gate as geometry evidence.

### Exact next sequence

On this host, stop at source/static validation. On an approved quiet host:

1. run the selected-target and dataset-binding unit gates;
2. run request/fixed-step receipt propagation gates, including cumulative pages;
3. rebuild and attest the sealed Metal extension;
4. execute the procedural material-only schema-v3 `F=8/64/300` matrix;
5. build the bounded public cache and populated dataset binding;
6. add/run the separate public-target companion verifier;
7. run the fixed-camera full-geometry parity/two-step/recompile gate; and
8. only after those pass, wire the lane into the unified paper runner.

This pass ran no Python, imports, tests, builds, Metal/MPS, CUDA, data decode,
or training work. The binding validator, mapped source, receipt changes, tests,
and manuscript edits remain source-only and unrun.

## Expansion Pass 8: Lowered cut-jet certificate for continuous geometry gradients

### Context and current belief

The remaining mathematical gap is not an alternative renderer or a missing
depth marginalization. The stable-stratum implementation already computes the
exact VJP of the fixed compiled surrogate at its `J` compiler nodes. What is
open is a continuous-in-time bound between that surrogate geometry/ray
Jacobian and the exact physical fixed-word Jacobian.

Current belief:
    The smallest sound certificate should work in **lowered cut-jet
    coordinates**, not seed one global dual variable for every world site.

Confidence:
    medium-high for the theorem and chain-rule reduction; medium for practical
    interval tightness; unverified for implementation cost because no certifier
    was executed in this pass.

The runtime remains unchanged:

```text
kinetic sites/camera
  -> certified owner word and active cuts
  -> J exact node words
  -> compact affine-Lie interpolation
  -> streamed sample cotangents
  -> one word VJP
```

The proposed cut-jet object exists only while compiling/certifying one
track-local chart.

### Definitions and units

Use the normalized dimensionless coordinates already required by the theorem
ledger. On one stable chart, an adjacent active face cut is

```text
z_r(t) = -B_r(t) / A_r(t),
A_r(t) = a_r0 + a_r1 t + a_r2 t^2,
B_r(t) = b_r0 + b_r1 t + b_r2 t^2.
```

The affine ray direction and physical ray speed are

```text
d(t) = d_0 + t d_1,
s(t) = ||d(t)||_2.
```

With near/far cuts appended, run `r` has physical length

```text
ell_r(t) = s(t) [z_(r+1)(t) - z_r(t)].
```

For `R` runs define the lowered geometry coordinate

```text
q = {
  (a_r0,a_r1,a_r2,b_r0,b_r1,b_r2) for r=1..R-1,
  d_0 in R^3,
  d_1 in R^3
}.
```

Thus one chart-local proof problem has at most

```text
n_q = 6(R-1) + 6
```

independent coordinates before removing constant/zero polynomial terms.
Origins, sites, velocities, and weight trajectories influence `q` through the
existing exact sparse lowering. Ray direction is kept explicitly because it
affects both the cut polynomials and physical speed.

### Claim 1: analyticity on a certified stable chart

Assumptions:

```text
inf_t |A_r(t)| >= delta_A > 0 for every active cut,
inf_t ||d(t)|| >= delta_d > 0,
inf_t ell_r(t) >= delta_ell > 0,
fixed owner word, fixed near/far, finite bounded material,
no event/root isolator neighborhood intersects the certified interval.
```

Claim:
    The exact P0 transfer `G(t,q)` and its lowered Jacobian `D_q G(t,q)` are
    analytic in `t` on the chart and smooth in `q` in a nonzero neighborhood
    controlled by the margins above.

Derivation:
    `A_r` and `B_r` are polynomials. Nonzero `A_r` makes `-B_r/A_r`
    analytic. Positive ray-speed squared keeps the principal square root
    analytic on the real interval and away from its complex branch locus.
    Positive lengths and finite densities make every segment optical depth,
    exponential transfer, ordered product, and affine-Lie chart composition
    analytic. Composition and differentiation preserve analyticity.

Failure cases:
    denominator zero, ray collapse, zero/negative run length, a topology seam,
    cone-chart loss, or a complex singularity too near the interval. Those
    cases split the interval, raise rank, route exact replay, or fail closed;
    they do not justify silently accepting a weak certificate.

### Claim 2: a lowered interval-dual certificate is sufficient

Let `G_J(t,q)` be the compiled surrogate with node times, barycentric weights,
rank, and dispatch fixed. Use outward-rounded interval forward AD over `(t,q)`
to enclose on every accepted leaf:

```text
e_0 = sup_t ||G(t,q)-G_J(t,q)||_infinity,
e_q = sup_t max_(output k, lowered coordinate h)
      |partial_qh G_k(t,q)-partial_qh G_J,k(t,q)|.
```

The exact branch evaluates `A`, `B`, `z`, speed, lengths, and the ordered word
directly. The compiled branch must match the actual compiler linearization:

1. use stored float64 node lengths as the node-transfer primal;
2. attach the exact real-arithmetic node-length tangent at the same snapshot;
3. apply the stored fixed node-to-coefficient fit;
4. evaluate the actual second-form barycentric decoder; and
5. differentiate neither node times, rank, chart endpoints, nor dispatch.

This distinction matters. Re-evaluating a mathematically ideal node primal and
calling it the compiled derivative would certify a nearby algorithm rather
than the source implementation.

Soundness follows from outward interval arithmetic: accepted leaves cover the
entire closed chart, and the maximum of their upper endpoints bounds every
entry of the real-arithmetic lowered Jacobian error. Requested frame samples
are never used to select rank or split leaves.

### Claim 3: sparse composition closes the world-parameter bound

Let local world parameters be `theta`, and let

```text
H = D_theta q(theta)
```

at the frozen snapshot. `H` is sparse because each active cut references two
sites and one ray. For any supported direction set `V`, define

```text
L_qtheta = sup_(v in V, ||v||_theta <= 1) ||H v||_1.
```

Then

```text
sup_t sup_(v in V, ||v||_theta <= 1)
||D_theta G(t)[v] - D_theta G_J(t)[v]||_infinity
  <= e_q L_qtheta
  = epsilon_1.
```

Proof:

```text
D_theta(G-G_J)[v] = D_q(G-G_J)[H v],

||D_q(G-G_J)[H v]||_infinity
  <= max_(k,h)|partial_qh(G-G_J)_k| * ||H v||_1
  <= e_q L_qtheta.
```

The map `H` must include both paths by which affine ray direction affects the
renderer: its contribution to the active `A/B` cut coefficients and its direct
contribution to `s(t)`. Treating those as independent lowered coordinates is
conservative; the sparse composition restores their correlation for actual
world directions.

Combined with the existing normalized-loss composition lemma, `e_0` and
`epsilon_1` yield a global VJP-error bound without a dense global dual or an
extra requested-frame-count factor.

### Memory and work contract for the certifier

The dense-global interval-dual oracle is not production admissible. The
production certifier must stream one `(track,chart)` problem at a time and cap
before allocation:

```text
dual dimension                      <= n_q,max,
simultaneously live interval state  O(n_q R),
retained result per chart           O(1) bounds + digests,
retained requested-frame state      0,
global world-dual tensor            absent.
```

The sparse `H` norm can be accumulated from active two-site/ray incidences; it
does not require materializing a dense `n_q x n_theta` matrix. A proof receipt
must record the parameter norm, blockwise `L_qtheta`, denominator/speed/length
margins, leaf count, deepest split, exact source/atlas digests, and whether any
dense exceptional row was used.

### Branches and backtracks

#### Branch A: entrywise lowered bounds are tight enough

Cheap test:
    Two- and three-run stable charts with translating and rotating cuts;
    compare continuous interval upper bounds to dense exact-JVP witnesses and
    finite differences for positions, velocities, weights, ray origin, and ray
    direction.

If supported:
    Implement the track-local cut-jet certifier and instantiate theorem-ledger
    D6. No new renderer formulation is needed.

#### Branch B: independent cut coordinates are too conservative

Symptom:
    `e_q` is small entrywise but `e_q L_qtheta` rejects useful charts because
    independent cut perturbations ignore cancellation/shared-site structure.

Response:
    Seed bounded blocks of actual sparse world directions or certify the
    induced action directly. Do not seed the complete global world. Retain the
    same runtime and chart decomposition.

#### Branch C: interval dependency explosion dominates

Symptom:
    leaf count or arithmetic cost exceeds the declared cap while dense
    witnesses remain accurate.

Response:
    split the chart at compiler-independent locations, use Taylor models or
    Bernstein coefficient bounds, or route exact streamed replay. A sampled
    witness alone must not be promoted to a continuous theorem.

#### Branch D: geometry rank genuinely grows with physical complexity

Symptom:
    accepted `J` or chart count rises with motion/event complexity even when
    requested `F` is fixed.

Interpretation:
    This is allowed by the theorem. The claim fixes physical world/camera/
    interval/tolerance and varies only sampling density. It becomes a method
    economics failure only if realistic `J/R` never reaches break-even.

#### Branch E: rank changes only when requested samples are densified

Symptom:
    `J`, event count, or certificate leaves change when only `F` changes.

Interpretation:
    The implementation is leaking the requested grid into compilation. Fail
    the strong sublinear gate and repair the compiler; do not weaken the claim
    after the fact.

### Falsification matrix

```text
test                         support condition
---------------------------  -----------------------------------------------
K partition                 same loss/VJP for K=1,4,...
F densification             same structural/certificate digests and J/E/R
cut-jet witness             observed max error <= certified upper bound
world composition           world JVP error <= e_q L_qtheta
near denominator failure    fail closed or split, never finite false bound
near ray collapse           fail closed or split
birth/death seam            excluded/one-sided; never classical D6 claim
stored-node perturbation     certifier follows stored-primal linearization
runtime parity              separate float32/Metal error budget
```

### Decision implication

This pass does **not** justify new runtime state, a new public method name, or
a Schur-like depth elimination. It supplies the missing mathematical target
for full geometry: a track-local continuous cut-jet certificate composed
through the existing sparse geometry lowering. If this certificate is
practically too loose, first change the proof basis or route exact replay. Redo
the representation only if measured `J/E/R` or native economics—not the mere
existence of retained depth—invalidates the architecture.

No Python, imports, tests, builds, data conversion, native execution, or
training ran in this expansion pass.

## Shader-design interlude: reuse the native sparse boundary pattern for kinetic geometry

### Context and source discovery

The next shader question was whether direct kinetic WorldFoam needs another
mathematical representation before full geometry can be memory-light. Source
inspection found a stronger answer: an older fixed-word backend already
implements the correct *systems pattern* on Metal:

```text
node transfer cotangents
    -> sparse Mobius-incidence bars
    -> active boundary bars
    -> site bars
```

The relevant staged calls are:

```text
fixed_word_p0_lie_node_vjp_accumulate_launch_only
fixed_word_p0_sparse_mobius_boundary_finalize_launch_only
fixed_word_p0_site_geometry_finalize_launch_only
```

They allocate shared world bars once, accumulate only active incidences, and
scatter through the exact site pairs that generated the forward boundaries.
They do not validate every inactive site during the warm reverse. This is the
same cold-certificate/warm-differential split derived in Expansion Pass 9.

That old path is not itself the direct kinetic solution. Its boundary/site
parameterization does not expose the current direct-kinetic parameters

```text
p_i(t) = p_i0 + t v_i,
w_i(t) = sum_l a_i,l t^l,
```

and therefore cannot simply be relabelled as the new full-geometry trainer.
It is, however, a concrete native implementation pattern to reuse instead of
inventing a second lifecycle.

Current belief:
    The direct-kinetic geometry reverse needs one new owner-local lowering
    stage, not new ordered-transfer math.

Confidence:
    high for the calculus and asymptotic shape; medium for numerical parity
    until the rebuilt native kernel is compared with the float64 dense oracle.

### Exact node-local kinetic reverse

For one certified row, compiler node `t`, and owner word

```text
i_1, ..., i_R,
```

define

```text
p_i = p_i0 + t v_i,
w_i = sum_l a_i,l t^l,
o = o_0 + t o_1,
d = d_0 + t d_1,
s = ||d||.
```

For adjacent owners `l=i_r`, `q=i_(r+1)`, let

```text
n = 2 (p_q - p_l),
D = n dot d,
I = n dot o + ||p_l||^2 - ||p_q||^2 - w_l + w_q,
z = -I / D.
```

With `z_0=near`, `z_R=far`, the physical run length is

```text
ell_r = s (z_r - z_(r-1)).
```

Given node-length bars `bar_ell_r`, the internal cut and speed bars are

```text
bar_z_r = s (bar_ell_r - bar_ell_(r+1)),
bar_s   = sum_r (z_r-z_(r-1)) bar_ell_r.
```

For one internal cut, implicit differentiation of the active equality gives

```text
q_implicit = -bar_z_r / D,
x_cut      = o + z d,

bar_p_l += q_implicit 2 (p_l - x_cut),
bar_p_q += q_implicit 2 (x_cut - p_q),
bar_w_l -= q_implicit,
bar_w_q += q_implicit,
bar_o   += q_implicit n,
bar_d   += q_implicit z n.
```

The ray-speed contribution adds

```text
bar_d += bar_s d / s.
```

Finally lower node-state bars to persistent kinetic parameters:

```text
bar_p_i0 += bar_p_i,
bar_v_i  += t bar_p_i,
bar_a_i,l += t^l bar_w_i,

bar_o_0 += bar_o,
bar_o_1 += t bar_o,
bar_d_0 += bar_d,
bar_d_1 += t bar_d.
```

Every term touches only adjacent owners already present in the certified word
and the row ray. There is no mathematical read of an inactive site. Inactive
competitors belong to cold topology certification and to post-update
recompilation, not to this VJP.

### Production kernel contract

The first native kinetic geometry kernel should consume one already-resident
equal-rank block:

```text
source_site_ids                 int64 [S_b]
row_track_ids / row chart ids  integer [Q_b]
word_offsets                   int32 [Q_b+1]
word_owner                     int32 [W_b]
row_node_times                 float [Q_b,J]
node_physical_lengths          float [J,W_b]
grad_node_physical_lengths     float [J,W_b]
compact positions0             float [S_b,3]
compact velocities             float [S_b,3]
compact weight coefficients    float [S_b,L_w]
row ray coefficients           float [Q_b,12]
row near/far                   float [Q_b,2]
```

and accumulate into caller-owned compact or request-global bars:

```text
grad_positions0                float [S_b,3]
grad_velocities                float [S_b,3]
grad_weight_coefficients       float [S_b,L_w]
grad_row_ray_coefficients      float [Q_b,12]  (disabled for fixed camera)
```

Equal rank means equal `J`, not equal node values, so node times must remain
row-local or be recoverable from a row-bound schedule token. One thread may
own one `(row,node)` pair. Its word interval is disjoint in the
`[J,W_b]` length bar. Site owners repeat across rows/nodes, so the first Metal
implementation may use atomic accumulation into compact site bars, matching
the existing sparse boundary-to-site stage. A deterministic segmented
reduction is an optional later optimization, not a prerequisite for the
memory theorem.

The kernel must not allocate a dense row-by-global-site result. A dense global
site gradient is allowed only once as the actual optimizer output; its size is
`Theta(S(6+L_w))` and does not depend on frames. The forbidden object is the
temporary `Theta(Q_b S (6+L_w))` row expansion.

### Peak logical memory target

For a streamed block, ignoring allocator-private storage:

```text
M_kinetic_geometry <=
    4 J W_b                         node-length bar
  + 4 S_b (6+L_w)                  compact kinetic input
  + 4 S_b (6+L_w)                  compact/output bars
  + 48 Q_b                         fixed row-ray input
  + 48 Q_b * I[trainable rays]     optional ray bars
  + O(Q_b + W_b + J Q_b)           topology, domains, node times
```

There is no `F_requested` term. With one native block in flight, the full
step retains only the global optimizer bars plus the maximum block envelope.
The unavoidable streamed sample/target block remains a separate `B_p x K`
term.

### Numerical branch and backtrack

The calculus is exact in real arithmetic, but the current compiler produces
node lengths in CPU float64 and lowers them to native float32. A Metal kernel
that recomputes cuts in float32 may not reproduce the dense float64 oracle
bit-for-bit.

Two implementation branches remain:

1. **Recompute owner-local cuts on Metal.**
   Smallest retained payload; validate recomputed physical lengths against the
   forward table before scattering. Reject a block whose discrepancy exceeds
   its declared tolerance.

2. **Lower certified cut jets.**
   Store per-node active cut depth/denominator or an equivalent compact jet.
   This increases the block payload by `O(J(R-1))` but removes ambiguous
   recomputation and is still frame-free.

The second branch is safer for paper claims if float32 recomputation produces
unstable parity near small denominators. The first branch should be tried
first only under the already-certified denominator, speed, and positive-length
margins. Neither branch changes the public representation or transfer monoid.

### Acceptance and falsification

Promotion requires all of the following:

```text
dense_global_site_accumulation_elements == 0
all_site_owner_validation_evaluations == 0
maximum_simultaneous_native_length_bars == 1
requested_frame_sampling_used == false
geometry output/bar bytes equal for F=8,64,300 on one fixed world
sparse CPU vs dense CPU bars within declared float64 tolerance
native sparse vs sparse CPU bars within declared float32 tolerance
non-owner perturbation with unchanged certificate never enters warm VJP data
changed owner/world/ray/certificate fails before output-bar mutation
geometry update invalidates every old chart/block and forces cold recompile
```

An `S` sweep at fixed owner word must show warm differentiable interactions
and bridge scratch following `R`, not `S`. Cold certificate time may still
follow `S`; it must be reported separately and amortized or recompiled after
geometry updates as required.

### Decision implication

Do not redo ordered-transfer math. Reuse the existing native staged-adjoint
lifecycle and replace only its geometry frontend with the direct-kinetic
owner-local lowering above. Keep the current dense CPU geometry reducer as the
independent oracle until source, native, update/recompile, and allocator gates
all pass.

No Python, imports, tests, builds, data decoding, native execution, or training
ran in this expansion pass.

## Red-team interlude: hidden frame axis and acceptance repair

### Falsified headline

The selected native node-forward, material-word VJP, and bounded sample
kernels are frame-free in their compiled-word state. The source-level
end-to-end path was **not** frame-free before this pass.

The red-team found three distinct axes that earlier prose had blurred:

```text
F_dataset    calibrated camera/time records in the physical dataset grid
F_requested  observations selected for one benchmark/training step
F_stored     frames physically stored by the target cache/source
```

The old schema-v3 trial set all three relevant fixture counts to the same `F`
and then varied `F=8,64,300`. That changed both the physical provider and the
requested sampling density. It was therefore not a clean test of the theorem,
which fixes world, camera program, physical interval, and compiled structure
while varying requested samples.

### Hidden source scaling found

Before the repair:

1. every cached CPU artifact retained one warm camera identity/layout/version
   signature per dataset frame and included the tuple in metadata bytes and
   artifact hashes;
2. every provider warm assertion reserialized the complete frame-time tuple;
3. those warm assertions were nested through source/request/chunk/session
   validation;
4. the static-camera factory compared all camera records once per spatial
   track; and
5. cold artifact admission replayed every track against every dataset frame,
   with several nested cold validations repeating provider work.

If `C` is the temporal chunk count, the old warm-validation layer alone could
pay at least

```text
Omega((Q+C) F_dataset),
```

and because the trial set `F_dataset=F_requested` with approximately

```text
C = Theta(V P F_requested / N_max),
```

the host validation work could become quadratic in the nominal sweep variable.
The unrun verifier's exact artifact-byte invariance check would correctly have
rejected the old candidate, because the retained artifact signatures grew
with `F_dataset`.

### Source-only repair applied

The acceptance experiment now fixes:

```text
F_dataset = 300
physical interval = [0,1]
world, static camera, compiler/rank policy = identical
```

and varies only deterministic endpoint-including requested subsets:

```text
F_requested = 8, 64, 300.
```

The driver records `dataset_frame_count`, the subset rule, and the exact
selected-index digest. The verifier reconstructs the expected subset,
requires the 300-frame dataset grid on every row, and compares structural work
and logical bytes only after that identity is proven.

The source cache/provider path was tightened as follows:

- artifacts no longer retain or hash an `F_dataset`-long camera-signature
  tuple;
- provider warm generation recomputation consumes the already cold-computed
  camera/path digests plus constant-size frame-domain facts rather than the
  full frame-time tuple;
- cold provider preparation records one static-camera-path certificate boolean
  per view while it already scans camera contents;
- the static track factory validates the representative camera semantics and
  consumes that certificate instead of rescanning all frames per pixel;
- static-path artifact admission checks the affine program at the two temporal
  endpoints, because provider cold certification has already proved all camera
  records identical; dynamic paths retain the full cold scan; and
- nested key/seal/result validation uses warm scalar certificates after one
  cold provider certification inside the compile callback.

Camera mutation between cold request boundaries remains forbidden. Warm
artifact checks deliberately do not rescan the grid; the next outer
`provider.assert_current()` detects content mutation before a new replay.

The step-level accounting no longer asserts
`structural_node_word_work_invariant_in_requested_f=True`. It reports that
cross-row verification is required. Only equality of actual structure/work
receipts across the fixed-dataset matrix can earn the conclusion.

### What this does and does not prove

This repair makes the **experiment** identify the intended narrow theorem:
expensive compiled word/node work and retained artifact payload should stay
fixed when only requested sampling is densified. It does not make all
end-to-end metadata constant in requested frames:

```text
provider frame times/cameras        O(F_dataset), fixed across the sweep
requested SpacetimeBatch samples    O(F_requested)
canonical batch positions           O(F_requested)
target/output/residual work          Omega(P F_requested)
```

A strict `O(K)` requested-observation metadata claim would require replacing
the explicit batch tuple and canonical-position tuple with a procedural
range/stride manifest. That is a later optimization, not a prerequisite for
the paper's precise claim that the expensive world/ordered-word reverse and
its interaction tape are independent of requested density.

Full geometry remains a separate gate. Its native `[J,W_b]` bar is bounded and
frame-free, but the current CPU reducer still pays all-site validation and
dense row bars. Expansion Pass 9 gives the exact owner-local replacement.

### New falsification conditions

The fixed-dataset `8/64/300` run must fail if any of these changes across rows:

```text
dataset/provider/camera-path identity,
event/chart/word/rank digests,
artifact accounted bytes,
node-forward or material-word VJP interactions,
native word VJP launch count,
maximum frame-invariant live logical bytes.
```

Only sample launches, sample-node interactions, target traffic, explicit
requested-frame metadata, and measured total wall work may grow with
`F_requested`.

All fixes in this pass are source-only and unrun. No Python, imports, tests,
builds, data conversion, native execution, Metal/MPS/CUDA, or training ran.

## Expansion Pass 9: owner-local geometry VJP and certificate separation

### Why this pass exists

The current full-geometry bridge is already independent of requested frame
count after sample-to-node reduction, but its CPU row reducer is not yet the
spatially lean executor suggested by the theorem. For every `(track,chart)`
row it currently:

1. recomputes the certified node geometry;
2. audits every non-owner competitor at left cut, right cut, and midpoint;
3. allocates full `[S,3]`, `[S,3]`, and `[S,L_w]` row-result bars; and
4. adds those dense row results into another caller-owned full-world result.

This does not introduce an `F` axis, so it does not refute the temporal
sublinearity claim. It does introduce avoidable `O(S)` row scratch and
`O(row_count*S)` dense addition work. The relevant optimization is not new
transport mathematics. It is a separation between a **cold topology proof**
and a **warm owner-supported differential**.

Current belief:
    Once a continuous owner-word certificate is sealed to the exact world,
    ray, interval, and chart snapshot, the fixed-word physical-length VJP has
    support only on the owners occurring in that word and on the affine ray.

Confidence:
    high for the support theorem; medium-high for the production lifecycle,
    because the current source already recompiles after every geometry/ray
    update but does not yet expose this exact sparse receipt type.

### Support theorem

Let a certified word contain distinct owners

```text
i_1, i_2, ..., i_R.
```

Its internal cuts are the adjacent equal-power roots

```text
z_r = -B_(i_r,i_(r+1)) / A_(i_r,i_(r+1)),  r=1,...,R-1.
```

At one compiler node, with physical run lengths

```text
ell_r = ||d|| (z_r - z_(r-1)),
```

the reverse of a cotangent `bar_ell_r` first gives

```text
bar_z_r = ||d|| (bar_ell_r - bar_ell_(r+1))
```

for every internal cut, plus the direct ray-speed bar. Each `z_r` depends
only on:

```text
site i_r,
site i_(r+1),
the affine ray at the node.
```

Therefore the union of parameter support across the whole word is exactly a
subset of its `R` distinct owners plus the ray. Non-owner competitors affect
whether the word is valid, but they do not enter the classical derivative of
the already-fixed word.

Consequently an exact row result may be represented as:

```text
active_owner_ids                   int64 [R]
grad_active_positions0             f64   [R,3]
grad_active_velocities             f64   [R,3]
grad_active_weight_coefficients    f64   [R,L_w]
grad_affine_ray                    f64   [12]  (optional at the caller)
node_physical_lengths              f64   [J,R]
```

No dense row-local world bar is mathematically required. Since owner ids in
one unbounded line-envelope word are unique, one may accumulate by the word's
run index without an intra-row hash table. Cross-row repetition is handled by
one caller-owned indexed add into the request/global bars.

### Proof/certificate boundary

Deleting the all-competitor audit without replacing its authority would be
unsound. The safe split is:

```text
cold compile / recertification
    all-site owner and event proof
    -> certificate digest bound to world/ray/chart contents

warm fixed-word VJP
    validate certificate identity + immutable tensor versions
    validate node denominator/speed/positive-length margins
    -> owner-local exact differential only
```

The continuous certificate, not a string supplied by an arbitrary caller,
must establish:

```text
the declared word owns every open run over the whole chart,
all active adjacent ties are the declared cuts,
no inactive competitor can enter before a certified margin,
the certificate covers the exact chart interval and near/far convention,
its world/ray/content digests equal the live immutable snapshot.
```

Every geometry, weight-trajectory, or camera-ray update invalidates this
authority and triggers full structural recompilation. Material-only updates
may reuse it because material does not alter power ownership. This is the
same conservative update rule already selected for production; sparse warm
VJP does not require output-sensitive chart repair.

### Resulting work and memory bound

For one active native block, after the native `[J,W_b]` length bar exists:

```text
warm differentiable work      Theta(sum_rows J R_row)
warm row scratch              O(max_rows (J R + R(6+L_w)))
caller-owned world bars       O(S(6+L_w))
requested-frame state         0
cold certificate work         reported separately; currently all-site
```

The first implementation should stream one row at a time, so the `max_rows`
factor above is one. A still tighter seam scatters each owner-local row result
directly into the request-owned bars and returns only a tensor-free reduction
receipt. Transactional request ownership remains intact because the request
bars, not the whole-step bars, own the scatter until commit.

The end-to-end lower bound is unchanged: target reads, predictions/residuals,
and output writes remain at least `Omega(PF)`. The improvement is specifically
that expensive world/ordered-word reverse interactions and their tape are
independent of requested sampling density, while the warm geometry reverse
also becomes output-sensitive in active owner incidences instead of dense in
all sites.

### Required fail-closed identities

The sparse result/receipt should bind at least:

```text
world generation and content digest,
camera/ray path digest,
track id and chart index,
owner-word digest and exact owner ids,
continuous topology certificate digest,
native block and native VJP provenance,
node-time / near / far / rank identity,
node-length cotangent provenance,
request accumulator generation.
```

The warm function must reject a certificate from another row even when the
owner tuple happens to match. It must also reject a changed world tensor
version before touching caller-owned bars.

### Falsification and acceptance

Source/runtime gates for this optimization are:

```text
sparse-versus-current dense row VJP       exact/close for every bar
sparse direct scatter versus row sum      exact/close across repeated owners
non-owner support                         identically absent from sparse result
stale world/ray/certificate               rejected before caller-bar mutation
fixed camera                              no ray-bar allocation or retained ray bar
F = 8/64/300                              identical sparse reverse counts/bytes
S sweep at fixed word                     row scratch follows R, not S
certificate disabled                      route current audited path or fail closed
```

The existing dense audited function should remain temporarily as an
independent oracle. Promotion requires parity against it before the production
coordinator selects the trusted-certificate sparse path.

### Decision implication

This is the next useful mathematical-to-systems closure after the scientist
review. It does not change the canonical runtime, add a public method name, or
claim a new renderer. It turns the existing fixed-word support theorem into a
smaller geometry-reverse ABI. Full recompilation remains the production rule;
local simple-root repair remains deferred until measurements justify it.

No Python, imports, tests, builds, data conversion, native execution, or
training ran in this expansion pass.

## Expansion Pass 10: direct-kinetic sparse reverse selected

The shader-design interlude above resolves the implementation branch opened by
Expansion Pass 9. The existing native sparse Mobius-incidence/boundary/site
pipeline is the lifecycle pattern to preserve, while its static geometry
frontend is replaced by the owner-local direct-kinetic differential. The
canonical production chain is now:

```text
cold all-site kinetic topology certificate
  -> frame-free equal-rank owner blocks
  -> bounded streamed sample-to-node cotangents
  -> one ordered-word material/length VJP per active block
  -> one owner-local kinetic geometry scatter per active block
  -> one global material/geometry update
  -> discard every structural object
  -> fresh cold compile and reseal before the next geometry step
```

This closes the need-for-new-math decision: no new depth marginalization and
no new public formulation is required. What remains falsifiable engineering is
sparse CPU/dense-oracle parity, native direct-kinetic lowering, stale-program
rejection after an update, and fixed-dataset allocator measurements.

No runtime or accelerator work ran in this pass.

## Expansion Pass 11: fuse the node-word and kinetic-geometry reverse

### Why the staged bridge is not the final memory endpoint

The current full-geometry route materializes one native
`grad_node_physical_length[J,W_b]` buffer, fences, copies it to CPU, and then
runs geometry reduction. That buffer is already independent of requested
frames, so it proves the temporal theorem. It is still avoidable bandwidth and
storage.

The native ordered-word VJP already computes every run's physical-length
cotangent

```text
bar_ell_r = density_owner(r) * bar_tau_r
```

inside the `(row,node)` thread. The kinetic cut reverse needs only adjacent
differences of those values and one scalar speed reduction. It can therefore
be fused before `bar_ell_r` ever leaves the thread.

### Streaming reverse identity

For one node, initialize

```text
z_left = near,
previous_bar_ell = undefined,
bar_speed = 0.
```

Scan runs front-to-back using the same constant-prefix ordered-transfer VJP.
For run `r`, compute `bar_ell_r` and read its compiled physical length
`ell_r`. With ray speed `s`:

```text
coordinate_length_r = ell_r / s,
z_right = z_left + coordinate_length_r,
bar_speed += coordinate_length_r * bar_ell_r.
```

When `r > 0`, the cut immediately before run `r` has already-known depth
`z_left` and cotangent

```text
bar_z_(r-1) = s * (previous_bar_ell - bar_ell_r).
```

Apply the adjacent-owner implicit cut VJP at `z_left`, then set

```text
previous_bar_ell = bar_ell_r,
z_left = z_right.
```

After the word scan, add

```text
bar_d += bar_speed * d / s
```

and lower the accumulated node-state bars to `p0`, velocity, polynomial
weight, and optional affine-ray coefficients. No suffix tape and no
`[J,W_b]` output is needed.

This scan remains `Theta(J W_b)` work with constant per-thread scalar state,
excluding the caller-owned output bars. It combines three operations that the
staged oracle currently separates:

```text
ordered material VJP,
physical-length cotangent construction,
owner-local kinetic geometry VJP.
```

### Numerical authority

Using cumulative `ell_r/s` for cut depth makes the geometry reverse correspond
to the exact physical-length table consumed by the native forward. The active
cut denominator still comes from the current owner pair and ray. A stricter
cut-jet payload may provide compiler-lowered `z,D` values if cumulative
float32 drift or denominator recomputation exceeds tolerance.

The fused kernel must fail admission before launch when the cold certificate,
world generation, compact owner mapping, node schedule, or near/far identity
does not match. A device kernel cannot recover from a failed structural
predicate after it has begun atomic writes; all such checks belong to token
preparation and warm identity/version validation.

### Preferred and oracle paths

```text
production:
    streamed sample -> node cotangents
    -> fused material + owner-local kinetic VJP
    -> caller-owned world bars
    [no length-bar tensor]

debug/parity oracle:
    streamed sample -> node cotangents
    -> staged material + [J,W_b] length bars
    -> sparse CPU reducer
    -> dense audited CPU oracle
```

The sparse CPU reducer remains valuable even after fusion because it isolates
kernel errors from topology/certificate errors. It is not the desired hot
path.

### Revised final memory gate

The intermediate source gate may allow one bounded `[J,W_b]` bar. The final
production full-geometry gate should require:

```text
native_length_bar_tensor_bytes == 0
native_length_bar_device_to_cpu_copy_count == 0
dense_global_site_accumulation_elements == 0
all_site_owner_validation_evaluations == 0
fused_owner_cut_reverse_interactions == J * sum_rows (R_row - 1)
fused_ray_speed_reverse_interactions == J * row_count
maximum_in_flight_sample_block_count == 1
requested_frame_sampling_used_by_world_reverse == false
```

The actual global geometry gradient remains `O(S(6+L_w))`; that is optimizer
state, not a frame or row expansion. Atomic accumulation affects determinism
and throughput but not this memory result.

### Falsification tests

```text
fused native vs staged native+CPU sparse bars
fused native vs dense float64 oracle
K and F sweeps with identical world-reverse counts/bytes
R=1 word (speed term only; no cut interaction)
R=2 sign test for bar_z = s(bar_ell_1-bar_ell_2)
repeated owner across different rows/charts (atomic scatter sum)
fixed-camera mode (no ray-gradient buffer or writes)
near-zero speed/denominator admission rejection
stale topology/world/node schedule rejected before launch
```

### Decision implication

The final memory-light shader should be a fused direct-kinetic full VJP, not a
permanent CPU reduction of native length bars. Implement the sparse CPU path
first for correctness and lifecycle closure, then use it as the oracle for the
fused native kernel.

No runtime, build, import, or accelerator command ran in this expansion pass.

## Expansion Pass 12: scientist synthesis, fused fixed-camera source, and lifecycle/claim correction

The external scientist's review supports the present consolidation. Of the
five proposed formulations, the translated optical-depth measure
`(kappa,nu)` is the one strong new proof object, the maximal similarity-gauge
result is a useful theorem extension, and the other three are rigorous views of
the already-selected kinetic event-chart architecture. The repository should
therefore keep one canonical executable chain:

```text
direct kinetic event charts
-> ordered affine transfer words (beta,m)
-> bounded J temporal nodes
-> streamed sample residuals and node cotangents
-> one ordered-word/world reverse
```

The translated measure is used to explain ordering, noncommutativity, seams,
and tangent structure; it is not a runtime payload or a sixth implementation
branch. External literature novelty remains unestablished. Full structural
recompilation remains the safe production rule after geometry/ray changes;
output-sensitive repair is deferred until a theorem and measurements justify
it.

### Source status after independent audits

The integrated fixed-camera full-geometry source coordinator still uses the
staged route: one bounded `[J,W_b]` physical-length cotangent is fenced,
reduced immediately by the certificate-backed owner-local CPU bridge, and
released. A separate suffixed fixed-camera fused-v1 Metal/host/Python source
entry point now performs the material word VJP and owner-local kinetic scatter
in one pass. It eliminates the `[J,W_b]` **cotangent** tape/output/copy, not the
frame-independent compiled primal `[J,W_b]` length table. It is deliberately
absent from selected renderer routing and the staged coordinator, and is
unbuilt, unrun, unattested, and unpromoted.

The fused-v1 public result has only four caller-owned bars:

```text
site material
global position0
global velocity
global polynomial weight coefficients
```

It is strictly fixed-camera. The ray is still a primal input for cut geometry,
but there is no camera-ray cotangent flag, buffer, alias, return, or write. The
previous Pass-11 gate requiring
`fused_ray_speed_reverse_interactions == J * row_count` is therefore
superseded:

```text
fixed-camera fused v1:
    ray_cotangent_surface_exposed == false
    ray_cotangent_writes == 0
    ray_speed_gradient_interactions == not_applicable

future trainable-camera ABI:
    track-keyed ray aggregation required
    ray-speed reverse and dense-oracle parity required separately
```

The provenance-bearing adapter now binds the sealed live world, native
lowering, chart sources, row identities, compiler-issued certificate digests,
and the raw prepared token. It revalidates those certificates against live
content immediately before launch; it does not rerun the active all-site
compiler. Publication promotion must require active-compiler provenance, with
the exhaustive compiler retained as an oracle only. The raw variant preparer
accepts no caller-supplied certificate-shaped string and is explicitly a
low-level unpromoted surface.

One remaining fail-closed defect is intentionally not papered over. Stale
world/compiler/source identities can be rejected before launch, but several
numeric tie/closure/denominator guards still run per device thread. Those
checks are defensive, not transactional: one thread could return after another
has already atomically accumulated. Promotion therefore requires either
whole-launch host admission or a proof that every device guard follows from
the sealed host certificate.

The fused launch token now distinguishes:

```text
aliased runtime topology bytes
aliased world bytes
owned row payload bytes
owned config bytes
unique retained launch bytes
owned persistent bytes
```

It explicitly reports zero retained frame, sample, target, prediction, and
dense-row-time bytes. Runtime-resident MPS offsets, owners, and source ids are
aliased when dtype/layout already match, avoiding a second topology copy. This
is exact logical accounting only; allocator slabs, private/register/spill
state, framework caches, and peak RSS remain unmeasured.

### Combined transaction defects closed at source scope

An independent lifecycle audit found and repaired several real P1 defects:

1. A successful transaction now revokes the step result, authorization seal,
   and accumulator, then releases every retained tensor reference before cold
   replay. A second step can no longer overlap the previous authorization
   merely because Python still holds the result object.
2. Authorization binds the exact provider identity and generation, not merely
   content-compatible payloads. A foreign content-identical provider is
   rejected.
3. Transaction preflight includes the immutable candidate-world geometry
   clone and a conservative tensor-sized validation temporary.
4. Store-owned accounting adds old-store residency and the fresh-store bound;
   it no longer uses an unjustified `max` while both can coexist.
5. The ready-generation handoff is one-shot and requires an explicit
   zero-retained-byte attestation for caller-held old artifacts, checkpoints,
   or retired generations.
6. The complete explicit next-step manifest is cold-compiled and sealed;
   later cache misses must compile fresh. This invalidates all old structure
   without implying eager compilation of every view in the dataset.
7. The full-geometry step no longer accepts a caller-authored geometry label.
   Its geometry generation is a canonical digest of the live provider world's
   generation, site-content digest, and site count; the result also binds those
   fields and the exact site-table identity to its accumulator.
8. Live combined state, retired state, and raw-only checkpoints now recompute
   that same canonical geometry ID from their bound world metadata. Rehashing a
   forged foreign ID therefore cannot pass their integrity checks. Focused
   forged-state/checkpoint tests are source-written but remain unrun on this
   pressured host.

The accounting scope is deliberately narrow: transaction-owned tensor state
plus store-owned/accounted entries. Caller-retained-byte attestation is
trust-based; compiler/provider transient scratch, Python objects, page cache,
and allocator peak are excluded. Transaction application also still assumes a
serialized caller, although next-step claiming is lock-protected. Restore
parsing and production trainer routing remain open.

### Public-data source correction

Earlier passages saying no converter exists are superseded. A bounded
Neural3D/PyAV offline converter and mapped RGB8 adapter now exist source-only.
The target camera honors its own path/start offset, visible conversion scratch
is bounded conservatively, and converter disk accounting includes payload plus
raw/verification spool coexistence. There is still no populated public cache,
production `from_manifest` trainer route, decoder-vs-cache parity result, or
allocator/RSS/page-cache measurement.

### Evidence boundary and next quiet-host sequence

No Python import, pytest, native build, Metal/MPS/CUDA launch, video decode, or
training command ran in this pass. The host had only tens of MiB of free pages,
heavy swap use, and active media/browser processes, so source-only inspection
was the only incident-safe choice.

When a quiet host is available, run in this order:

1. focused CPU tests for the sparse reducer, combined transaction, mapped
   adapter, fused recurrence, and source contracts;
2. rebuild the native variant and verify exact schema registration/import;
3. compare fused fixed-camera v1 against staged sparse and dense float64
   oracles for `R=1`, the `R=2` cut-sign case, repeated owners, all four bars,
   and stale provenance;
4. close numeric-guard transactional admission before promotion;
5. run two complete geometry steps with loss decrease, checkpoint/restore,
   retirement, fresh recompilation, and one-shot authorization;
6. populate one real 512-wide Neural3D mapped cache, bind it to the trainer,
   and prove PyAV/OpenCV selected-pixel plus evaluation parity;
7. run `F=8/64/300` on one fixed 300-frame world while recording logical
   receipts, allocator peak, sampled process-group RSS, and host pressure; and
8. only then integrate the fused route into the paper runner and produce public
   quality/scaling rows.

The current conclusion is strong but bounded: no new formulation is needed to
obtain the desired memory shape. The math and source architecture now support
frame-independent expensive word/world reverse for a fixed compiled surrogate,
while camera/target/interpolation work remains necessarily linear and streamed.
Native execution and measured end-to-end evidence are still the gates.

## Expansion Pass 13: adjudicated claim ladder and transactional native admission

The scientist's scorecard is accepted without changing the selected method.
The five-formulation exercise produced one load-bearing new proof object, one
useful gauge theorem, and three clarifying views of machinery already present.
The repository work produced the more important implementation closure. The
combined result should be presented as a three-level claim ladder rather than
one broad claim.

### Level A: exact ordered-transfer algebra

For a finite fixed P0 owner word, the translated optical-depth measure obeys

```text
(kappa_A,nu_A) odot (kappa_B,nu_B)
  = (kappa_A+kappa_B, nu_A+shift(kappa_A)_#nu_B),

L(kappa,nu)
  = (exp(-kappa), integral exp(-u) dnu(u)),

L(A odot B) = L(A) star L(B).
```

This is an algebraic theorem. It explains ordering, noncommutativity, compact
pointwise transfer, and boundary tangent masses. It does **not** by itself
prove temporal rank, memory scaling, chart correctness, or trainability. The
runtime continues to store owner words and the four-scalar quotient
`(beta,m)`; materializing `nu` would enlarge the system without helping the
hot path.

### Level B: fixed-surrogate temporal factorization

Fix the compiled chart topology, node times, ranks, interpolation rule, camera
program, and physical interval. Let node transfer states be `g_j(theta)` and a
requested sample decode as

```text
y_f = D(sum_j w_j(t_f) g_j(theta)).
```

For sample cotangents `bar_y_f`, accumulation gives

```text
bar_g_j = sum_f w_j(t_f) D'(g(t_f))^T bar_y_f,
bar_theta = sum_j D_theta g_j(theta)^T bar_g_j.
```

Consequently requested-frame density appears in streamed decode and reduction,
but not in the expensive ordered-word/world reverse once `bar_g_j` exists. A
bounded implementation has the intended shape

```text
work = Theta(F_requested * local_decode)
     + Theta(sum_(p,c) J_(p,c) r_(p,c)),

memory = O(state(S) + compiled(B_p,R_b,J) + node_bars(B_p,J)
           + world_output_bars(S) + live_samples(B_p,K)),
```

with no resident frame-image, prediction, sample-by-run, or frame-by-world
cotangent tape. This theorem is conditional on the compiled surrogate and on
bounded `J`, `R_b`, `B_p`, and `K`; it is not a claim that all costs are
sublinear in scene duration or physical complexity.

### Level C: optimizer-step lifecycle

Geometry updates change the world from `theta_n` to `theta_(n+1)` and may
change charts, owner words, events, and ranks. The safe implemented policy is

```text
authorization bound to C(theta_n)
-> one material/geometry update
-> revoke old authorization and tensor roots
-> retire C(theta_n)
-> cold compile and certify C(theta_(n+1))
-> expose exactly one next-step generation.
```

The canonical geometry generation digest is not bookkeeping decoration. It is
the discrete proof obligation preventing a gradient produced by one compiled
world from authorizing an update or checkpoint labelled as another world. The
full-step result, live state, retired state, and checkpoint now all bind the
same function of world-generation digest, site-content digest, and site count.

This lifecycle still differentiates the fixed surrogate used during a step;
it does not claim a classical total derivative through event-time or topology
selection. Full recompilation between steps makes the next forward correct. It
does not retroactively create missing structural derivatives.

### Native atomic admission is now the next correctness problem

The fused fixed-camera v1 performs global atomic additions. Its per-thread
numeric checks occur before that thread's first atomic, but not before every
other thread's first atomic. Therefore a late failing thread can leave a
partially accumulated result. The kernel is not promotable merely because all
individual guards are mathematically sensible.

Three possible closures are ordered by conservatism:

1. **Separate validation launch plus one status scalar.** Scan every
   `(row,node,run)` predicate without writes to output bars, fence/read one
   failure flag, and launch the atomic reverse only on global success. This
   duplicates frame-independent `O(JR_b)` validation work but adds only
   constant status memory and is the safest first native gate.
2. **Certificate-implied host admission.** Prove that compiler certificate
   margins survive every binary64-to-binary32 conversion and every native
   evaluation rounding error. Then seal the resulting downgraded margins into
   the launch token and make the fused kernel's guards unreachable on an
   admitted token. This can remove the extra launch, but only after a complete
   rounding bound exists for length positivity, near/far closure, tie residual,
   denominator, cosine, ray speed, density, chart state, and incoming node
   cotangents.
3. **Scratch outputs then commit.** Accumulate per-block bars into private
   scratch and add them to global bars only after validation. This is safe but
   can recover the very memory traffic the fusion was intended to avoid; it is
   a fallback, not the preferred promotion route.

Per-thread early return with direct global atomics is not a fourth valid
option. Until one of the closures above exists and is tested, fused v1 remains
an unselected source candidate.

### Falsification gates for the combined claim

The paper claim should fail, rather than be weakened rhetorically, if any of
these occurs on a rebuilt quiet-host run:

```text
fused != staged sparse != dense float64 for any of the four output bars
requested F changes node-forward or ordered-word reverse counts
requested F increases retained compiled/world-reverse tensor bytes
one rejected row/node leaves any caller-owned output bar modified
second optimizer step reuses an old provider/artifact/generation
full recompile does not bind every explicit next-step manifest request
mapped public targets differ from the canonical decoder at selected pixels
allocator/RSS growth tracks F materially after fixed dataset metadata is removed
```

The next mathematical work is correspondingly narrow: either derive a
certificate-to-float32 guard implication theorem or retain the explicit
validation pass. It is not a sixth representation and it is not
output-sensitive chart repair. The latter remains deferred until baseline
full-recompile measurements show it is worth the extra proof surface.

The paper-facing translated-measure section now states the assumptions,
identity/associativity proof, Laplace homomorphism, zero-width seam corollary,
weighted-total-variation transfer certificate, and opacity-tail bound
formally. Crucially, the tail statement includes a tangent-measure term and
keeps runtime truncation disabled: large prefix opacity certifies a small
primal rear contribution, not automatically a small training gradient. The
two new inequalities still need their source-written proof-oracle regressions
executed before becoming compiler certificates. The tangent-tail statement
now explicitly fixes the
prefix/tail membership and owner order, normalizes parameter directions under
a declared norm, requires a uniform bound over the admitted unit ball, and
adds the bounded/Lipschitz loss-cotangent assumption needed to promote a color
JVP bound into a loss-gradient/VJP certificate. A finite optimizer-step claim
would require those bounds uniformly over the whole admitted neighborhood.
The proof-only CPU oracle and its focused test now source-encode the
weighted-TV, tangent-variation, primal-tail, and fixed-split directional-tail
bounds against exact P0 transfers. They remain unrun on this host and are not
enabled in native execution.

The static native guard audit sharpened the admission boundary. Structural
counts/CSR/owner/global IDs are covered by the sealed Python route, but a raw
native call can bypass part of that proof. The continuous owner certificate is
topological and does not certify native binary32 arithmetic. Post-cast
time/ray/near-far separation, material density, optical-depth products,
coordinate length, active-tie residuals, denominator/cosine margins,
near/far closure, node-chart values, and incoming node cotangents remain
dynamic numerical predicates. The last item makes a purely cold compiler
certificate insufficient unless the cotangent producer is itself sealed.

The raw preparer now rejects threshold values that become zero, infinite, or
otherwise invalid after conversion to binary32, and the static source contract
records that check. This closes only the seven-scalar conversion hole. The
required promotion design remains: validate exact arithmetic and every proposed
atomic contribution into one reason-bit status, cover all active blocks before
the first global write, then run status-gated accumulation in the same native
operation/serialized stream. The fused path remains unselected until that
exists and passes an injected-failure test proving all output bars remain
unchanged.

No runtime, import, build, accelerator, decode, or test command ran in this
expansion pass.

## Expansion Pass 14: source-only asymptotic memory ledger and the limits of “sublinear over time”

This pass cross-checks the concrete static PowerFoam/“Brilliant Foam” lineage
against the fixed-camera compiled WorldFoam route. It supersedes two stale
details above: the public ragged-sample preparation seam is the Pass-5
`4MJ+28M+20` formula, not the older `4MJ+36M+20` formula, and fused v1 now has
a two-launch validation/write transaction plus a fifth, four-byte status
return. The latter remains source-only and unbuilt; it is no longer merely the
unsafe per-thread-guard design described in Pass 13.

The conclusion is deliberately narrower than “WorldFoam is sublinear over
time”:

1. For a fixed world, camera, physical interval, chart partition, owner words,
   ranks, and chunk caps, WorldFoam removes requested frame count `F` from the
   expensive node forward and ordered-word reverse state. This structural
   result is supported by current source.
2. Static PowerFoam can also render a static world one frame at a time with a
   peak independent of `F`. WorldFoam's memory advantage is against the
   repository's explicit per-frame `MetalPowerFoamVideo` parameterization, not
   against an honestly streamed static renderer.
3. Strict end-to-end sublinear memory in `F` is false in current source because
   replay/provider metadata is `O(F)` and a resident target source is
   `Theta(PF)`. Full-coverage work and output are unconditionally
   `Omega(PF)`. If increasing `F` also increases physical duration, events,
   charts, ranks, and owner words can themselves grow with duration.
4. Cold compiler peak has no exact byte theorem. The artifact store bounds
   retained products, but explicitly excludes temporary compiler scratch,
   Python containers, exact-polynomial/root objects, and arbitrary-precision
   numerator/denominator storage.

No separate “Brilliant Foam” implementation exists in the inspected tree. The
authoritative concrete static lineage for this comparison is
`third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py` plus
`third_party/powerfoam-metal/csrc/metal/powerfoam_metal.mm`; the dynamic
paper-baseline wrapper is `src/train/powerfoam_metal_trainer.py`.

### Symbols for this pass

This pass follows the requested scaling axes and therefore uses `N` for world
primitives, unlike the older Pass-3 use of `N` for launch observations:

```text
F       requested frame samples over the fixed interval
F_d     dataset/provider frame count
V       selected view count
P       spatial (view,pixel) track count; full coverage has PF observations
N       global world primitive/site count
L       kinetic power-weight coefficient count, currently L <= 3
A       static PowerFoam CSR adjacency-entry count
b       static PowerFoam frames simultaneously batched
T       total raster tiles in one static batched launch
Q       emitted static tile/cell candidate incidences
E       exact kinetic event/root-guard count, aggregated over compiled tracks
q       one equal-rank WorldFoam block
r_q     track-chart rows in q
j_q     temporal nodes per row in q
w_q     flattened ordered owner-word entries in q
s_q     compact sites referenced by q
rho_q   maximum owner-word length among rows in q
U       union-local sites in the current spatial request
M       observations in one live sample/target launch, M <= B_p K
B_p     maximum simultaneously active spatial tracks
K       maximum temporal observations per live track chunk
C_store retained CPU artifact-store byte cap
```

All byte equations below are sums of distinct named tensor storages at the
stated source boundary. They are not allocator/RSS peaks unless explicitly
labelled as measured. Tensor headers, Python objects, Metal command buffers,
allocator slabs, and rational bit complexity are outside those exact sums.

### Static PowerFoam: exact named tensor boundary

For P0 features of width `C`, the static model and CSR are exactly

```text
M_static_world = 4N(5+C) + 4(A+N+1).
```

For RGB, `C=3`, hence the cell table is `32N` bytes and the complete named
world/topology table is `32N+4(A+N+1)`. Direct training forward saves the
world/CSR aliases, int32 `sorted_ids[b,N]`, int32
`screen_bounds[b,N,4]`, float32 rays `[b,H,W,6]`, ten-int32 and six-float32
configs, and the float32 `log_t` and int32 `pixel_stop` checkpoints. Together
with the returned RGB/features and alpha, the exact post-forward named tensor
boundary is

```text
M_static_direct_post_fwd
  = M_static_world
  + 20bN
  + 24bP
  + 4bP(C+3)
  + 64.
```

Here `4bP(C+3)` is output `C`, alpha, `log_t`, and `pixel_stop`. Backward adds
model gradients of exactly `4N(5+C)` and consumes or, when absent, creates
upstream output/alpha bars of `4bP(C+1)`. Thus the exact additional named
backward boundary is

```text
Delta_static_direct_bwd = 4N(5+C) + 4bP(C+1).
```

This is still not a native allocator peak because kernel-private and allocator
storage are not exposed.

The tiled training path retains tile offsets/ids and has the exact forward
output/checkpoint term

```text
4bP(C+3) + 4T + 4(T+1) + 4Q.
```

For the `emit_sort` candidate builder, the Python locals keep three
`[T+1]` int32 arrays and, at the return boundary, int64 sort keys, int32
unsorted ids, int64 `argsort` order, and final int32 ids. The exact named
builder co-residency is therefore

```text
M_static_emit_sort_builder = 12(T+1) + 24Q.
```

These equations expose two distinct static choices:

```text
b=1, frame-at-a-time: peak = O(N+A+P+Q)
b=F, video batch:      peak = O(FN+FP+Q).
```

The static renderer does not require the second choice. By contrast,
`MetalPowerFoamVideo` makes `raw_xy`, `raw_z`, `raw_radii`,
`raw_densities`, and `raw_features` parameters with a leading frame axis. Its
minimum raw P0 RGB table is exactly `32FN`; `initial_*`, `contrib_ema`,
`point_error_ema`, richer feature modes, gradients, and optimizer moments add
more `F`-scaled storage. This wrapper, not streamed static PowerFoam, is the
proper source match for the claim that the kinetic world removes an `FN`
parameter table.

### WorldFoam world state and cold compilation

The fixed-site kinetic geometry is `8N(6+L)` bytes. After releasing the cold
initializer, geometry plus the complete material training state is

```text
M_world_steady = 8N(12+L),
```

which is `120N` at `L=3`. A live material-only step adds the physical RGBA bar
and scalar loss:

```text
M_world_material_step_base = 8N(14+L)+4,
```

or `136N+4` at `L=3`, before optimizer history and compiled artifacts. The
current CPU full-geometry accumulator, with fixed cameras and no ray bar, is

```text
M_cpu_full_step_bars = (64+8L)N+4,
```

or `88N+4` at `L=3`. The intended float32 fused whole-step bars are smaller:

```text
M_fused_whole_step_bars
  = 16N material + (24+4L)N geometry + 4 loss
  = (40+4L)N+4.
```

Cold topology compilation is one `(view,pixel)` track at a time. The active
compiler reports

```text
T_compile,track
  = O(U_c N R_max)
  + O(G_c (N log N + N R_max)),
```

where `U_c` is the number of unique witnessed owner words and `G_c` the
cumulative root-complement discoveries. This excludes exact-root bit
complexity. Its pair-polynomial cache admits at most `N(N-1)/2` entries per
track. `sources_by_owner_word`, `source_roots`, isolated-root groups, active/
inactive/endpoint guards, root-complement witnesses, cells, charts, and exact
`Fraction` endpoints remain simultaneously relevant Python state. Their byte
cost depends on object overhead and numerator/denominator bit length, not only
on `N` and `E`.

For one unique temporal schedule of rank `j`, the exact retained CPU float64
schedule tensors are

```text
M_schedule(j) = 8(j^2+2j)
```

for node times, fit matrix, and barycentric weights. For one track-chart with
word length `w`, int64 owners and float64 node physical lengths add

```text
M_track_chart_geometry(j,w) = 8w + 8jw.
```

Schedules shared by many track programs must be charged once per unique
tensor storage, not once per reference. During chart lowering, the list of
`j` float64 `[w]` length tensors and the final stacked `[j,w]` tensor coexist,
so the exact named tensor term at that construction seam is

```text
M_chart_length_build(j,w) = 16jw,
```

in addition to the live schedule and owners. Exact node times and transition
depths are also retained as tuples of `Fraction` objects; they have `O(jw)`
object count but no fixed byte coefficient.

For one lowered equal-rank block, the exact CPU payload is

```text
M_source,q = 8s_q + 57r_q + (4+4j_q)w_q + 20.
```

If copied to MPS, its exact launch tensor sum is

```text
M_runtime,q = 8s_q + 4r_q + (4+4j_q)w_q + 24.
```

The CPU source remains live while the runtime copy exists. The bounded artifact
store charges reachable CPU tensors plus canonical metadata and excludes
targets, observations, samples, gradients, and device runtimes. It explicitly
does **not** enforce or measure cold compiler scratch. Therefore the honest
compile-peak identity is

```text
M_compile_peak
  = M_preexisting_artifacts
  + max_t(
      M_pair_cache(t)
    + M_predicate_sources(t)
    + M_exact_roots_and_guards(t)
    + M_chart_objects(t)
    + M_named_compile_tensors(t)
    + M_python_and_allocator(t)
    ),

M_preexisting_artifacts <= C_store.
```

There is no source-derived exact byte upper bound for the second line. A store
cap is not a compiler cap. Across all `P` tracks, cold work is the sum of the
per-track compiler work and is at least `Omega(P)`; bounded block compilation
and eviction can bound peak residency, but cannot make total compile work
sublinear in `P`.

### Warm node forward and retained active state

For block `q`, native node forward consumes compact RGBA `[s_q,4]` and returns
node chart `[r_q,j_q,4]`. Its exact block tensors and work are

```text
M_node_fwd,q = 16s_q + 16r_q j_q,
T_node_fwd,q = Theta(j_q w_q).
```

The current material executor keeps the active blocks of a spatial request
until their streamed sample cotangents are complete. Once a node cotangent of
the same shape exists, the exact accounting formula for active material state
is

```text
M_active_material
  = sum_q (16s_q + 32r_q j_q)
  + 16U
  + 16 max_q(s_q)
  + 4 number_of_active_blocks
  + 16.
```

The first `16r_qj_q` is primal node transfer and the second is its cotangent.
There is no requested-frame axis. Peak still scales with every active block,
`r_q`, and `j_q`; it is bounded in `P` only if the coordinator really limits
the spatial request and releases it before the next one.

### Streamed camera sampling and targets

For `M <= B_p K` observations at rank `j`, the retained ragged sample block is
exactly

```text
M_sample_retained = 4Mj + 24M.
```

The public native prepare adds a device row-id copy and configs, giving the
correct current seam

```text
M_sample_public_peak = 4Mj + 28M + 20.
```

The bounded float64 interpolation evaluator's declared source-visible upper is

```text
M_interp_upper = max(
  M(8j+12) + 4096+512j+8j^2 + K_sub(1024+512j) + 16K_sub,
  M(16j+32)
).
```

The native-prepare `4M+20` bytes may coexist after materialization. Sample
forward and reduction work are both `Theta(Mj)`; cumulative observation and
target work over full coverage is `Theta(PF)`.

The v3 acceptance fixture's procedural direct-pixel source declares a `32M`
source-visible temporary bound. Its CPU RGB target and MPS RGB target chunks
are `12M` each. A general resident source is qualitatively different:
`ResidentPowerFoamTargetSource.frames[V,F_d,3,H,W]` is exactly

```text
M_resident_target_video = 12 V F_d P_view.
```

It also uses a `52M` selected-pixel source boundary. Path/video fallback can
materialize one full `12P_view` frame plus bounded index/output rows. Thus the
frame-density theorem requires a nonresident selected-pixel source; merely
streaming the accelerator transfer does not remove a resident CPU video.

The replayable observation source retains no observation tensor, but it does
retain Python tuples with the charged logical size

```text
M_replay_metadata = 24(F+V).
```

The provider separately retains `F_d` scalar frame times and `VF_d` calibrated
camera records. Current source calls this allowed linear camera metadata and
reports zero persistent frame **tensor** bytes. The distinction matters:
“zero frame tensor” is true; “zero frame-dependent memory” is false.

### Staged full-geometry reverse

The staged route first forms one float32 node-length bar `[j_q,w_q]` and then
reduces it through the CPU float64 kinetic geometry bridge. Let
`rho=rho_q`. For fixed cameras, `_preflight_memory` declares the following
exact conservative logical-tensor formula:

```text
M_staged_bridge,q
  = 4j_q w_q
  + 8j_q rho
  + (56+8L)s_q
  + [8rho(7+L) + 96 + 8j_q + 8j_q rho]
  + 8rho(6+L)
  + 8(24rho+64)

  = 4j_q w_q
  + (56+8L)s_q
  + 16j_q rho
  + 8j_q
  + 8(37+2L)rho
  + 608.
```

The terms are, in order: one native `[j,w]` length bar; the largest CPU
float64 row copy; compact source ids plus compact float64 position, velocity,
and weight bars; the maximum row source; the maximum row parameter bar; and a
conservative node-scratch overbound. The source asserts exactly one
simultaneous `[j,w]` length-bar tensor, no full CPU `[j,w]` clone, no dense
global geometry bar inside this bridge, and no retained frame/sample/target/
prediction/material tensor. Work remains `Theta(j_qw_q)` up to exact-rational
and row-validation constants. This is an exact formula for the source's
logical preflight, not proof that allocator peak equals the formula.

### Intended fused reverse

For one sealed fused fixed-camera token, the exact retained launch tensors are

```text
M_fused_token,q
  = (48+4L)s_q
  + 60r_q
  + 20r_qj_q
  + 4w_q
  + 4j_qw_q
  + 56.
```

Expanded by ownership:

```text
aliased topology = 8s_q + 4r_q + 4w_q + 4j_qw_q + 4
aliased world    = 16s_q + 16r_qj_q
owned row data   = (24+4L)s_q + 56r_q + 4r_qj_q
owned configs    = 52.
```

The sealed adapter therefore creates only
`(24+4L)s_q+56r_q+4r_qj_q+52` persistent row/config bytes beyond the aliased
runtime/world tensors. At invocation it additionally receives node cotangent
`16r_qj_q`, compact material bar `16s_q`, and the once-per-step float32 global
geometry bars `(24+4L)N`. Native source creates exactly one distinct int32
validation receipt, four bytes. Counting the token itself, the exact named
phase boundary is

```text
M_fused_named,q
  = M_fused_token,q
  + 16r_qj_q
  + 16s_q
  + (24+4L)N
  + 4.
```

The primal `4j_qw_q` node-length table remains, but there is no `[j_q,w_q]`
cotangent. The Metal reverse carries only scalar current/prefix transfer and
local bars per thread, so algorithmic private state is `Theta(1)` in owner-word
length; native private bytes are not measured.

Current source now enqueues a validation kernel and then a status-gated write
kernel on the same MPS stream. The write checks the shared status before every
atomic, so a rejected block is intended to leave all four caller-owned bars
unchanged. Objective-C++ returns four aliases plus the status scalar; Python
expects exactly those five tensors, exposes `accepted_bars()`, and synchronizes
through `validation_status_i32.item()`. The ABI is source-consistent. It is
still not evidence: the native variant is marked source-only until rebuild and
sparse-oracle parity, and no result artifact closes transactional behavior or
allocator peak.

### Complete scaling-axis inventory

The remaining state by requested axis is:

```text
F:
  24(F+V) replay metadata;
  O(F_d) provider times and O(VF_d) camera records;
  12VP_viewF_d if ResidentPowerFoamTargetSource is selected;
  12PF target/prediction/output if a caller retains full coverage.

J:
  8(j^2+2j) unique schedule;
  8jw source physical lengths and 4jw native lengths;
  16rj node chart and another 16rj node cotangent;
  4Mj interpolation weights;
  8j^2 and bounded j-dependent interpolation scratch;
  staged 4jw length cotangent, absent from fused.

E:
  exact root, guard, endpoint, witness, cell, and chart Python objects;
  arbitrary-precision polynomial/Fraction bits;
  indirect growth of chart count and the sums over j, w, and r;
  no dense requested-frame-by-event tensor.

W:
  8w CPU owners, 4w native owners;
  8jw CPU lengths, 4jw native primal lengths;
  staged 4jw cotangent only;
  Theta(jw) node forward and ordered-word reverse work.

P:
  one track program/ray binding per resident track, including 96 bytes of
  float64 affine-ray coefficients per program;
  sums of track-chart schedules/owners/lengths and exact certificates;
  bounded peak only through B_p, store eviction, and release fences;
  Omega(P) cold compile work and Omega(PF) full-coverage sample work.

N:
  8N(12+L) steady WorldFoam geometry/material state;
  current or fused global bars above;
  compact s_q subsets per active block;
  up to N(N-1)/2 exact pair-cache entries during each track compile;
  static PowerFoam world/CSR and per-launch bN ordering tensors.
```

No source tensor proportional to requested `F*w`, `F*j*w`, or `FN` exists in
the selected compiled WorldFoam reverse. That statement does not erase the
listed `F`, `PF`, `P`, `E`, `J`, `W`, or compiler-object terms.

### Falsifiable acceptance equations

For requested-density rows `F_i in {8,64,300}`, hold fixed the 300-frame
provider grid and the complete structural fingerprint

```text
Sigma = (world digest, camera digest, physical interval, N, P, L,
         active block ids, {r_q,s_q,j_q,w_q,rho_q}_q,
         B_p, K, artifact policy, compiler/rank/fallback receipts).
```

Then a valid narrow frame-density result must satisfy exactly

```text
Sigma(F_i) = Sigma(F_0)

M_source,q(F_i)  = 8s_q+57r_q+(4+4j_q)w_q+20
M_runtime,q(F_i) = 8s_q+4r_q+(4+4j_q)w_q+24
M_active_material(F_i) = the invariant formula above

node_forward_launches(F_i)      = constant
node_forward_interactions(F_i)  = constant
word_VJP_launches(F_i)          = constant
word_VJP_interactions(F_i)      = constant
maximum live M(F_i)             <= B_p K
persistent sample/target/prediction tensor bytes = 0.
```

Expected linear quantities must not be hidden inside the invariant category:

```text
observation_count(F_i)          = P F_i
transferred_target_bytes(F_i)   = 12 P F_i
sample-to-node interactions     = sum_launches M_l j_l
replay metadata                 = 24(F_i+V)
full retained RGB video, if any = 12 P F_i and is an immediate failure.
```

The current v3 contract's measured allocator tolerances are

```text
RSS/MPS growth from F=8 to F=300 <= 32 MiB,
sampled memory scale             <= 1.25,
frame-invariant logical slack    = 0.
```

Those are empirical tolerances, not exact-memory equations. The schema sets
`require_exact_allocator_peak_measurement=false` and
`require_native_private_scratch_measurement=false`; it certifies only
`fixed_site_material_only_manual_vjp`, explicitly sets full-geometry
certification false, and attests three material-route kernels. Its checked-in
trial is a tiny two-site, rank-four, `384x384`, `B_p=512`,
`M<=4096`, one-artifact-entry fixture. No result files currently exist under
`outputs/worldfoam_memory_scaling/`. It cannot be cited as full-geometry,
fused, compiler-peak, or general-`N/P/E/J/W` evidence.

Full-geometry promotion additionally requires, for every block,

```text
staged logical preflight = M_staged_bridge,q above
maximum simultaneous [j_q,w_q] cotangents = 1
fused retained token = M_fused_token,q above
fused [j_q,w_q] cotangent allocations = 0
fused validation receipt bytes = 4
rejected fused launch changes each caller-owned bar by exactly 0 bytes
norm_inf(fused_bar-staged_bar) <= epsilon_fused_staged
norm_inf(staged_bar-dense64_bar) <= epsilon_staged_dense
for each material/position/velocity/weight bar, with both epsilons declared.
```

Cold-compile promotion requires a separate bound, because none exists now:

```text
peak pair-cache entries <= N(N-1)/2 per track
peak retained artifact bytes <= C_store
peak compiler scratch bytes <= a declared C_compile
max exact-integer/rational byte length <= a declared B_exact
peak process RSS during compile <= baseline + C_store + C_compile.
```

Every term must be measured or recursively accounted; setting
`C_compile = C_store` without observing scratch is invalid.

Finally, the broad “sublinear over time” statement is falsified unless its
scope is restricted. For a duration-growing family, a necessary persistent
structural condition after streaming metadata is

```text
E(F)
+ sum_unique_charts (j_c(F)^2)
+ sum_track_charts (w_pc(F) + j_c(F) w_pc(F))
= o(F).
```

This count condition is necessary, not sufficient: exact-root numerator and
denominator bit lengths must also keep the byte-valued
`M_exact_roots_and_guards(F)` sublinear. Current source establishes neither
condition, and the present replay metadata is already `Theta(F)`. For
full-coverage rendering/training,

```text
T_total(F) >= Omega(PF)
```

regardless of representation. The defensible paper statement is therefore:
**for denser sampling of one fixed compiled physical program, WorldFoam's
expensive ordered-world forward/reverse state and work are independent of
requested `F`, while bounded camera/target/sample work remains linear.** It is
not an end-to-end sublinear-time theorem and not a general longer-duration
memory theorem.

No runtime, import, build, native load, decode, or test command ran in this
expansion pass.

## Source-only fused full-geometry admission closure

The external scientist's useful correction is now the project boundary:
translated optical-depth measure `(kappa, nu)` is a theorem/certification
language, while the runtime remains kinetic event charts, ordered affine
transfer words `(beta, m)`, bounded interpolation nodes, streamed node
cotangents, and one compiled word/world reverse. We did not add a sixth runtime
formulation. The safe production geometry rule remains complete recompilation;
local simple-root repair is optional future work gated by measurement.

The fixed-camera fused-v1 source now removes the staged `[J,W]` physical-length
cotangent. One prepared multi-block transaction owns fresh zero compact/global
scratch, rejects duplicate block generations and storage aliases, checks a
logical scratch budget before allocation, initializes one four-byte status,
then enqueues `validate all -> accumulate all -> finalize all` without a host
barrier between phases. The finalizers scan every output ledger for NaN or
infinity after the atomics. One successful completion fence and a zero status
produce a sealed receipt retaining only accepted output bars; raw phase results,
node bars, prepared tokens, and status are released from the transaction.

This is fail-stop admission, not rollback or exact summation. Prewrite rejection
leaves the owned zero scratch unchanged. Postwrite rejection can leave mutated
scratch, which is quarantined and cannot authorize an optimizer commit. If an
abort/completion fence fails, live roots are retained globally and every later
public fused adapter operation now fails immediately until process restart.
The strong lifetime claim applies only to this prepared transaction; the legacy
one-block convenience remains unpromoted.

Still open are the higher-layer facts the adapter cannot invent: the executor
must prove the supplied ordered blocks are its complete active manifest; the
accepted bars must feed the existing out-of-place fail-atomic candidate updater;
native same-stream ordering and float-atomic nonfinite propagation need rebuilt
fixtures; sparse/dense gradient parity and allocator peak need measurement; and
the fused route is fixed-camera only. The host had roughly 70--110 MiB free
during this pass, so no Python import, pytest, extension build, Metal/MPS launch,
training, or decode was attempted.
