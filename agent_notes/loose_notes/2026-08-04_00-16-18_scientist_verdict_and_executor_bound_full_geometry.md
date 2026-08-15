# Scientist verdict and executor-bound full geometry

## User input

The independent scientist's verdict is directionally correct and matches the
repository decision already recorded on August 3:

- keep kinetic event charts, ordered transfer words, bounded temporal node
  replay, streamed node cotangents, and one block reverse as the canonical
  runtime architecture;
- use the translated optical-depth measure as the strongest mathematical
  theorem/certificate layer, while retaining the four-scalar `(beta, m)`
  runtime payload;
- use constructible-chart language only where it sharpens the moving-root and
  seam statements;
- keep similarity factoring conditional on an end-to-end reduction in event,
  rank, node, and camera-certificate cost; and
- do not open a sixth formulation before closing the geometry integration.

The scientist also correctly tightened the evidence boundary: the previously
integrated dense/block-major path proved one material-word VJP per active
block, not yet a rebuilt dataset-bound native full-geometry training step.

## Source work continued under host saturation

The host remained unsafe for execution (`load averages: 79.78 79.74 84.23`).
No pytest, Python process, dataset decode, native build, MPS, Metal, CUDA, or
training workload was launched.

Static integration work continued along the already-selected seam:

1. `kinetic_native_material_step_executor.py` now owns mutually exclusive
   `material_only` and `full_geometry` reverse modes. The full-mode receipt is
   tied to the actual executor session, runtime/block, node-bar identity,
   observed sample-launch count, streamed sample count, and bounded `[J,W]`
   physical-length bar.
2. The full-mode receipt now also seals the per-block loss-scalar identity and
   tensor version. The geometry finalizer cannot accept an arbitrary cloned or
   caller-substituted scalar after the streamed reductions.
3. `test_kinetic_full_geometry_step_cpu_fake_native.py` no longer constructs a
   private `_NativeDeferredState`, calls the old sample helper, or supplies its
   own coverage counts. It launches node forward and every ragged sample
   through `KineticNativeMaterialStepSession`, asks that same session for the
   sole full VJP, and passes only the executor-sealed execution receipt to the
   geometry finalizer.
4. The failure fixture now checks that a foreign loss scalar is rejected before
   the fence and that a failing fence poisons and clears all request-local bars
   before any global material or geometry commit.

This closes the known source-level provenance bypass. It does **not** produce a
green gate because the changed files have not run on the saturated host.

## Current next step

On a quiet host, run only the three focused CPU contract files for the executor,
full-geometry finalizer, and dense cached request. If green, route the existing
dense cached request through `full_geometry`, immediately fence/reduce each
bounded length bar, and commit request-local material plus geometry bars only
after exact replay coverage and executor sealing. Do this in the existing
dense route; do not introduce another whole-step coordinator.

Native allocator and timing evidence remains a later approved-host gate. The
claim remains source-level and CPU-fake-native until that evidence exists.

## Expansion Pass 2: Exact surrogate VJP, continuous geometry error, and seams

### Scientist comparison resolved

The follow-up scientist review does not change the implementation direction.
It makes the division of labor sharper:

- the repository architecture is the canonical executable method;
- the translated optical-depth measure is the strongest new mathematical
  formulation and belongs in the proof spine;
- `(beta,m)` remains the runtime quotient;
- no sixth formulation and no foam analogue of Gaussian Schur marginalization
  should be invented merely to sound novel; and
- the next mathematical work is a quantitative geometry/ray tangent theorem
  plus an exact seam-dispatch rule, not another representation branch.

This is a supported synthesis, not a compromise between incompatible plans.
The measure theorem explains ordered transfer. The kinetic executor realizes
its compact homomorphic image.

### Observed implementation fact: the compiled-surrogate VJP factorization

For one stable owner chart `c`, let:

```text
h_c(t, theta) = encode(G_c(t, theta)) in R^4
```

where `G_c` is the exact fixed-word transfer and `encode` is the affine-Lie
chart. Let `t_cj`, `j=1..J_c`, be compile nodes and `l_cj(t)` the actual
fit-derived second-form barycentric cardinal weights. The implemented
surrogate is:

```text
hhat_c(t, theta) = sum_j l_cj(t) h_c(t_cj, theta)
Ghat_c(t, theta) = decode(hhat_c(t, theta)).
```

For samples `s` dispatched to chart `c(s)`, the exact reverse of this fixed
surrogate is:

```text
bar_h_cj = sum_{s: c(s)=c} l_cj(t_s)
            Ddecode(hhat_c(t_s))^T bar_G_s

bar_theta = sum_c sum_j Dh_c(t_cj, theta)^T bar_h_cj.
```

This is the current memory-light factorization:

- stream samples and reduce into `O(sum_c J_c)` node cotangents;
- discard targets, residuals, and sample weights after each bounded launch;
- run the ordered-word/world reverse once at each compile node;
- scatter the resulting local bars into step-owned global bars.

No approximation is introduced by this reverse *relative to the compiled
surrogate*. The approximation question is only how closely `Ghat` and its
physical tangent track the continuous exact `G` inside a stable chart.

### Claim: local forward/tangent bounds imply a normalized global loss-VJP bound

Definitions for one sample:

```text
e0 = sup_t ||Ghat(t,theta) - G(t,theta)||
e1 = sup_t ||Dtheta Ghat(t,theta) - Dtheta G(t,theta)||_op
M0 = sup_t ||Dtheta G(t,theta)||_op
q  = d ell / d G
Mq = sup ||q||
Lq = Lipschitz constant of q with respect to G.
```

Then, inside a frozen stable stratum:

```text
||Dtheta ell(Ghat) - Dtheta ell(G)||
 = ||Jhat^T qhat - J^T q||
 <= ||Jhat-J|| ||q|| + ||Jhat|| ||qhat-q||
 <= e1 Mq + (M0+e1) Lq e0.
```

For a weighted objective `L = sum_s a_s ell_s` with nonnegative
`sum_s a_s = 1`, uniform bounds give:

```text
||grad_theta Lhat - grad_theta L||
 <= e1 Mq + (M0+e1) Lq e0.
```

The bound therefore does not acquire an artificial factor of requested frame
count under the actual globally normalized loss. For mean RGB MSE, `q` is
linear and `Lq` is explicit. A paper theorem should state the chosen norm and
the RGB/transfer cone used to bound `Mq`.

### Sparse local-to-global scatter

Each ray/chart depends only on a local parameter vector `theta_pc`; canonical
world parameters are `Theta`, with local gather/scatter map `A_pc`:

```text
theta_pc = A_pc Theta.
```

The safe global bound is:

```text
||grad_Theta Lhat - grad_Theta L||
 <= sum_pc ||A_pc|| [e1_pc Mq_pc + (M0_pc+e1_pc)Lq_pc e0_pc].
```

For pure index gathers and Euclidean norms, a sharper bound can use the maximum
site-incidence overlap degree rather than the full sum. The implementation
should report both the conservative `L1` sum and the observed overlap degree;
the paper need not rely on the sharper statement initially.

### What the existing continuous certificate does and does not prove

`continuous_lie_jet_certificate.py` already propagates interval duals through
a fixed-word representation parameterized by boundary planes, affine rays,
material, and lowered Mobius coefficients. It can certify a continuous
world-Jacobian approximation in that coordinate system.

The current native full-geometry path instead exposes direct kinetic site
parameters:

```text
positions0, velocities, quadratic weight coefficients, affine ray coefficients.
```

The missing bridge is not a new renderer. It is one of:

1. extend the interval dual certificate through the exact map from direct
   kinetic sites/rays to boundary/Mobius coefficients; or
2. certify that map separately, then compose its Jacobian bound with the
   existing transfer certificate by the chain rule.

The second option is smaller and easier to falsify. It must include lower
bounds on every cut denominator and fiber speed. If those margins cross zero,
the stable-stratum theorem fails closed and the compiler must split or
recompile.

### Exact rational samples versus algebraic event roots

The owner compiler retains each algebraic event as a rational polynomial plus
a certified isolating interval. For a rational sample time `q` and a unique
simple root `alpha`, exact comparison does not require materializing `alpha`:

```text
if q <= isolator.lo: q < alpha
if q >= isolator.hi: q > alpha
otherwise evaluate p(q) exactly and compare its sign with p(isolator.lo/hi).
```

If `p(q)=0`, the sample is exactly the rational root and the right-continuous
chart owns the seam. The source already contains this sign-orientation logic in
`kinetic_multichart_transfer_program.py`; however compact float64 charts are
built only on the safe interval outside unresolved isolators. The warm/native
dispatch therefore still rejects samples inside those neighborhoods because
it has no certified transfer payload there.

There are two honest completion routes:

- **Exact narrow fallback:** exact-dispatch the sample and evaluate/reverse the
  fixed owner word directly for the unresolved-neighborhood sample. Refine the
  isolator so this fallback is rare and account for it explicitly.
- **Declared avoidance:** require the finite requested sample set to avoid all
  unresolved isolators, refining guards against the requested rational grid
  until this is proved. This is valid for a frozen dataset protocol but is not
  an arbitrary-densification claim.

The first route is the stronger paper result. It adds no persistent `O(F)`
state, but fallback work must appear in the sample complexity and telemetry.

### Systems corrections made during this pass

The new executor ownership rule exposed a real compatibility defect: the older
lazy material coordinator supplied one shared loss scalar to all native blocks,
while the hardened executor correctly rejects cross-block accumulator reuse.
The lazy coordinator now allocates one scalar per active block, streams samples
into it, and reduces it into the step-owned loss after that block's reverse.
The extra live state is `4 * active_block_count` bytes, independent of frame
density.

The executor now also:

- rejects aliases by underlying storage rather than Python object identity;
- retains each block-local loss scalar until session sealing so a post-launch
  mutation is detectable;
- verifies read-only inputs immediately after native sample preparation and
  again after launch; and
- binds one global loss denominator/scale/normalization id across every sample
  block in a session.

These are source changes only. The host again reached load averages near 80--90
with unrelated Node/vitest and UI processes, so no Python or device gate ran.

### Falsification gates after the host recovers

1. Run the three focused CPU executor/dense/full-geometry files.
2. Add a synthetic direct-kinetic parameter direction and compare the composed
   interval bound against dense finite-difference/JVP witnesses over a complete
   stable chart.
3. Place rational samples on both sides of, inside, and exactly at an
   algebraic-root isolator. Require exact chart orientation and either a sealed
   fallback receipt or a deliberate fail-closed avoidance certificate.
4. Sweep requested `F` at fixed chart/rank. Verify heavy word launch counts and
   peak `[J,W]` state are invariant, while only sample/I/O work grows.
5. Measure native allocator peaks only in a quiet approved window. Logical byte
   accounting is not allocator evidence.

### Current belief

**Confidence: high** that no new Schur-like foam formulation is needed for the
memory-light result. The missing work is transaction-safe integration, the
direct-kinetic tangent certificate, seam fallback/avoidance, and native
allocator evidence.

**Could be wrong if:** the required rank `J` or event count grows empirically
with requested temporal density rather than physical camera/world complexity.
That is the decisive scaling falsification test; if it happens, the current
surrogate is not sublinear in the intended regime even if its executor is.

## Expansion Pass 3 — full-step source integration and the real native blocker

### Scientist verdict applied

The external scientist's recommendation agrees with the narrowed project
direction:

- retain kinetic event charts -> ordered words -> J-node temporal surrogate ->
  streamed node cotangents -> one word VJP;
- treat the translated optical-depth measure and its boundary-mass tangent as
  the genuine new mathematical formulation;
- keep runtime at the affine transfer quotient `(beta,m)`;
- do not invent a sixth formulation or a foam analogue of Gaussian Schur
  elimination; and
- defer conditional similarity gauges, sheaf language, and product trees until
  a concrete seam or parallel-scan theorem requires them.

The resulting work item was systems integration, not another representation.

### Source-level full geometry now reaches a step-owned accumulator

The mutually exclusive executor modes now feed the dense cached replay path.
For each active compiled block, the source candidate:

1. builds the node chart once;
2. streams every request sample into the same node cotangent and a block-local
   loss scalar under one global normalization;
3. launches exactly one material-only or material-plus-length word reverse;
4. in full-geometry mode, fences and immediately reduces the bounded `[J,W]`
   length bar to site position, velocity, polynomial-weight, and affine-ray
   bars;
5. returns one sealed request-local combined delta; and
6. merges deltas in canonical replay order into zero-owned, world-bound step
   bars, exposing them only after an exact full-manifest replay receipt.

The request never mutates optimizer-visible bars. Duplicate, foreign,
out-of-order, wrong-normalization, wrong-ray, storage-alias, or post-replay
failures poison and clear the step state. This closes the source transaction
boundary that the earlier free-standing geometry reducer lacked.

This is still not a production trainer claim. The new files were only reviewed
statically because the host remained unsafe; their focused tests have not run.
The upstream replay source also retains `O(F)` scalar frame metadata even
though its tensors and request deltas are bounded independently of F.

### Asynchronous residency correction

The first dense draft released each K-local Python object after launch but
performed only one final lane fence. That does not bound MPS allocator
residency: queued command buffers may retain every target, weight, row, and
config tensor until the final synchronization.

The source candidate now requests a completion fence after every sample launch
before releasing K-local inputs. MPS preparation accepts only the canonical
`torch.mps.synchronize()` function and provenance id; CPU fake-native tests
retain an injected fence. This deliberately starts with one launch in flight.
It may reduce throughput, but it gives the first allocator experiment an
unambiguous hypothesis. After real measurement, a small fixed-depth queue may
replace per-launch synchronization if its peak is still bounded independently
of F. Logical byte counts and Python `del` statements remain insufficient
allocator evidence until that real path runs.

### Native ABI audit and fail-closed preparation

The Metal source contains the four required compiled operations:

```text
kinetic_precompiled_length_p0_lie_node_forward_launch_only
kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only
kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only
kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only
```

The installed `_C.cpython-311-darwin.so` predates those sources and registers
none of them. The previous executor checked five Python callables, including
the Python-only sample preparer; this could pass even when `torch.ops` had no
compiled kernels.

The real wrapper now exposes a cold compiled-ABI attestation. For MPS runtime
preparation it requires:

- exactly one selected `_C` library and no load error;
- the library mtime to be at least every compiled C++/Objective-C++/loaded
  Metal source mtime;
- exact dispatcher schemas for the four compiled operations; and
- a `CompositeExplicitAutograd` implementation for each.

CPU fake-native adapters intentionally retain the callable-only seam. The
import verifier calls the same attestation. No attestation or native op was run
in this pass; the current binary is expected to fail until rebuilt.

### Smallest remaining execution sequence

1. Wait for a genuinely quiet host, then run only the focused executor, dense
   request, full-geometry request, and adapter/import-verifier CPU tests.
2. Rebuild the Metal extension and require exact ABI attestation before tensor
   materialization.
3. Bind the production fence to a real MPS completion primitive and run a tiny
   forward -> streamed loss-only sample -> material/full VJP -> CPU geometry
   reduction parity gate.
4. Measure synchronized allocator snapshots and sampled high-water memory at
   `F=5/41`, then `16/64/128/300`, with chart/rank/word counts held fixed.
5. Only after those gates, connect dataset-bound records, initialization,
   checkpoint/evaluation, and the unified runner.

The remaining obstacle is therefore not an unknown foam formulation. It is a
stale native build, real completion/backpressure evidence, focused runtime
verification, and then trainer plumbing.

## Expansion Pass 4: static residency red-team and ownership closure

The scientist's final comparison sharpened the scope rather than changing the
architecture. The translated optical-depth measure is the one genuinely strong
new mathematical formulation. Kinetic lower envelopes, constructible chart
language, product trees, and similarity gauges are useful derivations or
conditional tools, not reasons to fork the runtime. The canonical chain remains

```text
kinetic event charts
  -> ordered owner words
  -> J-node affine-transfer surrogate
  -> streamed sample-to-node cotangents
  -> one ordered-word VJP per active block.
```

A source-only static red-team then found several lifecycle gaps that the
earlier logical tensor counts did not close:

- full target frames could be copied to MPS and retained by queued command
  buffers before the first sample fence;
- compact reverse bars could queue across active blocks even though the
  preflight budgeted only one;
- a whole-request delta could be scattered into the step bars and released
  without a completion fence, allowing deltas to queue across spatial
  requests;
- optimizer authorization did not revalidate the exact material/background
  snapshot;
- the executor accepted a bare full-geometry acknowledgement without proof
  that `[J,W]` had been fenced and reduced; and
- native sample preparation added named row/config scratch absent from the
  public logical count.

The source candidate now closes those ownership boundaries:

1. Each full frame is decoded/resized on CPU, selected pixels are gathered into
   one bounded CPU `[N,3]` chunk, and only that chunk is transferred to the
   device.
2. Every sample launch is completion-fenced before K-local launch state is
   released.
3. Every active-block material/loss scatter is fenced before compact reverse
   scratch is released.
4. Every request-delta scatter is fenced before the delta is destructively
   consumed; a sealed tensor-free commit receipt and chain digest bind the
   whole-step cursor.
5. The step retains read-only references to the material/background snapshot
   and rejects mutations before authorization. Authorization is explicitly a
   point-in-time permission, not a promise to repair bars after later mutation.
6. Full-geometry execution is released only against the exact sealed geometry
   reduction and fence provenance. Executor telemetry says
   `fenced_and_reduced_not_globally_committed`; the request/step layers own the
   later commit proofs.
7. The public named sample boundary now counts the sealed `4NJ+24N` block plus
   the native prepare's `12N+20` bytes of CPU/MPS row and config scratch.

This still is not a whole-process memory proof. A free target-loader wrapper can
allocate or retain unreported state; PIL/OpenCV/NumPy decoder internals, the
float64 barycentric-weight materialization path, driver/command-buffer storage,
and allocator peak are unmeasured. Production promotion therefore remains
false until a sealed exact-type target loader and real allocator telemetry
replace those open scopes.

No Python, import, pytest, build, Metal/MPS, CUDA, dataset, or training command
ran during this pass. The last live preflight still showed load averages around
`62--76`, about `75 MiB` of free VM pages, and only `36 GiB` free disk. Static
whitespace and call-site checks were the only safe verification.

The correct stop point is now explicit: do not invent another formulation or
add another renderer fork. On a quiet host, run the focused CPU/fake-native
transaction gates, rebuild and content-attest the native extension, then run a
tiny parity/allocator death curve. If those pass, connect one dataset-bound
`worldfoam_native4d` lane. If they fail, fix the measured boundary rather than
opening more theory.

## Expansion Pass 5: asynchronous failure paths are fail-stop

A final read-only lifetime audit found four cases where successful-path byte
accounting was insufficient: lane construction could fail before the caller
owned the partial device objects; cleanup could drop references after a failed
fence; executor poisoning/abort could release native state without completion;
and the older full-geometry finalizer could be mistaken for an accelerator
lifecycle seam.

The source contracts now make the ownership rule uniform:

1. Poisoning marks the executor failed but retains every token/world/native
   result reference.
2. `abort` requires an explicit completion callback and nonempty provenance;
   it releases only after that callback returns `None`. A failed fence retains
   the references and permits a later fenced-abort retry.
3. Partial lane construction fences before its locals unwind. If that fence,
   an abort fence, or the outer lane-release fence fails, the world-bound step
   accumulator retains the original exception, traceback, and live roots in a
   sealed quarantine. The step is poisoned, optimizer authorization is
   impossible, and the process must restart.
4. The older standalone full-geometry assembly/finalizer is now explicitly
   CPU/fake-native-only and rejects non-CPU work, sessions, and tensors at its
   public boundaries. Production MPS ownership belongs to the dense
   request-delta route.

The successful fence count and memory formula are unchanged. Quarantine is not
a normal-step memory term; it is an intentional bounded leak after loss of a
completion proof, chosen over use-after-free or unbounded asynchronous queue
growth. Result accounting now also states that the target loader remains an
arbitrary callable and that decoder allocator peak, simultaneous float64 sample
materialization scratch, and whole-step Python-object peak are unmeasured.

Only static call-site and whitespace checks ran. Runtime behavior, native fence
semantics, and allocator peaks remain open for a quiet approved host. The final
preflight had improved load averages (`9.37/12.33/27.22`) but only `4,347`
free 16-KiB VM pages (about `68 MiB`), so Python/import/test/device work was
still unsafe; disk had `40 GiB` available.

### Exact production-trainer boundary

A separate static trace confirmed that the implementation ends at a sealed
`PaperKineticDenseOptimizerAuthorization`. Bounded dataset providers, paper
sampling, lazy program-bundle harnesses, the CPU compiler/lowering, dense
request replay, material/geometry bars, and exact manifest authorization are
real. The production `PaperKineticWorldInitializer` and
`PaperKineticTrackProgramFactory` still have only test implementations, and no
native-4D parameter container/optimizer consumes the authorization. There is
also no prediction-returning streamed evaluator, kinetic checkpoint schema,
paper artifact writer, or `worldfoam_native4d` runner dispatch.

The smallest honest integration sequence is fixed-resolution/fixed-site first:
implement the initializer and track factory, run one fenced optimizer update
with cameras frozen, rebuild/reseal all structure after each geometry update,
prove a two-step loss decrease plus identical checkpoint resume, then add
frame-at-a-time heldout evaluation and the existing paper evidence schema. Only
after native parity and allocator telemetry should the distinct runner lane be
registered. Progressive growth is last.
