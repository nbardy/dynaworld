# WorldFoam Measure--Connection Scientist Review

Date: 2026-08-05 KST

## Trigger

The user supplied a new scientist memo proposing a unified fiber-bundle,
connection/curvature, factorization, jet, and monodromy view of WorldFoam and
asked for independent subagent review, retention of all useful mathematics,
and a judgment about whether this is the best current approach.

Source:

```text
/Users/nicholasbardy/.codex/attachments/
  2492c6e4-bcde-416a-80b0-711c5a6101da/pasted-text.txt
SHA-256:
  965c7a1a28343914dd348a88afa1b30a976dabd6dbf80fb48a1076ad878334c5
```

The host safety pause remained active. No Python, import, test, build, Metal,
MPS, CUDA, dataset, or training workload ran.

## Independent Review Lanes

Three read-only subagents independently audited:

1. mathematical signs, assumptions, proofs, counterexamples, and theorem
   scope;
2. fit to the current compiler/native ABI, memory/work consequence, and exact
   oracle; and
3. literature/novelty boundary and paper claim ladder.

All three agreed that the constrained Lagrangian ray-fiber optical connection
is the memo's strongest new project-level idea, but not yet a speed or memory
result.

## Central Synthesis

The retained formulation is layered:

```text
ordered optical field
  -> translated optical-depth measure (kappa,nu)
  -> exact affine quotient (beta,m)
  -> independently constrained scene/camera flow
  -> flow-covariant U_tilde
  -> signed optical-curvature residual K_F.
```

The prior translated measure remains the order-explicit proof/tangent object.
The four-scalar affine transfer remains the exact compact runtime quotient.
The new connection explains when the vertical optical object is reusable
across time after a declared correspondence. It does not replace retained
depth, the event atlas, or the colored-overlap commutator.

## Critical Corrections

- The memo uses a left-ordered ODE, while executable WorldFoam is right-
  ordered/near-to-far. The correct repo curvature is
  `F^R_tz=dt A_z-dz A_t+[A_t,A_z]`, and its integral is
  prefix--curvature--suffix in repo order.
- The memo's written plaquette has the wrong small-loop sign for its
  orientation.
- `A_z` must include the physical ray-speed Jacobian.
- Flatness is equivalent to reuse of every transported subinterval under a
  normalized orientation-preserving flow, not to equality of one total ray.
- P0 singular curvature is `([wA_z]-r_dot[A_z]) delta`; the scalar factor form
  requires a continuous trace of `w`.
- Scalar depth flow cannot represent generic transverse camera/object motion.
- A freely fitted per-ray flow is gauge cheating.
- The factorization/cosheaf and program-stack language is not formally or
  operationally justified by current code.
- The order-`q` seam theorem needs separate signed one-sided width coefficients
  and common exterior jets.
- Separated real roots have canonical order and trivial real-order monodromy.

## Important ABI Correction

The first draft of the plan said to compile `U_tilde` through the existing
atlas. That is too broad.

`U_tilde=H_a U H_b^-1` generally lies in the affine group completion. It can
have attenuation above one and a signed moment. The current physical Lie atlas
requires `kappa>=0` and `0<=v<=kappa`, so it cannot consume `U_tilde`
unchanged.

The first oracle must instead use:

- physical transfer `U` with the existing cone;
- group-completion `U_tilde` with an unrestricted `beta>0` affine chart; and
- `K_F=dU_tilde/dt` as a signed four-component tangent, not a transfer.

Only reconstructed `U=H_a^-1 U_tilde H_b` receives the physical cone and
end-to-end primal/tangent certificate.

## Computational Meaning

The connection can only help by reducing certified temporal ranks/chart
counts or by exposing a separately exploitable sparse curvature source. It
does not reduce the four-scalar state and does not remove sample/output-linear
work.

The first ablation is direct `U` versus group-completion `U_tilde` under one
approximation family and one reconstructed-`U` primal/tangent tolerance. Only
if `U_tilde` wins should the CPU oracle compare signed `K_F`. No native
connection shader is justified first.

Promotion remains gated on at least `2x` total retained payload and ordered
work improvement against both direct alternatives, at least 20% measured
request-time improvement, admissible target-independent flow, full flow and
endpoint VJPs, stable endpoint inverses, and reconstructed physical-cone
validity.

## Durable Files

- `research_notes/worldfoam_paper/
  WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`
  remains the theorem-by-theorem corrected intake.
- `research_notes/worldfoam_paper/
  WORLD_FOAM_MEASURE_CONNECTION_SYNTHESIS_2026-08-05.md`
  now unifies the prior measure theorem with the new connection and records
  the full math, flow lift, ABI split, oracle, work model, claim ladder, and
  kill criteria.
- `TODO/worldfoam_memory_light_native4d.md` now records the unrestricted
  group-completion and signed-tangent oracle boundary.
- `WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md` records the same
  corrected C5 hypothesis.
- Both research indexes route future agents to the synthesis.

## Paper Decision

Claim now only that the flow-covariant identity is derived and that flatness
characterizes transported-subinterval reuse under an independently constrained
flow. Do not yet claim lower temporal rank, memory savings, a curvature
runtime, general moving-camera correspondence, or a holonomy renderer.

Keep the method name **WorldFoam**. Reserve holonomy for closed ray-time
plaquettes. If the oracle succeeds, the ray-fiber optical connection is the
stronger main-paper connection; the older cell-frame adjacency holonomy should
remain an exploratory diagnostic or appendix until its correlation experiment
wins.

## Next Safe Work

1. Finish safe-host validation of the direct memory-light `U` trainer.
2. Build the CPU float64 group-completion oracle on a safe host.
3. Run the exact correctness sentinels, including a nonphysical `U_tilde` with
   physical reconstructed `U`.
4. Compare `J_U`, `J_U_tilde`, and selected tangent ranks on real stable
   charts.
5. Compute `K_F` only after `U_tilde` shows a real compression signal.
6. Promote no native branch without the stated payload/work/time gates.

## Expansion Pass 2: Proof, Native-Lifetime, and Work-Count Reconciliation

The attachment was presented again for a fresh review. Its SHA-256 is exactly
the digest above, so this was not a second mathematical proposal. Three fresh
read-only lanes audited the connection math, exact memory/work theorem, and
native allocation/lifetime map. They independently reached the same main
decision as the first intake: the connection is the strongest new mathematical
layer, while direct physical `U` remains the production path to finish first.

### Exact fixed-program work theorem

Use different symbols for native track-chart rows and ordered run entries:

```text
R_row,b = track-chart rows in active block b
W_run,b = ordered owner/run entries in active block b
J_b     = certified temporal nodes in block b
F_r     = requested samples assigned to row/chart r
K       = maximum live sample launch
```

The node-forward kernel dispatches one thread for every `(row,node)` but each
thread scans the entire CSR owner word. Therefore

```text
forward threads        = sum_b R_row,b J_b
forward run-node work  = sum_b W_run,b J_b
reverse run-node work  = sum_b W_run,b J_b
ordered world work     = 2 sum_b W_run,b J_b.
```

The streamed sample/output slice is

```text
Theta(
  sum_r F_r J_r
  + sum_r N_fb,r J_r^2
  + P F
).
```

Thus the expensive ordered world forward/reverse remains invariant when only
requested sampling density grows over one fixed compiled interval. Total work
still has the unavoidable `Omega(PF)` target/output term. Repeating node
forward or reverse per `K` chunk would multiply world work by `ceil(F/K)` and
destroy the theorem; the intended lifecycle is forward once, accumulate all
sample cotangents, reverse once.

The dense structural report had counted `sum R_row,b J_b` as forward
"interactions." That was only a thread count and understated absolute forward
world work by the mean runs per row. Source telemetry now records both counts
and requires forward and material-reverse run-node interactions to match. No
runtime behavior changed and the correction remains unrun.

### Exact retained-state boundary

For the current material path, the principal active tensor bound is

```text
M_active
  = 16 U
  + sum_b (16 s_b + 32 R_row,b J_b)
  + 16 max_b s_b
  + 4 B
  + 16 bytes,
```

where `32 R_row,b J_b` is the float32 node transfer plus node cotangent.
Material-only reverse has no `[J,W]` cotangent. Staged full geometry adds a
bounded `[J,W]` length bar; fused v1/v2 remove that cotangent but retain the
compiled primal lengths. With one fenced sample launch, sample-axis state is
`O(KJ+K)`, not `O(FJ)`. Cheap replay identities still retain exactly
`24(F+V)` logical bytes, so only native ordered/reverse interaction state—not
the entire process—is currently frame-density invariant.

### What the connection can and cannot change

For a constrained horizontal flow, the exact repo-ordered identity remains

```text
d_t U
  = endpoint flux
  + integral prefix * F_tz^R * suffix dz.
```

This can only improve the executor through a lower certified temporal rank or
slower chart growth:

```text
direct U:       Theta(sum_b J_U,b       W_run,b)
group U_tilde:  Theta(sum_b J_Utilde,b  W_run,b) + flow/endpoints/reconstruction
curvature K_F:  Theta(sum_b J_F,b       W_run,b) + integration/reconstruction.
```

It creates no new asymptotic result in requested `F`. Two counterexamples are
load-bearing:

1. Under perfect flat advection, direct total `U` is already constant, so the
   advected slab proves the identity but cannot prove compression.
2. From `||U_tilde-U_tilde_hat|| <= T epsilon_F`, fixed global error over a
   longer duration may require `epsilon_F=O(1/T)`, which can increase `J_F`.

Sparse curvature also does not imply sparse training action: prefix/suffix
transports can depend on all active word parameters, leaving an `Omega(W_run)`
reverse lower bound without a separate compressed parameter-action theorem.

### Native source audit: actual immediate blockers

The new geometry does not close a measured native blocker. The highest-value
source work remains:

1. Replace dense one-shot CPU-only spatial/runtime construction with explicit
   caller-owned two-phase lifetimes and exact completion receipts.
2. Switch dense node forward from the return-allocating path to the existing
   caller-preallocated forward-into ABI.
3. Thread the exact sealed completion capability through construction,
   sample, reverse, optimizer, release, and quarantine; remove arbitrary
   callbacks and free provenance strings.
4. After native parity, expose existing fused-union v2 in the dense
   full-geometry coordinator so geometry scratch is union-local `O(U)` rather
   than global `O(S)`.
5. Shrink construction roots after a proven fence instead of retaining CPU
   payloads, transferred copies, and construction intermediates together.

No frame-linear native node, target, or reverse tensor was found. The current
expensive resident term is block/chart complexity, approximately
`sum_b(32 R_row,b J_b + 16 s_b)`, not requested frame count and not an
intrinsic 32-GB requirement.

### Literature sanity check

- `arXiv:math/0604428` is *Morse theory and tilting sheaves* and does not
  support the memo's local-triviality claim.
- The cited non-Abelian Stokes, factorization-algebra, exit-category, and
  braid-monodromy papers are legitimate background for their respective
  subjects. They do not establish a WorldFoam factorization cosheaf, program
  stack, compression theorem, or real-root braid runtime.
- The defensible project contribution remains the renderer-specific
  right-ordered affine specialization, constrained-flow flatness theorem, BV
  interface residual, translated-measure bridge, and falsifiable rank/work
  comparison.

### Revised decision

The math is fully retained and is probably the best conceptual formulation of
WorldFoam so far. The implementation decision is deliberately narrower:

```text
finish direct physical U and native lifetime correctness
  -> run fixed-program F=8/64/300 evidence
  -> compare U against group-completion U_tilde in a CPU float64 oracle
  -> compute signed K_F only if U_tilde first shows a compression signal
  -> build no connection shader unless total bytes/work improve >=2x and
     measured request time improves >=20% with all flow/endpoint VJPs charged.
```

Factorization/cosheaf, jet-stack, sensor-time patch, and optimizer-monodromy
ideas remain theorem language or offline diagnostics until their stated census
or kill tests succeed.

## Expansion Pass 3: Dense ownership repair after the math decision

Three independent read-only audits converged on a source-level safety repair
that does not require new shader math:

1. The dense request still used the legacy return-allocating node-forward ABI,
   even though the native caller-preallocated `forward_into` op and its bounded
   lifetime carrier already existed and were used by the sparse route.
2. Dense lane creation still called one-shot spatial/runtime constructors.
   Those constructors deliberately reject non-CPU use without caller-retained
   two-phase construction lifetimes, so the dense accelerator path could lose
   Python roots if a transfer enqueued work and then raised.
3. The sealed completion capability is internally coherent but isolated. It
   should first replace callbacks end-to-end in the CPU-only lazy material path;
   partially threading it through dense would leave construction, commit,
   geometry, optimizer, abort, and release seams falsely certified.

The dense coordinator source now:

- installs a sealed aggregate construction lifetime before materialization;
- retains the union-local construction lifetime and one runtime-construction
  lifetime per native block through the lane release boundary;
- charges the resulting accelerator overlap honestly: fresh CPU union/map
  predecessors plus their device destinations coexist until that boundary,
  and every runtime also retains one CPU epsilon scalar;
- quarantines current predecessors, partial destinations, partial bundle,
  runtimes, and executor if construction cleanup fencing fails;
- gathers compact materials under a caller-visible predecessor lifetime;
- allocates the node chart in the caller and uses
  `launch_node_forward_into` exclusively;
- retains gather, compact material, output, world, and token roots through the
  corresponding reverse/fused/abort fence;
- records install/retire counts and exact caller-owned node-output bytes; and
- includes a forward-into enqueue-then-raise plus failed-abort-fence quarantine
  fixture.

This is a source ownership improvement, not accelerator evidence. The dense
path still accepts arbitrary completion callbacks and free-form provenance,
all new tests are unrun, and allocator/native promotion remains closed. The
math decision is unchanged: finish direct physical `U`, then falsify the
`U`/`U_tilde`/`K_F` compression hypothesis before adding a connection shader.

## Expansion Pass 4: Exact end-state audit and sealed-release dependency

The user-level objective is broader than a bounded material fixture. It is not
complete until all of the following claims have direct evidence; a green source
test in one row cannot stand in for the matrix.

| Required claim | Exact evidence | Current status |
| --- | --- | --- |
| Depth/order correctness | Exact replay versus compiled forward and selected material/geometry/camera VJPs across stable words and supported simple events | CPU mathematics exists; latest integration is unrun and general event derivatives remain fail-closed. |
| Frame-density-independent expensive state | One fixed world/physical interval with `F=8/64/300`; identical compiled generations and constant node/word state bytes | Source shape and schema exist; fresh-process native artifacts are missing. |
| Frame-density-independent expensive world work | Per block, node forward and ordered-word VJP counts remain one while only `O(FJ)` sampling/output work grows | CPU/fake-native lifecycle proved earlier; rebuilt native telemetry is missing. |
| Material-training memory lightness | Fresh processes stay under the declared `<=2 GiB` MPS allocator limit and sampled process-group RSS stays below `4 GiB`, with decoded targets and saved tensors attested | Producer/verifier source exists; no accepted rows. |
| Full-geometry memory lightness | Fused union-local geometry reverse avoids a global geometry bar and passes the same `F` sweep, parity, poison-fence, and allocator gates | Fused-v2 source exists but is neither built nor integrated into the dense production request. |
| Safe asynchronous ownership | Every construction/launch/reverse/commit/release epoch is tied to the exact native ops, device, owner generation, launch generation, and a one-shot completion receipt; unknown completion quarantines all roots | Caller-owned lifetimes exist; arbitrary callbacks still authorize release. |
| Real trainability | Two uninterrupted steps and checkpoint/restart parity reduce loss through the sealed coordinator/updater without stale structure | Source lifecycle exists; runtime gate is missing. |
| Paper relevance | Frozen-world replay-versus-compiled scaling plus heldout quality/cost rows | Not yet produced for native-4D WorldFoam. |

The intended scaling theorem remains deliberately split:

```text
expensive world state/work at fixed compiled interval:
    M_world = O(sum_b (R_b J_b + J_b W_b + S_b))
    W_world = 2 sum_b J_b W_b

unavoidable requested-sample slice:
    M_sample = O(K J + K)
    W_sample = O(F J + P F)
```

This is sublinear in requested frame density only for the expensive world-side
rasterization/reverse. Total rendering cannot be sublinear in the number of
requested output pixels because the outputs themselves are `Omega(PF)`.

### Why the sealed completion capability is now P0

An arbitrary `Callable[[], None]` plus a provenance string proves nothing. It
can be a no-op, synchronize the wrong device, or return while a producer stream
still owns the tensors. Shape accounting is therefore unsound until release is
authorized by a capability with these relational checks:

```text
receipt.capability_generation == capability.generation
receipt.owner_generation      == capability.owner_generation
receipt.native_ops_identity    == id(capability.native_ops)
receipt.native_abi_digest      == capability.native_abi_digest
receipt.device/scope           == capability.device/scope
receipt.stage                  == expected release stage
receipt.launch_generation      == exact preceding launch epoch
receipt.sequence               == next unconsumed sequence
```

One capability may be sequentially reused without retaining history, but at
most one successful receipt may be outstanding. Consuming it advances one
monotone counter. A failed fence poisons the capability; a missing, repeated,
foreign, wrong-stage, or wrong-launch receipt releases nothing. CPU is only a
call-return source contract. MPS/CUDA minting must remain closed until the
canonical native module and actual dispatch domain are attested.

The safe integration order is lazy material-only CPU/fake-native first, then
native ABI attestation, then dense material, then staged/fused geometry. A
partial dense conversion must not be called sealed because its construction,
delta commit, post-accept, abort, optimizer, and lane-release epochs form one
larger ownership transaction.
