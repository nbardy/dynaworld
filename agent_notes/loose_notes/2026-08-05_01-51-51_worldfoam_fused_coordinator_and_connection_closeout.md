# WorldFoam fused coordinator and connection closeout

Date: 2026-08-05 KST

## Scope and safety

This work chunk audited the latest external scientist proposal and continued
the source-only memory-light WorldFoam integration. The source was:

```text
/Users/nicholasbardy/.codex/attachments/
  2492c6e4-bcde-416a-80b0-711c5a6101da/pasted-text.txt
SHA-256:
  965c7a1a28343914dd348a88afa1b30a976dabd6dbf80fb48a1076ad878334c5
```

The machine remained unsafe after prior memory/resource incidents. No Python
process, import, pytest, native build, Metal/MPS/CUDA launch, or training run
was started. Every implementation statement below is source-written or
statically inspected unless explicitly labelled as older evidence.

## Scientist proposal: what survived the audit

The strongest part is a constrained Lagrangian optical connection on the
ray-depth fiber. It is the best new mathematical hypothesis in this intake,
but it is not yet a compression theorem.

WorldFoam's executable affine transfer is

$$
T(\beta,m)=
\begin{bmatrix}\beta I_3&m\\0&1\end{bmatrix},
\qquad
T(\beta_1,m_1)T(\beta_2,m_2)
=T(\beta_1\beta_2,m_1+\beta_1m_2).
$$

The repository scans near to far with the right-ordered convention

$$
U(b,a)=U(s,a)U(b,s),
$$

so

$$
\partial_bU=UA_z(b),
\qquad
\partial_aU=-A_z(a)U.
$$

The physical coordinate-depth generator must include ray speed:

$$
A_z=X(-\lambda,\eta),
\qquad
\lambda=\|\partial_z\Gamma\|\rho,
\qquad
\eta=\lambda c.
$$

For a horizontal generator $A_t$, the repo-native curvature is

$$
F^R_{tz}=\partial_tA_z-\partial_zA_t+[A_t,A_z].
$$

For fixed endpoints, the exact variation identity is

$$
\partial_tU-UA_t(b)+A_t(a)U
=\int_a^b U(s,a)F^R_{tz}(s)U(b,s)\,ds.
$$

For moving endpoints, with

$$
B_a=A_t(a)+\dot a A_z(a),
\qquad
B_b=A_t(b)+\dot b A_z(b),
$$

the identity is

$$
\frac{dU}{dt}
=UB_b-B_aU
+\int_a^b U(s,a)F^R_{tz}(s)U(b,s)\,ds.
$$

The constrained Lagrangian choice

$$
A_t=-wA_z
$$

gives

$$
F^R_{tz}=\partial_tA_z+\partial_z(wA_z).
$$

For $A_z=X(-\lambda,\eta)$, flatness is exactly the pair of continuity
equations

$$
\partial_t\lambda+\partial_z(w\lambda)=0,
\qquad
\partial_t\eta+\partial_z(w\eta)=0.
$$

When an independently specified Lipschitz $w$ generates an
orientation-preserving flow $\phi_t$, zero curvature is equivalent to
invariance of the pulled-back infinitesimal generator and therefore exact
reuse of every transported depth subinterval:

$$
F^R_{tz}=0
\iff
A_z(t,\phi_t(s))\partial_s\phi_t(s)=A_z(t_0,s),
$$

$$
U_t(\phi_t(s_1),\phi_t(s_0))=U_{t_0}(s_1,s_0).
$$

This is the theorem worth keeping. It says curvature measures the residual to
coherent transported-subinterval reuse.

## Corrections and non-results

- The raw note used the opposite multiplication convention. Its transported
  sandwiches, endpoint factors, gauge correction, and holonomy orientation
  cannot be copied into WorldFoam.
- Equality of one total ray transfer does not imply flatness. Nonzero
  curvature can cancel in depth. Flatness characterizes reuse of every
  transported subinterval, not one accumulated color.
- A free per-ray flow can hide the answer. The flow must come from an
  independent compact scene/camera model, and its parameters, fitting,
  storage, reconstruction, and gradients must be charged.
- A scalar depth velocity cannot represent generic transverse 3D motion. The
  full lift generally needs

  $$
  H=\partial_t+v_u\partial_u+v_v\partial_v+w\partial_z.
  $$

- For a moving P0 interface $z=r(t)$, the singular term is

  $$
  \bigl([wA_z]-\dot r[A_z]\bigr)\delta(z-r).
  $$

  It becomes $(w-\dot r)[A_z]\delta(z-r)$ only when $w$ has a continuous
  trace.
- Curvature does not remove the depth-order event compiler. Order swaps remain
  discriminant events.
- Fiber bundles, interval factorization, jet seams, and real-root continuation
  are useful organization. The proposed cosheaf/stack/monodromy machinery does
  not supply a runtime or memory bound.
- Open rays use ordered parallel transport. Holonomy should remain reserved
  for closed loops.
- Do not conflate the new **ray-fiber optical connection** with the paper's
  existing **cell-frame adjacency connection**. They have different bases,
  fibers, groups, curvatures, and tests.

The complete derivation, proofs, counterexamples, and oracle design are in
`research_notes/worldfoam_paper/WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`.
The theorem ledger now records C1--C5, D2a, and R1.

## Decisive experiment before a curvature runtime

Compare, under one identical continuous primal and tangent certificate:

```text
direct transfer U
vs endpoint/flow-corrected transfer U_tilde
vs transported curvature source K_F.
```

Charge flow, endpoints, reconstruction, gradients, physical-cone checks, and
conditioning. Do not build a runtime unless total payload and word work improve
by at least 2x against both direct representations and measured request time
improves by at least 20%. A failed gate leaves curvature as a theorem and
diagnostic, not as an implementation branch.

## Subagent findings

Three independent static audits converged:

1. The math audit confirmed the full scientist note is accounted for, found
   the convention and theorem corrections above, and required separating the
   two connection concepts.
2. The code map found that ordered affine transfer, event stratification,
   compact active sets, exact/frozen-program reverse, and a fused source route
   already exist. A curvature oracle does not exist.
3. The fused-path static audit found no current field/key/fake mismatch but did
   find a completion-callback lifecycle/snapshot hardening gap.

## Source integration completed in this chunk

The fixed-camera coordinator can now select the fused-direct-v1 full-geometry
reverse explicitly. Staged sparse remains the default. The selected mode is
bound into coordinator accounting, the step-result digest, and the combined
CPU updater's receipt. Source tests for the fused coordinator and combined
receipt path are written but unrun.

The fused adapter and material-step executor now statically bind callback-
visible transaction/session identities, tensor versions and storage bindings,
authoritative manifests, lifecycle counters, and failure roots. Reentrant
execute/abort is rejected. Mutation detected after the completion fence
quarantines and restores transaction scratch roots instead of accepting
callback-authored state. This is structural/lifecycle hardening, not a claim
of sandboxing arbitrary malicious raw-memory writes.

## Memory accounting and remaining risk

At fixed program complexity, the important formulas are frame-density
independent:

```text
degree-2 steady world + material                         = 120 S bytes
full step accumulator                                  = 88 S + 4 bytes
steady + authorized bars                               = 208 S + 4 bytes
staged full-geometry request addition                  = max_b 4 J_b W_b
fused prepared transaction
  = sum_b 4 [R_b(J_b+14) + S_b(6+C) + 13]
fused output scratch
  = 16 sum_b S_b + 4 S(6+C)
CPU float64 bridge
  = 12 S(6+C)
```

Here `S` is global sites, `S_b` active sites in a block, `J_b` chart nodes,
`W_b` word length, `R_b` rows/tracks, and `C` material coefficients.

Removing the staged `[J,W_b]` length cotangent is not itself proof of lower
whole-request peak. Fused execution can overlap all prepared blocks, compact
outputs, global float32 bars, the CPU float64 bridge, and the step accumulator.
Allocator telemetry must decide. If prepared blocks dominate, the next design
is either union-local geometry output with one global scatter or a block-
streamed fail-atomic transaction.

The cheap camera/sample slice remains `O(VF)` metadata/work and is not a full
video tape in the fixed-camera source path. The expensive world reverse is
bounded by compiled chart/word/active-site complexity rather than requested
frame count. That is the intended World-Tubes-like scaling claim; it still
needs native measurement.

## Exact continuation order

1. On a quiet host, run the focused CPU/fake-native transaction, coordinator,
   and combined-state tests.
2. Rebuild the native fused-v1 extension and attest the exact ABI.
3. Prove staged/fused float64 forward and geometry/material gradient parity,
   then run poison/fence failure cases.
4. Measure whole-request allocator peaks, not only logical tensor bytes.
5. Add/verify production trainer and evaluator routing plus checkpoint restore.
6. Run fixed-dataset `F=8/64/300` scaling, then public quality rows.
7. Only after the memory-light path is measured, build the tiny CPU
   `U`/`U_tilde`/`K_F` curvature oracle.

No paper/runtime claim should be promoted from this source-only chunk.
