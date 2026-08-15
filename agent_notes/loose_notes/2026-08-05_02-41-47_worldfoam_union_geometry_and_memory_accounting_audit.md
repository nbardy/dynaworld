# WorldFoam union geometry and memory-accounting audit

Date: 2026-08-05 02:41 KST

## Scope and host constraint

Continued the persistent memory-light WorldFoam goal after the external
scientist connection/fiber proposal was fully normalized in
`research_notes/worldfoam_paper/WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`.
The source attachment remains
`/Users/nicholasbardy/.codex/attachments/2492c6e4-bcde-416a-80b0-711c5a6101da/pasted-text.txt`
with SHA-256
`965c7a1a28343914dd348a88afa1b30a976dabd6dbf80fb48a1076ad878334c5`.

The host had already suffered repeated memory/resource incidents. This chunk
ran no Python, imports, pytest, builds, Metal, MPS, CUDA, or training. Work was
limited to shell reads, static diff checks, source edits, derivation, and
parallel source-only subagent audits.

## Main mathematical decision

The scientist's constrained Lagrangian optical connection remains a serious
future transport/certification hypothesis, but it is not required for the core
sublinear world-side backward. The existing fixed-certified-atlas factorization
already gives:

```text
streamed sample/target work: Omega(PF), cheap and bounded in K;
compiled ordered-word/world reverse: Theta(sum J_c R_c), invariant in requested F;
reverse state: no F x R or P x F tape.
```

WorldFoam must keep depth order alive. No Gaussian-style Schur marginalization
is needed or valid for this systems theorem.

The first new memory formulation is instead an exact output-index-space
factorization. For block compact-to-global scatter `P_b`, compact-to-request-
union scatter `Q_b`, and union-to-global scatter `P_U`, the existing cold union
certificate proves

```text
P_b = P_U Q_b,
sum_b P_b g_b = P_U (sum_b Q_b g_b).
```

Thus fused geometry can be accumulated in `[U,6+C]` instead of `[S,6+C]`
without changing the real-arithmetic cotangent. The exact float32-source plus
CPU-float64-bridge saving is

```text
12(S-U)(6+C),
```

or `108(S-U)` at `C=3`. If the CPU commit needs a newly allocated int64
union-to-global map, charge `8U`, so the net counted improvement is

```text
12(S-U)(6+C) - 8U.
```

This means `U/S` must be measured; the complete request is not universally
smaller when the union is nearly global. The detailed proof, three-index-space
ABI, fail-atomic transaction, source map, and falsification gates are now in
`research_notes/worldfoam_paper/WORLD_FOAM_UNION_LOCAL_FUSED_GEOMETRY_V2_DESIGN_2026-08-05.md`.

## Staged reverse accounting bug and correction

A static audit found that the dense staged admission added only the native
`4J_bW_b` cotangent to active state. The sparse reducer's complete logical
preflight was capped separately, so simultaneously live tensors were not
composed.

For fixed camera block `b`, with maximum row word count `rho_b`, compact site
count `s_b`, and weight width `C`, the corrected deterministic tensor upper
bound is

```text
H_b
  = 4J_bW_b
    + (56+8C)s_b
    + 16J_b rho_b
    + 8J_b
    + 8(37+2C)rho_b
    + 608
    + V_b,

V_b = 1 + max(J_b rho_b, 3s_b, Cs_b).
```

`V_b` is the explicit bool finite-mask plus scalar validation scratch. For
trainable rays, add `96(T_b+1)` to `H_b` and candidate `12T_b` inside `V_b`.

Source changes:

- exposed
  `preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory` as a
  tensor-allocation-free deterministic upper-bound query;
- made the reducer and dense request reuse that one formula;
- added explicit validation scratch to the memory receipt;
- staged admission now checks the bridge before lane construction and composes
  `max H_b` with active request/step state without double-counting `[J,W]`;
- lane admission is also preflighted before construction;
- accounting reports `lane + active`, the corresponding policy-cap sum, and
  explicitly denies allocator-peak status;
- the fixed-camera outer coordinator propagates and validates the new staged
  upper bound.

The tighter staged phase is

```text
L + A + G + X0 + max_b(16s_b + H_b),
```

while source admission conservatively uses
`L + A + G + X0 + 16 max s_b + max H_b`. Target/sample phases remain separate;
this is not yet a measured whole-request/process peak.

Tests were written, not run, for an independent closed-form fixed/trainable
preflight, additive active-cap failure before native lane build, propagation,
and reverse lane-plus-active arithmetic.

## Restart parser memory hardening

The new combined checkpoint parser originally scanned all geometry before byte
admission, trusted serialized state/payload byte totals until after five clones,
and bounded only persistent manifest interval metadata, not the request-local
track tuple.

It now source-checks, before any full tensor scan or clone:

```text
geometry_checkpoint = 8S(6+C)
checkpoint           = 16S + geometry_checkpoint
live_state           = 48S + geometry_checkpoint
state+checkpoint     = live_state + checkpoint
payload peak         = live_state + 2 checkpoint.
```

It requires serialized totals to equal these derived values, enforces current
policy caps, parses the bounded manifest first, caps both persistent interval
metadata and the maximum request-local int64 track tuple, and rejects oversized
source backing storage under both checkpoint and source-plus-owned-clone
coexistence bounds. Schema/provenance/top-level digests and tensor metadata now
fail before scans. Full finite/content/generation validation remains the final
integrity gate. Tests for lying accounting, oversized backing storage, and an
oversized request range were added but not run.

Nested material parameterization and optimizer policy are still serialized
authority. Live restore must take their expected current-config authority or
exact digests rather than trusting the checkpoint.

## Trainer critical path clarified

The combined lifecycle is CPU-only; simply allowing MPS tensors would trigger
implicit synchronization through existing `.item()` invariants and widen the
proof surface. The smaller first production bridge is:

1. keep raw material, optimizer state, and geometry on CPU;
2. seal and fence one CPU-state/version-bound MPS `[S,4]` material snapshot;
3. run the native fixed-camera step;
4. seal and fence one exact-result-bound CPU `[S,4]` material-gradient clone;
5. keep the combined SGD/recompile/checkpoint lifecycle synchronous on CPU.

This adds only `O(S)` transfers and no frame axis. Untracked `.to('mps')`
calls are insufficient because current APIs accept a material tensor and a
separate generation id without proving they match.

The first honest trainer target is the existing fixed-512 pixel-matched
control: fixed `384x512`, fixed `1024` sites, fixed optimizer policy. The
600-step progressive row changes resolution, site count, and LR policy; it
requires a separate stage-transition transaction and does not block the first
trainer/evaluator.

The exact remaining order is:

1. close the sample prepared-payload async lifetime/quarantine gap;
2. rebuild the stale extension and add a separate five-kernel selected fused
   full-geometry attestation;
3. run bounded staged-versus-fused native parity;
4. implement union-local fused v2 and its parity/quarantine/accounting gates;
5. add sealed CPU<->MPS material bridges;
6. implement live semantic combined restore and uninterrupted-vs-restart
   two-step parity;
7. build the fixed-512 trainer and streaming evaluator;
8. run separate full-geometry `F=8/64/300` fresh-process memory evidence;
9. register `worldfoam_native4d` as a distinct lane, preserving existing
   PowerFoam `worldfoam` evidence;
10. only then run the connection/curvature `U` vs `U_tilde` vs `K_F` kill gate.

## Remaining source correctness gap

The native sample prepare result is currently a local variable inside
`KineticNativeMaterialStepSession.launch_sample_accumulate` and is not
explicitly rooted through the caller's completion fence. Failure quarantine
also omits current target/sample and some popped compact scratch roots. The
safe repair is a one-launch sealed lifetime token settled immediately after
the fence. Retaining all tokens until session end would reintroduce linear
sample memory and is forbidden.

## Verification performed

- parallel source-only audits: exact staged/fused memory formulas, checkpoint
  parsing, production trainer seam, and union-v2 ABI;
- an independent proof re-audit confirmed the repo-native curvature,
  moving-endpoint, BV-interface, and holonomy signs. It tightened the coherent-
  motion theorem to a normalized `C^1` flow (`phi_t0=id`), stated the merely
  Lipschitz form almost everywhere/as a pulled-back measure, declared the seam
  order `q>=1` on common one-sided Taylor germs, and separated discontinuous-
  flow BV diagnostics from the admissible material-flow theorem;
- independent intake and systems audits added an explicit equal-certificate
  rank definition, primal/tangent seam-defect census, parked event-rewrite
  coherence equations, two non-vacuous numeric oracle sentinels, and the
  implementation order `U -> direct U_tilde -> K_F`. They also made explicit
  that transported curvature still needs ordered prefix/suffix work per node
  and cannot improve the existing requested-frame scaling theorem by itself;
- conflict-marker/trailing-whitespace scans on touched files;
- static line-by-line source inspection.

No runtime verification was performed. Every new source/test claim remains
unrun until a quiet approved host window.
