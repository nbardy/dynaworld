# WorldFoam CPU-lazy sealed completion integration

## Scope and safety

This pass continued the memory-light native-4D WorldFoam lane without running
any Python, import, pytest, build, Metal, MPS, CUDA, or training workload.  The
host has already suffered resource incidents, so work was restricted to shell
reads, static scans, and source edits.  Accelerator capability minting remains
hard closed.

The mathematical result did not change.  Direct ordered transfer `U` remains
the production representation, with expensive frozen-program material work

```text
W_world = 2 * sum_b J_b * W_run,b
```

independent of requested frame density at fixed compiled program complexity.
The constrained Lagrangian connection, flow-corrected `U_tilde`, and curvature
source `K_F` remain a later oracle/ablation after the direct runtime gates.

## Problem

The CPU-lazy coordinator previously accepted a caller-supplied completion
callback and free provenance text.  Object identity and a hash of that text do
not prove that a callback synchronized the producing backend, device, or
dispatch domain.  Releasing transfer, forward, sample, reverse, bundle, or
top-level roots under such a convention would make allocator and memory claims
unsound.

The safe lifecycle is:

```text
mint exact capability internally
  -> register exact epoch before work
  -> launch/materialize
  -> fence the same epoch
  -> revalidate receipt and retained subjects
  -> consume the one-shot receipt
  -> commit root release
```

A failed fence poisons the capability, retains one bounded quarantine, and is
never retried.  Receipt/capability ledgers are constant-sized rather than a
history over samples or frames.

## Source outcome

The public CPU-lazy material step no longer accepts a completion callback,
completion provenance, or backend provenance.  It derives a canonical binding
from the exact native ABI and device, mints one CPU-only sealed capability, and
registers phases before the work they cover:

- top-level output allocation and gradient zero;
- bundle materialization, including generator `next()` transfers;
- lane/runtime construction;
- sparse transfer, gather, forward, and each bounded sample launch;
- each active-block reverse;
- error settlement and cleanup zeroing.

Capability and receipt objects now bind their mint-time Python identity in
their digests and lifecycle ledgers.  `dataclasses.replace(...)` copies cannot
act as authority.  The capability additionally remembers the exact outstanding
receipt identity, while registered launch epochs were already identity-bound.

The bundle stream now performs a separately accounted, registered, fenced
terminal probe.  A premature end fails.  A bundle that overshoots declared
coverage or an undeclared trailing bundle is fenced, consumed, retired, and
then rejected before optimizer authorization.

The bundle and sample wrappers explicitly delete their yielded local before
materializing the next payload.  Their outer loops also delete the prior
payload before calling `next()` again.  This fixes a subtle two-payload peak:
ordinary generator rebinding evaluates the next materialization before
decrementing the previous local reference.

Release paths were split into prevalidation and post-consumption commit tails.
The successful lane reuses the final reverse completion boundary and does not
claim an extra fence call.  Accounting separately charges top initialization,
bundle materialization, the terminal exhaustion probe, lane construction,
sample settlement, and reverse settlement.

## Swarm findings that changed the patch

Static adversarial review found and corrected:

1. capability and receipt cloning through `dataclasses.replace`;
2. release-before-consume paths where a foreign-thread consumption failure
   could occur after roots were already cleared;
3. a fence-count equation that initially double-counted the reused lane-release
   boundary;
4. accelerator transfers hidden inside bundle/sample `next()` before epoch
   registration;
5. silent valid-prefix acceptance with undeclared trailing observations; and
6. generator-local retention that briefly overlapped two bundles/sample
   payloads and contradicted the one-lane/one-sample memory claim.

The exact sample iterator needs no terminal probe for current CPU correctness:
the sealed plan partitions every record once, dispatches each to exactly one
block, and partitions each block entry into exact chunks.  A future extra probe
would still be useful defensive hardening.

## Deliberate boundary

This is a CPU-lazy source contract, not accelerator or dense-path promotion.
Legacy low-level and dense geometry APIs still accept callback/provenance
authority and must not be described as sealed.  Before MPS/CUDA minting opens:

1. bind each releasable lifetime to its installing capability/owner and exact
   pre-launch epoch;
2. make authority-free convenience releases CPU-only;
3. settle sparse transfer plus executor roots in one post-fence composite
   revalidation, one receipt consumption, and two commits, because accelerator
   synchronization can release the GIL;
4. place any accelerator optimizer/update work under its own epoch or keep the
   optimizer and persistent state on synchronous CPU; and
5. bind canonical native build/ABI/device/launch-domain attestation and run
   failure, allocator, and release evidence on an approved quiet host.

## Files

Primary source:

- `research_experiments/world_foam_lane2/kinetic_sealed_completion_fence.py`
- `research_experiments/world_foam_lane2/kinetic_lazy_native_material_step.py`
- `research_experiments/world_foam_lane2/kinetic_native_material_step_executor.py`
- `src/train/paper_kinetic_union_local_bar_assembly.py`
- `src/train/paper_kinetic_sparse_sample_blocks.py`
- `src/train/paper_kinetic_lazy_program_bundles.py`

Focused source-written tests:

- `research_experiments/world_foam_lane2/test_kinetic_sealed_completion_fence.py`
- `research_experiments/world_foam_lane2/test_kinetic_lazy_native_material_step.py`

Ledger:

- `TODO/worldfoam_memory_light_native4d.md`

## Verification status

Static conflict-marker and whitespace checks are clean.  Focused behavior tests
are written but unrun.  No runtime, allocator, native ABI, loss-decrease, or
`F=8/64/300` evidence was produced, so no baseline or paper row is promoted.

The final frozen-source red-team found no CPU-path P0 under the declared
synchronous call-return contract.  It retained four accelerator blockers listed
above.  It also classified several post-consumption list clears, counter
increments, active-subject rereads, and cleanup loops as P1 failure-atomicity
hardening: if they throw after known completion, roots may leak into quarantine,
but they cannot be released before completion.  A unique terminal sentinel and
one-more-sample probe remain P2 defensive hardening.
