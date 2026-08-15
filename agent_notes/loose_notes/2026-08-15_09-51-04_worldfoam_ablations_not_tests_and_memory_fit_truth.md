# WorldFoam ablations, not tests, and memory-fit truth

## User correction

The user correctly rejected framing source/unit checks as the goal.  The paper
deliverables are the measured ablations.  Tests and dry plans are only
fail-closed preflight and must never be counted as experimental rows.

## Exact evidence state

- G4-v2 public heldout quality: `0/36` measured rows.
- G6 native memory/work: `0/21` measured rows, plus `0/3` auxiliary restart
  processes.
- The canonical G4 and G6 artifacts are absent.
- Paper B therefore correctly retains `public_quality=false` and
  `native_memory_fit=false` with explicit NOT MEASURED placeholders.
- This means the new shared-adjoint WorldFoam memory fit is unmeasured, not a
  measured failure.

An older July 22 PowerFoam-Metal smoke did fit locally at four frames and two
optimizer steps, with about 34.3 MB current MPS allocation and 1.145 GB driver
allocation.  That used the old raytrace route, small primitive counts, and low
resolution.  It is not evidence for the new `S=1024`, `F<=300` shared-adjoint
implementation.

## G4-v2 source completion

The tractable selected-ray matrix is now source-complete:

- three Neural3D scenes;
- seeds `17/29/43`;
- WorldFoam compiled, WorldFoam framewise replay, World Tubes, and dynamic
  3DGS;
- exactly 36 fresh-process rows;
- 300 steps, four spacetime samples/step, 1024 common selected pixels/sample;
- `1,228,800` target pixels and `3,686,400` RGB-MSE scalars per row;
- complete 300-frame `384x512` heldout evaluation;
- route-specific raster work reported instead of claimed equal;
- 2-GiB hard MPS ceiling and 4-GiB child-inclusive process-group RSS ceiling;
- execute-only 8-GiB free-disk/available-RAM, <=2-GiB swap, load<=8 host gate,
  rechecked before every row.

The real-native bounded pilot is also source-complete.  It runs Coffee Martini
seed 17 for both WorldFoam routes, one optimizer step, 4096 targets/route, all
300 heldout times over 128 spatial tracks, and one complete-track bitwise
frame-major/cross-time parity check.  It remains explicitly
`pilot_only=true`, `public_quality_evidence=false`.

The final G4-v2 source capability SHA is
`a137a881e4aec568647a505bf2e1e20f428132f39a892c2f5ab4143e1cf18082`.
The dry matrix contains all 36 rows but aborts before row 1 until the pilot and
stale native extension blockers close.

## G6 source completion

G6 is source-complete but runtime-unmeasured:

- Metal full-geometry direct and union VJPs;
- 133/133 source schema/implementation contract;
- primary shared-adjoint, same-representation sequential replay, and restart
  transactions;
- 12 primary rows + 9 controls + 3 restart processes;
- independent source/native/hardware/parity/lifecycle/memory verifier;
- hard 2-GiB MPS and 4-GiB process-group RSS limits.

The logical state is small at `S=1024`: 114,688 B live, 81,920 B checkpoint,
and 278,528 B conservative live-plus-checkpoint-clone.  These numbers do not
measure native scratch, compiler objects, allocator behavior, Python state, or
RSS.  Only the fresh-process G6 matrix can promote the memory claim.

The retained native extension is stale: it exposes 103 schemas and predates
the 133-schema source.  The clean-host bundle force-rebuilds and attests it
before any evidence row.  Most new G6 source is also currently untracked; the
exact source set must be committed/frozen before a publication run even though
the artifact already hash-binds repository-relative bytes.

## Paper-asset integrity repairs

The Paper-B asset layer was corrected before accepting future measurements:

1. G6 no longer compares sequential inner replay time against fused full
   transaction time.  It reports common route-core timing and identical
   parent-watchdog process end-to-end timing separately.  Route-local
   transaction and compile timing stay separately labelled in JSON/CSV.
2. The bundle verifier now reopens every retained input, reruns current
   independent validators/extractors, rebuilds every generator-owned
   JSON/CSV/Markdown/TeX/SVG in memory, and requires byte identity.  Re-hashing
   an edited result and rebinding the outer manifest no longer passes.
3. The joint synthetic G0/G3 claim now requires both accepted constant ordered
   transfer and accepted synthetic visibility evidence.
4. G4 captions distinguish quality mean+-standard deviation from mean-only cost
   cells; cost dispersion remains in CSV/JSON.
5. G4/G6 SVGs have title and description metadata.

The honest incomplete bundle was regenerated and independently verified.  The
WorldFoam ICLR package binding was updated.  Focused aggregate preflight is
`40 passed`; this is machinery verification only and contributes zero paper
ablation rows.

## Host incident state

At the last audit this Mac had only about 4.1 GiB free disk and 18.3/19.0 GiB
swap in use, with severe compressor history.  No native build, MPS pilot,
cache conversion, or training row was launched.  The 8-GiB launch headroom is
an incident guard, not a claim that WorldFoam needs 8 or 32 GiB.

## Next execution sequence

1. Commit/freeze the exact G4/G6 source revision.
2. Move to a quiet eligible Apple-silicon Mac.
3. Rebuild and attest the 133-schema native extension.
4. Produce the Coffee Martini cache capability and native runtime
   capability/evidence.
5. Run and verify the two-route G4-v2 pilot.
6. If the pilot stays within caps, run and verify all 36 G4 rows.
7. Run and verify the G6 21-row matrix plus three restart processes.
8. Regenerate Paper B tables/figures/package strictly from accepted artifacts.

Do not call source completeness, tests, dry plans, analytic bytes, or the old
PowerFoam smoke a completed WorldFoam memory ablation.
