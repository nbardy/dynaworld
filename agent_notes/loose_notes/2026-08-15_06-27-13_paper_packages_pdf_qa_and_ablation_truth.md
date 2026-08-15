# Paper packages, PDF QA, and ablation truth

Date: 2026-08-15 KST

## Context

The user correctly rejected a status update that foregrounded passing tests.
The project goal is measured paper ablations. Tests and strict verifiers are
necessary guards, but they are not experiment rows and must never be counted as
evidence.

This note freezes the evidence boundary after the two concise manuscripts,
generic-PDF QA, and strict venue-package gates were brought into alignment.

## Evidence vocabulary

Observed fact:

- A **test** checks a behavior contract in code.
- A **dry plan** checks orchestration and emits zero evidence.
- A **logical byte formula** describes selected tensors under stated
  assumptions.
- An **ablation row** is a measured execution of a frozen experimental
  condition.
- An **accepted paper row** is a measured row that also passes its independent
  artifact, source, protocol, and provenance checks.

Decision:

No test count, source verifier, dry-run report, fake-native row, smoke result,
or logical byte formula may populate an ablation table.

## WorldFoam evidence now accepted

### G0/G3 synthetic ordered-transfer ablation

Artifact:

```text
outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/summary.json
```

This is a real deterministic float64 CPU experiment, not a unit-test proxy. It
contains:

- 224 depth-layer rows;
- 168 comparator rows;
- 56 adaptive rows;
- 448 measured rows total;
- eight named scenes crossed with seven camera programs.

Accepted findings:

- 128-layer fifth-percentile context PSNR: `37.9252 dB`;
- crossing-family mean-MSE improvement over representative-depth sorting:
  `82.2477x`;
- crossing-family mean-MSE improvement over depth marginalization:
  `528.953x`;
- physical-Jacobian gauge error: `3.32998e-7`;
- error when that Jacobian is omitted: `0.305335`, or about `916,927x`
  larger.

Claim boundary:

This supports retained-depth ordered-transfer correctness, convergence,
crossing behavior, and the physical measure factor. It does not support native
runtime, native peak memory, public-data quality, or full kinetic-compiler
acceptance.

### M3/M5 material-family ablation

Artifact:

```text
artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json
```

This contains 36 measured held-out rows over seeds `17/29/43`. M3 exactly fits
the positive-P2 generating family (`5.25789e-17` median held-out loss), while
M5 exactly fits the convex-log-P2 family (`6.18840e-15`). Both serialize six
scalars / 24 bytes. There is no universal winner, so the result does not
authorize a native six-way renderer fork.

## WorldFoam G6 memory truth

Current measured result:

```text
0 / 21 required evidence rows
```

The frozen protocol is implemented for:

- `S=1024` sites;
- `P=512` selected tracks;
- `384x512` images;
- `F=8/64/300` requested frames;
- staged sparse versus fused union-v2 at `F=8`;
- fused union-v2 at `F=8/64/300`;
- same-representation sequential replay at `F=8/64/300`;
- three repeats per mode/frame condition;
- three additional checkpoint/restart processes.

That is 12 primary rows plus nine control rows, or 21 evidence rows, executed
through 24 sequential fresh processes after adding the three lifecycle jobs.

The dry plan is correctly fail-closed. It emits zero evidence and currently
reports:

```text
native_extension_older_than_bound_native_sources
```

The native source declares 133 schemas. The stale binary registers 103 and is
missing 30. Therefore a runtime result cannot be obtained until the extension
is rebuilt and attested.

### Logical accounting, not a measurement

At `S=1024`:

```text
material live state                         48 B/site
trainable geometry                         64 B/site
combined live state                       112 B/site = 114,688 B
combined checkpoint                        80 B/site =  81,920 B
live state plus checkpoint                192 B/site = 196,608 B
payload-clone peak                        272 B/site = 278,528 B
```

The intended asymptotic split is:

```text
expensive world-side state and reverse     bounded in requested frame count F
selected targets/rays and small telemetry  linear in F, streamed in bounded chunks
sequential replay control world work       O(F)
compiled shared-adjoint world work         sublinear / reused within certified strata
```

This is the design reason to expect a memory-light result. It is not proof that
Python compiler objects, Metal scratch, MPS allocator caching, bridge tensors,
or process RSS stay within the paper envelope.

### Acceptance measurement

G6 alone authorizes the phrase "fits the target memory envelope." It requires:

- all 21 measured rows from fresh processes;
- maximum MPS working set at or below `2 GiB`;
- maximum process-group RSS at or below `4 GiB`;
- bounded `F=8 -> F=300` peak-memory growth under the frozen ratio/delta
  checks;
- staged/fused, sequential-control, and restart numerical parity;
- exact source, native binary, hardware, camera, world, and track bindings.

## Why G6 was not launched on this Mac

Current observed host state at the final audit:

- about `9 GiB` disk free;
- `15.425/16 GiB` swap used;
- only 5,635 free 16-KiB VM pages at the sample;
- severe compressor and swap activity;
- unrelated long-lived workloads.

The incident guard requests 8 GiB available memory and at most 2 GiB swap use.
That is safety headroom, not a 32-GiB WorldFoam representation requirement.
Weakening the guard would risk another machine-level incident and would also
contaminate the memory measurement.

A B200 is not a drop-in alternative: the implemented ABI is Metal/MPS. Moving
G6 to CUDA requires a separately validated CUDA backend with parity against
the same representation and contract.

## World Tubes evidence boundary

The concise manuscript is limited to accepted theorem/fixture correctness and
structural-reuse evidence. The bounded camera curve has genuine numerical rows
through a `178 degree` half-span and an honest compiler death boundary at
`179 degrees`, but its current artifact remains a candidate because clean
source provenance is false. Frozen same-world scaling and seven schema-v2
public contexts remain unmeasured.

## Manuscripts and PDFs

The concise sources now explicitly separate tests from ablations:

```text
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_ICLR_MAIN_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_ICLR_MAIN_DRAFT.md
```

Generic QA PDFs were structurally and visually inspected page-by-page:

```text
output/pdf/world_tubes_iclr_generic_qa.pdf
output/pdf/worldfoam_iclr_generic_qa.pdf
```

They have embedded fonts, no Type-3 fonts, and portable opaque RGB evidence
images. The WorldFoam page-10 table/footer collision was repaired and the
small evidence figures were enlarged. These are generic manuscript QA builds,
not official ICLR packages.

The official ICLR 2027 archive URL still returns 404. The World Tubes venue
directory is therefore an intentionally non-buildable, hash-bound scaffold;
no ICLR 2026 style was renamed or promoted. The strict WorldFoam gate likewise
fails until G4, G6, the official style, clean source, and venue PDF exist.

## Strict gates

The integrated low-cost verification pass completed with `50 passed`. This
only confirms that the package gates reject incomplete evidence. Current strict
audits correctly return `accepted=false`:

- World Tubes: missing frozen/public accepted rows, clean source, official
  ICLR 2027 style, official venue build, and final visual-QA bindings;
- WorldFoam: `G4=0`, `G6=0`, incomplete evidence, clean source, official style,
  venue source/PDF, and page-complete visual QA.

## Falsification branches

Hypothesis A:
    The native shared-adjoint implementation is genuinely frame-memory-light.

Supporting result:
    All fused rows stay within the fixed peak envelope, with bounded growth
    from `F=8` to `F=300`, while sequential replay performs proportional world
    work.

Falsifying result:
    MPS peak or process RSS grows materially with `F`, or a hidden compiler,
    bridge, target, sample, or gradient tensor remains frame-resident.

If falsified:
    Inspect the receipt-backed memory categories, remove the identified
    frame-resident owner, and rerun the same frozen matrix. Do not change the
    benchmark dimensions to make the row pass.

Hypothesis B:
    The logical state is small but native allocator/compiler overhead exceeds
    the practical envelope.

Supporting result:
    Logical bytes remain invariant while allocator/RSS peaks violate the
    absolute or growth limits.

If supported:
    Treat this as an implementation failure, not a mathematical failure. Use
    the peak categories to decide whether bounded caching, scratch lifetime,
    or compiler streaming needs repair.

## Next executable sequence

On a quiet eligible Mac:

1. Rebuild `world_foam_lane2_fused_slab_v0` for Python 3.11.
2. Require source and import/ABI verifiers to agree on all 133 schemas.
3. Rerun the dry plan and require zero blockers and zero emitted evidence.
4. Execute the guarded 24-process sequence without changing the frozen
   dimensions or ceilings.
5. Independently verify all 21 measured rows.
6. Only then generate the G6 table/figure and add a memory-fit sentence.

For the broader submission, G4 public quality and the missing World Tubes
schema-v2 frozen/public rows remain separate required experiments.

