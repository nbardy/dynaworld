# Four-frame profile and exact slice-record construction

The preceding goal turn was progress: root 18f1d1d retained five passing Metal
layouts and three capacity-negative cases. This continuation profiles the valid
F32/chunk4 layout, removes measured generic copy overhead, and verifies the
same layouts again. One lead, no subagents; accelerator jobs remain sequential.
All source, world, image/gradient tolerances and resource guards stay fixed.

## Bounded profile

The synchronized Python profile at outputs/benchmarks/2026-09-13_f32_chunk4_profile/
passes existing numerical/identity/storage/memory validators and keeps exact
parent atlas bytes. It records five compilations and 40 slice/forward/native-
backward calls (correctness plus one warmup and three timed trials). Offline
W&B y8j073l3; sampled process-tree/launcher RSS 1,663,516,672 bytes, zero new swap.
The stopped Kineto/record_shapes mode was not retried. These instrumented times
are diagnostic totals across all profiled calls, not ordinary F32 benchmark time:

- Compilation: 15.48s profile total; visibility stratification 7.90s cumulative,
  support rebinning 4.86s and fallback marking 2.85s.
- Slicing: 2.79s profile total; generic dataclasses.replace spends 1.11s cumulative
  across 429,550 calls, including 429,470 cell copies and 80 parent replacements.
- Native forward wrapper: 4.08s profile total; interval packing 2.28s cumulative
  and synchronized device completion 1.56s. Packing remains the larger CPU target.
- Native backward wrapper: 1.34s, primarily 1.30s in synchronized device completion.

## Change and paired CPU check

STAR 9943865 replaces generic replace(cell,...) with explicit construction of
ProjectiveTraceTileTimeCell in slice_projective_trace_cell_atlas_frames. Its tile
coordinates, clipped interval, remapped primitive IDs/order, depth intervals,
fallback bit and fallback reasons are unchanged. Parent atlas copying, tensor
selection, support, depth ordering, interval packing and native code are untouched.
The change is six insertions/two deletions in one source file, not new math.

compare_slice.py extracts the previous slice function and its iterator from
the retained pre-edit source into an independent namespace. Across F4/8/16/32,
all dataclass fields and tensor values match exactly for chunks 1/2/3/4/8,
every one-frame window, whole/inner windows and an empty beyond-end window.
Three alternating F32/chunk4 CPU timing trials include iterator setup and all
eight slices. Median 0.306592->0.269597s (1.137x; 12.07% less time), with
samples old [0.300935,0.336297,0.306592] and new [0.296954,0.260129,0.269597].
This mechanical CPU check has no W&B or GPU-performance claim.

## Full Metal repeat

The same frozen 2048-tube 800-update world, Coffee Martini cam06, 96x128,
CPU LRU 8, configured tile_t=1, capacity 256 and one-warmup/three-repeat timing
contract are retained. The runtime config changes only output_dir relative to
the checked-in device-chunk control; it is saved alongside the actual W&B config.
All 189 regression tests pass in 4.00s. Five rendered layouts pass the existing
independent image/world-VJP, checkpoint, target/camera, payload, retained atlas,
route-memory and timing checks. Both F4 rows retain non-unit slice parity.
The three previously proved capacity rejections remain rejected; the parent
atlas identities are unchanged, so this is not an all-layout acceptance claim.

Separate-run medians, seconds; E+B excludes I/O/transfer/loss but includes CPU
setup. Full includes compile and measured forward/backward segments, not
optimizer work or inter-segment cleanup. Small changes include run variation.

| F | Chunk | E+B before | After | Full before | After |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.14469 | 0.14448 | 0.43130 | 0.43120 |
| 32 | 1 | 1.87755 | 1.79437 | 4.85160 | 4.82425 |
| 4 | 2 | 0.11942 | 0.12473 | 0.41461 | 0.41848 |
| 32 | 2 | 1.64620 | 1.58073 | 4.65392 | 4.64157 |
| 32 | 4 | 1.21665 | 1.18709 | 4.23945 | 4.18126 |

F32/chunk4 E+B falls 1.21665->1.18709s (2.43%), full 4.23945->4.18126s
(1.37%). This is a small practical gain, consistent with the paired CPU saving;
F4/chunk2 is slightly worse. Do not attribute unrelated phase variation to
record construction. Replay remains faster at 0.32341s E+B for F32/chunk4.
Maximum RGB error 8.64267349e-7 and per-parameter normalized VJP 1.85973848e-6.
Peak process-tree plus launcher RSS 2,230,321,152 bytes (~2.08GiB), zero new swap.
The unchanged 3-GiB RSS, 2-GiB allocator, 256-MiB swap-growth, 600-second and host/disk
limits pass. Final offline W&B wnv4ty8m; artifacts under
outputs/benchmarks/2026-09-13_direct_slice_cells/. Native SHA remains
6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.

No optimizer fit or new quality result. Numerical layouts stay 5 passing/3
capacity-negative, earlier fixed-density rows 4/4, public contexts/lanes 0/7 and
0/21. BASELINES is unchanged. No sublinear scaling or replay speed win follows.
Run-bound sources, including the previous implementation, are archived and
hash-verified. Root owns this note, scoped status edits and the STAR gitlink;
unrelated streaming, runner, browser and paper WIP remains unstaged.

## Next decision

Before another compiler micro-optimization, audit the existing native batched
UVT renderer as a stronger static-camera control on this exact frozen world.
The current replay intentionally projects/renders one frame at a time; the
static compiler already produces an affine UVT sequence once, and the ordinary
STAR renderer can consume a sequence. Test value/world-gradient parity first,
then charge projection, binning, native forward/backward and target handling
under the same caps. Start with the existing contiguous 32-frame lattice;
do not assert equivalent non-unit sparse sampling or moving-camera closure.
This is a control using an existing renderer, not a new model family. The
remaining measured interval-packing cost is still a candidate if the control
shows that this static affine case needs an interval atlas at all. The broad
overnight goal remains active.
