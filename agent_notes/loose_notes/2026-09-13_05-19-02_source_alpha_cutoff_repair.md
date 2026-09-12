# Source-compatible alpha membership repairs the fitted F32 compiler gate

The previous goal turn made progress: the source-overfit review recovered and
checked positive historical evidence, and exact session 61770 returned terminal
exit zero for the pending cutoff-repair sweep. Existing independent validators
then accepted its four local rows (previously three). This continuation checks
those retained results and integrates the previously uncommitted repair. One
lead, no subagents, sequential accelerator/native jobs, unchanged resource caps,
offline W&B, and no newly specified token budget. The broad goal remains active.

## Demonstrated failure and correction

Completing the spatial Gaussian square changes float32 arithmetic order. The
retained trace-771 example gives source alpha 0.003921571187674999 versus compiled
0.003921567928045988, on opposite sides of 1/255. Omitting this contributor causes
a 0.0006172061 RGB error although the continuous alpha difference is only
3.26e-9. Its compositing jump is T*a*(c-S), for prefix transmittance T and suffix
composite S. Analytic equality does not preserve hard branch membership.

STAR commit 01bd9e0 retains original [ma_uvt,q_uvt] in a detached optional Nx9
float32 tensor, alpha_cutoff_reference_uvt. CPU reference/fallback and native
interval/row forward and interval backward use the original quadratic ONLY to
choose inclusion at the alpha cutoff. Continuous alpha and its derivatives
still use the compiled centered/spatial expression. This preserves the ordinary
piecewise derivative without falsely routing continuous source values through
a different compiled derivative. The hard boundary itself is discontinuous;
this does not create a derivative there.

The reference is regenerated from live UVT tensors after world updates, sliced
and remapped with trace ids, and carried through static/trainer/CPU/quadrature
constructors. Missing opacity_time_centered propagation in quadrature and a
backend constructor was also corrected. Generic atlases with no source reference
keep their existing rule. Native plain-cell reserved1 flags the reference; three
plain-cell operator ABIs gain its tensor. Camera-family native APIs are unchanged.
This repairs the observed cutoff case, not every hypothetical support-pruning,
sort, or transmittance-stop rounding boundary.

The new reference costs 36 bytes per active trace, now included in storage and
payload accounting. Storage readers accept the historical six/seven tensor
record layouts and the new eight-record layout, while verifying actual bytes.
All seven previous tensor records and cell topology remain byte-exact for every
before/after row. The new reference bytes independently match the retained
original MPS projection selected by source id. Snapshot hashes, fixed checkpoint,
cameras, target/time contracts, tolerances and native binary are checked.

## Actual verification and local curve

The first regression run failed four cases at the unchanged threshold (CPU/MPS,
fast/fallback); neighboring opacity values passed. CPU-only after repair passes
six cases. A guarded native rebuild passes; the first full gate passes 112 tests.
Strengthened coverage changes the initial q, live-updates it to the counterexample,
slices time, checks RGB and ma/q/opacity/color VJPs, and checks actual Metal rows.
The extended gate passes 159 tests in 3.95 s, including producer/trainer, depth,
visibility, quadrature and existing correctness tests. The earlier test source
is retained to explain its pre-extension source-hash difference. No tolerance
was relaxed. These tests are preflight, not publication rows.

The same 800-update, 2048-tube fitted Coffee Martini world is evaluated on cam06
at 96x128, F={4,8,16,32} spanning the same physical 32-frame interval. Resident
chunks remain one frame, CPU target LRU eight frames. Each row has one warmup
and three alternating paired timed trials. No training occurs in the sweep.

| F | Max RGB error | Max parameter normalized VJP error | Atlas bytes | Compile s | Compiled forward s | Compiled backward s | Replay fwd+bwd s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8.34e-07 | 1.76e-06 | 2,970,771 | 0.99144 | 0.30724 | 0.26044 | 0.04528 |
| 8 | 8.64e-07 | 1.21e-06 | 6,560,634 | 2.54584 | 0.72462 | 0.61447 | 0.07570 |
| 16 | 8.64e-07 | 8.7e-07 | 12,544,489 | 4.93927 | 4.87547 | 1.50828 | 3.39724 |
| 32 | 8.64e-07 | 6.33e-07 | 20,431,725 | 8.16230 | 10.59741 | 2.75986 | 6.80828 |

All eight main checks pass at every F; the configured extra F4 selected-time
slice check also passes. All rows have zero fallback and overflow. F32 max RGB
error falls 6.17e-4 -> 8.64e-7. Accepted LOCAL density rows increase 3/4 -> 4/4.
This is not the full public protocol: publication_eligible stays false, public
counts remain 0/7 contexts and 0/21 lanes, and BASELINES is unchanged.

Compile/forward/backward remain slower than replay. Forward includes CPU target
loading/decoding, transfer and loss, and the eight-frame LRU thrashes at F16/32.
No separate-process decode time was subtracted. The next useful measurement is
an explicit in-trial forward breakdown with synchronization boundaries, keeping
end-to-end totals and the existing streaming memory contract intact.

Artifacts: outputs/benchmarks/2026-09-13_alpha_cutoff_repair/ holds raw rows, atlases, reports, validation, native/build/test
logs, resource receipts, source snapshots and offline W&B zufaogo3. The native
binary SHA is 6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.
Peak process-tree plus launcher RSS: build 2,094,923,776; extended tests 911,179,776;
sweep 2,062,155,776 bytes. All have zero new swap and no guard trip. All exact
handles are terminal. The Kineto/record_shapes lane remains stopped.

## Integration and continuing interpretation

The submodule commit owns six clean source files plus one tracked payload
inventory line. A second inventory line is inside an already-uncommitted helper
and stays with that unrelated benchmark WIP; the run-bound after/ snapshot
preserves the full tested source. An initial scoped-stage preparation expected
two occurrences in HEAD and failed before mutation; inspection found only one
tracked occurrence, so staging was narrowed to that line. No unrelated WIP was
committed. Root integration owns the backend/storage changes, two regression
files, this note, scoped status edits and the submodule pointer.

Successful historical source-space fits remain real (29.82 dB, and the smaller
21.77-dB recipe reproduced). The newer world-space fit is 20.84/15.30 dB on
training/validation views and remains blurry. Compiler equivalence, training
convergence, novel-view quality and runtime scaling are distinct claims.
