# Individual support bounds unblock the fitted world; temporal gradients still fail

## Starting evidence and scope

The preceding conversational turn rechecked known source-fit results for the
user; it did not advance the overnight goal's evidence counts. This continuation
returned to the actual failed four-time compiler run on the retained 800-update,
2048-tube Coffee Martini world. The failure log was terminal, and the process
inventory showed only unrelated market recorders before execution. There were
no concurrent accelerator jobs. The stopped shape-recording profiler stayed
stopped. No new training or quality-baseline promotion occurred.

The failed call was the full-versus-sliced atlas check, before final atlas
serialization: packed projective interval tile capacity overflow. No retained
atlas existed from that attempt. Ordinary evaluation of the same world had
already rendered completely, so the compiler's candidate support was the next
suspect. The failure is distinct from a host resource stop.

## Reproduced cause and bounded fix

The UVT lowerer computed the maximum alpha-support radius across every tube
and both spatial axes, then assigned it to every trace. On the saved world
this radius is 96.54515 pixels; the median individual square radius is 17.998.
CPU reconstruction of the old actual compiler produced 1958 active traces,
3840 cells, and overflowing lists in all 192 spatial tiles at each selected
time. Maximum candidates for frames [0,10,21,31] were [591,938,921,578].

STAR commit a316cc6 retains separate conservative u/v radii for each trace.
The existing support-event rebinning function uses those radii consistently
for boundary roots and interval boxes. Explicit scalar uv_padding remains a
minimum, and filtered inactive source rows retain the correct radius mapping.
No opacity, alpha cutoff, world state, native capacity, or renderer law changed.
No Metal/C++ rebuild was needed.

The bound is ordinary Gaussian algebra, not new gauge mathematics. Completing
the spatial square gives alpha_i(x,t) = p_i(t) exp(-0.5 d^T A_i d), with
d = x - center_i(t). For alpha >= epsilon, d^T A_i d <= R_i^2, where
R_i^2 = 2 log(max_sample p_i(t)/epsilon). Cauchy-Schwarz in the A_i metric
gives |d_j| <= sqrt(R_i^2 (A_i^-1)_{jj}). Using each tube's maximum over the
supplied sample times is conservative for those samples. The fix removes the
extra maximum over unrelated tubes and spatial axes. It does not assert
support coverage for arbitrary unsupplied times or prove sublinear compilation.

The retained post-fix Metal atlas has per-frame maxima [78,114,119,87] at the
unchanged capacity 256, with zero overflow. It has 1853 cells and zero fallback
tile samples. All contributors above the declared alpha threshold must still
be included; raising capacity or silently truncating was not used.

## Runtime evidence and remaining negative

The existing frozen replay/compiled contract completed against the exact
checkpoint, camera, target images, alpha law and four selected times. It is
still accepted=false:

| Check | Measured | Limit / status |
| --- | ---: | --- |
| Maximum RGB difference | 1.400709e-6 | 1e-5, pass |
| Global normalized world VJP difference | 4.262453e-6 | 1e-5, pass |
| Maximum per-parameter normalized VJP difference | 3.235681e-5 | 1e-5, fail |
| Fallback fraction | 0 | 0.2, pass |
| Same-parent slicing maximum RGB difference | 0 | pass |
| Same-parent slicing maximum per-parameter VJP difference | 1.677275e-5 | 1e-5, fail |

The failing parameter is raw_lambda_t; all other world parameters pass the
per-parameter route gate. Same-parent sliced losses agree exactly. This is not
a successful compiler acceptance result merely because most checks pass.

The single un-warmed measurement records compilation 4.5063 s, compiled forward
0.3764 s and backward 9.6449 s, versus replay forward/backward 0.07413 s total.
It is a diagnostic, not a speedup or scaling exponent. The larger fitted world
exposes substantial backward cost even with zero fallback.

## Identical-backward control

Three further passes reused one parent atlas, the same four one-frame chunks,
world and robust-L1 loss. No optimizer ran. Images and losses agree exactly
across all three passes. Per-pair raw_lambda_t normalized differences are
9.7363e-6, 1.0358e-5 and 1.2180e-5: two pairs exceed the frozen gate even when
the two sides use the identical route. All three pairs and raw gradients are
retained; no best-pair selection or acceptance retry was used.

Atlas coefficient/opacity/precision/color gradient variability is around
0.9e-7 to 1.4e-7, while projected-q gradient variability is around 6.6e-6 to
7.8e-6 and raw_lambda_t variability reaches 1.22e-5. The first lambda-gradient
norm is 0.001686, so it is not an essentially zero reference. This demonstrates
non-repeatability and amplification through the lowering adjoint; it does not
prove that all of the larger replay-versus-compiled discrepancy is noise.

One candidate source is the expanded temporal envelope. For coefficients
(c0,c1,c2) = (lambda*t0^2,-2*lambda*t0,lambda), its lambda adjoint combines
t0^2*g_c0 - 2*t0*g_c1 + g_c2. Nearly cancelling terms can amplify small
coefficient-gradient perturbations when t is near t0. A centered temporal
formulation or a stable adjoint is worth isolating against a higher-precision
reference. This is a hypothesis, not an implemented or accepted repair.

## Tests, artifacts, resources and provenance

The new mixed-size moving-tube regression failed on old code with false tile
overflow. It then passed complete dense-reference RGB and ma/q/opacity/color
VJPs on CPU and Metal. An inactive source row protects radius remapping.
The focused CPU producer/binning/depth suite passed 58 tests, 19 MPS skips;
the actual guarded depth/support Metal suite passed all 33 tests. Initial new
test wiring had a reference-API keyword error, then changing capacity in a
shared process exposed the native library's cached capacity. The corrected
test checks tight packing separately while retaining a consistent native
configuration. The failed Metal log remains capacity_fixture_failure.log.

All outputs are under outputs/benchmarks/2026-09-13_frozen_quality_world/:
the original overflow log/source hashes carry .overflow_failure suffixes;
projective_trace.before.py and projected_cpu.pt preserve the diagnosed inputs;
support_diagnosis.json binds old candidate counts; report.json and its atlas
record the negative post-fix run; gradient_repeatability/ retains three raw
gradient passes and all pair comparisons. Existing timing/storage/slicing
validators validate the report's internal consistency, including its negative
acceptance status; report_validation.json records this without a new verifier.
Source hashes still match, and tests/config are hashed. Serialized atlas bytes:
2,900,060, SHA 2bd6dfeb2ce6b8c9fc85410010c010f1a8ede1a7cbe3a7552f406ae6313db2d9.

Offline W&B: original overflow jy8gqiaj, completed negative compiler a5g8ks9f,
identical-backward control ax1lezzl. Mechanical tests and CPU support diagnosis
omit W&B because they make no training/benchmark claim. Peak tree/launcher RSS
was 392,101,888 bytes for CPU diagnosis, 602,734,592 for the Metal regression,
2,140,684,288 for the compiler check and 2,327,838,720 for repeated backward.
All completed receipts show zero new swap and no tripped guard. The unchanged
3-GiB process and 2-GiB allocator limits apply on this physical 24-GiB host.

The saved world and its 20.84/15.30 dB train/heldout quality remain unchanged.
Next diagnose temporal-gradient conditioning/repeatability before expanding
the frozen scaling sweep. Publication counts remain 0/7 contexts and 0/21
lanes; BASELINES stays unchanged. The broad overnight goal remains active.
