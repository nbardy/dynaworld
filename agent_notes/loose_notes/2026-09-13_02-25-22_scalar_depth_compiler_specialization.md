# Exact scalar-depth specialization removes redundant compiler work

The previous goal turn made measured progress with sparse reference fallback.
This turn follows its actual profile: UV depth events and repeated tile-corner
evaluation dominated compilation even though the saved atlas has zero spatial
depth slopes. STAR `ccf6c1c` implements the exact specialization and preserves
the general path. This is a compiler implementation improvement, not new gauge
mathematics, new training evidence, or completion of the broad goal.

## Why the specialization is exact

The existing depth model is

    z_i(u,v,t) = z_ci(t) + s_ui(t)*(u-u_i(t)) + s_vi(t)*(v-v_i(t)).

If every coefficient of both spatial slope polynomials is exactly zero,
then z_i(u,v,t)=z_ci(t). All four tile-corner depths are equal. A pair's
order depends only on time, so there is no spatial line to locate; coincident
scalar depths remain an ambiguity over the whole tile. The existing scalar
near-depth checks still decide fallback using the unchanged epsilon 1e-6.

The marker recognizes exact zeros in the full [N,6] spatial-polynomial tensor.
It does not use a magnitude threshold or look only at one sampled time. It
uses the scalar depth-range path and omits the redundant UV event report.
This is a local compiler decision: no retained tensors are removed or edited.
Nonzero spatial coefficients use the existing general path. Image-tile bounds
are now validated before either path so skipping corner evaluation cannot
bypass out-of-image rejection. The source edit is 12 insertions/1 deletion.

Regression coverage includes explicit zero tensors versus absent spatial data,
scalar near-ties, inherited fallback, constant spatial crossings, a spatial
slope which is zero initially but becomes nonzero later, and invalid tiles.
The existing spatial splitting, live per-pixel sorting, depth-order and world
VJP checks remain passing.

## Repeated local Metal timing

The same checked-in `frozen_fallback_cost_20260913.jsonc` uses one warmup and
three synchronized paired trials. The comparison is against the immediately
preceding sparse-fallback run, on the same 256-tube/four-frame/96x128/cam06
checkpoint, selected times, loss, alpha law, capacity128, and two-frame device
chunks. These before/after processes are local controls, not an interleaved
implementation comparison or the full clean-source frame-scaling sweep.

| Median seconds | Sparse fallback before | Scalar specialization | Before / after |
| --- | ---: | ---: | ---: |
| Compilation | 8.01556 | 0.94148 | 8.51x |
| Compiled forward | 2.12509 | 2.14532 | 0.99x |
| Compiled backward | 1.80433 | 1.81730 | 0.99x |
| Compiled forward + backward | 3.90343 | 3.96569 | 0.98x |
| Compilation + forward + backward | 11.90172 | 4.94081 | 2.41x |
| Replay forward + backward | 0.03680 | 0.04279 | 0.86x |

The three compilation samples are 0.94148,0.93474,0.97511 s. Combined
compile/forward/backward samples are 4.97405,4.79770,4.94081 s. Medians of
sums need not equal sums of medians. Rendering is effectively unchanged while
compilation improves strongly; replay fluctuation also illustrates host noise.
Even after both recent fixes, the compiled route remains much slower than
replay on this tiny world. No sublinear exponent or speed superiority follows.
Offline W&B run `i6pm2a9p` retains config, tags, report and timing distributions;
the preceding comparison run is `ahjazger`.

## Correctness and retained identity

Both F4 and nonuniform F3 [0,2,3] rows pass all eight local checks. F4 max RGB
error is 6.8545341e-7, global world-VJP error 7.2145873e-7, and maximum
parameter-group error 6.8233291e-7. F3 gives the same max RGB error, global
VJP 9.5662026e-7 and max parameter VJP 1.0318185e-6. F3 chunk slicing matches
single-frame slicing exactly in RGB, with global VJP error 4.8325336e-7 and
max parameter error 5.7327333e-7. Existing independent timing, retained-storage,
and selected-time-slicing validators accept the corresponding artifacts.

The strict-loaded world file SHA remains
`8ca44d076cf17e6aed274f2986a5cb033a7220fa2a1c14a41ca1e589345949f9`,
with logical world SHA
`ccf3d00cfe5463a8b14b2dccfb36ee6e1b2695fdb392cb29e08a7e779e50c64a`.
Raw retained tensor payload hashes/shapes/dtypes, every sampled tile's active
trace set, and fallback tile masks are exactly unchanged for F4 and F3.
F4 topology is identical after stripping only the redundant
`visibility_uv_depth_line` reason. Its fallback remains 92/720=12.7778%,
with 2007 cells and 15085 raw cell entries. F3 remains 17.0370% fallback.

The F4 serialized file is now 1,038,473 bytes, down 2,322 bytes solely from
removing labels; the 32,768-byte tensor payload is unchanged. This is not a
new geometric compression result. The actual artifact SHA is
`4476bfbce89f8748874a6decdbd86be3008b77f622040fb6f35f7d468cad89f5`.

## Gates, resources, and profile

- Focused CPU depth/correctness/UVT-producer suite: 70 passed, 33 MPS skips.
- General spatial visibility suite: 18 passed.
- Guarded actual Metal depth/fallback suite: 23 passed.
- Repeated timing, selected-time check, and one compiler profile all completed.
  Maximum tree/launcher RSS was 1,578,958,848 bytes, zero new swap and no guard
  trip. Existing 3-GiB RSS, 2-GiB MPS, host, disk and 600-second limits remain.
  No native build, concurrent accelerator job, application termination, cloud
  upload, or subagent was used.

The instrumented compiler call falls 8.3345 -> 1.1183 s (profiler overhead
included). Fallback marking falls 7.2376 -> 0.0726 s; the UV report is absent.
Time-event stratification is now 0.5524 s, with 33459 pair-root calls. These
profiles attribute cost; use the separate repeated measurements for timing.

Artifacts and launch scripts are retained in
`outputs/benchmarks/2026-09-13_scalar_depth_compiler/`. `comparison.json`
records the input/tensor/membership checks, timing distributions, resource
receipts, and profile. W&B is omitted only for mechanical tests, slicing, and
profiling; the repeated performance control uses offline W&B.

## Next work

The dominant remaining cost is ~3.97 s in forward/backward, largely the Torch
reference fallback, versus ~0.043 s replay. Inspect/profile that route before
another scaling sweep. Preserve source-centered float32 depth and source-id
ties: changing them previously caused a measurable order swap. A faster
fallback must include every contributing trace in each selected tile and
retain complete world-VJP coverage; do not bypass the compiler by silently
substituting the baseline renderer. No threshold should be relaxed to make
this case appear faster.

Paper counts stay 0/7 contexts and 0/21 lanes; BASELINES is unchanged. Source
fits and shared-world quality controls remain separate evidence. The active
overnight objective is still open.
