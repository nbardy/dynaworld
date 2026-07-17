# 09 - Metal Acceptance Plan

The Metal path should progress in small gates.

## Gate A: Projective Trace Evaluator

Implement:

```text
projective_trace_eval(coeffs, times, eps) -> [N, S, 4]
```

where:

```text
coeffs: [N,9] = [u0,u1,u2, v0,v1,v2, z0,z1,z2]
times: [S]
out[n,s] = [u, v, z, valid_sign]
```

This proves homogeneous/rational camera-time math runs in Metal and matches
Torch.

The bundle-level gauge invariant now has a CPU/Torch report before the Metal
hot path:

```text
research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.md
```

It is not a Metal timing row. It is the acceptance guard for the statement
that changing the screen-fiber coordinate, such as depth to log-depth, must
preserve `pi_* Gamma^* rho` when the measure Jacobian is included and must
reject orientation-reversing depth gauges as visibility boundaries.

## Gate B: Rational Center To Affine UVT Fit

Fit a local affine UVT center from rational samples:

```text
ma, velocity, fit_error = fit_affine_trace(projective_samples)
```

The result can still feed the existing renderer, but now the compiler can choose
orbit windows by rational fit error.

Current status:

```text
fit_projective_trace_polynomial(coeffs, times, degree)
split_projective_trace_windows(coeffs, times, degree, thresholds)
```

exist as CPU/Torch compiler helpers. They fit affine or quadratic local traces
for `[u, v, h_z]`, return residual/validity certificates, and split long time
intervals into accepted local chart windows. They are not yet Metal hot-path
fit kernels, but accepted windows now feed tile-time/cell atlas binning paths.

## Gate C: Projective Bounds

Compute conservative image/time tile support from rational center plus local
covariance. This replaces naive per-frame support enumeration.

Current status:

```text
bound_projective_trace_window(window)
bound_projective_trace_windows(windows)
```

exist as CPU/Torch compiler helpers. They compute continuous polynomial UV/depth
bounds from accepted chart fits and inflate those bounds by residual
certificates. They do not yet include full Gaussian footprint covariance, but
tile quantization and renderer bin integration exist for the current cell-atlas
prototype.

## Gate D: Depth And Denominator-Aware Visibility

Extend `depth_beta` contract with:

```text
depth_uncertainty
denominator_min/max
chart_gauge_id
```

or a sidecar table if the current hot renderer tensors must remain stable.

Current status:

```text
make_projective_trace_visibility_sidecar(window)
compare_projective_trace_depth_order(sidecar_a, sidecar_b)
```

exist as CPU/Torch compiler helpers. They record denominator margin/root flags,
depth range, depth slope range, monotonicity, uncertainty, and chart gauge id.
The focused synthetic test detects both stable front/back order and crossing
depth strata.

```text
make_projective_trace_appearance_sidecar(alpha_max, color)
bound_projective_trace_visible_swap_cost(order, appearance_a, appearance_b)
```

also exist as CPU/Torch compiler helpers. They implement the alpha/color visible
swap bound for ambiguous pairs and mark either `safely_commutable` or
`needs_fallback`. Renderer fallback masks are not wired yet.

## Gate E: Renderer Integration

Only after Gates A-D, add rational trace support to binning/rendering. The first
renderer integration can still evaluate rational center and local Gaussian
footprint per sample, with fallback to the current affine `q_uvt` path when the
chart is marked affine.

Current compiler-side status:

```text
bin_projective_trace_support_bounds(bounds, image_width, image_height, tile_size)
assemble_projective_trace_tile_time_atlas(records)
```

exist as CPU/Torch prototypes. They emit compressed tile-time records from
accepted projective support bounds, carry optional fallback flags from
visibility masks, and assemble per-tile active sets with depth order metadata.
This is not a Metal or renderer hot-path integration yet.

The first dense-reference correctness test also exists:

```text
tests/test_star_uvt_projective_correctness.py
```

It checks atlas coverage and stable depth order against dense per-frame
projective projection samples. This is still a compiler-side gate, not a Metal
hot-path renderer.

The same test file now also exercises:

```text
render_projective_trace_tile_time_atlas_reference(...)
```

That helper is a CPU/Torch reference renderer. It consumes atlas cells and
matches dense per-frame compositing for a small stable-depth screen-space
Gaussian scene. Gate E is therefore ready for a guarded Metal/hot-path
integration attempt, with this helper as the reference.

The first guarded bridge into the existing STAR UVT renderer contract also
exists:

```text
projective_trace_windows_to_uvt_tubes(...)
```

It lowers accepted degree-1 projective chart windows into
`ma/q_uvt/depth0/depth_beta` tensors. The CPU brute-force q-UVT renderer matches
the atlas reference renderer on the focused stable-depth scene. Next Metal work
should first run the same lowered tensors through `render_uvt_tubes(...)` on
MPS, when available, before adding a projective atlas-specific Metal kernel.

That guarded MPS smoke now passes on this machine:

```text
test_projective_affine_q_uvt_bridge_matches_metal_renderer_if_available
```

The remaining Gate E work is no longer "can projective charts reach Metal at
all?" The next question is whether nonlinear/projective atlas cells get their
own Metal evaluator, or whether chart splitting plus explicit interval gating
can lower enough of the orbit into existing q-UVT tubes.

The first explicit interval-gated split-chart bridge now exists on the CPU
oracle path:

```text
ProjectiveTraceUVTBridge.active_start / active_stop
render_projective_trace_uvt_bridge_reference(..., use_window_gates=True)
```

This proves the right sidecar contract for split affine charts. Metal still does
not consume the interval gate sidecar; without that gate, split segment tubes
leak outside their chart domains.

The interval sidecar first reached Metal through a span-gated wrapper:

```text
projective_trace_uvt_bridge_active_spans(...)
render_projective_trace_uvt_bridge_metal_gated(...)
```

That wrapper partitioned the frame axis at active interval boundaries, called
the existing `render_uvt_tubes(...)` Metal path for each constant-active-set
span, and copied only that span into the output.

The native shader-side interval gate now exists:

```text
torch.ops.star_uvt_v0.render_gated(...)
render_uvt_tubes_gated(...)
```

It passes per-tube `[active_start, active_stop)` int32 buffers into the Metal
renderer. The gated binning kernel clamps tube support to active frame
intervals, and the gated render kernel skips inactive tubes per sample so split
chart segments do not leak inside multi-frame tiles. The projective q-UVT bridge
now calls this native gated path instead of the span-partition wrapper.

The remaining Gate E work is now nonlinear/projective atlas-cell Metal
evaluation and backward coverage. Degree-1 chart windows can lower into the
existing q-UVT renderer with native interval gates; higher-order/rational chart
cells still need either local lowering or their own hot evaluator.

The matching direct VJP interval gate now exists:

```text
torch.ops.star_uvt_v0.direct_atomic_backward_gated(...)
direct_atomic_backward_gated(...)
direct_backward_projective_trace_uvt_bridge_metal_gated(...)
```

It uses the same per-tube `[active_start, active_stop)` buffers as
`render_gated`, clamps backward tile-time binning to active intervals, and skips
inactive tubes during per-sample VJP accumulation. This gives split affine chart
segments a first native forward/backward q-UVT path. It is direct atomic
backward coverage, not yet a full optimized or trainer-integrated backward
strategy.

The first bridge-level trainability smoke now passes. It renders a target from
split projective chart windows, starts from a different bridge color, computes
MSE image gradients, runs the native interval-gated bridge VJP, applies one
color update, and verifies loss decrease. This proves the gated q-UVT bridge can
drive a real optimization step; it is still not real trainer integration.

The first nonlinear/projective atlas-cell Metal evaluator now exists:

```text
pack_projective_trace_tile_time_bins(...)
torch.ops.star_uvt_v0.render_projective_trace_tiles(...)
render_projective_trace_tile_time_atlas_metal(...)
torch.ops.star_uvt_v0.direct_projective_trace_backward(...)
direct_backward_projective_trace_tile_time_atlas_metal(...)
```

This path consumes compiler-side tile-time active sets instead of affine q-UVT
tubes. Each packed tile slot carries primitive id plus an exact sample-domain
interval. The shader evaluates the quadratic homogeneous trace at the current
sample time, uses `h_z(t)` as projective depth, and composites a screen-space
Gaussian footprint. A focused quadratic-chart test matches the CPU atlas oracle
on MPS, so degree-2 projective cells now have a native forward renderer.

The matching direct projective VJP now differentiates the compiled local
footprint through color, opacity, and the nine homogeneous coefficients while
treating tile membership/order as compiled constants. A focused quadratic-chart
test matches this native VJP against Torch autograd.

The first projective atlas-cell coefficient trainability smoke now passes. It
keeps color fixed, renders a shifted-coefficient target, applies the native
direct VJP, line-searches a small coefficient step, and verifies the
Metal-rendered loss drops. This is the first optimization smoke for nonlinear
projective atlas-cell geometry-like parameters.

The packed projective atlas now also has a frame-count scaling probe:

```text
count_projective_trace_dense_per_frame_tile_pairs(...)
research_project/benchmarks/projective_atlas_scaling_probe.py
```

The deterministic 45-degree orbit fixture gives:

```text
4 -> 64 frames
dense per-frame tile pairs:     35 -> 555
ideal interval atlas entries:   13 -> 13
Metal tile_t=4 slab entries:    13 -> 208
```

So Gate C's interval-compressed object has the desired sublinear world-side
shape, while the older Metal-compatible `tile_t=4` schedule expands the same
interval into one entry per temporal slab.

The first interval-compressed projective cell forward kernel now exists:

```text
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_tiles(...)
render_projective_trace_cell_interval_atlas_metal(...)
torch.ops.star_uvt_v0.direct_projective_trace_cell_interval_backward(...)
direct_backward_projective_trace_cell_interval_atlas_metal(...)
```

It consumes spatial tile bins directly and checks per-entry
`[active_start, active_stop)` intervals in the shader. The focused MPS test
matches the CPU atlas oracle, and the scaling probe matches slab-render image
sums from `4 -> 64` frames. On that tiny orbit fixture, interval Metal wall time
changes `24.8067ms -> 29.3612ms`, while slab Metal changes
`20.0995ms -> 37.2617ms`.

The matching interval-compressed direct VJP now consumes the same spatial tile
bins and active intervals. Focused tests match Torch autograd for color,
opacity, and cell trace coefficients; a coefficient-only training smoke applies
the native interval VJP and verifies Metal-rendered MSE drops. The latest tiny
scaling artifact records interval backward `6.7827ms -> 35.6822ms` over
`4 -> 64` frames while packed interval entries stay `13 -> 13`. This is a
forward/backward acceptance gate for the cell object; training still needs a
segment producer that supplies nontrivial gauge-domain intervals.

The trainer harness now has a first acceptance bridge for that segment-producer
shape:

```text
render_projective_cell_interval_atlas_metal_backward(...)
```

Its focused smoke builds split projective chart windows, lowers them to an
interval cell atlas with multiple `[active_start, active_stop)` domains, renders
through the interval-compressed Metal forward path, backprops through the
interval direct VJP, and verifies an optimizer step on cell trace coefficients
reduces loss. Treat this as trainer-harness acceptance; production acceptance
still requires a compiled-atlas producer, real STAR UVT training-loop routing
through the production state helper, and a chart recompilation policy.

The first chart-lifecycle guard now exists:

```text
projective_trace_cell_atlas_coverage_report(...)
projective_trace_cell_atlas_visibility_report(...)
rebin_projective_trace_cell_atlas(...)
mark_projective_trace_cell_visibility_fallbacks(...)
projective_trace_cell_atlas_fallback_stats(...)
stratify_projective_trace_cell_atlas_visibility(...)
projective_trace_cell_atlas_complexity_stats(...)
projective_trace_cell_atlas_budget_report(...)
projective_trace_cell_support_event_report(...)
projective_trace_cell_visibility_event_report(...)
projective_trace_cell_sensor_time_event_partition(...)
projective_trace_cell_sensor_time_partition_quadrature(...)
projective_trace_cell_sensor_time_partition_rolling_quadrature(...)
rebin_projective_trace_cell_atlas_support_events(...)
stratify_projective_trace_cell_atlas_visibility_events(...)
refresh_projective_cell_interval_atlas_if_stale(...)
ProjectiveCellIntervalTrainerState
```

This is compiler/training-control logic, not a hot shader. The coverage report
checks whether live cell trace coefficients still visit only frame/tile pairs
covered by compiled atlas cells. The visibility report checks whether the
compiled front-to-back order still matches live per-sample depths. The rebin
helper preserves the live coefficient/opacity/color tensors and rebuilds
support cells plus depth interval metadata. Focused tests move support into a
new tile and flip depth order without changing support; both stale cases are
detected and repaired. The trainer-harness refresh smoke then moves an MPS
coefficient tensor with an optimizer step, refreshes metadata, renders through
the interval-compressed Metal autograd path, and verifies gradients still
return to that same tensor. The trainer-state smoke now owns atlas/config/times
plus refresh cadence, refreshes after optimizer steps, and proves both support
and order refresh preserve the live tensor. Ambiguous depth ties now become
explicit fallback metadata: strict refresh raises, opt-in refresh marks
affected cells as `visibility_ambiguous_depth`, and pack/render paths reject
those cells unless fallback is explicitly allowed. The reference cell-atlas
renderer now gives fallback cells concrete semantics by sorting marked
tile/sample regions with live evaluated depth, and the trainer state reports
fallback coverage before any mixed Metal fast/fallback kernel exists. Refresh
also tries visibility-stratum splitting before fallback, converting crossing
depth order into stable time-run cells. Budget diagnostics report interval
compression, stratum split count, fallback fraction, and named budget failures;
refresh can now enforce those budgets and raise before rendering.
Support refresh now uses continuous screen/tile boundary roots to split moving
traces into time-local tile runs, avoiding broad all-frame tile rectangles when
the center crosses tile boundaries.
Continuous cell-local visibility event roots are now reported by solving
`z_i(t)-z_j(t)=0` on active intervals for affine/quadratic depth models, and
refresh now tries an event-root stratifier before the sampled stratum split.
Exact roots on a frame sample are isolated as singleton cells, which keeps
fallback localized to the actual tie/event sample. This is the first production
shape for the orbit, finite-exposure, and rolling-shutter boundary certificate.
The compiler now also exposes a continuous sensor-time partition that merges
support roots, visibility roots, and caller-supplied exposure/shutter split
times into intervals, independent of frame-index cells.
The partition can now be lowered into normalized finite-exposure midpoint
quadrature or per-row rolling-shutter quadrature with readout offsets. Those
schedules now have a differentiable CPU/Torch render oracle:

```text
render_projective_trace_cell_atlas_quadrature_reference(...)
render_projective_trace_cell_atlas_rolling_quadrature_reference(...)
```

The oracle evaluates direct cell traces at fractional sensor times, sorts by
live depth, composites, and accumulates sample weights. It is reference
semantics for exposure/rolling schedules. A first interval-Metal bridge now
exists too:

```text
ProjectiveTraceCellQuadratureLowering
lower_projective_trace_cell_atlas_quadrature(...)
render_projective_trace_cell_atlas_quadrature_interval_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_metal(...)
```

It lowers quadrature samples to integer sample-indexed interval cells, then
uses `render_projective_trace_cell_interval_atlas_metal(...)` and weighted
sample accumulation. Rolling shutter now has a batched lowering:

```text
ProjectiveTraceCellRollingQuadratureLowering
lower_projective_trace_cell_atlas_rolling_quadrature(...)
render_projective_trace_cell_atlas_rolling_quadrature_batched_reference(...)
```

The batched lowering merges row schedules into unique sample times and a
`row_weights[Q,H]` matrix. The current Metal wrapper uses that shared schedule,
and the row-weighted Metal kernel now consumes that matrix directly:

```text
render_projective_trace_cell_interval_atlas_rows_metal(...)
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_rows(...)
```

It dispatches over output pixels, skips zero-weight sample/row pairs, composites
the interval atlas for nonzero pairs, and writes the final rolling image
without materializing `[Q,H,W,3]`. Mixed fast/fallback forward rendering now
has a first production-shaped scheduler:

```text
split_projective_trace_cell_atlas_fallback_cells(...)
projective_trace_cell_atlas_fallback_tile_sample_mask(...)
render_projective_trace_cell_atlas_quadrature_interval_mixed_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal(...)
```

It renders non-fallback cells with interval Metal, renders the full active list
for marked fallback tile/sample regions with live-depth reference ordering, and
patches whole regions before exposure/rolling accumulation. Trainer state now
also exposes `fallback_render_mode="mixed"`, which keeps non-fallback regions
on native interval Metal autograd and patches fallback regions with a
differentiable live-depth Torch reference. The first-class STAR UVT feature
trainer now rejects `projective_interval.enabled` unless a real
`ProjectiveTraceCellTraceAtlas` producer is explicit, preventing silent fallback
to the old affine feature-tube renderer. The first compatible-tube producer now
exists: `uvt_tubes_to_projective_trace_cell_atlas(...)` completes the UVT
quadratic, extracts the moving center, lowers exact isotropic affine tubes into
cell-polynomial atlas rows, and then runs support/visibility event compilation.
`make_projective_cell_interval_trainer_state_from_uvt_tubes(...)` wraps that
producer and constructs the trainer state directly from UVT tensors.
It intentionally rejects anisotropy and pixel-varying depth. Continuous
temporal opacity is now represented in the atlas as `opacity_time_coeffs` and
used by both the CPU/Torch reference path and the interval Metal forward,
row-weighted forward, and direct backward paths. That producer is now routed
through the real STAR UVT feature trainer for the exact RGB-width route: the
trainer pins spatial precision to `sigma_px`, keeps temporal/motion/opacity/
feature gradients live, renders feature color through the interval atlas, and
renders a second white-trace atlas for total alpha. The route now has an
explicit metadata-cache policy: `refresh_policy="cadence"` preserves fixed
full-atlas rebuild cadence, while `refresh_policy="measured"` reuses compiled
cells across steps and updates only live differentiable tensors. Cached live
updates call the trainer-state refresh oracle, so stale support/order/fallback/
budget metadata is repaired before Metal rendering. A controlled
optimizer-style MPS gate now covers actual stale support rebin across four
measured-cache update steps. A real synthetic `run_training` A/B smoke now also
shows measured mode skips the cadence rebuild while matching the cadence loss
curve. The saved 8-step cache-policy artifact lives at
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md`:
measured reduces full atlas rebuilds `4 -> 1`, keeps final loss identical, and
cuts no-first-step mean time by about `1336 ms`, while still rebinnning support
on every live update. The next Metal gate is reducing that metadata churn under
ordinary tube motion; after that, extend the anisotropic footprint/pixel-depth
representation and, if needed, add a row-compacted launch that avoids scanning
all unique times for every row.
The production trainer bridge now starts at
`src/train/star_uvt_projective_interval_backend.py`, where
`feature_uvt.projective_interval` is normalized into refresh, budget, and
fallback policy and can instantiate `ProjectiveCellIntervalTrainerState` from a
compiled atlas.
`support_guard_policy="slack_budgeted"` is the first support-event-distance
guard allocator: it keeps base-active traces in crowded guarded tiles and
spends spare slots on extra traces nearest to that tile boundary, rather than
primitive id order.
The aggregate cache-policy benchmark now has a verifier contract in
`projective_interval_cache_policy_benchmark.py`: all four slack-budgeted cap128
tail-epsilon artifacts verify with identical cadence/measured final loss, zero
overflow/fallback/visibility stratification, measured rebuild reduction,
tail-certified reuse with `support_stale_overshoot_epsilon=0`, and monotone
support-rebin counts across the epsilon bracket. Focused verifier tests passed
`9 passed in 0.14s`.
The exposure/rolling Metal bridge now also has a saved verifier artifact:
`outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.md`.
It checks finite-exposure lowering, rolling row-time deduplication, and mixed
fallback patching against CPU oracles. On this MPS machine the finite interval,
rolling row-weighted interval, finite mixed fallback, and rolling mixed fallback
Metal paths all match within `5.96e-8`; rolling uses `7` unique sample times for
`8` row samples.
The matching exposure/rolling backward artifact is
`outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.md`.
It verifies that final-image adjoints can be pushed to sample adjoints by
quadrature weights or `row_weights`, then accumulated through one interval-cell
Metal VJP. The finite and rolling VJPs match Torch autograd with max absolute
gradient error `1.43e-6` and max relative error `6.38e-7`; rolling keeps the
same `7/8` shared sample schedule. The verifier now recomputes the rolling
reuse ratio, validates positive sample image/adjoint support and nonzero
coeff/opacity/color reference gradients, checks Metal aggregate errors against
their subrows, and recomputes the summary.
The mixed fallback backward artifact is
`outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.md`.
It intentionally does not add a public raw-op "autograd" wrapper around the
forward-only interval Metal op. Instead the report composes the existing
trainer-harness interval Metal VJP for fast cells with live-depth Torch
reference gradients for marked fallback tile/sample regions, then applies
finite-exposure or rolling row weights. On this MPS run both finite and rolling
mixed fallback backward cases pass with fallback fraction `0.5`, max output
error `5.96e-8`, max gradient absolute error `2.15e-6`, max gradient relative
error `7.41e-7`, and rolling row-time reuse `11/12`.
The combined focused projective plus interval-gated trainer suite now passes
`119` tests.

## Test Matrix

```text
static camera / static point
small camera motion
orbit 30 degrees
orbit 180 degrees split into windows
near denominator boundary
rolling shutter orbit
crossing occluders
```

Promotion requires:

```text
Metal/Torch parity <= 1e-5 for Gate A
affine/quadratic fit residual falls with chart splits for Gate B
orbit chart count grows sublinearly with frame count for Gate C
visibility fallback stays below 20% on ordinary synthetic scenes for Gate D/E
native interval-gated q-UVT Metal parity passes for split chart segments
native interval-gated direct VJP matches masked single-tube references
interval-gated q-UVT bridge one-step color training smoke lowers MSE
packed projective atlas-cell Metal forward matches the CPU atlas oracle
packed projective atlas-cell direct VJP matches Torch autograd
packed projective atlas-cell coefficient update lowers rendered MSE
packed projective atlas interval entries grow sublinearly against dense per-frame project/bin pairs
interval-compressed projective cell Metal forward matches the CPU atlas oracle
current Metal slab expansion is measured against the interval renderer
coverage-report/rebin detects moved support cells and repairs them
trainer-harness refresh keeps Metal autograd valid after optimizer support motion
visibility-report/rebin detects depth-order flips and repairs them
trainer-state owns refresh cadence and preserves live tensors across render/backward
ambiguous-depth cells are fallback-marked and rejected unless fallback is allowed
fallback stats report fallback coverage and the CPU reference fallback sorts live depth
visibility-stratum splitting repairs crossing order without fallback
complexity/budget stats expose interval ratio, stratum count, and fallback fraction
refresh lifecycle can enforce the complexity budget before render/backward
continuous visibility event roots drive exact-root cell splits before fallback
continuous support/tile-boundary roots drive time-local support rebinning
continuous sensor-time partitions merge support, visibility, and exposure events
finite-exposure and rolling-shutter quadrature schedules lower from partitions
finite-exposure and rolling-shutter quadrature schedules render through a CPU/Torch oracle
quadrature schedules lower to sample-indexed interval atlases for the interval Metal renderer
rolling quadrature schedules batch unique sample times with a row-weight matrix
row-weighted rolling interval Metal kernel skips zero-weight sample/row pairs
mixed finite-exposure/rolling forward patches whole fallback tile/sample regions
trainer-state mixed fallback keeps differentiable reference gradients
production feature trainer rejects enabled projective interval backend without atlas producer
compatible STAR UVT tubes can now produce ProjectiveTraceCellTraceAtlas rows
temporal opacity envelopes are represented in the atlas trace payload
interval Metal consumes temporal opacity and returns `grad_opacity_time_coeffs`
feature_dim=3 STAR UVT trainer configs can route through the compatible projective interval producer
projective trainer route renders total alpha by a white-trace interval atlas and backprops through it
projective trainer route can reuse compiled cell metadata across steps via `refresh_every`
cached live atlas updates run measured support/order/fallback/budget refresh before render
tail-alpha image-error verifier locks certified tail reuse below omitted-alpha bound and rejects core-loss/overlapping-tail stale reuse; focused tests 7 passed in 9.09s and three saved artifacts verify
anisotropic tail-bound verifier locks rotated/summed SPD footprint support bounds and rejects core-loss reuse; focused tests 6 passed in 8.81s and two saved artifacts verify
bundle gauge-invariance verifier locks `pi_* Gamma^* rho` across ordinary-depth and log-depth fiber gauges on a revolving camera: saved max relative error 3.50e-13 with Jacobian, missing-Jacobian control >=0.600, monotone order preserved, orientation-reversing order rejected, row/order summaries recomputed, and stale row errors rejected; focused value+gradient bundle tests 21 passed in 6.45s and saved artifact verified
bundle gauge-gradient verifier locks primitive derivatives of `pi_* Gamma^* rho` across ordinary-depth and log-depth fiber gauges: saved max gradient relative error 2.33e-12 with Jacobian, missing-Jacobian gradient control >=0.592, finite-difference mean[0] relative error 1.42e-10, finite-difference consistency checked, and stale gradient summaries rejected; focused value+gradient bundle tests 21 passed in 6.45s and saved artifact verified
shared-work goal audit verifies saved orbit, trained high-motion, exposure/rolling quadrature, exposure/rolling backward, and mixed fallback backward artifacts before checking bandwidth/backward ratios: the restored default orbit artifact now uses 8/16/32/64 frames, keeps payload growth 1.0x vs per-frame 8.0x for a payload-growth ratio of 0.125, and verifies final fixed/per-frame ratios payload/trace/segment=0.0625, CPU=0.091, forward=0.117, backward=0.158; trained shared interval-entry growth <=1.462x vs per-frame >=9.852x, final trained entry ratio <=0.149, trace-count ratio=0.1, forward ratio<=0.266, backward ratio<=0.094, and shared/replay interval-entry growth ratio=0.148; verifier now recomputes summaries, rejects stale row/ratio evidence, requires rolling unique-time reuse below 1.0, all four forward Metal cases, both ordinary backward Metal cases, both mixed fallback backward cases, mixed fast/fallback coverage, explicit payload/trace/segment/entry-growth reuse, and saved-report/current-input agreement via `--verify-current-inputs`; trained+shared audit tests pass 33 passed in 6.20s, and `outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json` verifies by CLI; the old 4/8/16/32 orbit timing failure is quarantined under `outputs/benchmarks/2026-05-25_star_uvt_revolving_orbit_fixed_chart_scaling_current_timing_fail/`
camera-family gauge verifier extends the bundle math from one camera path to a one-parameter local family over `Q x Omega x T`: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_gauge/summary.json` verifies value invariance (max rel 8.42e-14), primitive-gradient invariance (max rel 2.40e-12), q-gradient invariance (1.60e-11), q finite-difference (1.49e-10), and missing-Jacobian controls for values/gradients/q-gradients
camera-family shared-work scaling verifier compares one local `Q x Omega x T` chart against replaying one `Omega x T` atlas per q sample: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_shared_work_scaling/summary.json` verifies family payload/chart growth stays 1.0x while per-q replay grows 16.0x, final payload ratio is 0.106, final chart ratio is 0.0625, and max family fit residual is 0.306px
2D camera-family gauge verifier extends the fiber-gauge derivative check to `Q2 x Omega x T`: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_gauge/summary.json` verifies value invariance (max rel 8.42e-14), primitive-gradient invariance (max rel 2.28e-12), `q_phase` gradient invariance (1.82e-11), `q_height` gradient invariance (1.10e-11), and both camera-coordinate finite-difference checks below 3.26e-10
2D camera-family shared-work scaling verifier compares one local `Q2 x Omega x T` chart against replaying one `Omega x T` atlas per q-pair: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_shared_work_scaling/summary.json` verifies family payload/chart growth stays 1.0x while per-q-pair replay grows 64.0x, final payload ratio is 0.0625, final chart ratio is 0.015625, and max family fit residual is 0.111px
2D camera-family Metal lowering verifier slices one shared `Q2 x Omega x T` coefficient table into existing `Omega x T` interval Metal atlases over a 5x5 q grid: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_lowering/summary.json` verifies 25 forward/backward Metal rows, nonzero image and coeff/opacity/color gradients, family/replay payload ratio 0.178, and peak slice/replay payload ratio 0.04; this is slice lowering, not native Q2/Qn Metal evaluation
2D camera-family Metal chain-rule verifier accumulates per-slice interval Metal VJPs back into one shared Q2 family adjoint: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_chain_rule/summary.json` verifies 25 forward/backward Metal rows, shared/replay gradient payload ratio 0.24, max finite-difference relative error 4.91e-05, and nonzero shared-family gradient support; this is shared-family backward accumulation over Metal slices, not native Q2/Qn Metal evaluation
2D camera-family materialized-batch verifier packs all 25 q-pair slices into one ordinary interval Metal atlas: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_materialized_batch/summary.json` verifies one forward/backward launch, image abs error 0.0 versus per-slice reference, shared-family gradient relative error 9.34e-08, forward/backward launch ratios 0.04, materialized/replay trace payload ratio 1.0, and true family/materialized trace payload ratio 0.178; this proves launch reuse while intentionally leaving native Q2/Qn family-coefficient Metal evaluation open
2D camera-family native eval/VJP verifier evaluates Q2 family trace coefficients directly inside Metal: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_eval/summary.json` verifies native `family_coeffs[N,9,B] @ q_basis[Q,B]` trace evaluation plus direct VJPs into both shared tensors, family/materialized coefficient payload ratio 0.24, family-plus-q/materialized coefficient payload ratio 0.5733333333333334, max value relative error 6.58e-08, max family-gradient relative error 5.72e-08, and max q-basis-gradient relative error 2.58e-07; this proves native family trace eval/VJP by itself, while the interval-forward/backward rows cover compositing and interval-cell VJP
2D camera-family native interval-forward verifier composites Q2 family traces directly inside the Metal interval renderer: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_forward/summary.json` verifies native forward rendering/compositing/visibility from shared `family_coeffs[N,9,B]` plus `q_basis[Q,B]`, family/materialized trace coefficient payload ratio 0.16615384615384615, full native-family forward/materialized trace payload ratio 0.4461538461538462, max image absolute error 0.0, max image relative error 0.0, and equal native/materialized image abs sums 1992.59228515625
2D camera-family native interval-backward verifier accumulates Q2 family trace VJPs directly inside the Metal interval renderer with compiled visibility/order held fixed: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward/summary.json` verifies native interval-cell backward into shared `family_coeffs[N,9,B]` plus `q_basis[Q,B]`, native-family/materialized-gradient payload ratio 0.2926315789473684, native family-coefficient/materialized-gradient payload ratio 0.11368421052631579, max family-gradient relative error 2.3355269149760716e-06, max q-basis-gradient relative error 8.51117079037067e-07, and nonzero family/q-basis gradient support
2D camera-family tile/order reuse verifier compresses stable sampled-Q tile/order metadata: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse/summary.json` verifies one local tile/order topology plus q-index applicability expands back to all 25 materialized q-pair cells, conservative family-union depth intervals preserve order with min gap 0.6033999919891357, materialized tile/order metadata grows 25.0x, shared topology metadata grows 1.0x, and shared/materialized metadata ratio is 0.11692307692307692; this is the stable-topology case, while the next row covers an order split
2D camera-family tile/order strata verifier compresses sampled-Q metadata when depth order changes across q: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata/summary.json` verifies 25 materialized q-pair cells compress into two certified topology strata, materialized metadata grows 25.0x, shared metadata grows 2.0x, shared/materialized metadata ratio is 0.15692307692307692, and minimum stratum union depth gap is 0.33200000002980246
2D camera-family active-set strata verifier compresses sampled-Q metadata when support/culling changes across q: `outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata/summary.json` verifies 25 materialized q-pair cells compress into three certified active-set topology strata, materialized metadata grows 25.0x, shared metadata grows 3.0x, shared/materialized metadata ratio is 0.19692307692307692, and minimum active-set union depth gap is 0.2630399994850159
real active-set distribution verifier measures checked-in high-motion compiled atlases instead of synthetic q-family strata: `outputs/benchmarks/2026-05-25_star_uvt_projective_real_active_set_distribution/summary.json` verifies three saved trained high-motion artifacts over 4/8/16 frames, nine trained-checkpoint atlas rows, all source videos present, all underlying verifiers passing, fallback-free rows, max cells per active-set group 3, max active-set-group/dense-tile-pair ratio 0.04009499860296172, and max cell/group ratio 1.3214953271028038
projective goal-progress audit now maps the active all-night objective onto verified artifacts instead of implying completion: `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json` verifies gauge invariance, gauge gradients, one- and two-parameter camera-family gauge, one- and two-parameter camera-family shared-work scaling, Q2 camera-family Metal slice lowering, Q2 camera-family Metal shared-backward chain rule, Q2 camera-family materialized single-launch batching, Q2 camera-family native trace eval/VJP, Q2 camera-family native interval forward compositing/visibility, Q2 camera-family native interval backward/VJP, Q2 camera-family stable-topology tile/order reuse, Q2 camera-family two-strata tile/order reuse, Q2 camera-family three-strata active-set reuse, real active-set distribution, interval-trainer smoke, real-video trainer smoke, real-video guarded-support matrix, real-video multiscene trainer matrix, five-source real-video multiscene extended functional matrix, real-video multiscene frame-scaling matrix, five-source real-video multiscene extended frame-scaling diagnostic, real-video multiscene quality tether, five-source real-video multiscene extended quality tether, real-video multiscene media tether, five-source real-video multiscene extended media tether, the Bq4 fresh-process median gate, the real-video acceptance envelope, the real-video timing-variance envelope, the real-video compiled-adjoint replacement artifact, and shared-work evidence; records `shared_work.current_input_errors = []`; proves thirty-four requirement rows; keeps `full_goal_completion` open for final completion-audit promotion rather than for a known missing local math/kernel row; the saved goal-progress artifact verifies with `--verify-current-inputs`; it now cross-checks Bq4 fresh-process status, post-warmup pair count, and medians across the acceptance and timing-variance envelopes so both summaries cannot drift independently; focused acceptance-envelope + timing-variance-envelope + goal-progress + Bq4 fresh-process tests pass 70 passed in 4.76s; focused multiscene/tether/guarded/audit tests, including the extended-frame-scaling diagnostic, timing breakdown, phase profile, render-forward residual, render-forward shape, and Bq4 traced-rerun/repeat-stability/sequence-order/policy-order/fresh-process reports, pass 142 passed in 8.17s; focused Q2 Metal lowering/chain-rule/materialized-batch/native-eval/native-interval-forward/native-interval-backward/tile-order-reuse/tile-order-strata/active-set-strata/real-active-set-distribution + goal-progress tests previously passed 106 passed in 8.30s; latest progress/gap/replacement/promotion suite passes 82 passed in 4.02s and the wider timing-protocol/frame-breadth/media/acceptance/compiled-adjoint/gap/promotion/goal-progress bundle passes 121 passed in 4.72s
exposure/rolling quadrature verifier locks rendered-field integration semantics: finite-exposure lowering matches CPU exactly, rolling batched row weights match rowwise CPU exactly with 7 unique sample times for 8 row samples, four available Metal paths match within 5.96e-8, and finite/rolling mixed fallback patches visibility_ambiguous_depth cells before accumulation; it now recomputes interval/dense ratios, fallback cell fractions, fallback tile/trace sample subsets, and Metal summary max/count; focused tests 11 passed in 34.79s and saved artifact verified
exposure/rolling backward verifier locks shared-adjoint semantics: final image adjoints map to sample adjoints by quadrature weights or row_weights, one interval-cell Metal VJP accumulates gradients, finite/rolling VJPs match Torch autograd with max abs grad error 1.43e-6 and max relative grad error 6.38e-7, rolling keeps 7 unique sample times for 8 row samples; it now recomputes the rolling reuse ratio, validates positive sample image/adjoint support and nonzero coeff/opacity/color reference gradients, checks Metal aggregate errors against their subrows, and recomputes the summary; focused tests 11 passed in 25.19s and saved artifact verified
mixed exposure/rolling fallback backward verifier locks differentiable fallback semantics: non-fallback cells use trainer-harness interval Metal VJP, visibility_ambiguous_depth tile/sample regions use live-depth Torch reference gradients, patched samples are accumulated by exposure or row weights, finite/rolling mixed cases both pass with max output error 5.96e-8, max abs grad error 2.15e-6, max relative grad error 7.41e-7, and rolling row-time reuse 11/12; focused tests 7 passed in 15.38s and saved artifact verified
synthetic `run_training` frame-scaling artifact keeps measured rebuilds 1/1/1 over 4/8/16 frames with matching cadence loss and zero overflow; verifier locks pass/loss/rebuild/live-update/staleness/no-overflow/no-fallback and synthetic timing-win contract, focused tests 6 passed in 11.58s, saved artifact verified
real-video high-motion `run_training` frame-scaling artifact keeps measured rebuilds 1/1/1 vs cadence 2/2/2 over 4/8/16 frames, matching end loss exactly and zero overflow; the current default artifact lives at `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling/summary.json`, reports measured/cadence no-first-step ratios 0.881/0.352/0.692, and still has support rebins on all measured live updates
guarded real-video high-motion trainer artifacts with slack_budgeted guards 0.25/0.5/1/2px + tail001 keep measured rebuilds 1/1/1, eliminate support rebins/stale refreshes 0/0/0, match cadence loss exactly, and keep zero overflow; guard0.25 is the smallest certified no-churn guard, and the 2026-05-25 rerun keeps every guarded measured/cadence no-first-step ratio below cadence with max 0.5895631229975254
aggregate guarded-support matrix `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json` verifies 5 artifacts, 15 measured rows, default measured support rebins 9, guarded measured support rebins 0, guarded stale refreshes 0, max guarded rebuild ratio 0.5, max guarded no-first-step ratio 0.5895631229975254; the aggregate verifies by CLI and is now a top-level audit row
real-video multiscene trainer matrix `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json` verifies three source-distinct checked-in segments, six cadence/measured rows, exact cadence-loss agreement, measured rebuild ratio 0.5, max measured/cadence no-first-step ratio 0.549583769671522, and zero measured support rebins, stale refreshes, overflow, fallback marks, and visibility stratifications; this is now a top-level audit row but still not broad real-scene quality acceptance
real-video multiscene extended functional matrix `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json` verifies five source-distinct checked-in segments, ten cadence/measured rows, max motion score 7.018424034118652, exact cadence-loss agreement, measured rebuild ratio 0.5, and zero measured support rebins, stale refreshes, overflow, fallback marks, and visibility stratifications; max measured/cadence no-first-step ratio is 1.50811535915855, so this is top-level functional broadening evidence, not a timing-win row and still not broad real-scene quality acceptance
real-video multiscene frame-scaling matrix `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json` verifies the same three source-distinct checked-in segments over 4/8/16 frames, 18 cadence/measured rows, frame-growth factor 4.0, exact cadence-loss agreement within 2.98e-8, measured rebuild ratio 0.5, measured rebuild growth 1.0, max measured/cadence no-first-step ratio 0.6901796551165242, max measured timing-growth/frame-growth ratio 0.4376975869236762, and zero measured support rebins, stale refreshes, overflow, fallback marks, and visibility stratifications; this is now a top-level audit row but still not broad real-scene quality acceptance
real-video multiscene extended frame-scaling diagnostic `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json` verifies the strict five-source frame-growth artifact failed only the two expected timing gates while preserving correctness/cache/support invariants: five distinct YouTube sources, 30 cadence/measured rows over 4/8/16 frames, frame-growth factor 4.0, exact cadence-loss agreement, measured rebuild ratio 0.5, measured rebuild growth 1.0, zero measured support rebins/stale refreshes/support tail/overshoot, and zero overflow/fallback/visibility stratification; max measured/cadence no-first-step ratio is 1.188933546093892 and max measured timing-growth/frame-growth ratio is 1.0009153415685994, so this is a top-level diagnostic/caveat row, not timing-win evidence and still not broad real-scene quality acceptance
real-video multiscene extended timing breakdown `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown/summary.json` decomposes that failed five-source source into 15 cadence/measured pairs: 3 no-first pairs exceed 1.0, only one normalized frame-growth scene exceeds 1.0 by 0.0009153415685994037, all failing pairs are cache/support clean, max rebuild ratio is 0.5, max loss delta is 0.0, and support/stale/fallback/overflow/visibility stratification remain zero; this narrows the next timing work to evaluation/noise/phase-shape rather than cache invalidation
real-video multiscene extended phase profile `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile/summary.json` reads saved per-step timings for the three no-first misses plus two growth endpoints: source/case no-first delta 0.0, max step ratio 1.188933546093892 on Bq4rmeIvJbs_seg_000 4f, max render-forward ratio 1.3566329017525305 on the same row, max backward ratio 1.0839184402497806 on Bq4rmeIvJbs_seg_000 16f, no-first dominant phases render_forward_ms:2 and colorize_loss_ms:1, all profiled rows cache/support clean; the immediate optimization target is Bq4 render-forward timing, not cache/support repair
real-video multiscene extended render-forward residual `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_residual/summary.json` compares all 15 saved cadence/measured case pairs and verifies all cadence/measured tile_stats are identical, all no-first misses preserve identical tile workload, max tile-stat delta is 0.0, max render-forward ratio and max render-forward-per-clipped-ref ratio are both 1.3566329017525305 on Bq4rmeIvJbs_seg_000 4f, and workload_explains_render_forward_miss_count is 0; this rules out saved candidate/support distribution as the render-forward miss explanation and points next at Bq4 render-forward substep instrumentation/replay
real-video multiscene extended render-forward shape `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape/summary.json` reads the saved per-step timings and verifies all three no-first misses are single-spike driven in render-forward and whole-step time; after dropping the largest positive render-forward delta, the worst no-first miss render ratio is 0.8418254365135661, max no-first render spread is 5.383083741915209, max no-first render spike delta is 728.0996670015156 ms, and chunk_traces_present_pair_count is 0, so substep attribution needs a traced Bq4 rerun rather than atlas math changes
real-video Bq4 traced spike rerun `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun/summary.json` reruns the Bq4 4f/16f cadence/measured spike-step cases with `trace_global_steps`, verifies all expected steps are traced and all traced chunks carry projective interval substep timing, keeps cache/support clean, records `traced_bq4_spike_reproduced=false`, and reports max measured/cadence no-first ratio 0.5785517503959672, max projective interval total ratio 1.2736600499593582, and max feature-state-update ratio 1.250134158419622; this moves the next timing target to repeat/stability and live-update feature-state-update phase cost, not fiber/chart theory changes
real-video Bq4 16f trace repeat stability `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability/summary.json` repeats the Bq4 16f cadence/measured traced pair three times, verifies all expected steps are traced, all chunks carry projective interval substep timing, and cache/support remains clean, and records no persistent bump: no_first_spike_reproduced_count 0, projective_total_bump_count 0, feature_state_update_bump_count 0, max no-first ratio 0.45165397508134686, max projective interval total ratio 0.9101288137358652, and max feature-state-update ratio 0.7882220153002857; this weakens the one-shot 16f feature-state-update concern and points remaining timing work at mixed-sequence/warm-state launch variance
real-video Bq4 trace sequence order `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order/summary.json` runs two repeats of mixed_4_to_16 and reverse_16_to_4, verifies all expected steps are traced and cache/support remains clean, records paired_16f_ratio_count 4 and no 16f no-first bump with max no-first ratio 0.45600195672964483, but finds order-sensitive substep variance: mixed_4_to_16 has max 16f projective-total ratio 0.9606946419165872 and feature-state ratio 1.0006466493572015, while reverse_16_to_4 has max 16f projective-total ratio 1.844612661591509 and feature-state ratio 1.73336471126077; this supports warm-state/launch-order variance as a substep profiling caveat, not a fiber/chart math failure
real-video Bq4 warmed trace policy order `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order/summary.json` warms the process with traced Bq4 4f/16f cadence/measured cases, then runs two repeats each of cadence_then_measured and measured_then_cadence 16f target pairs, verifies all expected steps are traced and cache/support remains clean, and records a warmed timing caveat: paired_ratio_count 4, no_first_bump_count 1, projective_total_bump_count 3, feature_state_update_bump_count 3, max no-first ratio 1.7836530508238704, max projective-total ratio 1.7184222253396344, and max feature-state-update ratio 1.9605903379413647; measured_then_cadence has measured in slot 0 and is worse, so this is policy/order/warm-state timing variance rather than a simple second-slot effect or atlas math failure
real-video Bq4 fresh-process trace isolation `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json` runs three isolated-process repeats over both 16f policy-order target pairs with warmup_discard_repeats=1, verifies fresh_process=true on every row, all expected steps traced, projective interval substep timing present, and cache/support clean; it records paired_ratio_count 6, no_first_bump_count 0, projective_total_bump_count 1, feature_state_update_bump_count 2, max no-first ratio 0.7087283466117477, max projective-total ratio 2.2454207580524894, and max feature-state-update ratio 1.2948922914387324; post-warmup median acceptance passes with status pass, post_warmup_pair_count 4, median no-first ratio 0.5645123618278631, median projective-total ratio 0.8356591487478802, and median feature-state-update ratio 0.846418513757801, so timing acceptance should use fresh-process medians/warmup-discard while retaining max-ratio outliers as caveats rather than new fiber/gauge math failures
real-video multiscene quality tether `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json` verifies saved case payloads from the source-distinct frame-scaling matrix: nine cadence/measured pairs, exact loss-curve and RGB-loss-curve agreement, exact end-loss/end-PSNR agreement, all required gradient-flow flags, min measured PSNR gain 0.02227306365966797, max loss-curve delta 0.0, and max end-PSNR delta 0.0; this is now a top-level audit row but still not broad real-scene quality acceptance
real-video multiscene extended quality tether `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_quality_tether/summary.json` verifies saved case payloads from the five-source extended functional matrix: five cadence/measured pairs over five distinct YouTube sources, exact loss-curve and RGB-loss-curve agreement, exact end-PSNR agreement, all required gradient-flow flags, min measured PSNR gain 0.04466235637664795, max loss-curve delta 0.0, and max end-PSNR delta 0.0; this is now a top-level audit row but still not broad real-scene quality acceptance
real-video multiscene media tether `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json` runs the actual contact-sheet media writer on the same source-distinct checked-in segment set at 8 frames: three cadence/measured media pairs, max contact-sheet pixel delta 0, matching PNG hashes, valid two-row target/pred layout, max artifact-MSE-vs-payload-loss delta 0.001525666420389149, min target/pred row stds 0.14441643529730494 / 0.07265247844694266, max final full-RGB loss delta 0.0, max loss-curve delta 0.0, min measured PSNR gain 0.04511058330535889, all required gradient-flow flags, max measured/cadence no-first-step ratio 0.9316588494614714, rebuild ratio 0.5, and zero overflow/fallback/visibility stratifications; this is now a top-level audit row but still not broad real-scene quality acceptance
real-video multiscene extended media tether `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether/summary.json` runs the actual contact-sheet media writer on the five-source extended set at 8 frames: five cadence/measured media pairs, max contact-sheet pixel delta 0, matching PNG hashes, valid two-row target/pred layout, max artifact-MSE-vs-payload-loss delta 0.001525666420389149, min target/pred row stds 0.14441643529730494 / 0.07178262974117959, max final full-RGB loss delta 0.0, max loss-curve delta 0.0, min measured PSNR gain 0.04466235637664795, all required gradient-flow flags, rebuild ratio 0.5, and zero overflow/fallback/visibility stratifications; max measured/cadence no-first-step ratio is 1.2065694734694634, so this is a top-level media/quality row, not timing-win evidence and still not broad real-scene quality acceptance
real-video acceptance envelope `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json` consolidates the functional trainer matrices, source-distinct frame scaling, five-source frame-scaling timing diagnostic, four-count frame-count breadth diagnostic, source-distinct/five-source/broad10 quality tethers, actual five-source plus broad10 media tethers, and Bq4 fresh-process median timing gate into one non-completion artifact: twelve underlying verifiers pass, broad quality distinct source count is 10, broad media distinct source count is 10, broad frame-count count is 4, functional/media scene count is 5, max support rebins and stale refreshes are 0, max rebuild ratio is 0.5, min quality PSNR gain is 0.02227306365966797, the expected strict timing failures are preserved, fresh-process post-warmup medians no-first/projective-total/feature-state-update are 0.5645123618278631/0.8356591487478802/0.846418513757801, no-first bump count is 0, `fresh_process_median_timing_win_claimed=true`, and `does_not_prove_completion=true`
real-video timing-variance envelope `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json` consolidates the strict five-source timing misses, render-forward residual/shape diagnostics, Bq4 traced rerun/repeat/sequence/policy-order probes, and fresh-process isolation into one non-completion artifact: nine underlying verifiers pass, source scene count is 5, strict failure count remains 2, all timing misses are cache/support clean, workload explains 0 render-forward misses, drop-spike render-forward ratio is 0.8418254365135661, traced Bq4 spike reproduced is false, fresh-process median acceptance is pass with median no-first/projective-total/feature-state-update ratios 0.5645123618278631/0.8356591487478802/0.846418513757801, `strict_timing_win_claimed=false`, and `does_not_prove_completion=true`
projective goal-completion gap report `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json` turns the open `full_goal_completion` row into a machine-checked evidence-gap contract instead of a vague caveat: it verifies goal-progress through current-input acceptance, verifies real-video acceptance-envelope, broad10 real-video trainer matrix, broad10 quality tether, broad10 media tether, real-video timing-variance-envelope, real-video timing-protocol acceptance, real-video compiled-adjoint replacement, and shared-work inputs; proves formal memory/audit, sublinear world-side proxy, broad real-scene quality acceptance, full compiled-adjoint trainer replacement, and timing acceptance protocol rows; records open_gap_ids `["full_goal_completion"]`, broad_quality_source_gap 0, broad_media_source_gap 0, broad_quality_frame_count_gap 0, strict_timing_failure_gap 0, timing_acceptance_gap 0, compiled_trainer_source_gap 0, and compiled_trainer_replacement_gap 0; preserves `completion_ready=false` plus `does_not_prove_completion=true` because it is the source input to the final promotion audit; verifies by CLI with `--verify-current-inputs`
projective goal-completion promotion audit `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json` is the authoritative completion artifact: it consumes the gap report, verifies current inputs, proves six objective rows (scope/key math, sensor-time trace compiler, sublinear non-pixel work, broad real-video acceptance, compiled-adjoint training, final promotion), and records `status=complete`, `completion_ready=true`, `is_goal_complete=true`, `does_not_prove_completion=false`, and `open_requirement_ids=[]`; it verifies by CLI with `--verify-current-inputs`; focused progress/gap/replacement/promotion tests pass 82 passed in 4.02s, and the wider timing-protocol/frame-breadth/media/acceptance/compiled-adjoint/gap/promotion/goal-progress bundle passes 121 passed in 4.72s
broad10 real-video trainer matrix `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json` runs the actual guarded projective-interval trainer on 10 distinct source videos at 8 frames: 20 cadence/measured rows, all source videos present, all rows pass, measured losses match cadence, max measured/cadence rebuild ratio 0.5, zero support rebins, zero stale refreshes, min/max motion score 0.5781455039978027/7.018424034118652, and max measured/cadence no-first-step ratio 1.9762875807881346; this closes the completion-gap trainer source-count gap while preserving the warm-state timing caveat
broad10 real-video quality tether `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json` reads the saved broad10 trainer cases and verifies 10 source-distinct cadence/measured quality pairs: all rows pass, all gradient flags are present, measured loss/RGB-loss curves match cadence within the explicit 2e-8 float32-tick tolerance, end PSNR matches cadence, every measured row improves PSNR, and min measured PSNR gain is 0.03675997257232666; this closes the completion-gap quality source-count gap
broad10 real-video media tether `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json` runs the actual contact-sheet writer on 10 distinct source videos at 8 frames: 10 cadence/measured media pairs, max contact-sheet pixel delta 0, matching PNG hashes, nontrivial target/pred rows, all gradient flags present, zero overflow/fallback/visibility stratification, rebuild ratio 0.5, max loss/RGB-loss curve delta 1.4901161193847656e-08, max final RGB-loss delta 2.9802322387695312e-08, and max final RGB-PSNR delta 5.960464477539062e-07 under explicit media scalar tolerances; this closes the completion-gap media source-count gap
real-video frame-count breadth diagnostic `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic/summary.json` accepts the 4-count multiscene frame-scaling matrix as breadth evidence rather than timing evidence: source scene count 3, source row count 24, frame counts 4/8/16/32, frame growth 8.0, all rows pass/loss-decrease/cache-support/fallback-free invariants, zero support rebins and stale refreshes, measured/cadence loss matching, max measured/cadence rebuild ratio 0.5, sublinear no-first growth ratio 0.22855493152192446, `strict_failed_only_expected_timing=true`, and `no_first_timing_win=false`; this closes the completion-gap frame-count coverage gap
real-video timing-protocol acceptance `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json` promotes fresh-process median timing with warmup discard as the accepted timing contract for the current projective interval evidence envelope: final_timing_protocol_accepted true, timing_acceptance_gap 0, broad quality/media/frame-count context passes, frame-count breadth passes, fresh no-first/projective-total/feature-state-update medians 0.5645123618278631/0.8356591487478802/0.846418513757801, strict warm-state failure count 2 is demoted to diagnostic caveat, strict timing win remains false, cache/support/workload invariants are clean, and the artifact explicitly does not prove full goal completion
real-video compiled-adjoint replacement `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json` proves the current practical trainer replacement: the source contract verifies the trainer chooses `_render_projective_interval_feature_tubes_autograd`, the harness uses `_ProjectiveCellIntervalBackward`, interval forward calls `render_projective_trace_cell_interval_atlas_metal`, interval backward calls `direct_backward_projective_trace_cell_interval_atlas_metal`, and visibility/tile membership are compiled constants; the saved artifact verifies 20 broad10 case payloads, all projective-interval main path, all RGB direct-loss autograd, all renderer gradient flags, forward/backward timing, measured cache reuse, zero fallback/support churn, 10 broad trainer sources, 10 broad quality/media sources, four frame-count points, and shared-work ratios below threshold; records final_compiled_adjoint_replacement_accepted true and compiled_trainer_replacement_gap 0; scope note: this is the practical direct-atomic RGB route backed by the compiled interval adjoint, not deterministic compact static-STAR promotion or full goal completion
trained high-motion checkpoint geometry compiles to fallback-free interval atlases with dense tile pairs 1542->6016 but interval entries 392->573
trained high-motion checkpoint timing compares shared interval to per-frame replay: at 16f 573 vs 3862 entries and 57.0/60.3ms vs 355.2/367.9ms forward/backward
larger 64px/128t trained high-motion timing: at 16f shared interval uses 1371 entries vs 9605 per-frame replay entries, with 469.7/303.3ms vs 802.0/1779.1ms forward/backward
larger 96px/256t cap256 trained high-motion timing: fixed benchmark tile env sync for non-default cap256; at 16f shared interval uses 2831 entries vs 20547 per-frame replay entries, with 117.1/169.9ms vs 1282.6/1810.5ms forward/backward
trained high-motion scaling verifier locks top-level config/frame fields, exact trained/per-frame frame coverage, ratio and summary recomputation, fallback-free trained rows, nonzero learned velocity, opacity bounds, timing-gradient signals, structural interval/trace wins across scales, and final-scale timing wins; trained+shared audit suite 33 passed in 6.20s, all three saved artifacts verify
revolving-camera fixed-chart frame growth keeps chart/trace counts constant while dense samples grow
revolving-camera fixed-chart backward densification keeps Metal VJP attached to the same chart params
measured revolving-camera fixed-chart artifact records count/payload/timing/autograd rows through 32 frames
revolving-camera fixed-chart scaling verifier locks constant fixed chart/trace/payload topology, row-level interval-ratio and CPU-compile consistency, zero fallback, slower interval growth than dense samples, fixed/per-frame CPU and Metal timing ratios below 0.5, nonzero direct Metal gradients into coeff/opacity/color/spatial precision, and nonzero autograd gradients into orbit screen-fiber q_uv/q_uvt terms; focused verifier 10 passed in 28.79s, saved artifact verified, orbit+verifier suite 24 passed in 126.68s
```
