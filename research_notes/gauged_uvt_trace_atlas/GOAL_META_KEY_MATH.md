# Goal, Meta Goals, Key Math

Date: 2026-05-24

This is the compact memory note for the Gauged UVT Trace Atlas thread. It is
the "do not lose the plot" version future agents should read before proposing
more UVT/WorldFoam camera-path work.

## Memory Contract

Keep these four anchors together:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The active objective is not complete until a real renderer path shows clean
forward/backward behavior and useful sublinear non-pixel world-side scaling on
a multi-frame orbit, rolling-shutter, or finite-exposure workload.

## Goal

Make revolving, rolling, finite-exposure, and otherwise complex known camera
programs first-class in STAR UVT / WorldFoam by compiling the world through the
camera program into a reusable sensor-time atlas.

Do not compile frames. Do not compile a video. Compile observation traces:

```text
world primitive -> camera-ray bundle -> fiber-integrated UVT trace
```

The target product is:

```text
many renders / shutter samples / rolling rows from a known or low-dimensional
camera path with amortized projection, binning, support, and visibility work
```

not arbitrary single-frame novel-view rendering.

## Meta Goals

- Keep the math richer than "affine UVT plus fallback"; fallback is a safety
  rail, not the theory.
- Treat current STAR UVT tensors as one local coordinate expression of a richer
  gauged object.
- Let gauges/projections carry camera complexity before splitting or falling
  back.
- Make a full orbit an atlas of event-certified gauge domains, not one global
  splat/tube.
- Preserve visibility as a stratified/order problem over sensor-time, not as a
  lost depth marginal.
- Make training a compiled adjoint problem eventually, but prove inference and
  trace correctness first.
- Keep WorldFoam in the same language: foam cells are world support regions
  pulled back through the same ray bundle.
- Demand falsifiable gates: projective trace parity, orbit chart residuals,
  support bounds, visibility fallback fraction, and renderer quality vs dense
  reference.

## Key Math

Sensor-time base:

```text
B = Omega x T
y = (u, v, tau)
```

Camera-ray bundle:

```text
pi: E_Gamma -> B
pi^{-1}(y) = F_y
```

Camera program into world spacetime:

```text
Gamma: E_Gamma -> M
M = R^3 x R
```

Local gauge / trivialization:

```text
chi_a: E_Gamma|C_a -> C_a x D_a
(y, z_a) = (u, v, tau, z_a)
```

`C_a` should be read as a gauge domain with validity certificates. It is not
only a fit window and not only a stable-sort patch. It certifies projection
regularity, trace error, support bounds, tile-time membership, depth/order
behavior, interval gates, and backward support.

Core invariant:

```text
UVT trace = pi_* Gamma^* world_primitive
bar_rho_i(y) = integral_{F_y} rho_i(Gamma(y,z)) dmu_y(z)
```

The invariant is now guarded by a small revolving-camera gauge test:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.md
```

It integrates the same spacetime Gaussian through an orbit camera in depth and
log-depth fiber gauges. Including the fiber-measure Jacobian gives
`max_rel_error = 3.50e-13`; omitting it gives at least `0.600` relative error.
So the phrase "UVT screen fiber" has a concrete implementation contract:
gauge transitions are allowed only with the correct measure transform and
monotone depth-order certificate. The verifier recomputes row/order summaries
and rejects stale row errors, missing Jacobian controls, bad gauge-order
derivatives, and invalid near/far/sample metadata.

The derivative contract is guarded too:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.md
```

For the same depth/log-depth gauges, primitive gradients for mean,
log-precision, and log-amplitude match to `2.33e-12` relative error when the
Jacobian is included. Omitting the Jacobian makes gradients wrong by at least
`0.592`, and a finite-difference check for `mean[0]` matches autograd to
`1.42e-10`. This is the clean-derivatives version of the fiber-gauge invariant.
The gradient verifier checks row gradient norms, finite-difference consistency,
value/gradient missing-Jacobian controls, and stale summary fields. Focused
value+gradient bundle tests pass `21 passed in 6.45s`, with both saved
artifacts verified by CLI.

The shared-work / bandwidth side is guarded by an aggregate audit:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.md
```

It verifies the underlying orbit and trained high-motion reports first, then
checks the ratios that map most directly to the active objective. Orbit fixed
payload growth is `1.0x` versus per-frame replay payload growth `8.0x`, so
the explicit fixed/replay payload-growth ratio is `0.125`; the restored default
orbit artifact now measures `8,16,32,64` frames and verifies with final
fixed/per-frame ratios of `0.0625` for payload/trace/segment, `0.091` for CPU
compile, `0.117` for forward, and `0.158` for backward.
Across three trained high-motion artifacts, shared interval-entry growth is at
most `1.462x`, per-frame replay entry growth is at least `9.852x`, final
shared/per-frame entry ratio is at most `0.148`, final trace-count ratio is
`0.1`, final forward ratio is at most `0.266`, final backward ratio is at most
`0.094`, and the shared/replay interval-entry growth ratio is `0.148`. The
audit now also includes the exposure/rolling forward, exposure/rolling
backward, and differentiable mixed-fallback backward verifier artifacts: it
requires their underlying verifiers to pass, rolling unique-time reuse below
`1.0`, all four forward Metal cases, both ordinary backward Metal cases, both
mixed-fallback backward cases, and the same value/gradient error thresholds as
their focused reports. It recomputes its summary from orbit/trained/exposure
rows and checks monotone frame counts, finite positive ratios, explicit
payload/trace/segment/entry-growth reuse, CPU/forward/backward thresholds,
mixed fast/fallback coverage, and the explicit
sublinear-backward-with-fallback theory contract. The combined trained-scaling
+ shared-work audit tests pass `33 passed in 6.20s`; the regenerated aggregate
artifact verifies by CLI, including current-input staleness rejection:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json \
  --verify-current-inputs
```

That command proves the saved aggregate report still matches the current
default input artifacts, not just its own internal row/summary consistency. The old
`4,8,16,32` orbit failure remains quarantined under
`outputs/benchmarks/2026-05-25_star_uvt_revolving_orbit_fixed_chart_scaling_current_timing_fail/`
as evidence that too-small timing probes can be dominated by MPS launch noise.

The current goal-progress audit is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.md
```

It loads and verifies the gauge-invariance, gauge-gradient, one- and
two-parameter camera-family gauge, one- and two-parameter camera-family
shared-work, Q2 camera-family Metal lowering, chain-rule backward,
single-launch materialized batch, native family trace eval/VJP, native family
interval forward, native family interval backward, stable Q2 tile/order reuse,
split-strata Q2 tile/order reuse, real active-set distribution,
interval-trainer, real-video trainer, real-video guarded-support matrix,
real-video multiscene trainer matrix, five-source real-video multiscene
extended functional matrix, real-video multiscene frame-scaling,
five-source extended frame-scaling diagnostic, real-video multiscene quality
tether, five-source real-video multiscene extended quality tether,
real-video multiscene media tether, five-source real-video multiscene
extended media tether, the real-video acceptance envelope, the real-video
timing-variance envelope, the real-video compiled-adjoint replacement
artifact, and shared-work artifacts, then maps the active objective into
requirement rows. Thirty-four
requirements are
currently proved: formal camera-path compiler contract, fiber-gauge trace
invariant, clean fiber derivatives, one-parameter local camera-family bundle
math over `Q x Omega x T`, two-parameter local camera-family bundle math over
`Q2 x Omega x T`, one-parameter camera-family shared metadata,
two-parameter camera-family shared metadata over `Q2 x Omega x T`,
two-parameter camera-family slice lowering into interval Metal,
two-parameter camera-family shared-backward chain-rule accumulation from
interval Metal VJPs, two-parameter camera-family single-launch materialized
Metal batching, two-parameter camera-family native Metal trace eval/VJP from
shared family coefficients plus q-basis values, two-parameter camera-family
native Metal interval forward rendering/compositing/visibility from shared
family coefficients plus q-basis values, two-parameter camera-family native
Metal interval backward/VJP into shared family coefficients and q-basis values
with compiled visibility/order held fixed, stable-topology Q2 tile/order
metadata reuse, two-strata Q2 tile/order metadata reuse for a depth-order
change, three-strata Q2 active-set metadata reuse for a support/culling
change, checked-in real-video active-set distribution, Metal time-shared
forward/backward, finite-exposure/rolling fallback, compiled-adjoint trainer
smoke, real-video trainer smoke, real-video guarded-support matrix,
real-video multiscene trainer matrix, five-source real-video extended
functional matrix, real-video multiscene frame-scaling matrix, five-source
extended frame-scaling diagnostic with expected timing failures, real-video
multiscene quality tether, five-source real-video multiscene extended quality
tether, real-video multiscene media tether, five-source real-video multiscene
extended media tether, real-video acceptance envelope, real-video
timing-variance envelope, real-video compiled-adjoint replacement, and
sublinear world-side work proxies. The
real-video acceptance envelope is intentionally not a completion claim: it
consolidates twelve underlying functional, frame-scaling,
frame-count-breadth, quality-tether, media-tether, and Bq4 fresh-process
reports; verifies five functional/media scenes, 10 broad quality/media
sources, and four real-video frame-count points; keeps support rebins and
stale refreshes at zero; keeps max rebuild ratio at `0.5`; preserves the
expected strict timing failures; records min quality PSNR gain
`0.02227306365966797`; passes the Bq4 fresh-process median timing gate with
post-warmup median ratios `0.5645123618278631`, `0.8356591487478802`, and
`0.846418513757801`; and stores `does_not_prove_completion=true`.
The timing-variance envelope is also intentionally not a completion claim: it
consolidates nine timing-diagnostic artifacts; preserves the two strict
five-source timing misses as expected timing failures; keeps every timing miss
cache/support clean; verifies workload changes explain zero render-forward
misses; records the traced Bq4 spike as unreproduced; and passes isolated
fresh-process median acceptance with median no-first ratio
`0.5645123618278631`, median projective-total ratio `0.8356591487478802`, and
median feature-state-update ratio `0.846418513757801`.
The local camera-family guard has max value relative error `8.42e-14`, max
primitive-gradient relative error `2.40e-12`, and q-gradient relative error
`1.60e-11`. The two-parameter camera-family guard has max value relative
error `8.42e-14`, max primitive-gradient relative error `2.28e-12`,
`q_phase` gradient relative error `1.82e-11`, `q_height` gradient relative
error `1.10e-11`, and both camera-coordinate finite-difference checks below
`3.26e-10`. The local camera-family shared-work guard compares one
`Q x Omega x T` chart against replaying one `Omega x T` chart per q sample:
shared payload growth is `1.0x`, per-q replay payload growth is `16.0x`,
final shared/replay payload ratio is `0.106`, final chart ratio is `0.0625`,
and the family fit residual is `0.306px`. The two-parameter shared-work guard
compares one `Q2 x Omega x T` chart against replaying one `Omega x T` chart
per q-pair: shared payload growth is `1.0x`, per-q-pair replay payload growth
is `64.0x`, final shared/replay payload ratio is `0.0625`, final chart ratio
is `0.015625`, and the family fit residual is `0.111px`. The Q2 Metal lowering
guard slices one shared `Q2 x Omega x T` coefficient table into the existing
ordinary `Omega x T` interval Metal path across a `5x5` q grid. It verifies
25 forward/backward Metal rows, nonzero image and coeff/opacity/color gradients,
family/replay payload ratio `0.178`, and peak slice/replay payload ratio
`0.04`. This is slice lowering, not native Q2 Metal evaluation. The Q2 Metal
chain-rule guard verifies the backward reuse step: per-slice interval Metal
coefficient VJPs accumulate into one shared `Q2 x Omega x T` family adjoint
with shared/replay gradient payload ratio `0.24`, max finite-difference
relative error `4.91e-05`, and nonzero shared-family gradient support. This is
shared-family chain-rule accumulation over Metal slices, not native Q2 Metal
evaluation. The Q2 materialized-batch guard packs all 25 q-pair slices into one
ordinary interval Metal atlas and verifies a single forward/backward launch
matches the per-slice reference exactly in image space, with max shared-family
gradient relative error `9.34e-08`. This proves launch reuse over a sampled
camera family, while intentionally preserving the materialized/replay trace
payload ratio at `1.0`; the family table would be `0.178x` of that payload,
which is why the next guard exercises the previously missing
shader-side contraction:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_eval/summary.json
```

It evaluates all `Q x N x S` homogeneous trace samples directly from
`family_coeffs[N,9,B]` and `q_basis[Q,B]`, and accumulates direct VJPs into
both shared family coefficients and q-basis values. The saved artifact reports
family/materialized coefficient payload ratio `0.24`,
family-plus-q/materialized coefficient payload ratio `0.5733333333333334`,
max value relative error `6.58e-08`, max family-gradient relative error
`5.72e-08`, and max q-basis-gradient relative error `2.58e-07`. This is native
family trace evaluation and VJP by itself; the interval guards below cover
compositing and interval-cell VJP. The forward interval compositor now has its
own native-family guard:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_forward/summary.json
```

It consumes `family_coeffs[N,9,B]` and `q_basis[Q,B]` directly inside the
Metal interval renderer over the same `5x5` q grid, derives per-q trace
coefficients shader-side, depth-sorts/composites through the interval-cell
path, and matches the materialized single-launch reference exactly in image
space. The saved artifact reports `100` batched frames, family/materialized
trace coefficient payload ratio `0.16615384615384615`, full native-family
forward/materialized trace payload ratio `0.4461538461538462`, max image
absolute error `0.0`, max image relative error `0.0`, and equal native versus
materialized image abs sums `1992.59228515625`. This proves native interval
forward rendering/compositing/visibility over family coefficients. The matching
native-family interval backward artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward/summary.json
```

accumulates interval-cell VJPs directly into `family_coeffs[N,9,B]` and
`q_basis[Q,B]`, with tile membership and depth order held as compiled
constants. It reports native-family/materialized-gradient payload ratio
`0.2926315789473684`, native family-coefficient/materialized-gradient payload
ratio `0.11368421052631579`, max family-gradient relative error
`2.3355269149760716e-06`, max q-basis-gradient relative error
`8.51117079037067e-07`, and nonzero family/q-basis gradient support.

The stable-topology Q2 tile/order metadata reuse artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse/summary.json
```

stores one local tile/order topology plus q-index applicability for all 25
q-pair cells, then expands it back to the materialized local topology. It uses
conservative family-union depth intervals as the order certificate, with min
union gap `0.6033999919891357`. Materialized tile/order metadata grows `25.0x`;
shared topology metadata grows `1.0x`; shared/materialized metadata ratio is
`0.11692307692307692`. This is the stable-order case, not the full solution
for q-family regions where active sets or depth order split into multiple
strata.

The split-strata Q2 tile/order metadata reuse artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata/summary.json
```

deliberately changes depth order across the q family and compresses 25
materialized q-pair cells into two topology strata. It verifies both strata
expand back to the materialized local topology, with conservative per-stratum
family-union depth certificates. Materialized metadata grows `25.0x`, shared
metadata grows `2.0x`, shared/materialized metadata ratio is
`0.15692307692307692`, and the minimum stratum union depth gap is
`0.33200000002980246`.

The active-set Q2 metadata reuse artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata/summary.json
```

deliberately changes support/culling topology across the q family and
compresses 25 materialized q-pair cells into three active-set topology strata.
It verifies every stratum expands back to the materialized local topology and
keeps a conservative union-depth order certificate. Materialized metadata grows
`25.0x`, shared metadata grows `3.0x`, shared/materialized metadata ratio is
`0.19692307692307692`, and the minimum active-set union depth gap is
`0.2630399994850159`.

The checked-in high-motion real active-set distribution artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_active_set_distribution/summary.json
```

aggregates three saved trained high-motion projective interval artifacts over
`4,8,16` frames. It verifies all underlying report verifiers pass, all source
videos exist, all nine trained-checkpoint rows are fallback-free, max cells per
active-set group is `3`, max active-set-group/dense-tile-pair ratio is
`0.04009499860296172`, and max cell/group ratio is `1.3214953271028038`.
This moves active-set topology evidence from synthetic q-family strata to real
compiled traces, without claiming broad real-scene quality acceptance.

The synthetic trainer
smoke runs the projective-interval `run_training` route over `4,8,16` frames;
measured live-cache rebuilds are `1,1,1` versus cadence `2,2,2`,
measured/cadence no-first-step ratios stay below `0.840`, and the max end-loss
delta is `2.98e-8`. The high-motion real-video trainer smoke uses the same
route on checked-in frames and keeps rebuilds `1,1,1` versus cadence `2,2,2`,
max measured/cadence no-first-step ratio `0.881`, and max end-loss delta
`0.0`. The source-distinct multiscene real-video trainer matrix now runs the
same guarded projective-interval trainer contract on three checked-in video
segments:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json
```

It covers three distinct YouTube sources and six rows, matches cadence losses
exactly, keeps measured rebuild ratio `0.5`, keeps max measured/cadence
no-first-step ratio `0.550`, and has zero measured support rebins, stale
refreshes, overflow, fallback marks, and visibility stratifications. This
broadened the trainer evidence beyond the single high-motion clip. The extended
functional matrix broadens the same contract to five checked-in sources:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json
```

It covers five distinct YouTube sources and ten rows, includes a high-motion
clip with motion score `7.018424034118652`, matches cadence losses exactly,
keeps measured rebuild ratio `0.5`, and has zero measured support rebins, stale
refreshes, overflow, fallback marks, and visibility stratifications. Its max
measured/cadence no-first-step ratio is `1.50811535915855`, so it is functional
broadening evidence rather than a timing-win row. The source-distinct
frame-scaling matrix adds the same guarded contract across three sources and
`4,8,16` frames:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json
```

It covers 18 cadence/measured rows, frame-growth factor `4.0`, max
measured/cadence no-first-step ratio `0.690`, max measured timing-growth versus
frame-growth ratio `0.438`, measured rebuild growth `1.0`, measured/cadence
rebuild ratio `0.5`, cadence-loss agreement within `2.98e-8`, and zero support
rebins, stale refreshes, overflow, fallback marks, and visibility
stratifications. The five-source extended frame-scaling diagnostic reads the
strict five-source frame-growth artifact as a caveat report:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json
```

It covers five distinct YouTube sources, 30 cadence/measured rows,
frame-growth factor `4.0`, exact cadence-loss agreement, measured/cadence
rebuild ratio `0.5`, measured rebuild growth `1.0`, zero support rebins, zero
stale refreshes, zero support tail/overshoot, and zero fallback/overflow/
visibility stratification. The strict source status remains failed, but only
for the expected timing gates: max measured/cadence no-first-step ratio
`1.188933546093892` and max timing-growth/frame-growth ratio
`1.0009153415685994`. Therefore this row proves correctness/cache/support
stability on the harder five-source frame-growth set, not a timing win. The
pair-level timing-breakdown report is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown/summary.json
```

It identifies `3/15` no-first measured/cadence pairs over `1.0` and only one
normalized frame-growth scene over `1.0`: `Iagm3K8QtFw_seg_000` at
`1.0009153415685994`. All timing-miss pairs still have rebuild ratio `0.5`,
loss delta `0.0`, and zero support rebins, stale refreshes, fallback,
overflow, and visibility stratification. Current belief: the five-source
timing miss is not cache invalidation or support churn; it is likely
evaluation cost, run-to-run timing variance, or a per-scene/frame phase shape.
The phase-profile report adds saved per-step timing evidence:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile/summary.json
```

It profiles the three no-first misses and the two growth endpoints. The case
step means match the source no-first rows exactly
(`max_source_case_no_first_abs_delta = 0.0`). The two Bq4 misses are render-forward dominated: max render
ratio `1.3566329017525305` at `4f` and render ratio `1.111793076402963` at
`16f`; the C8k `8f` miss is only `1.0249968931082667` on step time and is
dominated by `colorize_loss_ms`. All profiled rows remain cache/support clean
and loss-tethered.

The render-forward residual report is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_residual/summary.json
```

It tests the sharper hypothesis that the Bq4 render-forward miss is caused by
more candidate/support work. The result is negative: all 15 cadence/measured
pairs have identical saved `tile_stats`, the three no-first misses have
identical tile workload, max tile-stat delta is `0.0`, and
`workload_explains_render_forward_miss_count = 0`. The remaining positive
signal is per-work-unit render-forward latency: max render-forward ratio
`1.3566329017525305` and max render-forward-per-clipped-ref ratio
`1.3566329017525305`, both on `Bq4rmeIvJbs_seg_000` at `4f`. Current next
target: instrument/replay render-forward substeps for Bq4 before changing
atlas math.

The render-forward shape report is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape/summary.json
```

It tests persistence vs spike structure in the saved per-step timings. All
three no-first misses are single-spike driven in render-forward time and in
whole-step time. Dropping the largest positive render-forward delta sends the
worst no-first miss render ratio below 1 (`0.8418254365135661`), even though
the original max render ratio is `1.3566329017525305`. The saved source has no
chunk traces (`chunk_traces_present_pair_count = 0`), so the next math/Metal
move is not "add a richer chart" but "rerun with `trace_global_steps` on the
Bq4 spike steps and split interval-cache lookup / trace eval / compositing /
synchronization timing."

The Bq4 traced spike-step rerun is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun/summary.json
```

It reruns the Bq4 `4f` and `16f` cadence/measured spike-step cases with
`trace_global_steps`. All expected global steps are traced, every traced chunk
has projective interval substep timing, cache/support remains clean, and the
saved spike does not reproduce at the no-first-step level:
`traced_bq4_spike_reproduced = false`, with measured/cadence no-first ratios
`0.4538476088322886` and `0.5785517503959672`. The caveat is one traced
projective interval substep: projective interval measured/cadence total ratios
are `0.5054386427773483` at `4f` and `1.2736600499593582` at `16f`, with
feature-state-update ratios `0.44341185194975186` and `1.250134158419622`.
Current belief: the old saved Bq4 miss is a small-step timing spike, while
live-update feature-state-update cost still deserves repeat/stability profiling.

The repeat/stability follow-up is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability/summary.json
```

It repeats the Bq4 `16f` cadence/measured traced pair three times. All expected
steps are traced, all chunks have substep timings, and cache/support remains
clean. The prior one-shot `16f` feature-state-update bump does not persist in
this 16f-only schedule: `no_first_spike_reproduced_count = 0`,
`projective_total_bump_count = 0`, `feature_state_update_bump_count = 0`, with
max ratios `0.45165397508134686`, `0.9101288137358652`, and
`0.7882220153002857` respectively. Current belief sharpens: the timing caveat
is mixed-sequence/warm-state launch variance, not a persistent live-update
feature-state-update hotspot.

The sequence-order follow-up is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order/summary.json
```

It runs two repeats each of `mixed_4_to_16` and `reverse_16_to_4`. All expected
steps are traced and cache/support remains clean. No `16f` no-first spike
reproduces (`no_first_bump_count = 0`, max `16f` no-first ratio
`0.45600195672964483`), but substep timing is order-sensitive:
`mixed_4_to_16` max `16f` projective-total ratio is `0.9606946419165872` and
feature-state-update ratio is `1.0006466493572015`, while `reverse_16_to_4`
max `16f` projective-total ratio is `1.844612661591509` and feature-state-update
ratio is `1.73336471126077`. Current belief: this is not a renderer math
failure; it is launch/warm-state phase variance that can affect substep
profiling while preserving the no-first measured/cadence win.

The policy-order isolation follow-up is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order/summary.json
```

It warms the process with traced Bq4 `4f`/`16f` cadence/measured cases, then
runs `16f` target pairs in both `cadence_then_measured` and
`measured_then_cadence` orders. All expected steps are traced and cache/support
remains clean. It reproduces a warmed timing caveat:
`no_first_bump_count = 1`, `projective_total_bump_count = 3`, and
`feature_state_update_bump_count = 3` across four target pairs. The worst
`measured_then_cadence` row has measured first, not second, with no-first ratio
`1.7836530508238704`, projective-total ratio `1.7184222253396344`, and
feature-state-update ratio `1.9605903379413647`. So the caveat is not merely
"second slot is slower"; it is a policy/order/warm-state interaction. This is a
timing acceptance issue, not a reason to change the fiber/gauge formulation.

The fresh-process isolation follow-up is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
```

It runs isolated Python/MPS processes for the `16f` target cases. The saved
artifact now uses three repeats over `cadence_then_measured` and
`measured_then_cadence`, with `warmup_discard_repeats = 1` for the acceptance
view. All rows are marked fresh-process, expected global steps are traced,
projective interval substep timing is present, and cache/support remains clean.
Across all six pairs, no no-first bump remains (`max_no_first_ratio =
0.7087283466117477`), but max substep outliers remain: one projective-total
bump with max ratio `2.2454207580524894`, and two feature-state-update bumps
with max ratio `1.2948922914387324`. The median evidence is favorable:
all-pair medians are `0.6530516888499702` no-first,
`0.8356591487478802` projective total, and `0.7124745747568637` feature-state
update; after discarding repeat 0, timing acceptance reports status `pass`,
post-warmup pair count `4`, median no-first `0.5645123618278631`, median
projective total `0.8356591487478802`, and median feature-state update
`0.846418513757801`. Current belief: use fresh-process median/warmup-discard
for acceptance, while treating max-ratio substep spikes as a timing outlier
caveat rather than a fiber/gauge math failure.
The
quality-tether report reads the saved case payloads from
that same matrix:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json
```

It verifies nine cadence/measured pairs, exact loss-curve and RGB-loss-curve
agreement, exact end-loss and end-PSNR agreement, all required gradient-flow
flags, positive measured PSNR gains, min PSNR gain `0.02227306365966797`, and
zero max loss-curve/end-PSNR delta. This still does not prove broad real-scene
quality acceptance.
The extended quality-tether report reads the saved case payloads from the
five-source functional matrix:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_quality_tether/summary.json
```

It verifies five cadence/measured pairs across five distinct YouTube sources,
exact loss-curve and RGB-loss-curve agreement, exact end-PSNR agreement, all
required gradient-flow flags, positive measured PSNR gains, min PSNR gain
`0.04466235637664795`, max loss-curve delta `0.0`, and max end-PSNR delta
`0.0`. This extends the live-cache/cadence quality tether to the five-source
functional broadening artifact while still not proving broad real-scene
quality acceptance.
The media-tether report exercises the actual contact-sheet media writer:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json
```

It verifies three source-distinct cadence/measured media pairs with
pixel-identical contact sheets, matching PNG hashes, valid two-row target/pred
layout, artifact-derived target/pred MSE matching payload final RGB loss within
`0.001525666420389149`, nontrivial target/pred rows with min stds
`0.14441643529730494` / `0.07265247844694266`, matching final full-RGB media
loss and PSNR, max loss-curve delta `0.0`, min measured PSNR gain
`0.04511058330535889`, all required gradient-flow flags, max measured/cadence
no-first-step ratio `0.9316588494614714`, measured rebuild ratio `0.5`, and
zero overflow, fallback marks, and visibility stratifications. This still does
not prove broad real-scene quality acceptance.
The extended media-tether report runs the actual contact-sheet media writer on
the five-source extended set:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether/summary.json
```

It verifies five source-distinct cadence/measured media pairs with
pixel-identical contact sheets, matching PNG hashes, valid two-row target/pred
layout, artifact-derived target/pred MSE matching payload final RGB loss within
`0.001525666420389149`, nontrivial target/pred rows with min stds
`0.14441643529730494` / `0.07178262974117959`, matching final full-RGB media
loss and PSNR, max loss-curve delta `0.0`, min measured PSNR gain
`0.04466235637664795`, all required gradient-flow flags, measured rebuild
ratio `0.5`, and zero overflow, fallback marks, and visibility
stratifications. Its max measured/cadence no-first-step ratio is
`1.2065694734694634`, so it is five-source media/quality evidence, not a
timing-win row.
It intentionally keeps `full_goal_completion` open as a pre-final progress
artifact. The current gap audit now closes broad real-scene quality acceptance,
timing acceptance, and the practical compiled-adjoint trainer replacement; the
separate final completion audit below promotes that remaining row into an
accepted completion claim. The focused Q2 Metal lowering, chain-rule,
materialized-batch, native-eval, native-interval-forward,
native-interval-backward, tile-order-reuse, tile-order-strata,
active-set-strata, real-active-set-distribution, and goal-progress newly touched
tests pass `106 passed in 8.30s`; the current multiscene/tether/guarded/audit
focused verifier subset, including the extended-frame-scaling diagnostic,
timing-breakdown, phase-profile, render-forward-residual, render-forward-shape,
and Bq4 traced-rerun/repeat-stability/sequence-order/policy-order/fresh-process
reports, passes `142 passed in 8.17s`; the latest focused
acceptance-envelope + timing-variance-envelope + goal-progress + Bq4
fresh-process subset passes `70 passed in 4.76s` and now cross-checks Bq4
fresh-process status, post-warmup pair count, and medians across both top-level
timing envelopes; the refreshed progress/gap/replacement/promotion suite passes
`82 passed in 4.02s`, and the wider timing-protocol/frame-breadth/media/
acceptance/compiled-adjoint/gap/promotion/goal-progress bundle passes
`121 passed in 4.72s`; the wider projective
evidence matrix was last run before
the native-interval-forward/backward/tile-order rows at `151 passed, 8 skipped
in 4.62s`; and the saved
goal-progress artifact verifies by CLI with
`--verify-current-inputs`.

The current completion-gap contract is now explicit:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

It is a non-completion artifact that machine-checks evidence gaps instead of
letting the goal blur. It preserves the same four memory anchors, verifies the
goal-progress input through that report's own current-input acceptance mode,
also verifies the acceptance-envelope, timing-variance-envelope,
timing-protocol, shared-work, broad10 real-video trainer, broad10 quality,
broad10 media, and compiled-adjoint replacement inputs. It now reports five
proved rows:
`formal_goal_memory_and_audit`, `sublinear_world_side_work_proxy`,
`broad_real_scene_quality_acceptance`, `full_compiled_adjoint_trainer_replacement`,
and `timing_acceptance_protocol`. The concrete current gaps are
`broad_quality_source_gap=0`, `broad_media_source_gap=0`,
`broad_quality_frame_count_gap=0`, `strict_timing_failure_gap=0`,
`timing_acceptance_gap=0`, `compiled_trainer_source_gap=0`, and
`compiled_trainer_replacement_gap=0`. It still keeps
`completion_ready=false` and `does_not_prove_completion=true`, because the
top-level goal-progress audit intentionally keeps `full_goal_completion` open
until a final completion audit, not a gap-row audit, is accepted.

The final completion-promotion audit is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json
```

It consumes the current completion-gap report, verifies that report against
current inputs, and promotes the lower non-completion stack into six proved
objective rows: scope/key-math preservation, sensor-time trace compiler
evidence, sublinear non-pixel work evidence, broad real-video acceptance,
compiled-adjoint training evidence, and final completion promotion. It records
`status=complete`, `completion_ready=true`, `is_goal_complete=true`,
`does_not_prove_completion=false`, and `open_requirement_ids=[]`.
The broad10 trainer matrix:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json
```

verifies 10 distinct source videos and 20 cadence/measured trainer rows through
the actual guarded projective-interval trainer path, with all rows passing,
matched cadence loss, max rebuild ratio `0.5`, zero support rebins, and zero
stale refreshes. Its max no-first timing ratio is `1.9762875807881346`, so it
is broad trainer-correctness/source evidence, not timing-win or quality/media
acceptance evidence.
The broad10 quality tether:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json
```

verifies 10 source-distinct cadence/measured quality pairs from the broad10
trainer payloads. It preserves all gradient-flow flags, positive measured PSNR
gain on every pair, min PSNR gain `0.03675997257232666`, end loss/PSNR deltas
`0.0`, and max loss/RGB-loss curve deltas `1.4901161193847656e-08` under the
explicit `2.0e-08` float32-tick tolerance. This closes the quality source-count
gap but is still quality-only.
The broad10 media tether:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json
```

verifies 10 source-distinct cadence/measured media pairs through the actual
contact-sheet writer. It preserves pixel-identical contact sheets and matching
PNG hashes, valid nontrivial target/pred rows, all gradient-flow flags, zero
overflow/fallback/visibility stratification, rebuild ratio `0.5`, min measured
PSNR gain `0.03675997257232666`, max loss/RGB-loss curve delta
`1.4901161193847656e-08`, max final RGB-loss delta
`2.9802322387695312e-08`, and max final RGB-PSNR delta
`5.960464477539062e-07` under explicit media scalar tolerances. This closes the
broad media source-count gap.
The frame-count breadth diagnostic:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic/summary.json
```

accepts the 4-count multiscene frame-scaling source artifact as breadth
evidence, not timing evidence. The source covers 3 distinct real-video
segments, 24 rows, four frame counts `4,8,16,32`, and `8.0x` frame growth. It
keeps all rows pass/loss-decrease/cache-support/fallback-free invariants,
measured/cadence loss matching, max measured/cadence rebuild ratio `0.5`, zero
support rebins, zero stale refreshes, and sublinear no-first growth versus
frame growth (`0.22855493152192446`). The source artifact itself remains
strict-timing failed, and the diagnostic explicitly records
`strict_failed_only_expected_timing=true` and `no_first_timing_win=false`.
This closes the fourth frame-count coverage gap while preserving strict timing
as a separate protocol decision rather than a frame-count breadth claim.
The timing-protocol acceptance artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json
```

promotes fresh-process median timing with warmup discard as the accepted timing
contract for the current evidence envelope. It requires the acceptance envelope
to keep 10 broad quality sources, 10 broad media sources, four frame-count
points, zero support rebins, zero stale refreshes, matching quality/media
tethers, and passing functional rows. It also requires the timing-variance
envelope to keep fresh-process status `pass`, at least four post-warmup pairs,
median no-first/projective-total/feature-state-update ratios at or below `1.0`,
strict misses limited to expected timing failures, cache/support-clean misses,
zero workload-explained render-forward misses, and no strict timing-win claim.
The saved artifact records `final_timing_protocol_accepted=true`,
`timing_acceptance_gap=0`, strict warm-state failure count `2` demoted to
diagnostic caveat, and medians
`0.5645123618278631` / `0.8356591487478802` / `0.846418513757801`.
The compiled-adjoint replacement artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json
```

proves the current practical trainer replacement. It ties the broad10 trainer
cases, broad10 quality/media tethers, acceptance envelope, timing protocol,
and shared-work audit to a source-level contract: the trainer selects
`_render_projective_interval_feature_tubes_autograd`, the trainer harness wraps
the interval atlas in `_ProjectiveCellIntervalBackward`, forward calls
`render_projective_trace_cell_interval_atlas_metal`, backward calls
`direct_backward_projective_trace_cell_interval_atlas_metal`, and visibility
order plus tile membership are compiled constants. The report verifies 20
case payloads, all using the projective interval main path, all RGB direct-loss
autograd cases, all renderer gradient flags present, positive forward/backward
timing, measured cache reuse, zero fallback/support churn, 10 broad trainer
sources, 10 broad quality/media sources, four frame-count points, and shared
work ratios below threshold. It records
`final_compiled_adjoint_replacement_accepted=true` and
`compiled_trainer_replacement_gap=0`. Scope note: this is not deterministic
compact static-STAR promotion; it is the practical direct-atomic RGB trainer
route backed by the compiled interval Metal adjoint.
This closes timing acceptance and compiled-adjoint replacement inside the
completion-gap contract. The gap artifact remains deliberately
non-completion-scoped, but the separate completion-promotion audit now consumes
it and records `completion_ready=true`, `is_goal_complete=true`, and
`does_not_prove_completion=false`. The saved gap and promotion artifacts both
verify by CLI with `--verify-current-inputs`. Focused progress/gap/replacement/
promotion tests pass `82 passed in 4.02s`; the wider timing-protocol/
frame-breadth/media/acceptance/compiled-adjoint/gap/promotion/goal-progress
bundle passes `121 passed in 4.72s`.

Finite-exposure and rolling-shutter evaluation are guarded by a focused
quadrature artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.md
```

It locks the rule that a shutter image is an integral of the rendered
sensor-time field:

```text
I_frame(u,v) = integral_tau Composite(K, u, v, tau) d tau
```

not a pre-visibility opacity integral. The finite-exposure lowering matches the
CPU oracle exactly, rolling shutter stores a row-weight matrix over
deduplicated sample times (`unique_to_row_sample_ratio = 0.875`), and the four
available Metal cases match the CPU oracle within `5.96e-8`. The mixed
finite/rolling cases mark `visibility_ambiguous_depth` cells and patch those
tile/sample regions through live-depth fallback before the exposure/row
accumulation. The strict verifier now recomputes interval/dense ratios,
fallback cell fractions, fallback tile/trace sample subsets, and the Metal
summary max/count; focused tests pass `11 passed in 34.79s`, and the saved
artifact verifies by CLI.

The corresponding finite-exposure and rolling-shutter adjoint contract is now
guarded too:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.md
```

The backward rule is:

```text
dL/d sample_image[q,row] = weight[q,row] * dL/d final_image[row]
```

where global shutter uses scalar quadrature weights and rolling shutter uses
the row-weight matrix. The report compares this shared sample-adjoint interval
VJP against Torch autograd on the lowered atlas. On this MPS machine the finite
and rolling Metal gradients match with max absolute error `1.43e-6` and max
relative error `6.38e-7`; rolling keeps the same deduplicated `7/8` sample
schedule as the forward report. The strict verifier now recomputes the rolling
reuse ratio, requires positive sample image/adjoint support, checks nonzero
coeff/opacity/color reference gradients, validates Metal aggregate errors
against their subrows, and recomputes the summary; focused tests pass
`11 passed in 25.19s`, and the saved artifact verifies by CLI.

Visibility-ambiguous finite-exposure and rolling-shutter fallback has its own
differentiable backward guard now:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.md
```

This is the current concrete answer to "is fallback mathematically ugly?" It is
an evaluator change on marked strata, not a break from the bundle trace. The
same lowered sensor-time atlas is used; non-fallback regions run through the
trainer-harness interval Metal VJP, while `visibility_ambiguous_depth`
tile/sample regions are patched with live-depth Torch reference gradients
before applying scalar exposure weights or rolling `row_weights`. On this MPS
machine both finite and rolling mixed backward cases pass with fallback
fraction `0.5`, max output error `5.96e-8`, max gradient absolute error
`2.15e-6`, and max gradient relative error `7.41e-7`; rolling still reuses a
deduplicated schedule (`11` unique times for `12` row samples in this focused
case). The focused report tests pass `7 passed in 15.38s`; the existing trainer
mixed-fallback tests pass `2 passed, 26 deselected in 33.43s`.

Current STAR UVT chart:

```text
ma, q_uvt, depth0, depth_beta, opacity, color_or_feature
```

is a local coordinate expression of this invariant trace, not the fundamental
representation.

Gaussian local chart:

```text
Gamma_a(y,z) ~= x0 + J_y delta_y + J_z delta_z
H = [J_y J_z]^T Lambda [J_y J_z]
S = H_yy - H_yz H_zz^{-1} H_zy
```

`S` is the UVT precision after fiber pushforward. Conditional depth/fiber mean:

```text
z_hat(y) = z0 + H_zz^{-1}(g_z - H_zy delta_y)
```

Projective/rational camera-time chart:

```text
h(t) = K(t) [R(t)|T(t)] X(t)
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
```

`h_z = 0` is a chart boundary, not a numerical annoyance.

Visibility strata:

```text
z_hat_i(y) = z_hat_j(y)
```

defines order-boundary surfaces. Unresolved order can be accepted only when the
visible swap bound is small:

```text
|Delta I_ij(y)| <= alpha_i(y) alpha_j(y) |c_i(y) - c_j(y)|
```

## Theory Commitments

- A full revolving camera is an event-certified gauge-domain cover:

```text
{C_a, chi_a, transition_ab}
```

not a single UVT Gaussian or one affine screen-time tube.

- Gauge choice comes before fallback. Try projective denominator, inverse-depth,
  ordinary-depth, object-local, or foam-local gauges before declaring a chart
  invalid.
- Residuals are certificates of chart validity. They are not the core method.
- Throwing away "charts" is only valid if the replacement still supplies event
  cells for denominator crossings, support birth/death, tile crossings,
  order swaps, and visibility ambiguity. Otherwise the renderer gives back
  projection/binning/order amortization.
- Chart transitions must preserve fiber/depth order when they are monotone:

```text
partial h_ab / partial z_a > 0
```

- Rolling shutter and finite exposure are base-domain integration problems:

```text
I_k(u,v) = integral w_k(u,v,tau) I(u,v,tau) d tau
```

Visibility must be evaluated before exposure integration unless order is proven
stable.

## Current Implementation State

Implemented and tested Gate A:

```text
projective_trace_eval(coeffs, times, eps) -> [N, S, 4]
```

where:

```text
coeffs = [u0,u1,u2, v0,v1,v2, z0,z1,z2]
out = [u, v, h_z, valid_sign]
```

This proves the first homogeneous/projective gauge primitive runs in Metal and
matches Torch. It does not yet fit footprints, build support bounds, or render.

Started Gate B:

```text
fit_projective_trace_polynomial(coeffs, times, degree)
    -> polynomial chart coefficients + residual/validity certificate
```

This is a CPU/Torch chart compiler helper, not a hot renderer path. It measures
whether a rational/projective trace can be represented by a local affine or
quadratic UVT chart over an orbit window, and reports denominator margin,
valid-sample fraction, UV residual, and depth residual.

Also added the first atlas-window splitter:

```text
split_projective_trace_windows(coeffs, times, degree, thresholds)
    -> accepted chart windows or unresolved fallback-candidate windows
```

This is still compiler-side Torch code. It is the first concrete object that
turns one long projective trace interval into a small list of valid local
charts.

The first synthetic orbit chart-count gate now passes on a stereographic
yaw-orbit trace. With `max_residual_uv = 0.015`, frame density
`F=16,32,64,128,256` produces `4,4,4,4,4` accepted windows on the same visible
span, while increasing orbit span at fixed `F=128` produces `1,2,3,7,7`
windows. Tightening residual at a fixed 90-degree span produces `2,6,8`
windows. The denominator certificate now evaluates the normalized quadratic on
the continuous interval, reporting a root/roundoff-boundary event through
`denominator_has_root` and the analytic `denominator_min_abs`. This catches
both roots between sampled frames and root-free between-sample minima that fall
below the requested chart margin.

Gate C now has a first compiler-side support-bound helper:

```text
bound_projective_trace_window(window)
bound_projective_trace_windows(windows)
    -> continuous polynomial chart bounds inflated by residual certificates
```

The focused test proves sampled rational orbit traces stay inside accepted
window UV/depth bounds, and unresolved denominator-boundary windows refuse
default bounds.

Gate D now has a first compiler-side visibility sidecar:

```text
make_projective_trace_visibility_sidecar(window)
compare_projective_trace_depth_order(sidecar_a, sidecar_b)
```

It records depth range, depth slope range, depth monotonic sign, depth
uncertainty, denominator margin/root flags, and chart gauge id. The focused
synthetic test proves stable front/back order is recognized and a pair of
crossing depth traces is marked as a visibility stratum / ambiguous order.

Gate D also has the first visible-swap bound:

```text
make_projective_trace_appearance_sidecar(alpha_max, color)
bound_projective_trace_visible_swap_cost(order, appearance_a, appearance_b)
```

It applies:

```text
|Delta I_ij| <= alpha_i alpha_j |c_i - c_j|
```

with optional color interval uncertainty. Ambiguous crossings below threshold
are marked safely commutable; visible crossings are marked `needs_fallback`.

Gate E now has a first CPU/Torch compiler-side tile-time binning prototype:

```text
bin_projective_trace_support_bounds(bounds, image_width, image_height, tile_size)
    -> compressed ProjectiveTraceTileTimeRecord list
```

It maps accepted support bounds into primitive/window/tile-rectangle/time-window
records, skips offscreen traces, preserves primitive ids, and carries an
optional fallback flag/reason from visibility masks. It is not renderer
integration yet.

Gate E also has the first tile-time atlas assembly object:

```text
assemble_projective_trace_tile_time_atlas(records)
    -> ProjectiveTraceTileTimeCell list
```

It expands compressed tile rectangles into tile-time cells with active primitive
sets, depth-sorted order metadata, depth intervals, fallback flags, and fallback
reasons.

The first dense-reference correctness gate now passes:

```text
tests/test_star_uvt_projective_correctness.py
```

It checks that projective atlas cells cover the same synthetic orbit samples as
dense per-frame projective projection/binning, and that stable atlas depth order
matches dense per-frame depth sorting.

The first minimal atlas evaluator gate now passes:

```text
render_projective_trace_tile_time_atlas_reference(...)
```

This CPU/Torch oracle composites ordered atlas candidates with a simple
screen-space Gaussian opacity model and matches dense per-frame compositing on a
small stable-depth scene. It is a correctness reference for future Metal
integration, not the hot path itself.

The first guarded STAR UVT bridge also passes:

```text
projective_trace_windows_to_uvt_tubes(...)
    -> ProjectiveTraceUVTBridge(ma, q_uvt, depth0, depth_beta, ...)
```

Accepted degree-1 projective chart windows lower exactly into the existing
STAR UVT affine tube contract. A focused test renders those lowered tubes with
`brute_force_render_uvt_tubes(...)` and matches the atlas reference renderer.
This proves affine camera-gauge charts can use the existing q-UVT renderer and
backward surface; nonlinear/projective charts still need atlas cells or a richer
Metal kernel.

The guarded MPS/Metal parity smoke for that bridge also passes. The focused
test lowers affine projective charts, renders them through
`render_uvt_tubes(...)` on MPS, and matches the CPU brute-force q-UVT reference.

The first explicit interval-gated split-chart bridge now passes:

```text
ProjectiveTraceUVTBridge.active_start / active_stop
render_projective_trace_uvt_bridge_reference(..., use_window_gates=True)
```

Split affine q-UVT segments carry exact sample-domain `[start, stop)` gates.
This is deliberately not encoded as temporal Gaussian precision, because soft
temporal precision would also attenuate valid chart endpoints. The focused test
uses a curved trace split into affine windows and proves gated q-UVT segments
match dense per-frame rendering while the ungated segments visibly leak.

Those explicit interval gates now reach the existing Metal renderer through a
span-gated wrapper:

```text
projective_trace_uvt_bridge_active_spans(...)
render_projective_trace_uvt_bridge_metal_gated(...)
```

The wrapper partitions the frame axis at all active interval boundaries, renders
each constant-active-set span through `render_uvt_tubes(...)`, and copies only
that span into the final image. This is not a native shader gate buffer yet, but
it proves the interval sidecar can drive Metal rendering without segment leaks.

Gate E now has a native shader-side interval gate:

```text
torch.ops.star_uvt_v0.render_gated(...)
render_uvt_tubes_gated(...)
```

It sends per-tube `[active_start, active_stop)` int32 buffers into the q-UVT
Metal renderer. `bin_screen_tubes_to_uvt_tiles_gated` clamps tile-time support
to each tube's active frame interval, and `render_uvt_tiles_gated` skips
inactive tubes per sample. The projective q-UVT bridge now uses this native
path, so split affine chart segments no longer need span-by-span external
render calls to avoid leakage.

The focused native interval gate passes:

```text
tests/test_star_uvt_projective_correctness.py::test_q_uvt_native_interval_gates_match_cpu_reference_if_available
tests/test_star_uvt_projective_correctness.py::test_projective_split_q_uvt_bridge_interval_gates_reach_metal_if_available
```

After the native forward gate, the full focused projective suite passed `34`
tests.

Gate E now has matching direct VJP interval coverage:

```text
torch.ops.star_uvt_v0.direct_atomic_backward_gated(...)
direct_atomic_backward_gated(...)
direct_backward_projective_trace_uvt_bridge_metal_gated(...)
```

It reuses the same `[active_start, active_stop)` interval buffers. The gated
backward bins only active tile-time support and skips inactive tubes during
per-sample VJP accumulation, so split chart segments do not receive gradients
outside their valid chart domains. The focused test compares the native gated
two-tube VJP against masked single-tube direct-backward references.

The first bridge-level trainability smoke now passes. It lowers split
projective chart windows into interval-gated q-UVT tubes, renders a target with
native gated Metal, computes an image MSE gradient, applies
`direct_backward_projective_trace_uvt_bridge_metal_gated(...)`, updates bridge
colors once, and verifies the loss drops. This is not full trainer integration,
but it proves the native interval-gated forward/backward bridge can drive a
real loss decrease without chart-domain leakage.

Gate F now has a first native nonlinear/projective atlas-cell renderer:

```text
pack_projective_trace_tile_time_bins(...)
torch.ops.star_uvt_v0.render_projective_trace_tiles(...)
render_projective_trace_tile_time_atlas_metal(...)
torch.ops.star_uvt_v0.direct_projective_trace_backward(...)
direct_backward_projective_trace_tile_time_atlas_metal(...)
```

The CPU compiler packs tile-time active sets into dense tile buffers, including
per-entry `[active_start, active_stop)` intervals so split chart windows do not
leak inside coarse tile-time groups. The Metal kernel evaluates the homogeneous
projective trace directly per sample:

```text
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
depth(t) = h_z(t)
```

and composites by per-sample projective depth. This is the first Metal path for
degree-2/rational projective atlas cells without lowering them into affine
q-UVT tubes.

The projective atlas-cell direct VJP now exists too. It treats tile membership
and visibility order as compiled constants, then differentiates the local
footprint:

```text
alpha = opacity exp(-0.5 ||p - (h_u/h_z, h_v/h_z)||^2 / sigma^2)
```

through color, opacity, and all nine homogeneous coefficients. The focused MPS
test matches the native direct VJP against Torch autograd on a quadratic chart.

The first projective atlas-cell coefficient trainability smoke now passes. It
renders a target from shifted homogeneous projective coefficients, keeps color
fixed, computes an MSE image gradient, runs the native direct VJP, applies a
small line-searched coefficient update, and verifies the Metal-rendered loss
drops. This proves the projective atlas VJP can move geometry-like camera-gauge
trace parameters, not only colors.

Gate F now also has a true cell-local gauge-domain evaluator:

```text
ProjectiveTraceCellTraceAtlas
projective_trace_windows_to_cell_trace_atlas(...)
eval_projective_trace_cell_torch(...)
torch.ops.star_uvt_v0.render_projective_trace_cell_tiles(...)
render_projective_trace_cell_atlas_metal(...)
```

Accepted windows lower into packed raw-time polynomial rows for:

```text
u(t), v(t), depth(t)
```

and tile-time cells index those row ids directly. This makes the accepted
gauge-domain cell itself the GPU-evaluable object, rather than only using the
atlas as a support/order wrapper around the original rational primitive.

Gate G now wires the interval-gated q-UVT path into the real source-view trainer
harness:

```text
render_uvt_tubes_metal_interval_gated_backward(...)
full_active_intervals(...)
uvt.render_backend = "metal_tile_interval_gated"
validate_uvt_backend_modes(...)
```

The current trainer integration uses full-video intervals as the degenerate
one-domain case for ordinary screen-time tubes, while preserving explicit
`active_start/active_stop` buffers for future projective/gauge-domain segment
producers. A real `src/train/train.py` smoke with
`metal_tile_interval_gated`, `index_add`, and `direct_atomic` selected the new
backend and decreased loss:

```text
0.15689826011657715 -> 0.12637178599834442
```

The combined focused projective plus interval-gated trainer suite passed:

```text
49 passed
```

Gate H now has a first frame-count scaling probe for packed projective atlas
cells:

```text
count_projective_trace_dense_per_frame_tile_pairs(...)
research_project/benchmarks/projective_atlas_scaling_probe.py
tests/test_star_uvt_projective_binning.py::test_projective_interval_packing_scales_sublinearly_over_frame_count
```

On the deterministic 45-degree orbit fixture, dense per-frame project/bin
entries grow `35 -> 555` from `4 -> 64` frames, while ideal interval-packed
atlas entries stay `13 -> 13`:

```text
frame_growth          16.0x
dense_pair_growth     15.857x
interval_entry_growth 1.0x
interval_pair_ratio   0.3714 -> 0.0234
```

The same probe also records the older Metal-compatible `tile_t=4` slab packing
as a limitation: slab entries grow `13 -> 208`, i.e. linearly with the number
of 4-frame slabs.

Gate I now adds a first interval-compressed Metal forward path for cell-local
projective traces:

```text
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_tiles(...)
render_projective_trace_cell_interval_atlas_metal(...)
```

This kernel dispatches over output pixel samples and indexes spatial tile bins
whose entries carry `[active_start, active_stop)` intervals, so it consumes the
interval atlas directly instead of re-expanding the entry into fixed temporal
slabs. The focused MPS parity test passes, and the 4-to-64-frame scaling probe
records matching image sums between slab and interval paths. On the tiny orbit
fixture, interval Metal wall time moves `24.8067ms -> 29.3612ms` over a `16x`
frame-count increase, while slab wall time moves `20.0995ms -> 37.2617ms`.

Gate I now also has interval-compressed direct VJP coverage:

```text
torch.ops.star_uvt_v0.direct_projective_trace_cell_interval_backward(...)
direct_backward_projective_trace_cell_interval_atlas_metal(...)
```

It consumes the same spatial tile bins with per-entry `[active_start,
active_stop)` intervals as the interval forward path. The focused MPS test
matches Torch autograd for color, opacity, and cell trace coefficients, and a
one-step coefficient trainability smoke renders a shifted-coefficient target,
applies the native interval VJP, line-searches coefficient updates, and verifies
loss drops with color fixed.

The combined focused projective plus interval-gated trainer suite now passes:

```text
49 passed
```

Gate J now adds the first trainer-harness bridge for nontrivial projective cell
intervals:

```text
render_projective_cell_interval_atlas_metal_backward(...)
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_wrapper_uses_split_windows_and_trains
```

The smoke builds split degree-1 projective windows, lowers them to
`ProjectiveTraceCellTraceAtlas` rows with multiple `[active_start, active_stop)`
domains, renders through the interval-compressed Metal forward path, backprops
through the interval-compressed direct VJP, runs `optimizer.step()` on cell
trace coefficients, and verifies loss drops. This is not yet the full
production STAR UVT trainer emitting projective chart segments, but it proves
the trainer-harness optimizer loop can consume a nontrivial gauge-domain
interval atlas directly.

Gate K now adds the first chart-lifecycle guard:

```text
projective_trace_cell_atlas_coverage_report(...)
projective_trace_cell_atlas_visibility_report(...)
rebin_projective_trace_cell_atlas(...)
mark_projective_trace_cell_visibility_fallbacks(...)
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_coverage_report_detects_motion_and_rebin_repairs
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_visibility_report_detects_depth_order_flip_and_rebin_repairs
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_visibility_ambiguity_marks_fallback_cells
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_visibility_stratifies_depth_crossing_without_fallback
refresh_projective_cell_interval_atlas_if_stale(...)
ProjectiveCellIntervalTrainerState
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_lifecycle_rebins_after_optimizer_motion
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_refresh_rebins_depth_order_without_replacing_tensor
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_state_owns_support_refresh_and_render
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_state_owns_depth_order_refresh
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_state_marks_ambiguous_visibility_fallback
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_refresh_stratifies_visibility_crossing_without_fallback
ProjectiveTraceCellAtlasFallbackStats
projective_trace_cell_atlas_fallback_stats(...)
stratify_projective_trace_cell_atlas_visibility(...)
ProjectiveTraceCellAtlasComplexityStats
ProjectiveTraceCellAtlasBudgetReport
projective_trace_cell_atlas_complexity_stats(...)
projective_trace_cell_atlas_budget_report(...)
ProjectiveTraceCellVisibilityEvent
ProjectiveTraceCellVisibilityEventReport
ProjectiveTraceCellSupportEvent
ProjectiveTraceCellSupportEventReport
ProjectiveTraceCellSensorTimeInterval
ProjectiveTraceCellSensorTimePartition
ProjectiveTraceCellSensorTimeQuadratureSample
ProjectiveTraceCellSensorTimeQuadrature
projective_trace_cell_support_event_report(...)
projective_trace_cell_visibility_event_report(...)
projective_trace_cell_sensor_time_event_partition(...)
projective_trace_cell_sensor_time_partition_quadrature(...)
projective_trace_cell_sensor_time_partition_rolling_quadrature(...)
rebin_projective_trace_cell_atlas_support_events(...)
stratify_projective_trace_cell_atlas_visibility_events(...)
ProjectiveCellIntervalAtlasRefresh.budget_after
ProjectiveCellIntervalTrainerState.enforce_complexity_budget
```

The coverage report evaluates live cell trace coefficients over their active
sample intervals and checks whether padded support still lands inside the
compiled tile-time cells. The rebin helper preserves the differentiable
coefficient/opacity/color tensors and rebuilds only support cells plus depth
interval metadata. The focused test moves a compiled cell trace into a new
screen tile, verifies the old atlas is reported stale, rebins the atlas, and
verifies coverage is repaired. The visibility report builds the same
per-sample/per-tile active order the renderer uses from compiled depth
intervals and compares it to live depths; the focused test flips front/back
depth without changing support, detects stale order, rebins, and verifies the
order is repaired. The trainer-harness refresh helper now wraps both lifecycle
checks for optimizer loops: after optimizer steps move either screen support or
depth order, the helper reports stale metadata, rebins without replacing the
coefficient tensor, and the Metal autograd smoke verifies gradients still flow
to that same tensor. `ProjectiveCellIntervalTrainerState` is the first
trainer-owned lifecycle surface: it stores the current atlas, render config,
times, refresh cadence, and last refresh report, exposes `render()`, and calls
support/order refresh from `after_optimizer_step()`. Focused tests prove it
repairs moved support on MPS before rendering/backprop and repairs a scheduled
depth-order flip on CPU without replacing the optimizer-owned tensor. Near-tie
visibility is now explicit fallback metadata: if rebin cannot remove ambiguity,
strict refresh raises, while `allow_ambiguous_fallback=True` marks affected
cells with `visibility_ambiguous_depth`; the Metal fast path still rejects
those cells. The CPU/Torch reference renderer now gives fallback cells their
intended semantics by sorting marked tile/sample regions with live evaluated
depth before compositing, while non-fallback regions keep compiled interval
order. `ProjectiveCellIntervalTrainerState` also exposes `fallback_stats()` and
`render_reference_with_fallback()` so fallback fraction is measurable before a
mixed Metal scheduler exists. Between rebin and fallback, the lifecycle now
tries visibility-stratum splitting: crossing order changes become smaller
stable time-run cells without replacing the live tensors. The atlas now reports
interval compression, visibility split count, fallback fraction, and named
budget failures so the trainer can tell a useful event-cell split from
visibility-complexity explosion. Refresh now returns `budget_after`, and
trainer state can opt into strict budget enforcement so an over-budget atlas
raises before the next render step. The support compiler now reports continuous
screen/tile boundary roots and refresh uses those roots to rebin moving traces
into time-local tile runs instead of one broad tile rectangle over the whole
active interval. The visibility compiler reports continuous cell-local roots
by solving `z_i(t)-z_j(t)=0` for affine or quadratic depth models, and refresh
prefers an event-root stratifier before the sampled stratum fallback. Exact
visibility roots that land on a frame sample are isolated as singleton cells,
so fallback can cover only the actual tie sample. Focused tests cover support
boundary roots at tile crossings, a support event rebin from `[0,4)` into
`[0,1), [1,3), [3,4)`, a visibility root at `5/3`, quadratic roots at `-1`
and `1`, stable no-event pairs, and the exact-root visibility split
`[0,1), [1,2), [2,4)`. The compiler also now has a continuous sensor-time
partition object that merges support roots, visibility roots, and caller-
supplied exposure/shutter split times into intervals independent of frame
indices; the focused fixture merges support roots at `1` and `3`, a visibility
root at `1.6`, and exposure endpoints at `0.5` and `2.5`. The combined focused
partition lowering now clips finite-exposure windows to event intervals and
emits normalized midpoint quadrature samples; the rolling-shutter helper applies
the same lowering per row with row-dependent readout offsets. Focused tests
cover a finite exposure `[0.25,2.75]` split over event cells with total weight
`1.0`, and a three-row rolling shutter schedule with shifted row windows. Those
quadrature schedules now feed a differentiable continuous-time CPU/Torch
reference renderer:

```text
render_projective_trace_cell_atlas_quadrature_reference(...)
render_projective_trace_cell_atlas_rolling_quadrature_reference(...)
```

The oracle evaluates direct cell traces at fractional sensor times, sorts by
live depth, composites, and accumulates sample weights. It also backprops
through trace coefficients, colors, and opacity in the focused test. The same
schedule can now lower into a sample-indexed interval atlas:

```text
ProjectiveTraceCellQuadratureLowering
lower_projective_trace_cell_atlas_quadrature(...)
render_projective_trace_cell_atlas_quadrature_interval_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_metal(...)
```

The lowering keeps trace coefficients as raw sensor-time functions, maps
quadrature samples to integer sample intervals for the existing interval Metal
kernel, and respects optional `domain_times` validity for split gauge rows.
Rolling shutter now has a batched schedule lowering too:

```text
ProjectiveTraceCellRollingQuadratureLowering
lower_projective_trace_cell_atlas_rolling_quadrature(...)
render_projective_trace_cell_atlas_rolling_quadrature_batched_reference(...)
```

It merges row schedules into unique sample times and a `row_weights[Q,H]`
matrix, so one interval render over unique times can feed all rows. This still
has a dedicated row-weighted Metal kernel now:

```text
render_projective_trace_cell_interval_atlas_rows_metal(...)
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_rows(...)
```

The kernel writes the final rolling image directly and skips zero-weight
sample/row pairs. It still loops over unique sample times per output pixel, so
a row-compacted launch can improve it later.
Mixed fallback forward rendering now exists:

```text
split_projective_trace_cell_atlas_fallback_cells(...)
projective_trace_cell_atlas_fallback_tile_sample_mask(...)
render_projective_trace_cell_atlas_quadrature_interval_mixed_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal(...)
```

The rule is semantic, not cosmetic: render non-fallback cells through interval
Metal, render the full active list for fallback tile/sample regions with
live-depth reference ordering, patch those whole regions, then apply exposure
or rolling row weights. This preserves alpha compositing under ambiguous
visibility; fallback is not an additive residual layer. The first mixed tests
cover both finite exposure and rolling shutter with a same-depth fallback tile
plus a separate fast tile.
`src/train/star_uvt_projective_interval_backend.py` now adds the first
production-facing bridge: `feature_uvt.projective_interval` config defaults and
validation, a `ProjectiveCellIntervalBackendConfig`, and a helper that builds
`ProjectiveCellIntervalTrainerState` from a compiled atlas, times, and trainer
config. It now also exposes
`make_projective_cell_interval_atlas_from_uvt_tubes(...)` and
`make_projective_cell_interval_trainer_state_from_uvt_tubes(...)`, which call
the lower-level `uvt_tubes_to_projective_trace_cell_atlas(...)` producer. That
producer completes the STAR UVT quadratic in the spatial variables, derives
the moving center as `ma_uv - A^{-1}b(t-ma_t)`, lowers exact compatible affine
UVT tubes into direct cell-polynomial rows, and then reuses support-event and
visibility-event atlas compilation. The combined focused projective plus
interval-gated trainer suite now passes:

```text
128 passed in 26.61s
```

The real STAR UVT feature trainer now has a narrow but real production route:
`feature_uvt.projective_interval.enabled=true` uses the compatible-tube
`ProjectiveTraceCellTraceAtlas` producer when `feature_dim=3`. The trainer pins
the screen-fiber spatial precision to the backend `sigma_px`, leaves temporal
precision/motion/opacity/feature gradients live, renders feature color through
the interval atlas, and renders a second white-trace atlas for total alpha so
the existing alpha-background/colorizer objective remains correct. The route is
full-frame only and requires `feature_target.image_vjp_mode=autograd`; configs
outside that contract still fail loudly instead of silently falling back to the
old affine renderer. It now has a first measured metadata-cache path: every
render rebuilds differentiable live trace tensors, but compiled
cells/support/order metadata can be reused across optimizer steps.
`projective_interval.refresh_policy="cadence"` keeps the old safe full-rebuild
cadence, while `refresh_policy="measured"` rebuilds only the first cached
compatible atlas and then lets the atlas refresh oracle decide whether metadata
must be repaired. On cached live updates that oracle checks coverage,
visibility/order, fallback, and complexity budget before rendering; if the
cached cells are stale, it rebins/stratifies/marks fallback while keeping live
tensors differentiable. The trainer now reports refresh policy plus
rebuild/live-update/staleness counters. The producer is
intentionally narrow for footprints and depth: by default it rejects
anisotropic spatial precision and pixel-varying depth slopes. Residual
temporal opacity is now represented in the atlas as
`opacity_time_coeffs=[k0,k1,k2]`, evaluated as
`exp(-0.5*(k0+k1*t+k2*t^2))` in the CPU/Torch reference path and in the
interval Metal forward/backward path. Native interval Metal now also returns
`grad_opacity_time_coeffs`, so temporal opacity is part of the fast trace VJP
instead of a reference-only truth.

The measured cache policy now has an optimizer-style stability gate: a tiny MPS
loop renders four cached measured-policy steps while SGD moves a live trace
center across a tile-support boundary. The gate verifies one full atlas build,
three live tensor updates, four alpha renders, three staleness checks, and one
support rebin without full rebuilding. This is still a controlled mechanism
gate, not a quality/timing promotion.

It now also has a real trainer A/B smoke on the ordinary synthetic
`run_training` route. With the same seed, target, four steps, and
`refresh_every=2`, cadence mode performs two full compatible-atlas rebuilds and
two live updates, while measured mode performs one rebuild and three live
updates. The measured loss curve and final loss match cadence within `1e-5`,
so reuse is behavior-preserving for this smoke.

The first saved cache-policy artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md
```

On the compatible 8f/64px full-frame route, cadence performs four full atlas
rebuilds and four live updates over eight steps; measured performs one full
rebuild and seven live updates. Final loss is identical (`0.0847767964`), and
no-first-step mean time improves by about `1336 ms` (`3473.3 -> 2137.2 ms`).
However, both policies report support rebin/stale refresh on every live update,
so the next optimization target is reducing metadata staleness churn under
ordinary tube motion.

The next pass adds a budgeted chart support guard:

```text
feature_uvt.projective_interval.support_guard_padding
```

The split is:

```text
coverage check support = uv_padding
compiled chart support = uv_padding + support_guard_padding
```

This is the concrete implementation of a gauge chart margin: correctness is
still judged against the actual trace footprint, but the compiled cell metadata
stores a conservative neighborhood so ordinary optimizer motion can remain
inside the same sensor-time chart.

Saved artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_cap256/summary.md
```

On the same compatible 8f/64px route, `support_guard_padding=2` with
`tile_capacity=256` keeps final loss identical (`0.0847767964`) and eliminates
stale refreshes/support rebins in both policies: cadence `4 -> 0`, measured
`7 -> 0`. Measured still keeps the full-atlas rebuild win (`4 -> 1`) and
improves no-first-step mean time `7468.6 -> 2496.7 ms`. The negative controls
matter: guard `2` and guard `8` at the old cap `128` overflow packed Metal
tiles. So chart guards are mathematically the right shape, but they must be
adaptive/budget-aware instead of globally inflated.

The first cap-aware implementation is:

```text
support_guard_policy = "budgeted"
```

It treats `support_guard_padding` as the maximum guard and searches downward
for the largest support padding that does not overflow the packed tile bins.
Saved artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_budgeted_cap128/summary.md
```

This avoids the fixed-guard cap128 overflow: the measured row passes, has zero
tile overflow, keeps final loss `0.0847767964`, and still uses one rebuild.
But it is not the desired final guard policy. Support rebins only move `7 -> 6`,
no-first-step mean slows to `6107.7 ms`, and the cadence row times out under
repeated global searches. This weakens the idea that a single global budgeted
guard is enough. The replacement model is local/headroom-aware guards:

```text
guard_i,C <= available_tile_headroom(C) under projected support of trace i
```

or a split/refit decision when a local guard would overfill the packed tile.

The first implemented local policy is:

```text
support_guard_policy = "local_budgeted"
```

It compiles the target guard, identifies only packed tiles that overflow, and
replaces those tile cells with the base-support atlas while preserving guarded
cells in tiles with headroom. Saved artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_local_budgeted_cap128_explicit/summary.md
```

This is cap-safe and cheaper than global bisection: both rows pass with final
loss `0.08477679640054703`, zero tile overflow (`max_tile_count=70` in the
case JSONs), and measured no-first-step `2468.3 ms`. It still does not solve
ordinary motion churn: measured support rebins remain `7/7`. Therefore the
next mathematical refinement is not "local tile or base tile"; it is
trace-local guard allocation inside crowded tiles, plus split/refit when
headroom is exhausted.

That refinement now exists as:

```text
support_guard_policy = "trace_budgeted"
```

It preserves base-active trace ids in overflowing tiles and spends remaining
tile slots on extra guarded trace ids. Saved artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_rerun/summary.md
```

This proves trace slot allocation is cap-safe, but not sufficient by itself:
both rows pass with final loss `0.08477679640054703`, zero overflow
(`max_tile_count=70`), measured no-first-step `2460.0 ms`, and measured
support rebins still `7/7`. The margin artifact then shows the stale boundary
is subpixel:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_margin/summary.md
```

For measured trace-budgeted cap128, max boundary overshoot was only
`0.0912 px`. This motivates a bounded support-staleness debounce:

```text
rebin iff max_boundary_overshoot_px > epsilon_support
```

Artifacts:

```text
epsilon 0.125 -> measured support rebins 3/7, max overshoot 0.1690 px
epsilon 0.25  -> measured support rebins 1/7, max overshoot 0.2986 px
epsilon 0.5   -> measured support rebins 0/7, max overshoot 0.4932 px
```

The `0.5px` row lives at:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_eps05/summary.md
```

It keeps final loss `0.08477679640054703`, zero overflow, and measured
no-first-step `1277.5 ms`. The stronger model is now:

```text
required_guard_i,C >= max_optimizer_displacement_i - tolerated_boundary_overshoot
```

subject to tile capacity and visual/error tolerance. Fixed guard2 at cap256
satisfies this exactly; cap128 can recover the churn win with a bounded
subpixel tolerance on this smoke.

The first image-level debounce contract is now:

```text
if support padding is a true footprint bound:
    0.05px boundary overshoot -> max RGB error < 1e-4
if support padding is only a center-tile marker:
    0.05px boundary overshoot -> max RGB error > 0.35
```

This matters because the epsilon is not intrinsically safe. It is safe only
relative to a certified support radius and alpha/error budget.

The first orbit-derived image-level contract now covers a tiny rational
revolving-camera trace as well. A small yaw-window trace compiles into one
tile, then a `0.10px` live coefficient update creates about `0.056px` padded
support overshoot across a tile boundary. Strict rebinning versus tolerant
reuse stays below `1e-4` max RGB error. This is a local chart certificate, not
a full-orbit guarantee, but it confirms that the debounce is compatible with
the projective/gauged camera-path formulation rather than only with an affine
UV toy.

The stronger certificate replaces pure pixel tolerance with a tail-alpha bound.
Let:

```text
r     = uv_padding
delta = support boundary overshoot
sigma = sigma_px
o_i   = opacity upper bound for trace i
```

For isotropic screen Gaussians, any omitted tile introduced only by this
support-boundary sliver has:

```text
alpha_omitted_i <= o_i * exp(-0.5 * (max(r - delta, 0) / sigma)^2)
```

The refresh path now accepts `support_stale_tail_alpha_epsilon`. If the maximum
omitted alpha bound is below that epsilon, support reuse is certified even when
the pixel overshoot would otherwise force a rebin. If `r=0`, the bound becomes
`o_i`, so center/core loss is still rejected. This is the mathematical version
of "gauge/projection carries the orbit, not arbitrary fallback."

The bound is now observable in artifacts:

```text
projective_interval_cache_last_support_tail_alpha_bound
projective_interval_cache_max_support_tail_alpha_bound
```

This closes the telemetry loop: measured refresh can be evaluated by both the
reuse outcome and the maximum omitted-alpha certificate.

The first max-per-trace cache-policy artifact using that loop was:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_tail001_cap128/summary.md
```

That artifact is now superseded. The broader certificate must aggregate
omitted tail alpha per missing sample/tile; taking a max over primitives is too
weak when many low-alpha tails overlap. The corrected aggregate artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.md
```

Settings:

```text
support_guard_policy = slack_budgeted
support_stale_tail_alpha_epsilon = 0.001
support_stale_overshoot_epsilon = 0.0
tile_capacity = 128
steps = 8
```

Measured versus cadence:

```text
end_loss:                       0.0847767964 == 0.0847767964
atlas rebuilds:                 1 vs 4
live updates:                   7 vs 4
stale refreshes / support bins: 0 / 0
max aggregate omitted-tail:     0.000736007 < 0.001
max support overshoot:          0.4932 px
tile overflow:                  0
```

Interpretation: the previous `0.5px` stale-support win still has a certificate,
but the corrected certificate is aggregate, not per primitive. The lower-budget
bracket is path-dependent:

```text
epsilon 0.00035:
    max aggregate omitted-tail bound = 0.000404648
    measured stale refreshes / support rebins = 2 / 2
    identical final loss, zero overflow

epsilon 0.00045:
    max aggregate omitted-tail bound = 0.000526049
    measured stale refreshes / support rebins = 1 / 1
    identical final loss, zero overflow

epsilon 0.0006:
    max aggregate omitted-tail bound = 0.000656625
    measured stale refreshes / support rebins = 1 / 1
    identical final loss, zero overflow

epsilon 0.001:
    max aggregate omitted-tail bound = 0.000736007
    measured stale refreshes / support rebins = 0 / 0
    identical final loss, zero overflow
```

Artifacts:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00045_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0006_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.md
```

The benchmark Markdown formatter now preserves sub-`1e-3` significant digits,
which matters because the certificate threshold lives exactly in that range.

The cache-policy artifact now has its own reusable report verifier:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00045_aggregate/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0006_aggregate/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.json
```

The report contract is:

```text
top level:
    support_guard_policy = slack_budgeted
    support_stale_overshoot_epsilon = 0
    support_stale_tail_alpha_epsilon > 0
    tile_capacity > 0

rows:
    exactly cadence and measured
    status = ok, pass = true, loss_decreased = true
    zero tile overflow
    zero visibility stratifications and fallback marks
    last_support_tail_alpha_bound in (0, epsilon]
    identical final loss across cadence and measured

amortization:
    measured rebuilds < cadence rebuilds
    measured live updates > cadence live updates
    measured no-first-step timing < cadence no-first-step timing
    if measured support_rebins = 0 then max_tail_bound <= epsilon
    if measured support_rebins > 0 then max_tail_bound > epsilon
```

Focused verifier tests passed `9 passed in 0.14s`. All four saved aggregate
artifacts verified through the CLI, and their epsilon bracket keeps the
expected monotone shape: looser tail budgets produce non-increasing support
rebin counts while the maximum observed tail bound grows.

The first image-level residual check is now:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.md
```

The aggregate image-error verifier is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.md
```

It keeps single-tail/orbit reuse below bound, rejects core loss, and adds a
64-trace overlap case: aggregate omitted-tail bound `0.01309515`; forced reuse
would produce `0.00141417` max RGB error. This is the first broader-scene
falsification loop for the certificate.

Positive cases compare strict rebinning against tail-certified reuse. Each max
RGB residual stays below the omitted-alpha bound:

```text
axis r4 sigma1 opacity0.5:    2.21e-5 <= 2.05e-4
axis r5 sigma1.25 opacity0.8: 5.50e-5 <= 3.15e-4
axis r6 sigma1.5 opacity0.9:  8.22e-5 <= 3.45e-4
orbit rational chart:         2.28e-5 <= 2.09e-4
```

The red-team case sets `uv_padding=0`, so the missing support is core, not
tail. The tail certificate returns `0.5` and rejects reuse under the `1e-3`
budget. A pixel-only `0.10px` stale-overshoot pardon would reuse and creates
`0.3987594` max RGB error. This is the current cleanest evidence that the
support gauge/projection math is doing useful certification work, not merely
renaming a fallback threshold.

The image-error verifier now exports
`verify_tail_alpha_image_error_report(...)` and
`assert_tail_alpha_image_error_report(...)`, plus a CLI-only validation mode:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.json
```

The report contract is intentionally stronger than `all_passed = true`:

```text
positive reuse:
    strict_rebinned = true
    certified_reused = true
    0 < support_tail_alpha_bound <= tail_alpha_epsilon
    max_abs_error <= 1.05 * support_tail_alpha_bound + 1e-7

negative controls:
    core_loss_rejected and overlapping_tail_aggregate_rejected exist
    certified_rebinned = true
    certified_reused = false
    support_tail_alpha_bound > tail_alpha_epsilon
    forced_bad_max_abs_error > tail_alpha_epsilon
```

Verification:

```text
focused tail-alpha verifier tests: 7 passed in 9.09s
base, tail00035 aggregate, and metal-precision-rerun saved artifacts: verified
```

The anisotropic extension is now specified and verified at the CPU/theory
level:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.md
```

For omitted tile rectangle `R`, local footprint center `mu`, SPD screen
precision `P`, opacity `o`, and RGB color with `||c||_inf <= 1`:

```text
max_{x in R} o exp(-0.5 (x-mu)^T P (x-mu)) ||c||_inf
= o exp(-0.5 min_{x in R} (x-mu)^T P (x-mu)) ||c||_inf.
```

For multiple traces omitted in the same tile, sum the per-trace bounds before
comparing to the support-tail budget. The verifier uses exact 2D convex
rectangle minimization by enumerating interior, edge-stationary, and corner
candidates. Results:

```text
diagonal anisotropic: 2.43e-5 <= 2.05e-4
rotated precision:    1.91e-5 <= 1.85e-4
two-trace sum:        1.66e-5 <= 2.29e-4
core loss rejected:   bound 0.5, error 0.4379515
```

The anisotropic report now exports
`verify_anisotropic_tail_bound_report(...)` and
`assert_anisotropic_tail_bound_report(...)`, plus CLI validation:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_metal_precision_rerun/summary.json
```

The contract is:

```text
positive anisotropic reuse:
    diagonal, rotated, and two-trace same-tile cases exist
    certified_reused = true
    0 < omitted_alpha_bound <= tail_alpha_epsilon
    max_abs_error <= 1.01 * omitted_alpha_bound + 1e-7
    omitted_tiles is non-empty
    two_trace_same_omitted_tile_sum bound exceeds each single-tail bound

negative control:
    anisotropic_core_loss_rejected exists
    certified_reused = false
    omitted_alpha_bound > tail_alpha_epsilon
    forced_bad_max_abs_error > 0.25
```

Verification:

```text
focused anisotropic tail-bound verifier tests: 6 passed in 8.81s
base and metal-precision-rerun saved artifacts: verified
```

Production implication: `sigma_px` is the scalar-isotropic special case
`P = sigma^{-2} I`. To make q-UVT / revolving-camera gauges first-class, the
compiled trace atlas should carry per-trace or per-cell `P_{uv}` and use this
rectangle certificate for support reuse. The current Metal path now renders
with per-trace `P_{uv}` when metadata is present, and the interval Metal VJP
now accumulates gradients into that precision because alpha depends smoothly on
the footprint metric.

The first metadata step now exists in code:

```text
ProjectiveTraceCellTraceAtlas.spatial_precision_uv: Tensor[N,3] | None
```

where each row is:

```text
(q_uu, q_uv, q_vv)
```

Contract:

```text
q_uu > 0
q_vv > 0
q_uu q_vv - q_uv^2 > 0
```

Atlas transforms preserve the field, and q-UVT lowering/live update fills it
from `q_uvt`. By default, q-UVT compatibility lowering still enforces the
legacy isotropic contract. The new opt-in path
`allow_anisotropic_spatial_precision` / `require_isotropic_spatial=False`
carries anisotropic precision and expands support using:

```text
R^2 = 2 log(alpha_peak / alpha_threshold)
P = [[q_uu, q_uv], [q_uv, q_vv]]
max |du| <= sqrt(R^2 (P^{-1})_00)
max |dv| <= sqrt(R^2 (P^{-1})_11)
```

Precision-only verification at that pass:

```text
precision focused tests: pass
broad projective/interval suite: 143 passed in 16.99s
reference cell render: consumes spatial_precision_uv
quadrature reference render: consumes spatial_precision_uv
stale-support certificate: consumes spatial_precision_uv
production interval Metal forward/backward: consumes spatial_precision_uv
q-UVT compatibility lowering: isotropic by default; anisotropic opt-in tested
scalar image-error rerun:
    outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.md
anisotropic bound rerun:
    outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_metal_precision_rerun/summary.md
```

The precision VJP is:

```text
r^2 = q_uu du^2 + 2 q_uv du dv + q_vv dv^2
alpha = opacity * exp(-0.5 r^2)

d alpha / d q_uu = -0.5 alpha du^2
d alpha / d q_uv = -alpha du dv
d alpha / d q_vv = -0.5 alpha dv^2
```

and the native direct backward returns `grad_spatial_precision_uv`. The
trainer-harness interval autograd route now passes `spatial_precision_uv` as a
real differentiable input when the atlas carries it, so anisotropic q-UVT
lowering can receive gradients back into `(q_uu,q_uv,q_vv)`.

The source-view trainer now makes this a real opt-in training route:

```text
allow_anisotropic_spatial_precision = False
    lock raw_precision[:,0:2] to sigma_px^{-2}
    mask spatial precision gradients

allow_anisotropic_spatial_precision = True
    skip the lock
    compile q-UVT with anisotropic support padding
    let interval Metal VJP update raw_precision[:,0:2]
```

Verification:

```text
locked/unlocked trainer bridge tests: 2 passed in 8.76s
broad projective/interval suite: 153 passed in 20.31s
```

## Implementation Update: Trainable Rotated UV Fiber Metric

The current pinned memory remains:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The q-UVT trace model now has a trainable UV cross precision term. This is the
first code-level answer to the revolving-camera/fiber-bundle concern: a screen
fiber footprint should not be restricted to axis-aligned ellipses in the local
camera gauge.

Let the local screen precision block be

```text
Q_uv = [[q_uu, q_uv],
        [q_uv, q_vv]].
```

The source-view tube model now parameterizes

```text
q_uu = softplus(raw_precision_u) + eps
q_vv = softplus(raw_precision_v) + eps
rho  = rho_max * tanh(raw_spatial_correlation), 0 <= rho_max < 1
q_uv = rho * sqrt(q_uu q_vv)
```

so

```text
det(Q_uv) = q_uu q_vv (1 - rho^2) > 0.
```

The coupled UVT precision still encodes the same center velocity:

```text
q_ut = -(q_uu v_u + q_uv v_v)
q_vt = -(q_uv v_u + q_vv v_v)
q_tt = q_t + q_uu v_u^2 + 2 q_uv v_u v_v + q_vv v_v^2
```

Recovering the velocity from the Schur block gives the original
`velocity_uv`, which is the invariance test for the gauge parameterization.

Training boundary:

```text
allow_anisotropic_spatial_precision = False
    zero raw_spatial_correlation
    mask raw_spatial_correlation gradients
    preserve legacy isotropic interval behavior

allow_anisotropic_spatial_precision = True
    keep the full SPD Q_uv
    compile anisotropic support padding
    propagate interval Metal VJP into q_uu, q_uv, q_vv
```

Verification:

```text
focused cross/locked/unlocked tests: 3 passed in 4.90s
py_compile: passed
broad projective/interval suite: 154 passed in 23.49s
```

The scalar rerun adds one more invariant: omitted tails sharing the same tile
and sample must be summed. In the overlapping-tail red-team, the aggregate
bound is `0.01309515`, so reuse is rejected under the `0.001` budget even
though each individual omitted tail is small. This is the right aggregation
law for alpha/error certificates over tile-time cells.

The depth companion contract is now:

```text
depth_affine_uv[N,6] = [zu0, zu1, zu2, zv0, zv1, zv2]

z(u,v,t) = z_c(t)
         + z_u(t) (u - u_c(t))
         + z_v(t) (v - v_c(t))
```

where `z_u,z_v` are quadratic time polynomials. This is the first tested
screen-fiber depth section over a local UVT gauge: the compiler can evaluate
conditional depth at a pixel, not only at a trace center.
`eval_projective_trace_cell_depth_at_uv_torch(...)` covers the CPU/Torch
certificate path, validation rejects malformed slope tensors, and support
rebinning, trainer refresh, quadrature lowering, and CPU detach preserve the
field.

The q-UVT producer now has an explicit opt-in for this screen-fiber depth
section:

```text
allow_depth_affine_uv = True
```

By default, compatible lowering still rejects nonzero `depth_beta[:,0:2]`.
When opted in, the producer maps the full q-UVT affine depth model

```text
z(u,v,t) = depth0
         + beta_u (u - ma_u)
         + beta_v (v - ma_v)
         + beta_t (t - ma_t)
```

into:

```text
z_c(t) slope = beta_t + beta_u velocity_u + beta_v velocity_v
depth_affine_uv = [beta_u,0,0,beta_v,0,0]
```

so the moving trace center keeps correct center depth while off-center pixels
retain the spatial depth plane. Live measured-cache atlas updates preserve and
recompute the same metadata when the reference atlas carries it.

Visibility certificates now consume this field over tile pixel ranges, not
only at trace centers. If affine depth can flip inside one tile, the visibility
report marks the tile/sample ambiguous, `mark_projective_trace_cell_visibility_fallbacks(...)`
can repair it by flagging fallback cells, and the CPU/Torch reference fallback
sorts by live per-pixel depth before compositing.

Interval Metal now also consumes `depth_affine_uv` in its dynamic per-pixel
selection sort:

```text
select_projective_cell_order_id_interval(...)
projective_cell_depth_at_pixel(...)
```

The hot path receives a dense zero tensor when the atlas has no depth plane, so
legacy scalar-depth atlases keep the same behavior. The native interval module
must be rebuilt after this ABI change.

The compiler now has a first explicit UV-order event certificate:

```text
ProjectiveTraceCellUVVisibilityEvent
ProjectiveTraceCellUVVisibilityEventReport
projective_trace_cell_uv_visibility_event_report(...)
```

At a fixed tile/sample and trace pair, it forms the affine depth-difference
line:

```text
Delta z(u,v,t_k) = a_u u + a_v v + a_0
```

and evaluates the line range on the tile's pixel-center rectangle. If the
range straddles zero, the report emits a UV visibility event with
`line_u,line_v,line_0,min_delta,max_delta`. Stable depth planes over the same
tile produce no event.

That certificate is now consumed by fallback marking:

```text
mark_projective_trace_cell_visibility_fallbacks(...)
    -> fallback_reasons include "visibility_uv_depth_line"
```

The first spatial split representation now exists:

```text
split_projective_trace_cell_atlas_uv_visibility_events(...)
```

It does not add oblique polygon cells. Instead, it retile-compiles a parent
atlas onto a finer child tile grid, then recomputes depth intervals and
front-to-back order for each child cell. In the canonical UV depth-line fixture,
one parent tile with an in-tile order line becomes two stable child cells:
left child order `(0,1)`, right child order `(1,0)`, no UV event, no fallback.
If the zero line still crosses a child tile, the same fallback marker keeps
`visibility_uv_depth_line`. So the concrete decision is now:

```text
cheap grid-refinement split first
otherwise explicit UV-depth-line fallback
```

The first adaptive policy/report now wraps that rule:

```text
adapt_projective_trace_cell_atlas_uv_visibility_events(...)
ProjectiveTraceCellUVVisibilitySpatialSplitReport
```

Because the current packed atlas/render path assumes one global tile size, the
policy cannot mix unsplit parent cells with child-grid event cells inside one
atlas yet. It instead keeps the parent atlas if no UV event exists; otherwise
it tries divisor child tile sizes from largest to smallest and accepts the
coarsest grid whose residual UV event-tile samples fit the requested budget.
If the minimum child grid still contains the zero line, the report is marked
`accepted=False` and unresolved child cells retain
`visibility_uv_depth_line` fallback. This makes the split-vs-fallback decision
measurable instead of implicit.

The report now includes the parent fallback baseline too:

```text
parent_fallback_cells
parent_fallback_fraction
fallback_cells
fallback_fraction
```

A first high-motion UV-line sweep fixture used two synthetic shutter samples.
The current report row is one step less synthetic: it decodes the checked-in
high-motion smoke video, computes adjacent-frame grayscale motion energy,
selects the strongest motion pairs, and uses their motion centroids as the UV
roots of a diagnostic pairwise depth line. This is still not trained scene
geometry, but it is parsed video-derived trace geometry rather than a hand-set
line sweep.

Current extracted row:

```text
video = data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
frames_read = 16
selected_pair_indices = (7,8,9)
root_positions_u ~= (4.395, 4.055, 4.123)
source = extracted_video_motion_centroid
```

At the parent grid, all three tile samples require UV-depth-line fallback:

```text
parent_uv_event_tile_samples = 3
parent_fallback_fraction = 1.0
```

Because the extracted roots cluster near the middle of the diagnostic tile, the
adaptive policy chooses child size `4` and recompiles two child cells with:

```text
residual_uv_event_tile_samples = 0
fallback_fraction = 0.0
```

This is a measured split-vs-fallback reduction driven by video motion
centroids. The next threshold is not another centroid proxy; it is extracting
the actual trainer/world trace geometry for the same video.

The same measurement now has an orbit-parameterized fixture. It uses

```text
q = tan(theta/2)
```

with the orbit trace depth polynomial from `_orbit_trace_coeffs(...)`, then
adds a moving pairwise depth offset so the UV order line crosses different
parent-tile positions over `q in {-0.5, 0, 0.5}`. The parent 8-pixel tile
again falls back for every orbit sample:

```text
parent_uv_event_tile_samples = 3
parent_fallback_fraction = 1.0
```

The adaptive policy tries child sizes `(4,2,1)`, chooses child size `2`, and
reduces the compiled child atlas to:

```text
residual_uv_event_tile_samples = 0
fallback_fraction = 0.0
```

This is not yet a real-scene orbit benchmark, but it ties the UV split report
to the revolving-camera gauge coordinate rather than to plain frame index.

The measurement is now a reusable report artifact builder:

```text
research_experiments/star_uvt_feature_tubes/projective_uv_visibility_split_report.py
outputs/projective_uv_visibility_split_report.json
schema_version = projective_uv_visibility_split_report_v1
```

It writes two before/after rows:

```text
high_motion_video_centroid_line_sweep:
    source = extracted_video_motion_centroid
    reference_video_path = data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
    selected_pair_indices = (7,8,9)
    parent_fallback_fraction = 1.0
    fallback_fraction = 0.0

orbit_parameterized_line_sweep:
    source = synthetic_orbit_q_tan_half_angle
    parent_fallback_fraction = 1.0
    fallback_fraction = 0.0
```

The report summary records `max_cell_growth = 4.0`,
`max_parent_fallback_fraction = 1.0`, `max_output_fallback_fraction = 0.0`,
and `any_needs_oblique_halfspace = false`. The high-motion row is parsed from
video, but only as a motion-centroid diagnostic; it does not yet claim to be a
trained STAR UVT or world-geometry trace.

A separate report now extracts actual STAR UVT trainer-harness trace geometry
from the same high-motion smoke video:

```text
research_experiments/star_uvt_feature_tubes/projective_high_motion_trace_geometry_report.py
outputs/projective_high_motion_trace_geometry_report.json
schema_version = projective_high_motion_trace_geometry_report_v1
```

It builds `ScreenTimeTubeModel.from_video_samples(...)` at smoke scale
(`64` tubes, `16` frames, `64px`) and lowers the resulting
`ma, q_uvt, depth0, depth_beta, opacity, color` tensors into the projective
cell atlas. The report now includes two initialization rows plus one tiny
in-report trained row. The trained row runs three dense CPU Adam steps before
trace extraction, so it proves tensors can move and still compile, but it is
not a persisted/full high-motion checkpoint.

Current rows:

```text
config_faithful_zero_velocity_init:
    velocity_init = zero
    cell_count = 793
    interval_to_dense_tile_pair_ratio = 0.063
    fallback_fraction = 0.0

block_match_motion_init:
    velocity_init = block_match_gated
    velocity_nonzero_count = 58 / 64
    velocity_max_px_per_frame = 5.657
    cell_count = 1496
    interval_to_dense_tile_pair_ratio = 0.293
    fallback_fraction = 0.0

block_match_motion_trained_dense_3step:
    velocity_init = block_match_gated
    train_steps = 3
    train_loss = 0.30096 -> 0.29547
    train_loss_ratio = 0.982
    trained_parameter_l1_delta = 67.95
    moved_parameters = center_uv, center_t, velocity_uv, raw_precision, raw_opacity, raw_color
    depth0_l1_delta = 0.0  # dense harness sorts on detached depth0
    velocity_nonzero_count = 64 / 64
    velocity_max_px_per_frame = 5.784
    cell_count = 1495
    interval_to_dense_tile_pair_ratio = 0.294
    fallback_fraction = 0.0
```

This tightens the next gate: the remaining gap is now a full saved trained
checkpoint or WorldFoam trace extraction, not merely video-derived geometry.

Verification:

```text
targeted depth-affine/spatial-precision Metal tests: 3 passed in 3.18s
broad focused STAR UVT projective suite: 149 passed in 19.91s
UV visibility event certificate + rebuilt interval backward ABI:
    broad focused STAR UVT projective suite: 150 passed in 14.23s
UV event-driven fallback marking:
    broad focused STAR UVT projective suite: 151 passed in 10.09s
UV finer-grid spatial split:
    targeted visibility checks: 4 passed in 1.80s
    broad focused STAR UVT projective suite: 153 passed in 15.90s
UV adaptive split-vs-fallback policy:
    targeted visibility checks: 4 passed in 3.06s
    broad focused STAR UVT projective suite: 156 passed in 16.69s
UV high-motion split-vs-fallback measurement:
    targeted visibility checks: 4 passed in 1.24s
    broad focused STAR UVT projective suite: 157 passed in 23.47s
UV orbit-parameterized split-vs-fallback measurement:
    targeted orbit/adaptive checks: 4 passed in 4.13s
    broad focused STAR UVT projective suite: 159 passed in 24.31s
UV split report artifact:
    targeted report/adaptive checks: 4 passed in 5.39s
    broad focused STAR UVT projective suite + report test: 163 passed in 25.46s
Video-derived high-motion report extraction:
    targeted report/adaptive checks: 4 passed in 4.53s
    broad focused STAR UVT projective suite + report test: 166 passed in 93.12s
High-motion STAR UVT trace-geometry extraction:
    targeted report tests with trained row: 2 passed in 35.36s
    broad focused STAR UVT projective suite + trace reports: 168 passed in 231.68s
```

Gradient note: `depth_affine_uv` is still treated as compiled metadata /
certificate structure. The current native VJP covers trace coefficients,
color, opacity, and temporal opacity, but not gradients into the depth-plane
slope metadata itself.

The paired visibility invariant is:

```text
support_stale_overshoot_epsilon only debounces support coverage
visibility_stale still forces rebin / stratify / fallback
```

A focused test combines a tolerated `0.05px` support overshoot with stale depth
intervals. The refresh still runs, repairs support, and changes the cell order
to the live front-to-back order. This prevents a subtle but dangerous failure:
using a support tolerance as an accidental visibility tolerance.

The same invariant now has a projective orbit fixture. Two tiny yaw-window
traces are lowered through the rational/projective path; a `0.10px` live
coefficient update creates bounded support drift that a support-only refresh
accepts under `support_stale_overshoot_epsilon=0.10`. With visibility checking
enabled, the stale depth intervals still produce four order mismatches, so the
refresh rebuilds the atlas and returns live order `(1, 0)` with zero remaining
order mismatches. The support gauge carries the orbit; the visibility
certificate still owns order.

The orbit visibility-root version is now covered too. Two rational yaw-window
traces cross in conditional depth at `t=0`, between the sampled frames. The
event report returns one continuous depth-order root; refresh is triggered by
visibility, not support, and the event stratifier splits the single orbit cell
into `(0,2)` with order `(0,1)` and `(2,4)` with order `(1,0)`. No fallback is
marked, order mismatches fall to zero, and the interval-to-dense trace-sample
ratio stays `0.5`. This is the formal replacement for a vague chart fallback:
visibility events become sensor-time cell boundaries.

The first policy that spends crowded-tile guard slots by this geometry is:

```text
support_guard_policy = "slack_budgeted"
```

For an overflowing target tile `C`, keep the base-active trace ids, then rank
extra guarded ids by:

```text
d_i,C = min_sample dist_inf(base_support_i(sample), tile C)
```

Small `d_i,C` means the trace is closest to crossing that support event, so it
gets headroom before farther traces. This is still a local deterministic
compiler rule, not yet a learned optimizer-motion predictor, but it replaces
id-order headroom spending with support-event geometry.

The first combined synthetic orbit stress now exercises all three current
certificates in one refresh:

```text
support drift:
    alpha_tail in (1e-4, 3e-4), so support reuse is certified
visibility:
    one rational yaw-window depth root at t=0, stratified into two order runs
guard allocation:
    tile_capacity = 32, base traces = 20, guard extras = 24
```

`trace_budgeted` spends the 12 available guard slots on the lower-id far traces.
`slack_budgeted` spends them on the 12 traces nearest the tile support event.
The slack run has zero remaining order mismatches, no fallback, max per-cell
active count equal to 32, and interval-to-dense trace-sample ratio about
`0.41`. This is still synthetic, but it is the first single test where
projective support, certified support reuse, visibility-root stratification,
and support-event-distance headroom all share one compiler decision.

## Implementation Update: Revolving Camera Fiber Metric Gate

The variable-camera segment compiler now has a direct orbit-facing test for the
fiber-bundle claim. The fixture is a synthetic elevated look-at orbit:

```text
frames = 16
camera path = look_at(eye(theta), target)
theta in [-60 deg, 60 deg]
frames_per_segment = 4
world tubes = anisotropic XY Gaussian supports
```

The compiler maps the world tubes into screen-time chart segments:

```text
segment_count = tube_count * 4
per_frame_segment_count = tube_count * 16
```

so it is still sharing projection/support work over time. But unlike a
diagonal screen metric, each segment carries the local pulled-back UV precision

```text
Q_uv(theta_chart) = [[q_uu, q_uv],
                     [q_uv, q_vv]]
```

with

```text
q_uu > 0
q_vv > 0
q_uu q_vv - q_uv^2 > 0
max |q_uv| > 1e-3
```

and for the first tube:

```text
min_chart q_uv < 0 < max_chart q_uv.
```

That sign change is the code-level evidence that a revolving camera induces a
rotating screen-fiber metric across the orbit, and that the compiler carries
the metric per chart instead of flattening the whole orbit into a single
axis-aligned footprint.

The same projected charted tubes render through the CPU UVT path with finite
nonzero output, which protects the tensor contract after adding the orbit
diagnostic.

Verification:

```text
orbit file: 7 passed in 2.18s
py_compile: passed
broad projective/interval suite: 158 passed in 14.19s
```

## Implementation Update: Revolving Camera Chart-Size Sweep

The orbit fiber gate now has a first quantitative share-vs-error sweep. The
reference is the same variable-camera route with one segment per frame:

```text
frames = 8
reference frames_per_segment = 1
tested frames_per_segment = 1, 2, 4, 8
```

The segment-count ratios are:

```text
frames_per_segment  segment_ratio
1                   1.000
2                   0.500
4                   0.250
8                   0.125
```

The test asserts that every shared route remains within:

```text
mean_abs_image_error < 0.009
mse_image_error      < 0.0011
max_abs_image_error  < 0.40
```

against the framewise projected reference. This is not yet a production
quality bar, but it is the first executable statement of the main objective's
sublinear-frame-cost condition in the revolving-camera lane:

```text
projection/support chart work falls with segment count
rendered image error is measured against the per-frame route
```

Verification:

```text
focused sweep: 1 passed in 10.70s
orbit file: 8 passed in 14.11s
py_compile: passed
broad projective/interval suite: 160 passed in 26.11s
```

## Implementation Update: Revolving Camera Interval Atlas Sweep

The revolving-camera chart-size sweep now goes through the interval atlas
object, not just the CPU UVT tube renderer. For each chart size, the compiler
does:

```text
world tubes + orbit camera program
    -> variable-camera UVT chart segments
    -> uvt_tubes_to_projective_trace_cell_atlas(...)
    -> support-event rebin + visibility stratification
    -> interval atlas stats and reference render
```

The measured atlas rows use the same `frames = 8` orbit fixture:

```text
frames_per_segment  trace_count  fallback_fraction  interval_ratio
1                   16           0.0                1.000
2                   8            0.0                <= 0.70
4                   4            0.0                <= 0.45
8                   2            0.0                <  0.35
```

The exact ratio values can move if support padding or visibility splitting
changes, but the monotone decrease and zero fallback are now protected. The
atlas reference render is also checked against the charted UVT render:

```text
mean_abs(atlas - charted_uvt) < 3e-5
max_abs(atlas - charted_uvt)  < 0.02
```

This is closer to the real objective than the image-only sweep because the
measured object is the reusable tile-time atlas. It says:

```text
projection chart segments decrease with chart size
interval trace entries per dense trace sample decrease
fallback remains zero on the synthetic orbit
the atlas render still matches the charted UVT semantics
```

The orbit family is also covered by the production interval Metal forward:

```text
render_projective_trace_cell_interval_atlas_metal(...)
```

matching the reference on the MPS path.

Verification:

```text
focused interval tests: 2 passed in 10.42s
orbit file: 11 passed in 9.96s
py_compile: passed
broad projective/interval suite: 163 passed in 33.16s
```

## Implementation Update: Revolving Camera Interval Backward

The orbit interval route now has a backward gate through the Metal autograd
bridge. The test builds orbit-derived chart tensors and marks them
differentiable:

```text
ma.requires_grad = True
q_uvt.requires_grad = True
opacity.requires_grad = True
color.requires_grad = True
```

Then it lowers those tensors into the interval atlas:

```text
uvt_tubes_to_projective_trace_cell_atlas(...)
```

and renders with:

```text
render_projective_cell_interval_atlas_metal_backward(...)
```

using an asymmetric image loss so the rotated footprint terms are exercised.
The protected gradient contract is:

```text
grad(ma) != 0
grad(opacity) != 0
grad(color) != 0
sum |grad(q_uu, q_uv, q_vv)| != 0
sum |grad(q_uv)| != 0
sum |grad(q_ut, q_vt, q_tt)| != 0
```

This is the clean-derivatives counterpart to the interval compression sweep:
the reusable atlas topology is static, but the differentiable trace parameters
still receive gradients through the shared interval Metal path. In particular,
the orbit-induced rotated screen-fiber metric is trainable through `q_uv`.

Verification:

```text
focused backward: 1 passed in 3.37s
orbit file: 12 passed in 18.07s
py_compile: passed
broad projective/interval suite: 164 passed in 28.66s
```

## Implementation Update: Revolving Camera Frame-Growth Work Units

The orbit lane now has an executable gate for the most important meta-goal:
share world-side work as the requested frame count grows. The test keeps the
camera-orbit chart budget fixed while increasing the number of output frames:

```text
frames = 8, 16, 32
temporal charts per tube = 4
tube_count = 2
```

The per-frame route still grows linearly:

```text
per_frame_segment_count = 16, 32, 64
```

but the compiled orbit route stays fixed:

```text
charted_segment_count = 8, 8, 8
atlas_trace_count      = 8, 8, 8
fallback_fraction      = 0, 0, 0
```

The interval atlas still has to cover more sampled output times, so its event
cells are not constant. The protected work-unit ratios are:

```text
frames  interval_entries  dense_trace_samples  interval_ratio
8       99                156                  0.6346
16      135               366                  0.3689
32      156               820                  0.1902
```

So a `4x` increase in frame count gives:

```text
dense_trace_samples  > 5x
interval_entries     < 2x
interval_ratio       falls by more than 65%
fallback             remains zero
```

This is not a claim that rendering materialized video is sublinear in pixels.
It is the narrower and correct claim:

```text
unavoidable pixel evaluation grows with frames
world-side projection/support/binning/atlas work is reused across frames
```

In the language of the pinned theory, the test holds the orbit camera-ray
bundle atlas fixed and densifies slices of the sensor-time base. The compiled
object is the trace atlas, not the video.

Verification:

```text
focused frame-growth gate: 1 passed in 12.13s
orbit file: 13 passed in 33.10s
py_compile: passed
broad projective/interval suite: 165 passed in 48.65s
```

## Implementation Update: Revolving Camera Backward Frame Densification

The forward frame-growth gate now has a small Metal-backed gradient sibling.
The test densifies frame samples while holding the orbit chart parameter set
fixed:

```text
frames = 4, 8
temporal charts per tube = 2
tube_count = 2
```

So both rows compile to:

```text
charted_segment_count = 4
atlas_trace_count      = 4
```

The renderer is the interval Metal autograd bridge:

```text
render_projective_cell_interval_atlas_metal_backward(...)
```

Each row backprops an asymmetric, time-weighted image loss and checks nonzero
gradients for:

```text
ma
opacity
color
q_uu, q_uv, q_vv
q_uv
q_ut, q_vt, q_tt
```

The point is not yet a backward timing claim. It is a derivative-topology
claim: densifying sampled frames can reuse the same compiled orbit chart
parameter tensors, and the interval VJP still reaches the pulled-back
screen-fiber metric and temporal Schur terms.

This closes one more gap in the pinned condition:

```text
share projection/support/binning/visibility/backward over time
```

Verification:

```text
focused backward gates: 2 passed in 4.28s
orbit file: 14 passed in 30.73s
py_compile: passed
broad projective/interval suite: 166 passed in 49.64s
```

## Implementation Update: Measured Orbit Fixed-Chart Scaling Artifact

The unit gates now have a measured synthetic artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.md
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

The benchmark compares two routes on the same elevated revolving-camera
fixture:

```text
fixed_chart:
    fixed temporal charts per tube = 4

per_frame:
    frames_per_segment = 1
```

For `frames = 4, 8, 16, 32`, the fixed-chart route reports:

```text
segment_count = 8, 8, 8, 8
trace_count   = 8, 8, 8, 8
payload_bytes = 608, 608, 608, 608
fallback      = 0, 0, 0, 0
```

while the per-frame route reports:

```text
segment_count = 8, 16, 32, 64
trace_count   = 8, 16, 32, 64
payload_bytes = 608, 1216, 2432, 4864
```

The fixed-chart interval work remains event-driven:

```text
frames  interval_entries  dense_trace_samples  interval_ratio
4       112               112                  1.0000
8       99                156                  0.6346
16      135               366                  0.3689
32      156               820                  0.1902
```

The regenerated artifact now directly times the CPU-side compile phases:

```text
route        project_ms 4->32    atlas_build_ms 4->32    cpu_compile_ms 4->32
fixed_chart 8.06 -> 4.38       19.73 -> 32.25        27.80 -> 36.64
per_frame   3.98 -> 35.09      17.95 -> 261.14       21.93 -> 296.22
```

The default saved artifact now uses `8,16,32,64` frames. The earlier
`4,8,16,32` regenerated artifact is quarantined as a too-small timing probe
because the 4-frame case let launch/packing overhead dominate the final timing
ratio.

So from `8 -> 64` frames:

```text
fixed_chart compile growth = 1.36x
per_frame compile growth   = 6.64x
```

At `64` frames:

```text
fixed/per_frame compile ratio  = 0.091
fixed/per_frame trace ratio    = 0.0625
fixed/per_frame payload ratio  = 0.0625
```

The saved prewarmed MPS eval timing remains a small synthetic diagnostic rather
than a broad speed claim. In the regenerated artifact, the `64f` fixed-chart
route is:

```text
forward ratio vs per_frame  = 0.117
backward ratio vs per_frame = 0.158
```

The same artifact also runs the Metal autograd topology check on fixed-chart
rows. It records nonzero `q_uv` and temporal `q_uvt` gradients at every frame
count, so the measured route still satisfies the derivative-side invariant.

Current interpretation:

```text
proved strongly:
    fixed orbit gauge atlas avoids per-frame trace/payload growth on this fixture

supported diagnostically:
    interval Metal forward/backward and CPU compile timing grow more gently than per-frame charting

not yet proved:
    real-scene end-to-end sublinear training wall time
```

The measured orbit artifact now has the same saved-report verifier pattern as
the cache/tail/high-motion reports:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

The contract is:

```text
topology:
    frame_counts are strictly increasing
    one fixed_chart and one per_frame row per frame count
    fixed_chart temporal_chunk_count = fixed_temporal_chunks
    per_frame frames_per_segment = 1
    fixed_chart segment_count, trace_count, and payload bytes stay constant
    per_frame segment_count, trace_count, and payload bytes grow

support/atlas:
    interval_ratio = interval_trace_entries / dense_trace_samples
    per_frame interval_trace_entries = dense_trace_samples
    fallback_fraction = 0 for all rows
    fixed_chart interval ratio is non-increasing
    final fixed_chart interval ratio falls by at least 65%
    fixed_chart interval-entry growth < dense trace-sample growth
    fixed_chart interval-entry growth < 2x

row consistency:
    iterations > 0
    warmup >= 0
    project_ms > 0
    atlas_build_ms > 0
    cpu_compile_ms = project_ms + atlas_build_ms
    when Metal runs, mps_atlas_build_ms > 0
    summary fields match summarize(sorted rows) for every current summary key

timing when Metal timings are present:
    final fixed/per-frame CPU compile ratio < 0.5
    final fixed/per-frame forward ratio < 0.5
    final fixed/per-frame backward ratio < 0.5

derivatives:
    fixed-chart autograd reaches ma, opacity, color
    fixed-chart autograd reaches q_uv and temporal q_uvt terms
    direct backward reaches coeffs, opacity, color, and spatial_precision_uv
```

Verification:

```text
focused verifier: 10 passed in 28.79s
saved artifact: verified by --verify-report
orbit + verifier suite: 24 passed in 126.68s
```

## Implementation Update: Real Trainer Route Frame-Scaling Cache Artifact

The compatible projective interval route now has a tiny artifact through the
actual STAR UVT feature trainer loop:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_interval_trainer_frame_scaling/summary.md
outputs/benchmarks/2026-05-25_star_uvt_projective_interval_trainer_frame_scaling/summary.json
```

The benchmark monkeypatches the video loader with synthetic frame tensors but
keeps `run_training`, the real projective interval producer, the Metal forward,
the Metal backward bridge, cache refresh policy, optimizer step, loss, and
trainer metrics in the loop.

For `frames = 4, 8, 16` and four optimizer steps:

```text
cadence rebuilds = 2, 2, 2
measured rebuilds = 1, 1, 1
measured/cadence rebuild ratio = 0.5 at every frame count
max measured-vs-cadence end-loss delta = 0.0
tile_overflow_sum = 0 for all rows
max_tile_count = 4 for all rows
```

This is not yet the end-to-end sublinear wall-time proof. The timing columns
are useful only as smoke diagnostics because the first MPS/trainer case can
carry cold-start cost. The durable result is narrower and more important for
the theory: the production-compatible trainer route can reuse compiled
sensor-time cache metadata across frame counts, preserve the same loss as the
cadence rebuild policy, keep tile pressure bounded, and still run live
staleness checks/updates before rendering.

That upgrades the memory contract from:

```text
synthetic orbit renderer can share work
```

to:

```text
the actual feature trainer has a measured cache-reuse path that preserves loss
```

The synthetic trainer artifact now has a reusable saved-report verifier too:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_interval_trainer_frame_scaling/summary.json
```

The contract is:

```text
topology:
    frame_counts strictly increase
    exactly one cadence and one measured row per frame count

trainer:
    status = ok
    rows pass
    loss decreases
    cadence/measured end loss matches per frame count
    zero tile overflow
    max tile count <= tile capacity

cache:
    measured rebuilds < cadence rebuilds
    measured live updates > cadence live updates
    measured staleness checks cover measured live updates
    support_rebins = stale_refreshes
    visibility stratifications = fallback marks = 0

smoke timing:
    measured no-first-step timing beats cadence on the synthetic MPS rows
```

Verification:

```text
focused synthetic verifier: 6 passed in 11.58s
saved synthetic artifact: verified by --verify-report
strict synthetic + real-video trainer verifier suite: 26 passed in 10.59s
real-video base, guard025, guard05, guard1, and guard2 artifacts: verified by --verify-report
guarded real-video guard025, guard05, guard1, and guard2 artifacts: verified by --verify-report --verify-guarded-support
aggregate guarded-support matrix:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json
aggregate matrix: 5 artifacts, 15 measured rows, default measured support rebins 9, guarded measured support rebins 0, guarded stale refreshes 0, max guarded no-first-step ratio 0.5895631229975254, max guarded rebuild ratio 0.5
```

This closes the synthetic production-trainer verifier gap. The real-video
artifact below remains the stronger high-motion evidence, but the synthetic
artifact is still useful because it isolates the production route under a small
deterministic generated target.

The same trainer-level cache-reuse claim now has a real-video artifact on the
checked-in high-motion clip:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling/summary.md
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling/summary.json
```

The benchmark runs the actual projective-interval `run_training` route for
`frames = 4, 8, 16`, with `size = 64`, `tube_count = 128`, four optimizer
steps, and cadence versus measured cache policy:

```text
cadence rebuilds:           2, 2, 2
measured rebuilds:          1, 1, 1
max measured/cadence loss delta: 0.0
tile_overflow_sum:          0 for all rows
max_tile_count:             18 for all rows
measured live updates:      3, 3, 3
measured staleness checks:  3, 3, 3
measured support rebins:    3, 3, 3
measured/cadence no-first-step ratios: 0.881, 0.352, 0.692
```

This is the first high-motion source-video trainer artifact showing that the
measured projective interval path can cut full atlas rebuilds in half, preserve
exact end loss versus cadence, and avoid overflow across growing frame
prefixes. It also shows the remaining weakness clearly: live updates still
trigger support rebins on every step, so support-lifecycle churn is not solved.

The support-lifecycle weakness now has guarded real-video followups:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard05_tail001/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard10_tail001/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard20_tail001/summary.json
```

All four use `support_guard_policy = slack_budgeted`,
`support_stale_tail_alpha_epsilon = 0.001`, and the same real high-motion
trainer setup. Guard025 uses a quarter extra support pixel, guard05 uses half,
guard1 uses one, and guard2 uses two.

Shared result:

```text
measured rebuilds:          1, 1, 1
measured support rebins:    0, 0, 0
measured stale refreshes:   0, 0, 0
max measured/cadence loss delta: 0.0
tile_overflow_sum:          0 for all rows
guard0.25 effective padding: 8.25px
guard0.25 measured/cadence no-first-step ratios: 0.373, 0.489, 0.516
guard0.5 measured/cadence no-first-step ratios: 0.398, 0.464, 0.542
guard1 measured/cadence no-first-step ratios: 0.393, 0.497, 0.521
guard2 measured/cadence no-first-step ratios: 0.417, 0.503, 0.590
```

Guard `0.25px` is now the smallest certified no-churn guard on this clip.
The 2026-05-25 rerun keeps every guarded measured no-first-step row below
cadence; the aggregate guarded-support report is the canonical compact evidence
for this matrix. The current interpretation is:

```text
proved:
    a small slack-budgeted guard can eliminate real-video trainer support churn
    while preserving exact cadence loss and bounded tile pressure

not free:
    guard size changes compile/render cost enough that it needs policy tuning
    rather than blindly increasing padding
```

The verifier matrix now includes guard025 as well as guard05/guard1/guard2.
The strict guarded-support CLI verifies guard025 directly:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_trainer_frame_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.json \
  --verify-guarded-support
```

The aggregate matrix verifies directly:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_guarded_support_matrix_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json
```

The broad real-video verifier now recomputes the summary and rejects stale or
weak rows: non-increasing frame counts, wrong row count, failed rows, missing
loss decrease, overflow, fallback/visibility marks, measured/cadence loss
drift, missing live staleness checks, support-rebin/stale-refresh mismatch, and
stale summary fields. Focused real-video verifier tests now pass:

```text
goal + guarded-support focused suite: 39 passed in 0.99s
```

## Implementation Update: Trained High-Motion Trace Geometry Scaling

The high-motion row now has a tiny saved trainer-smoke geometry artifact rather
than only video motion-centroid diagnostics:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/trained_high_motion_checkpoint.pt
```

The benchmark trains a tiny projective-interval STAR UVT feature model on the
checked-in high-motion smoke video, saves the checkpoint, reloads the learned
model tensors, and compiles those tensors into projective trace-cell atlases.
It also compiles the matched init model as a control.

For the trained smoke checkpoint over `4, 8, 16` frame prefixes:

```text
loss:                     0.298236 -> 0.296121
train pass:               true
train tile overflow:      0
trace_count:              64, 64, 64
fallback_fraction:        0.0, 0.0, 0.0
dense_per_frame_tile_pairs: 1542, 3061, 6016
interval_trace_entries:     392, 477, 573
interval/dense tile ratio:  0.254, 0.156, 0.095
```

So as the output prefix grows `4x`, the dense per-frame tile-pair work grows
`3.90x`, while the compiled interval trace entries grow only `1.46x`. This is
still a tiny smoke, not a final high-resolution wall-time claim or the full
persisted checkpoint gate, but it moves the evidence from hand-built
orbit/proxy geometry into an actual saved trainer-smoke artifact generated from
the high-motion video.

The regenerated artifact also adds a repeated per-frame interval baseline on
the same learned tensors. For the trained smoke checkpoint:

```text
frames:                         4       8       16
shared_interval_entries:        392     477     573
per_frame_replay_entries:       392     1956    3862
shared_forward_ms:              127.2    25.3    57.0
per_frame_forward_ms:           147.0   179.4   355.2
shared_backward_ms:             111.1    43.9    60.3
per_frame_backward_ms:          126.4   183.3   367.9
```

Treat these timings as diagnostic because the row is tiny and MPS timing is
noisy, but they prove the saved smoke checkpoint geometry can run through the
native interval forward/backward path over growing frame prefixes and directly
compare against framewise replay. At `16f`, the shared interval route is
`0.160x` the per-frame forward time and `0.164x` the per-frame backward time
in this tiny diagnostic.

Current interpretation:

```text
proved at smoke scale:
    trained STAR UVT feature tensors from real video can be compiled into a
    fallback-free projective trace atlas whose interval work grows slower than
    dense per-frame tile pairs, and those atlases execute native interval
    Metal forward/backward with finite gradients

still open:
    same claim at larger tube counts/resolution and real end-to-end wall time
```

Larger smoke scale now has a matching artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/trained_high_motion_checkpoint.pt
```

This rerun uses `size = 64`, `tube_count = 128`, four optimizer steps, native
interval Metal timing, and the same repeated per-frame replay baseline. The
trained checkpoint row:

```text
loss:                     0.317323 -> 0.316218
train pass:               true
train tile overflow:      0
cache rebuild/live/stale: 1 / 3 / 3
trace_count:              128, 128, 128
fallback_fraction:        0.0, 0.0, 0.0
dense_per_frame_tile_pairs: 3578, 7158, 14363
interval_trace_entries:     956, 1158, 1371
interval/dense tile ratio:  0.267, 0.162, 0.095
```

Against same-checkpoint per-frame replay:

```text
frames:                         4       8       16
shared_interval_entries:        956     1158    1371
per_frame_replay_entries:       956     4811    9605
shared_forward_ms:              295.0   183.2   469.7
per_frame_forward_ms:           301.3   770.8   802.0
shared_backward_ms:             113.6    57.5   303.3
per_frame_backward_ms:          189.8   657.5   1779.1
```

The 16-frame entry ratio is `0.143x`, the forward timing ratio is `0.586x`,
and the backward timing ratio is `0.170x`. The forward timings are still noisy
at this scale, but the backward row and interval-entry growth are aligned with
the theory: shared sensor-time trace work grows much more slowly than
framewise replay while retaining differentiable native Metal execution.

The next scale step exposed a practical but important harness bug: when the
benchmark used `tile_capacity = 256`, the trainer render still saw the old
`STAR_UVT_TILE_CAPACITY = 128` environment and failed before training. The
benchmark now synchronizes the Metal tile environment before `run_training`,
not only inside the standalone timing helper, and the focused test covers this
non-default cap path.

With that fixed, a larger cap256 artifact now passes the same verifier:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/trained_high_motion_checkpoint.pt
```

This rerun uses `size = 96`, `tube_count = 256`, `tile_capacity = 256`, four
optimizer steps, native interval Metal timing, and same-checkpoint per-frame
replay. The trained checkpoint row:

```text
loss:                     0.317038 -> 0.315874
train pass:               true
train tile overflow:      0
cache rebuild/live/stale: 1 / 3 / 3
trace_count:              256, 256, 256
fallback_fraction:        0.0, 0.0, 0.0
dense_per_frame_tile_pairs: 7820, 15628, 31255
interval_trace_entries:     2045, 2349, 2831
interval/dense tile ratio:  0.262, 0.150, 0.091
```

Against same-checkpoint per-frame replay:

```text
frames:                         4       8       16
shared_interval_entries:        2045    2349    2831
per_frame_replay_entries:       2045    10279   20547
shared_forward_ms:              140.8    200.7   117.1
per_frame_forward_ms:           766.5    948.0  1282.6
shared_backward_ms:             141.1    203.0   169.9
per_frame_backward_ms:          683.2    990.0  1810.5
```

The 16-frame entry ratio is `0.138x`, the forward timing ratio is `0.091x`,
and the backward timing ratio is `0.094x`. This remains a smoke-scale
diagnostic, but it is now a bigger stress than the 64px/128t run and verifies
that the same saved-checkpoint contract survives non-default tile capacity.

The trained high-motion scaling claim is now guarded by
`verify_trained_high_motion_trace_scaling_report(...)` in
`research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py`.
The verifier rejects reports without train loss decrease, with tile overflow or
fallback, with non-constant trained trace count, with interval entries growing
as fast as dense per-frame tile pairs, or with slower final-scale shared
timing/entry counts when a per-frame replay baseline is present. Non-final
timing rows remain finite-positive gradient/timing smoke checks because the
4-frame timing row can be dominated by MPS launch noise. It can be run without
rerunning training:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
```

Verification:

```text
trained high-motion + shared-work audit suite: 33 passed in 6.20s
all three saved trained high-motion scaling artifacts: verified
```

## Next Gates

1. Validate `slack_budgeted` plus bounded support-tail alpha debounce beyond the
   first real cache-policy artifact: the current stresses say the certificate
   accepts real tails, rejects center/core loss, remains compatible with a tiny
   orbit-derived trace, and does not suppress visibility repair.
2. Push the trained high-motion checkpoint geometry gate beyond the 96px/256t
   cap256 smoke into longer training, larger resolution, and cleaner repeated
   timing, then decide from residual fallback and cell-growth numbers when an
   oblique/fiber halfspace cell is worth adding, decide whether depth-plane
   slopes ever need gradients, and extend the same trace object to richer
   WorldFoam/instance cells.
3. Use the enforced budget policy in production trainer decisions: choose
   split/refit versus tile-local fallback from interval ratio, stratum count,
   and fallback fraction.
4. Keep `fallback_render_mode=mixed` as the trainer-state fallback policy and
   consider a row-compacted launch for the row-weighted rolling kernel.
5. Bridge WorldFoam by compiling cell-camera intersections through the same
   `pi_* Gamma^*` object.
