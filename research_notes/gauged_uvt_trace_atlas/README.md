# Gauged UVT Trace Atlas

Date: 2026-05-24

This folder turns the "UVT screen fiber" idea into a working theory and
implementation plan. The goal is not another screen-space STAR UVT variant. The
goal is a gauged camera-ray bundle whose local coordinate expressions compile
world primitives into sensor-time traces.

Start with `00_WHAT_IS_THIS_GOAL.md` for the first-read contract: what this is,
what the renderer is trying to make fast, and the formal derivative,
memory-bandwidth, and sublinear-growth conditions. Then read
`GOAL_META_KEY_MATH.md` for the compact memory version of the goal, meta goals,
math, theory commitments, current implementation state, and next gates.

Core equation:

```text
UVT trace = pi_* Gamma^* world_primitive
```

where:

```text
B = Omega x T                         sensor-time base
pi: E_Gamma -> B                      camera-ray bundle
Gamma: E_Gamma -> M                   camera program into world spacetime
Gamma^*                               pull world density/radiance onto rays
pi_*                                  integrate or summarize along ray fibers
```

The current STAR UVT contract stores a local coordinate expression:

```text
ma, q_uvt, depth0, depth_beta, opacity, color_or_feature
```

That is useful, but it is only the affine/Gaussian chart of a richer object.
Revolving cameras should be handled by charted projective/rational gauges, not
by one global affine UVT splat with late fallback.

Terminology correction: when these notes say "chart", read it as a **gauge
domain with validity certificates**, not as an ad hoc fitted patch. The durable
invariant remains `pi_* Gamma^* world_primitive`; the local domain only says
where a cheap expression, support bound, depth/order certificate, and backward
support are valid. See `GAUGE_DOMAINS_NOT_CHARTS.md` before changing this
architecture.

## Subtheory Index

1. `00_bundle_foundations/` defines the base, total ray space, fibers, gauges,
   pullback, pushforward, and chart transitions.
2. `01_camera_gauge_choices/` catalogues depth/projective/object/local gauges
   and when each reduces trace complexity.
3. `02_gaussian_fiber_pushforward/` derives the Schur-complement UVT Gaussian
   as a local expression of the invariant bundle operator.
4. `03_projective_rational_traces/` defines homogeneous camera-time traces,
   rational centers, denominator singularities, and the first Metal probe.
5. `04_revolving_camera_atlas/` explains why a full orbit is an atlas, not one
   chart, and how to choose orbit windows.
6. `05_visibility_strata/` formalizes order as strata in sensor-time, with
   commutation bounds and chart-level visibility.
7. `06_exposure_and_rolling/` treats finite exposure and rolling shutter as
   integration over the same base with row/time coupling.
8. `07_adjoint_training/` describes forward traces plus adjoint traces for
   training without replaying per-frame projection/sorting.
9. `08_worldfoam_bridge/` maps foam cells and instance charts into the same
   bundle language.
10. `09_metal_acceptance_plan/` records concrete kernels, tests, and promotion
    gates from diagnostic probe to renderer integration.

Root addendum:

```text
DEPTH_FIBER_CROSS_TRACK_NOTE.md
GAUGE_DOMAINS_NOT_CHARTS.md
DETAILED_NEXT_PLAN.md
CODE_IMPLEMENTATION_PLAN.md
paper/
```

`DEPTH_FIBER_CROSS_TRACK_NOTE.md` records the shared ray-depth fiber idea
across World Tubes and WorldFoam: World Tubes uses the fiber for Schur/fiber
marginalization plus conditional depth/order certificates; WorldFoam keeps the
fiber as the transmittance axis. `GAUGE_DOMAINS_NOT_CHARTS.md` explains why
"charts" should be interpreted as event-certified gauge domains, and what it
would actually mean to throw them away. `DETAILED_NEXT_PLAN.md` is the
post-final-audit execution plan: stop adding umbrella proofs, build a decisive
visual/runtime demo, stress visibility, then decide whether a native projective
atlas kernel is worth carrying. `CODE_IMPLEMENTATION_PLAN.md` turns that plan
into concrete files, functions, report schemas, verifier tests, pass/fail
thresholds, and the keep/kill gate for native Metal work. `paper/` contains the
arXiv-style "World Tubes in Gauged Camera Space" draft and the
baseline/ablation/chart plan for turning the verified research lane into a
submission.

## Current Implementation State

The first implementation pass has moved past a single probe. The durable state
to preserve is:

```text
Goal:
    compile 4D spacetime primitives through a known camera program into
    reusable UVT viewport-time traces.

Meta-goal:
    make gauges/projections carry orbit and rolling-camera complexity before
    splitting or falling back.

Key math:
    UVT trace = pi_* Gamma^* world_primitive
```

Implemented Gate A:

```text
projective_trace_eval(coeffs, times, eps) -> [N, S, 4]
```

with quadratic homogeneous coefficients per tube:

```text
h_u(t) = a0 + a1 t + a2 t^2
h_v(t) = b0 + b1 t + b2 t^2
h_z(t) = c0 + c1 t + c2 t^2
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
depth(t) = h_z(t)
```

This establishes the first GPU-tested projective gauge primitive.

Implemented bundle-gauge falsification probe:

```text
projective_bundle_gauge_invariance_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.md
```

This checks the core equation on a revolving-camera spacetime Gaussian by
integrating the same pulled-back primitive in ordinary depth and log-depth
fiber gauges. With the required Jacobian, `max_rel_error = 3.50e-13`; without
the Jacobian, relative error is at least `0.600`. The report also locks the
depth-order rule: monotone gauge changes preserve order, while orientation
reversal flips order and must become a gauge/visibility boundary. The verifier
now recomputes the summary from rows/order and rejects stale row errors,
missing measure-Jacobian controls, non-monotone gauge certificates, and bad
near/far/sample metadata.

Implemented bundle-gauge derivative probe:

```text
projective_bundle_gauge_gradient_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.md
```

This differentiates the same fiber-pushforward objective with respect to
primitive mean, log-precision, and log-amplitude. With the Jacobian, the
ordinary-depth and log-depth gradients match to `2.33e-12` relative error; the
missing-Jacobian gradient control is wrong by at least `0.592`; an independent
finite-difference check for `mean[0]` matches autograd to `1.42e-10`. The
gradient verifier now checks row gradient norms, finite-difference internal
consistency, missing-Jacobian value/gradient controls, and stale summary fields.
Focused value+gradient bundle tests pass `21 passed in 6.45s`; both saved
artifacts verify by CLI.

Implemented shared-work goal audit:

```text
projective_shared_work_goal_audit.py
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.md
```

This reads the saved orbit fixed-chart and trained high-motion artifacts,
verifies their own contracts, then checks the active-goal ratios directly. On
the orbit row, fixed-chart atlas payload stays constant (`1.0x`) while
per-frame replay payload grows `8.0x`; the restored default orbit artifact now
uses frame counts `8,16,32,64` and verifies with final fixed/per-frame ratios:
payload/trace/segment `0.0625`, CPU compile `0.091`, forward `0.117`, backward
`0.158`. Across the three trained
high-motion artifacts, shared interval entries grow at most `1.462x` while
per-frame replay entries grow at least `9.852x`; final shared/per-frame
interval-entry ratios stay below `0.149`, and final backward ratios stay below
`0.094`. The verifier now also requires exposure/rolling forward and backward
audit rows: their underlying verifiers must pass, rolling unique-time reuse
must stay below `1.0`, all four forward Metal paths and both backward Metal
paths must be present, and Metal value/gradient errors must stay under the
focused verifier thresholds. The audit recomputes the summary from
orbit/trained/exposure rows, checks frame-count monotonicity, finite positive
ratios, trace/segment/payload reuse, CPU/forward/backward thresholds, and
rejects stale objective summaries. Current focused audit tests pass
`33 passed in 6.20s` across the trained high-motion and shared-work audit
tests; the regenerated aggregate artifact verifies by CLI and now has a
current-input acceptance mode:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json \
  --verify-current-inputs
```

That mode recomputes the audit from the current default orbit/trained/exposure
artifacts and rejects a saved aggregate JSON that is internally valid but stale
relative to those inputs. The old
`4,8,16,32` orbit failure is quarantined under
`outputs/benchmarks/2026-05-25_star_uvt_revolving_orbit_fixed_chart_scaling_current_timing_fail/`
as launch-noise evidence, not a passing report.

Implemented projective goal-progress audit:

```text
projective_goal_progress_audit.py
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.md
```

This is not a completion claim. It maps the active objective onto verified
evidence and proves thirty-four current rows: formal camera-path compiler contract,
fiber-gauge trace invariant, clean fiber derivatives, one-parameter local
camera-family bundle math over `Q x Omega x T`, two-parameter local
camera-family bundle math over `Q2 x Omega x T`, one-parameter camera-family
shared-metadata scaling, two-parameter camera-family shared-metadata scaling
over `Q2 x Omega x T`, two-parameter camera-family slice lowering into the
existing interval Metal forward/backward path, two-parameter camera-family
shared-backward chain-rule accumulation from interval Metal VJPs,
two-parameter camera-family single-launch materialized Metal batching,
two-parameter camera-family native Metal trace eval/VJP from shared family
coefficients plus q-basis values, two-parameter camera-family native Metal
interval forward rendering/compositing/visibility from shared family
coefficients plus q-basis values, two-parameter camera-family native Metal
interval backward/VJP into shared family coefficients and q-basis values with
compiled visibility/order held fixed, stable-topology Q2 tile/order metadata
reuse, two-strata Q2 tile/order metadata reuse for a depth-order change,
three-strata Q2 active-set metadata reuse for a support/culling change,
checked-in high-motion real-video active-set distribution evidence,
Metal time-shared forward/backward,
finite-exposure/rolling fallback, a small synthetic compiled-adjoint trainer
smoke, a checked-in high-motion real-video trainer smoke, a real-video
guarded-support matrix, a source-distinct real-video multiscene trainer
matrix, a five-source extended functional matrix, a source-distinct
real-video frame-scaling matrix, a five-source extended frame-scaling
diagnostic that preserves correctness/cache/support invariants while failing
only the expected strict timing gates, a source-distinct real-video quality
	tether, a five-source extended quality tether, a broad10 quality tether, a
	source-distinct real-video media tether, a five-source extended media tether,
	a broad10 media tether through the actual contact-sheet writer, the Bq4
	fresh-process median timing gate, the
real-video acceptance envelope, the real-video timing-variance envelope, the
real-video compiled-adjoint replacement artifact, and sublinear world-side work
proxies. It records
`shared_work.current_input_errors = []`
and has its own current-input
acceptance mode:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json \
  --verify-current-inputs
```

That top-level mode recomputes the goal-progress report from the current
bundle, one- and two-parameter camera-family gauge, one- and two-parameter
camera-family shared-work, trainer-interval, real-video trainer, real-video
guarded-support, multiscene trainer, multiscene frame-scaling, extended
frame-scaling diagnostic, multiscene quality-tether, multiscene media-tether,
real-video acceptance-envelope, timing-variance-envelope, and shared-work
artifacts; the shared-work row in turn
recomputes its
orbit/trained/exposure inputs.

Implemented projective goal-completion gap report:

```text
projective_goal_completion_gap_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.md
```

This is also not a completion claim. It turns the previously open
`full_goal_completion` row into a machine-checked remaining-work contract:
goal-progress is verified through current-input acceptance, the
acceptance-envelope, timing-variance-envelope, shared-work, broad10 trainer,
broad10 quality/media, timing-protocol, and compiled-adjoint replacement inputs
are checked, and five proxy/acceptance rows are proved:
`formal_goal_memory_and_audit`, `sublinear_world_side_work_proxy`,
`broad_real_scene_quality_acceptance`,
`full_compiled_adjoint_trainer_replacement`, and
`timing_acceptance_protocol`. The only partial lower row is
`full_goal_completion`; the concrete gaps are all zero:
`broad_quality_source_gap=0`, `broad_media_source_gap=0`,
`broad_quality_frame_count_gap=0`, `strict_timing_failure_gap=0`,
`timing_acceptance_gap=0`, `compiled_trainer_source_gap=0`, and
`compiled_trainer_replacement_gap=0`. The saved gap artifact keeps
`completion_ready=false` and `does_not_prove_completion=true` because it is the
input to, not the replacement for, the final completion audit.

Implemented projective goal-completion promotion audit:

```text
projective_goal_completion_promotion_audit.py
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.md
```

This is the authoritative completion claim for the active gauged/projective UVT
thread. It consumes the gap report, verifies it against current inputs, and
promotes the lower non-completion stack into six proved objective rows:
scope/key-math preservation, sensor-time trace compiler evidence, sublinear
non-pixel work evidence, broad real-video acceptance, compiled-adjoint training
evidence, and final completion promotion. The saved report records
`status=complete`, `completion_ready=true`, `is_goal_complete=true`,
`does_not_prove_completion=false`, and `open_requirement_ids=[]`. Focused
progress/gap/replacement/promotion tests pass `82 passed in 4.02s`, and the
wider timing-protocol/frame-breadth/media/acceptance/compiled-adjoint/gap/
promotion/goal-progress bundle passes `121 passed in 4.72s`.

The two-parameter camera-family gauge row
keeps value and primitive gradients gauge-invariant over `Q2 x Omega x T`
with max value error `8.42e-14`, max primitive-gradient error `2.28e-12`,
`q_phase` gradient error `1.82e-11`, and `q_height` gradient error
`1.10e-11`. The
two-parameter camera-family shared-work row keeps one `Q2 x Omega x T`
payload constant while per-q-pair replay grows `64x`, with final payload ratio
`0.0625`, final chart ratio `0.015625`, and max fit residual `0.111px`. The
two-parameter camera-family Metal lowering row verifies a `5x5` Q2 grid:
one shared Q2 coefficient table lowers into ordinary `Omega x T` interval
Metal slices, all 25 forward/backward rows produce nonzero images and
coeff/opacity/color gradients, family/replay payload ratio is `0.178`, and
the peak slice/replay payload ratio is `0.04`. This is explicitly a
slice-lowering smoke, not native high-dimensional Q2 Metal evaluation. The
two-parameter camera-family Metal chain-rule row verifies the backward side of
that lowering: per-slice interval Metal coefficient gradients over the same
`5x5` Q2 grid accumulate into one shared Q2 family adjoint, shared/replay
gradient payload ratio is `0.24`, max finite-difference relative error is
`4.91e-05`, and shared-family gradient support is nonzero. This is still a
shared-family chain-rule smoke over Metal slices, not native Q2 Metal evaluation. The
two-parameter camera-family materialized-batch row packs all 25 q-pair slices
into one ordinary interval Metal atlas. It verifies one forward/backward launch
matches the per-slice reference with image abs error `0.0` and shared-family
gradient relative error `9.34e-08`, while keeping materialized/replay trace
payload at `1.0x`. That last number is intentional: the batch proves launch
reuse, not native family-coefficient compression. The true Q2 family table
would be `0.178x` of the materialized trace payload. The native family
eval/VJP row then contracts `family_coeffs[N,9,B]` with `q_basis[Q,B]` inside
the Metal shader, evaluates all `Q x N x S` homogeneous trace samples, and
accumulates direct VJPs into both shared tensors. The native eval artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_eval/summary.json
```

reports family/materialized coefficient payload ratio `0.24`,
family-plus-q/materialized coefficient payload ratio `0.5733333333333334`,
max value relative error `6.58e-08`, max family-gradient relative error
`5.72e-08`, and max q-basis-gradient relative error `2.58e-07`. This is native
family trace evaluation and VJP by itself; the interval rows below cover
compositing and interval-cell VJP. The native family interval forward row then consumes
`family_coeffs[N,9,B]` and `q_basis[Q,B]` directly inside the Metal interval
renderer, depth-sorts/composites through the interval-cell path, and matches
the materialized single-launch reference exactly:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_forward/summary.json
```

It reports `100` batched frames, family/materialized trace coefficient payload
ratio `0.16615384615384615`, full native-family forward/materialized trace
payload ratio `0.4461538461538462`, max image absolute error `0.0`, max image
relative error `0.0`, and equal native/materialized image abs sums
`1992.59228515625`. This proves native interval forward
rendering/compositing/visibility over family coefficients. The native family
interval backward row then runs the matching interval-cell VJP directly against
`family_coeffs[N,9,B]` and `q_basis[Q,B]`, with tile membership and depth order
treated as compiled constants just like the ordinary interval VJP:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward/summary.json
```

It reports `100` batched frames, native-family/materialized-gradient payload
ratio `0.2926315789473684`, native family-coefficient/materialized-gradient
payload ratio `0.11368421052631579`, max family-gradient relative error
`2.3355269149760716e-06`, max q-basis-gradient relative error
`8.51117079037067e-07`, and nonzero family/q-basis gradient support. This
proves native interval renderer backward/VJP over shared family coefficients
and q-basis values under the fixed compiled-visibility contract. The Q2
tile/order reuse row then checks the remaining sampled-Q metadata pressure in
the stable-topology case:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse/summary.json
```

It proves one local tile/order topology plus q-index applicability can expand
back to all 25 materialized q-pair cells, while conservative family-union depth
intervals preserve the stored order with min depth gap `0.6033999919891357`.
Materialized tile/order metadata grows `25.0x`; shared topology metadata grows
`1.0x`; shared/materialized metadata ratio is `0.11692307692307692`.
The split-strata tile/order row then deliberately flips depth order across the
Q2 family and stores two topology groups instead of one record per q-pair:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata/summary.json
```

It proves 25 materialized q-pair cells compress into two certified order
strata, with materialized metadata growth `25.0x`, shared metadata growth
`2.0x`, shared/materialized metadata ratio `0.15692307692307692`, and min
per-stratum union depth gap `0.33200000002980246`. The active-set strata row
then deliberately changes support/culling topology across the Q2 family:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata/summary.json
```

It proves 25 materialized q-pair cells compress into three certified
active-set topology strata, with materialized metadata growth `25.0x`, shared
metadata growth `3.0x`, shared/materialized metadata ratio
`0.19692307692307692`, and min active-set union depth gap
`0.2630399994850159`. These three metadata rows prove stable, order-split, and
active-set-split local metadata compression. The real active-set distribution
row then measures checked-in high-motion compiled atlases instead of another
synthetic q-family:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_active_set_distribution/summary.json
```

It verifies three saved high-motion artifacts over `4,8,16` frames, nine
trained-checkpoint atlas rows, existing source videos, passing underlying
artifact verifiers, fallback-free rows, max active-set cells per group `3`,
max active-set-group/dense-tile-pair ratio `0.04009499860296172`, and max
cell/group ratio `1.3214953271028038`. This proves the active-set topology
case is now measured on real compiled traces, but it is still not broad
real-scene quality acceptance. The
trainer smoke is an actual `run_training` route over `4,8,16` frames: measured
live-cache rebuilds stay `1,1,1` versus cadence `2,2,2`, measured/cadence
no-first-step ratios stay below `0.840`, and the max end-loss delta is
`2.98e-8`. The high-motion real-video `run_training` route now has the same
goal-progress evidence status: rebuilds stay `1,1,1` versus cadence `2,2,2`,
measured/cadence no-first-step ratios stay below `0.881`, and the max end-loss
delta is `0.0`. The source-distinct real-video multiscene trainer matrix uses
three checked-in segments from three YouTube sources and the guarded
projective-interval trainer contract:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json
```

It verifies six cadence/measured rows, exact cadence-loss agreement, max
measured/cadence no-first-step ratio `0.550`, rebuild ratio `0.5`, and zero
support rebins, stale refreshes, overflow, fallback marks, and visibility
stratifications. The extended functional matrix broadens that same guarded
contract to five checked-in sources, adding higher-motion bike and FPV clips:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json
```

It verifies ten cadence/measured rows, five distinct YouTube sources, max motion
score `7.018424034118652`, exact cadence-loss agreement, rebuild ratio `0.5`,
and zero support rebins, stale refreshes, overflow, fallback marks, and
visibility stratifications. Its max measured/cadence no-first-step ratio is
`1.50811535915855`, so it is functional broadening evidence rather than a
uniform timing-win claim. The source-distinct real-video frame-scaling matrix
then uses the same three checked-in video sources over `4,8,16` frames:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json
```

It verifies 18 cadence/measured rows, exact cadence-loss agreement within
`2.98e-8`, frame-growth factor `4.0`, max measured/cadence no-first-step ratio
`0.690`, max measured/cadence rebuild ratio `0.5`, measured cache rebuild growth
`1.0`, max measured timing-growth/frame-growth ratio `0.438`, and zero support
rebins, stale refreshes, overflow, fallback marks, and visibility
stratifications. The five-source extended frame-scaling diagnostic preserves
the harder functional set's correctness/cache/support behavior while refusing
to call it a timing win:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json
```

It verifies the failed strict five-source frame-scaling matrix failed only the
two expected timing gates, covers five distinct YouTube sources and 30
cadence/measured rows over `4,8,16` frames, keeps cadence-loss delta `0.0`,
measured rebuild ratio `0.5`, measured rebuild growth `1.0`, support
rebins/stale refreshes `0`, support tail/overshoot `0`, and
fallback/overflow/visibility stratification `0`. Its max measured/cadence
no-first-step ratio is `1.188933546093892` and max timing-growth/frame-growth
ratio is `1.0009153415685994`, so this is explicitly a diagnostic/caveat row,
not timing-win evidence. A pair-level timing-breakdown report isolates the miss:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown/summary.json
```

It finds `3/15` no-first measured/cadence pairs over `1.0`: `Bq4rmeIvJbs`
at `4f` (`1.188933546093892`), `Bq4rmeIvJbs` at `16f`
(`1.1381882094250788`), and `C8kTRrtE3KU` at `8f`
(`1.0249968931082667`). Only `Iagm3K8QtFw` misses normalized 4-to-16-frame
growth, at `1.0009153415685994`. All failing pairs are still cache/support
clean: rebuild ratio `0.5`, loss delta `0.0`, support/stale/fallback/overflow
`0`. The current timing hypothesis is therefore evaluation/noise/phase-shape
rather than cache invalidation or support churn. The phase-profile report then
reads the saved per-step timings for those rows:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile/summary.json
```

It profiles five rows: the three no-first misses plus the two growth endpoints.
The saved case step means match the source no-first rows exactly, max source/
case ratio delta `0.0`. The two Bq4 misses are render-forward dominated
(`render_forward_ms` ratios `1.3566329017525305` at `4f` and
`1.111793076402963` at `16f`), while the C8k `8f` miss is small
(`step_ms` ratio `1.0249968931082667`) and dominated by `colorize_loss_ms`
noise. All profiled rows keep rebuild ratio `0.5`, loss delta `0.0`, and
support/fallback/overflow/visibility issues `0`. A render-forward residual
report then checks whether the Bq4 render-forward miss is explained by saved
candidate/support workload:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_residual/summary.json
```

It compares all 15 cadence/measured saved case pairs and finds the cadence and
measured `tile_stats` are identical for every pair (`max_tile_stats_abs_delta =
0.0`). The three no-first timing misses are also exactly identical in tile
workload, and all three are render-forward misses; the max render-forward
ratio remains `1.3566329017525305` on `Bq4rmeIvJbs_seg_000` at `4f`, but
`workload_explains_render_forward_miss_count = 0`. This rules out the saved
tile candidate distribution as the explanation and points the next work at
render-forward per-work-unit latency or timing replay/instrumentation. A
per-step render-forward shape report then checks whether that residual is a
persistent slowdown or a small-step spike:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape/summary.json
```

It finds all three no-first timing misses are single-spike driven in both
render-forward time and whole-step time: after dropping the largest positive
render-forward delta, the worst no-first miss render ratio falls to
`0.8418254365135661`. The worst no-first miss has render step-ratio spread
`5.383083741915209` and render spike delta `728.0996670015156 ms`. The saved
strict source has no `chunk_traces` (`chunk_traces_present_pair_count = 0`), so
substep attribution requires a traced rerun of the Bq4 spike steps rather than
another chart/fiber math change. That traced Bq4 rerun is now saved:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun/summary.json
```

It reruns the Bq4 `4f` and `16f` cadence/measured spike-step cases with
`trace_global_steps`, verifies every expected spike step has a traced chunk,
verifies projective interval substep timing is present, and keeps cache/support
clean. The saved spike does not reproduce at the no-first-step level:
measured/cadence no-first ratios are `0.4538476088322886` and
`0.5785517503959672` (`traced_bq4_spike_reproduced = false`). The substep
timing still shows one live-update caveat: measured/cadence projective interval
totals are `0.5054386427773483` at `4f` and `1.2736600499593582` at `16f`, with
feature-state-update ratios `0.44341185194975186` and `1.250134158419622`.
So the next timing work is repeat/stability plus feature-state-update/live-update
phase profiling, not a new fiber/chart theory change. A 16f-only repeat
stability report now tests whether that one-shot feature-state-update bump is
persistent:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability/summary.json
```

It repeats the Bq4 `16f` cadence/measured traced pair three times. All expected
steps are traced, all chunks carry substep timings, and cache/support remains
clean. The bump does not persist in this schedule:
`no_first_spike_reproduced_count = 0`, `projective_total_bump_count = 0`, and
`feature_state_update_bump_count = 0`; max ratios are `0.45165397508134686`
for no-first step, `0.9101288137358652` for projective interval total, and
`0.7882220153002857` for feature-state update. The remaining timing caveat is
therefore mixed-sequence/warm-state launch variance rather than a persistent
16f live-update hot substep. The mixed-sequence follow-up is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order/summary.json
```

It runs two repeats of `mixed_4_to_16` and `reverse_16_to_4`, tracing both
frame sizes in each sequence. No `16f` no-first spike reproduces
(`no_first_bump_count = 0`, max `16f` no-first ratio `0.45600195672964483`),
but order-dependent substep variance appears: `mixed_4_to_16` has max `16f`
projective-total ratio `0.9606946419165872` and max feature-state-update ratio
`1.0006466493572015`, while `reverse_16_to_4` has max projective-total ratio
`1.844612661591509` and max feature-state-update ratio `1.73336471126077`.
This supports the warm-state/launch-order hypothesis for substep timing without
contradicting the higher-level measured no-first win. The policy-order isolation
follow-up is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order/summary.json
```

It warms the process with traced Bq4 `4f` and `16f` cadence/measured cases,
then runs `16f` target pairs in both `cadence_then_measured` and
`measured_then_cadence` order. All traces and cache/support checks pass. This
does reproduce a warmed timing failure: across the four target pairs,
`no_first_bump_count = 1`, `projective_total_bump_count = 3`, and
`feature_state_update_bump_count = 3`. The bumps do not simply follow "second
slot": `measured_then_cadence` has measured first and reports max no-first
ratio `1.7836530508238704`, max projective-total ratio `1.7184222253396344`,
and max feature-state-update ratio `1.9605903379413647`; `cadence_then_measured`
keeps no-first below `1.0` but still has one projective/feature-state bump.
This sharpens the timing caveat to policy/order/warm-state interaction while
leaving the atlas/fiber math unchanged. A fresh-process follow-up is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
```

It runs isolated Python/MPS processes for the `16f` target cases in both
`cadence_then_measured` and `measured_then_cadence`. The saved artifact now uses
three repeats and a discard-repeat-0 acceptance view. Every row is marked
fresh-process, all expected global steps are traced, projective interval
substep timing is present, and cache/support remains clean. Across all six
pairs, there is no no-first bump (`max_no_first_ratio = 0.7087283466117477`),
but substep outliers remain: `projective_total_bump_count = 1`,
`feature_state_update_bump_count = 2`, max projective-total ratio
`2.2454207580524894`, and max feature-state-update ratio
`1.2948922914387324`. The median view is better: all-pair medians are
`0.6530516888499702` no-first, `0.8356591487478802` projective total, and
`0.7124745747568637` feature-state update. After discarding repeat 0, the
timing-acceptance view passes on medians: status `pass`, post-warmup pair count
`4`, median no-first `0.5645123618278631`, median projective total
`0.8356591487478802`, and median feature-state update `0.846418513757801`.
So the current timing rule is sharper: fresh-process median/warmup-discard
supports the measured policy, while max-ratio timing claims still need outlier
handling and should not drive fiber/gauge math changes. The new quality-tether report reads the saved case payloads
from that matrix:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json
```

It verifies nine cadence/measured case pairs, exact loss-curve and RGB-loss-curve
agreement, exact end-loss and end-PSNR agreement, all required gradient-flow
flags, and positive measured PSNR gains on every pair. Its min PSNR gain is
`0.02227306365966797`; max loss-curve delta and max end-PSNR delta are both
`0.0`. This tethers live-cache output to cadence over the small source-distinct
frame-scaling matrix, but it is still not broad real-scene quality acceptance.
The extended quality-tether report applies the same payload-level quality check
to the five-source extended functional matrix:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_quality_tether/summary.json
```

It verifies five cadence/measured case pairs over five distinct YouTube
sources, exact loss-curve and RGB-loss-curve agreement, exact end-PSNR
agreement, all required gradient-flow flags, positive measured PSNR gains, min
PSNR gain `0.04466235637664795`, max loss-curve delta `0.0`, and max end-PSNR
delta `0.0`. This closes the quality-tether gap on the five-source functional
broadening artifact, but it is still not broad real-scene quality acceptance.
The media-tether report runs the actual trainer media path and compares cached
versus cadence contact sheets:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json
```

It verifies three source-distinct cadence/measured media pairs, pixel-identical
contact sheets (`max_abs_contact_sheet_delta = 0`, matching PNG hashes), valid
two-row target/pred contact-sheet layout, artifact-derived target/pred MSE
matching payload final RGB loss within `0.001525666420389149`, nontrivial rows
(`min_contact_sheet_target_std = 0.14441643529730494`,
`min_contact_sheet_pred_std = 0.07265247844694266`), matching final full-RGB
media loss and PSNR, max loss-curve delta `0.0`, min measured PSNR gain
`0.04511058330535889`, all required gradient-flow flags, max measured/cadence
no-first-step ratio `0.9316588494614714`, rebuild ratio `0.5`, and zero
overflow, fallback marks, and visibility stratifications. This proves the actual
rendered-media artifact path agrees with cadence on the small checked-in matrix;
it is still not broad real-scene quality acceptance.
The extended media-tether report runs the same actual contact-sheet media path
on the five-source extended set:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether/summary.json
```

	It verifies five source-distinct cadence/measured media pairs, pixel-identical
contact sheets (`max_abs_contact_sheet_delta = 0`, matching PNG hashes), valid
two-row target/pred contact-sheet layout, artifact-derived target/pred MSE
matching payload final RGB loss within `0.001525666420389149`, nontrivial rows
(`min_contact_sheet_target_std = 0.14441643529730494`,
`min_contact_sheet_pred_std = 0.07178262974117959`), matching final full-RGB
media loss and PSNR, max loss-curve delta `0.0`, min measured PSNR gain
`0.04466235637664795`, all required gradient-flow flags, rebuild ratio `0.5`,
and zero overflow, fallback marks, and visibility stratifications. The max
	measured/cadence no-first-step ratio is `1.2065694734694634`, so this is
	five-source media/quality evidence, not a timing-win row.
	The broad10 media-tether report runs the actual contact-sheet media path on
	ten source-distinct clips:

	```text
	outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json
	```

	It verifies ten cadence/measured media pairs, pixel-identical contact sheets,
	matching PNG hashes, nontrivial target/pred rows, all required gradient-flow
	flags, max loss/RGB-loss curve delta `1.4901161193847656e-08`, max final
	full-RGB loss delta `2.9802322387695312e-08`, max final full-RGB PSNR delta
	`5.960464477539062e-07`, rebuild ratio `0.5`, and zero overflow/fallback/
	visibility stratification. This closes the broad media source-count gap; it is
	still not a strict timing-win row.
The focused Q2 Metal lowering/chain-rule/materialized-batch +
native-eval/native-interval-forward/native-interval-backward/tile-order-reuse/
tile-order-strata/active-set-strata/real-active-set-distribution +
goal-progress tests previously passed `106 passed in 8.30s`; the current
media-tether + quality-tether + extended-quality-tether + multiscene
extended-media-tether + frame-scaling + multiscene + guarded-support +
extended-frame-scaling diagnostic + timing-breakdown + phase-profile +
render-forward-residual + render-forward-shape + Bq4 traced-rerun +
Bq4 repeat-stability + Bq4 sequence-order + Bq4 policy-order +
Bq4 fresh-process isolation +
goal-progress focused suite passes
`142 passed in 8.17s`
	for the newly touched audit/media verifier subset; the latest
	timing-protocol/frame-breadth/media/acceptance/compiled-adjoint/gap/
	promotion/goal-progress bundle passes `121 passed in 4.72s`. The final
	completion-promotion audit is now the authoritative closeout for the
	previous broad real-scene/full-trainer gap.

Implemented Gate B/C compiler helpers:

```text
fit_projective_trace_polynomial(coeffs, times, degree)
split_projective_trace_windows(coeffs, times, degree, thresholds)
bound_projective_trace_window(window)
bound_projective_trace_windows(windows)
```

which fit affine/quadratic local `[u, v, h_z]` charts from projective samples,
report residual, denominator, valid-sample certificates, split long time
intervals into valid chart windows, and build conservative UV/depth support
bounds.

Implemented Gate D/E compiler helpers:

```text
make_projective_trace_visibility_sidecar(window)
compare_projective_trace_depth_order(sidecar_a, sidecar_b)
make_projective_trace_appearance_sidecar(alpha_max, color)
bound_projective_trace_visible_swap_cost(order, appearance_a, appearance_b)
bin_projective_trace_support_bounds(bounds, image_width, image_height, tile_size)
assemble_projective_trace_tile_time_atlas(records)
render_projective_trace_tile_time_atlas_reference(...)
```

These build depth/denominator visibility sidecars, mark stable/crossing order,
bound visually harmless swaps, assemble tile-time active sets, and provide a
CPU/Torch dense-reference atlas renderer.

Implemented q-UVT bridge:

```text
projective_trace_windows_to_uvt_tubes(...)
ProjectiveTraceUVTBridge.active_start / active_stop
render_projective_trace_uvt_bridge_reference(..., use_window_gates=True)
render_projective_trace_uvt_bridge_metal_gated(...)
render_uvt_tubes_gated(...)
direct_atomic_backward_gated(...)
direct_backward_projective_trace_uvt_bridge_metal_gated(...)
pack_projective_trace_tile_time_bins(...)
render_projective_trace_tile_time_atlas_metal(...)
direct_backward_projective_trace_tile_time_atlas_metal(...)
```

Accepted degree-1 chart windows can now lower into the existing STAR UVT
`ma/q_uvt/depth0/depth_beta` renderer contract. Split chart segments carry
explicit sample-domain interval gates, and the native `render_gated` Metal op
passes those gates into shader-side binning and per-sample compositing. The
matching direct atomic backward path also respects the same interval gates, so
split affine chart segments have a first native forward/backward surface. A
bridge-level one-step color update smoke now lowers image MSE through that
native gated surface.

The first nonlinear/projective atlas-cell Metal renderer now exists too. It
packs compiler-side tile-time active sets with exact per-entry chart intervals,
then evaluates quadratic homogeneous projective traces directly in Metal and
composites by per-sample projective depth. This gives degree-2/rational atlas
cells a native forward path without lowering to affine q-UVT. The matching
direct VJP now covers color, opacity, and homogeneous projective coefficients
against a Torch autograd oracle. A coefficient-only trainability smoke now
renders a shifted-coefficient target, applies the native direct VJP, updates
homogeneous coefficients, and verifies loss drops with color held fixed.

The forked cell-evaluator lane adds `ProjectiveTraceCellTraceAtlas`, which
stores accepted gauge domains as direct raw-time polynomial rows for
`u(t)`, `v(t)`, and `depth(t)`. Those rows now render through
`render_projective_trace_cell_atlas_metal(...)`, so an accepted atlas cell is a
GPU-evaluable object rather than only a support/order wrapper.

The forked trainer lane adds `uvt.render_backend="metal_tile_interval_gated"`
to the real STAR UVT source-view trainer path. It routes through native
`render_gated` forward plus `direct_atomic_backward_gated` VJP, using full-video
intervals for ordinary screen-time tubes while preserving explicit interval
buffers for future projective/gauge-domain segment producers. The combined
focused projective plus trainer interval-gate suite now passes `46` tests, and
a real `src/train/train.py` smoke selected the new backend and decreased loss.

The packed projective atlas now has a first frame-count scaling contract. The
new helper:

```text
count_projective_trace_dense_per_frame_tile_pairs(...)
```

counts the ordinary time-sliced project-and-bin denominator, and the benchmark:

```text
research_project/benchmarks/projective_atlas_scaling_probe.py
```

compares it against interval-packed projective atlas entries. On the current
45-degree orbit fixture, dense per-frame tile entries grow `35 -> 555` from
`4 -> 64` frames, while ideal interval-packed entries stay `13 -> 13`. The
older Metal-compatible `tile_t=4` slab packing grows `13 -> 208`, which exposed
the implementation gap: the atlas object was sublinear, but the hot Metal tile
scheduler still expanded it into fixed temporal slabs.

That gap now has a first interval-compressed Metal forward/backward cut:

```text
render_projective_trace_cell_interval_atlas_metal(...)
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_tiles(...)
direct_backward_projective_trace_cell_interval_atlas_metal(...)
torch.ops.star_uvt_v0.direct_projective_trace_cell_interval_backward(...)
```

The interval renderer dispatches over output pixel samples and uses spatial tile
bins with per-entry `[active_start, active_stop)` intervals. On the tiny
4-to-64-frame orbit probe it keeps the interval entry count at `13`, matches
the slab renderer image sums, and records interval Metal wall time
`24.8067ms -> 29.3612ms` over `16x` more frames. The matching interval direct
VJP now matches Torch autograd for color, opacity, and cell trace coefficients,
and a one-step coefficient trainability smoke verifies Metal-rendered MSE drops
after a native interval-VJP update with color held fixed. The latest scaling
artifact records interval backward `6.7827ms -> 35.6822ms` over `4 -> 64`
frames while interval entries stay `13 -> 13`, so the derivative path consumes
the interval object even though pixel work still grows.

The trainer harness now has a first projective interval-cell autograd bridge:

```text
render_projective_cell_interval_atlas_metal_backward(...)
```

The focused trainer smoke builds split projective windows, lowers them into a
cell atlas with multiple active intervals, renders and backprops through the
interval-compressed cell path, runs `optimizer.step()` on cell trace
coefficients, and verifies loss drops. The combined focused projective plus
trainer interval-gate suite passed `49` tests at this gate.

The projective cell atlas now has a first support lifecycle guard:

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

This detects when live cell trace coefficients have moved outside the compiled
tile-time support or when live depths no longer match the compiled
front-to-back order. It refreshes only support cells and depth intervals while
preserving the differentiable coefficient/opacity/color tensors. Focused tests
move a trace into a new tile and flip two traces' depth order without changing
screen support; both stale cases are detected and repaired by rebinning. The
trainer-harness refresh smoke moves an MPS coefficient tensor with an optimizer
step, refreshes metadata, renders through the interval-compressed Metal
autograd path, and verifies gradients still flow into the same tensor. The
trainer-state smoke now owns atlas/config/times/refresh cadence, refreshes
after optimizer steps, and proves both moved support and depth-order changes are
repaired without replacing live tensors. Ambiguous near-tie visibility now
becomes explicit fallback metadata: strict refresh raises, opt-in refresh marks
affected cells as `visibility_ambiguous_depth`, and the Metal fast path rejects
those cells. The CPU/Torch cell-atlas reference renderer now executes fallback
cells by live-depth sorting only marked tile/sample regions, and the atlas/state
report fallback fraction and reasons. Before fallback, refresh now tries
visibility-stratum splitting: live depth-order crossings become smaller stable
time-run cells, e.g. `[0,2): (0,1)` then `[2,4): (1,0)`, without replacing
the differentiable tensors. Budget diagnostics now report interval entries vs
dense trace samples, stratum split cells, max cells per active-set group,
fallback fraction, and named budget failures. Refresh returns the post-refresh
budget report, and trainer state can enforce those budgets before rendering.
Support refresh now uses continuous screen/tile boundary roots, so a trace that
crosses tile boundaries can become smaller time-local tile runs rather than one
broad tile rectangle over all active frames.
Continuous cell-local visibility events are now reported by solving
`z_i(t)-z_j(t)=0` inside each active interval; focused tests cover a linear
root at `5/3`, quadratic roots at `-1` and `1`, and stable pairs with no
events. Refresh now uses those roots before the sampled stratum split, and
exact roots on frame samples become singleton cells so fallback can stay local
to the actual tie/event sample. This makes split/refit a gauge/projection
certificate rather than an ad hoc fallback trigger.
The compiler also now exposes a continuous sensor-time partition that merges
support roots, visibility roots, and caller-supplied exposure/shutter split
times into intervals. This is the first explicit bridge from sampled cell
indices toward finite-exposure and rolling-shutter chart scheduling.
That partition can now be lowered into normalized finite-exposure midpoint
quadrature, or into per-row rolling-shutter quadrature with row readout
offsets. Those quadrature schedules now feed a differentiable CPU/Torch
continuous-time reference renderer:

```text
render_projective_trace_cell_atlas_quadrature_reference(...)
render_projective_trace_cell_atlas_rolling_quadrature_reference(...)
```

The oracle evaluates direct cell traces at fractional sensor times, sorts by
live depth per quadrature sample, composites, and accumulates sample weights.
It is intentionally separate from the integer-frame tile-cell renderer, so
finite exposure and rolling shutter do not inherit frame-index assumptions.
The schedules now also lower into the interval Metal contract:

```text
lower_projective_trace_cell_atlas_quadrature(...)
render_projective_trace_cell_atlas_quadrature_interval_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_metal(...)
```

This bridge maps quadrature samples to integer sample-indexed interval cells
while preserving raw sensor-time trace evaluation and optional `domain_times`
activity for split gauge rows. Rolling shutter now has a batched schedule
lowering too:

```text
ProjectiveTraceCellRollingQuadratureLowering
lower_projective_trace_cell_atlas_rolling_quadrature(...)
render_projective_trace_cell_atlas_rolling_quadrature_batched_reference(...)
```

It merges per-row schedules into one sorted unique sample-time axis plus a
`row_weights[Q,H]` matrix, so duplicate sample times and interval atlas packing
are shared before the weighted row gather. The rolling Metal path now uses a
dedicated row-weighted interval kernel:

```text
render_projective_trace_cell_interval_atlas_rows_metal(...)
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_rows(...)
```

That kernel writes the final rolling image directly, skipping
`row_weights[q,row] == 0` sample/row pairs instead of materializing
`[Q,H,W,3]` and reducing it in Python.
Mixed fallback now has a concrete forward policy too:

```text
split_projective_trace_cell_atlas_fallback_cells(...)
projective_trace_cell_atlas_fallback_tile_sample_mask(...)
render_projective_trace_cell_atlas_quadrature_interval_mixed_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal(...)
```

The mixed renderer keeps non-fallback sample/tile regions on the interval Metal
path, but replaces whole fallback tile/sample regions with the live-depth
reference before exposure or rolling row-weight accumulation. That matters:
fallback is a visibility replacement region, not an additive layer.
The first production-facing bridge now lives in
`src/train/star_uvt_projective_interval_backend.py`: trainer configs normalize
`feature_uvt.projective_interval`, build a
`ProjectiveCellIntervalBackendConfig`, and can construct a
`ProjectiveCellIntervalTrainerState` from a compiled atlas without reaching
directly into harness-only setup code.
It now also exposes the first actual atlas producer:

```text
uvt_tubes_to_projective_trace_cell_atlas(...)
make_projective_cell_interval_atlas_from_uvt_tubes(...)
make_projective_cell_interval_trainer_state_from_uvt_tubes(...)
```

This producer completes the STAR UVT quadratic in the spatial variables,
extracts the moving screen center `ma_uv - A^{-1}b(t-ma_t)`, lowers compatible
isotropic affine UVT tubes into `ProjectiveTraceCellTraceAtlas` rows, then
reuses support-event and visibility-event atlas compilation. It is deliberately
not the full general solution yet: by default it rejects anisotropic spatial
precision and pixel-varying depth slopes, but residual temporal opacity is now
stored as an `opacity_time_coeffs` quadratic and consumed by both the CPU/Torch
reference path and the interval Metal forward/backward path. The Metal VJP
returns gradients for `opacity_time_coeffs`, and the reference producer check
still backprops through the temporal envelope into the UVT temporal precision
coefficient.
The compatible producer is now wired through the real STAR UVT feature trainer
for the exact RGB-width route. When `feature_uvt.projective_interval.enabled`
is true and `feature_dim=3`, the trainer pins spatial precision to the backend
`sigma_px`, keeps temporal/motion/opacity/feature gradients live, renders the
feature image through `ProjectiveCellIntervalTrainerState`, and renders a
second white-trace atlas to recover total alpha for the existing
alpha-background/colorizer objective. This is still a first route, not the
general endpoint: it is full-frame only, requires autograd image losses, and
now has a first cache/rebuild cadence rather than always rebuilding compiled
metadata. The helper:

```text
make_projective_cell_interval_live_atlas_from_uvt_tubes(...)
```

updates differentiable trace, opacity, temporal-opacity, and color tensors from
current UVT model tensors while reusing compiled cells from a reference atlas.
The trainer now exposes an explicit cache policy:
`projective_interval.refresh_policy="cadence"` keeps the older full-rebuild
cadence, while `refresh_policy="measured"` rebuilds the compatible atlas only
for the first cached render and otherwise reuses compiled cells with fresh live
tensors. Cached live updates call the trainer-state refresh oracle before
rendering, so support/order/fallback/budget staleness can rebin, stratify, or
mark fallback while preserving live differentiable tensors. This is the first
production path where the sensor-time object persists across optimizer steps
because measured atlas invalidity says it is still valid, not because a fixed
cadence has not expired yet.
The combined focused projective plus interval-gated trainer suite now passes
`119` tests.

Saved cache-policy artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md
```

On the compatible 8f/64px full-frame route, measured cache policy keeps final
loss identical to cadence while reducing full atlas rebuilds `4 -> 1`.
No-first-step mean step time drops by about `1336 ms`, but support metadata is
still repaired on every live update, so the next cache problem is reducing
staleness churn under ordinary tube motion.

Saved support-guard artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_cap256/summary.md
```

The new `projective_interval.support_guard_padding` separates two radii:
`uv_padding` is the correctness support checked during refresh, while
`uv_padding + support_guard_padding` is the conservative chart support compiled
into the atlas. On the same compatible route, guard `2` plus cap `256`
eliminates stale refreshes/support rebins (`4/7 -> 0/0` for cadence/measured),
keeps final loss identical (`0.0847767964`), and keeps measured rebuilds at
`1` versus cadence `4`. Guard `2` or `8` at cap `128` overflows packed Metal
tiles, so this is a real gauge-margin/memory tradeoff rather than a free
default.

The follow-up cap-aware policy is:

```text
projective_interval.support_guard_policy = "budgeted"
```

It treats `support_guard_padding` as a maximum and bisects downward until the
packed interval atlas fits the configured tile capacity. The first cap128
artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_budgeted_cap128/summary.md
```

That artifact avoids the old immediate overflow and the measured row passes at
the same loss with zero tile overflow. It does not solve the churn: measured
support rebins improve only `7 -> 6`, no-first-step mean slows to `6107.7 ms`,
and cadence times out because repeated global budget search is expensive. The
replacement policy is now:

```text
projective_interval.support_guard_policy = "local_budgeted"
```

It keeps the target guard in tiles with capacity headroom and falls back to
base support only for packed tiles that would overflow. The explicit cap128
artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_local_budgeted_cap128_explicit/summary.md
```

This passes both cadence and measured rows at identical final loss
(`0.0847767964`) with zero tile overflow (`max_tile_count=70` in the case
JSONs). It fixes the global-search cost, not the stale-support churn: measured
support rebins are still `7/7`. The next useful guard is therefore finer than
tile replacement: allocate guard headroom per trace/cell inside crowded tiles
or split/refit local offenders.

The first trace-headroom policy is:

```text
projective_interval.support_guard_policy = "trace_budgeted"
```

It keeps base-active traces in overflowing tiles and spends remaining capacity
on deterministic extra guarded trace ids. The artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_rerun/summary.md
```

It passes both rows at identical final loss with zero overflow
(`max_tile_count=70`) and measured no-first-step `2460.0 ms`, but measured
support rebins remain `7/7`. The new support-boundary overshoot telemetry then
shows why:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_margin/summary.md
```

The old `7/7` measured rebins came from tiny tile-boundary overshoots:
`max_support_max_overshoot_px=0.0912`. The debounce knob
`support_stale_overshoot_epsilon` brackets the effect:

```text
epsilon 0.125 -> measured rebins 3/7, max overshoot 0.1690 px
epsilon 0.25  -> measured rebins 1/7, max overshoot 0.2986 px
epsilon 0.5   -> measured rebins 0/7, max overshoot 0.4932 px
```

The `0.5px` artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_eps05/summary.md
```

It keeps the same final loss (`0.0847767964`), zero overflow, and measured
no-first-step `1277.5 ms`. Treat this as a bounded experimental tolerance,
not a universal default, until broader image/error checks prove the subpixel
sliver is visually negligible.

The first image-level debounce stress now makes that condition explicit. With
real support padding, a `0.05px` tile-boundary overshoot only drops a Gaussian
tail and the strict-rebinned versus tolerant-reused atlas differs by less than
`1e-4` max RGB. With an underspecified support radius, the same nominal
subpixel overshoot can drop the Gaussian core and cause `>0.35` max RGB error.
So the debounce rule is only safe when support padding is a real footprint
bound, not when it is merely a center-tile marker.

The first orbit-derived image-level check now exercises the same contract on a
small rational camera trace rather than a purely axis-aligned affine toy. A
tiny yaw-window trace compiles into one tile; a `0.10px` coefficient update
pushes padded live support about `0.056px` across the neighboring tile boundary.
Strict rebinning versus tolerant reuse still differs by less than `1e-4` max
RGB. This is not a full-orbit proof, but it does tie the debounce to the
revolving-camera/projective chart math instead of only to a static boundary
example.

The debounce now also has a support-tail alpha certificate. For an isotropic
Gaussian trace with support radius `r = uv_padding`, boundary overshoot `delta`,
and screen sigma `sigma`, the omitted tile's maximum alpha is bounded by:

```text
alpha_tail <= opacity * exp(-0.5 * (max(r - delta, 0) / sigma)^2)
```

`support_stale_tail_alpha_epsilon` lets the measured refresh path reuse stale
support only when this bound is below budget. This moves the rule closer to the
renderer's actual error model: a tiny missing tail may be reused, but a missing
core still rebins even if the pixel overshoot is small.

The trainer and benchmark now record the certificate too:

```text
projective_interval_cache_last_support_tail_alpha_bound
projective_interval_cache_max_support_tail_alpha_bound
```

So the next real cache-policy artifact can report not only "we skipped a
support rebin", but the alpha bound under which that reuse happened.

The first max-per-trace artifact was:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_tail001_cap128/summary.md
```

That artifact is now superseded. It validated the local idea, but the broader
certificate must aggregate omitted tail alpha per missing sample/tile; a max
over primitives is too weak when many small tails overlap.

The corrected aggregate artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.md
```

On the compatible 8f/64px full-frame route, `slack_budgeted` with
`support_stale_tail_alpha_epsilon=0.001` and zero pixel-overshoot tolerance
keeps the same final loss as cadence (`0.0847767964`), zero tile overflow, and
`0/7` measured support rebins. The measured row uses one atlas rebuild and
seven live updates, while cadence uses four fixed rebuilds. The corrected
certificate number is:

```text
max_support_tail_alpha_bound = 0.000736007
max_support_max_overshoot_px = 0.4932
```

The corrected lower-budget bracket is path-dependent. With the same
slack-budgeted cap128 setup, `support_stale_tail_alpha_epsilon=0.00035` now
records two support rebins because the max aggregate bound reaches
`0.000404648`; `0.00045` and `0.0006` still record one support rebin because
skipping earlier repairs lets later aggregate drift grow to `0.000526049` and
`0.000656625`. The corrected artifacts are:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00045_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0006_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.md
```

The benchmark Markdown formatter now preserves small significant digits, so
values such as `0.00035`, `0.000404648`, and `0.000736007` do not collapse to
the same displayed number.

The cache-policy benchmark itself now has an executable saved-report contract
in:

```text
research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py
```

Run it without rerunning training via:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.json
```

The verifier pins the aggregate cache claim: `slack_budgeted` support, zero
pixel-overshoot pardon, positive tail-alpha budget, cadence and measured rows
with identical final loss, zero overflow/fallback/visibility stratification,
measured full rebuilds below cadence, measured live updates above cadence, and
tail-certified support reuse. It also checks the epsilon bracket: saved rows
with lower tail budgets rebin when the aggregate omitted-tail bound exceeds the
budget, while `tail001` clears with zero support rebins. Focused tests passed
`9 passed in 0.14s`, and all four corrected aggregate saved artifacts verified
through the CLI.

The first image-error verifier for that certificate is:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.md
```

It compares strict support rebinning against certified stale-support reuse on
three affine boundary-tail cases and one tiny rational orbit chart. All reuse
cases stay below their omitted-alpha bound:

```text
axis r4 sigma1 opacity0.5:    tail 0.0002046117, max RGB error 0.0000221119
axis r5 sigma1.25 opacity0.8: tail 0.0003146833, max RGB error 0.0000550129
axis r6 sigma1.5 opacity0.9:  tail 0.0003447873, max RGB error 0.0000822361
orbit rational chart:         tail 0.0002094069, max RGB error 0.0000227757
```

The negative core-loss case is the important red-team: with `uv_padding=0`,
the tail certificate reports `0.5` and rebins, while a pure pixel-overshoot
pardon would reuse and produce `0.3987594` max RGB error. So the certificate is
not just prettier telemetry; it changes the accept/reject boundary in the
right direction for this local isotropic/projective suite.

The aggregate verifier adds the missing overlap red-team:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.md
```

At `tail_alpha_epsilon=0.00035`, single-tail affine cases and the rational
orbit case still reuse with image residual below the bound. A 64-trace
overlapping-tail case rejects reuse with aggregate bound `0.01309515`; forcing
that reuse would produce `0.00141417` max RGB error.

The image-error certificate now has a reusable report verifier in:

```text
research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py
```

Run it without recomputing cases via:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.json
```

It locks the actual contract: positive certified-reuse cases must strict-rebin
without certification, reuse with certification, keep tail bound in
`(0, epsilon]`, and keep max RGB error below the omitted-tail bound; core-loss
and overlapping-tail aggregate cases must reject stale reuse, have bound above
epsilon, and show forced-reuse image error above budget. Focused verifier tests
passed `7 passed in 9.09s`, and the base, tail00035 aggregate, and
metal-precision-rerun saved artifacts all verified.

The next gate generalizes the certificate math to anisotropic local footprint
precision:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.md
```

For a local Gaussian footprint with SPD precision

```text
P = [[p_uu, p_uv],
     [p_uv, p_vv]]
```

and an omitted tile rectangle `R`, the continuous maximum omitted alpha is
bounded by

```text
opacity * exp(-0.5 * min_{x in R} (x - mu)^T P (x - mu)).
```

The verifier enumerates the exact convex-rectangle candidates: interior,
stationary edge points, and corners. It passes:

```text
diagonal anisotropic tail: bound 0.0002046116, max error 0.0000242851
rotated precision tail:   bound 0.0001845283, max error 0.0000190703
two-trace same-tile sum:  bound 0.0002287796, max error 0.0000166098
core-loss negative:       bound 0.5, max error 0.4379515, reuse rejected
```

This certificate is now paired with a production interval Metal footprint path:
the verifier remains CPU/theory evidence for support reuse, while focused Metal
tests exercise the same per-trace precision in forward and backward rendering.

The anisotropic certificate now has the same reusable report contract pattern:

```text
research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py
```

Run:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.json
```

The contract requires diagonal, rotated, and two-trace summed anisotropic tails
to certify reuse with `0 < omitted_alpha_bound <= epsilon` and image residual
below the bound; it requires the two-trace bound to exceed each single-tail
bound, and requires the core-loss negative to reject stale reuse with a large
forced-bad residual. Focused tests passed `6 passed in 8.81s`, and the base
plus metal-precision-rerun saved artifacts verified.

That metadata bridge now exists. `ProjectiveTraceCellTraceAtlas` has optional

```text
spatial_precision_uv: Tensor[N,3] = (q_uu, q_uv, q_vv)
```

with validation that the UV precision block is positive definite, float32,
contiguous, and on the same device as `coeffs`. Support rebinning, visibility
stratification, fallback marking, quadrature lowering, detached CPU conversion,
and trainer-state atlas materialization preserve the field. q-UVT lowering and
live-atlas updates populate it from the source `q_uvt`.

Important boundary: hand-built projective cell atlases can now render/backprop
with per-trace UV precision in Metal. The q-UVT compatibility lowering path
still defaults to the legacy isotropic `sigma_px` contract, but it now has an
explicit opt-in for anisotropic spatial precision. That opt-in carries
`(q_uu,q_uv,q_vv)` into the projective interval atlas and expands support by
the alpha-threshold ellipse bound
`d^T P d <= 2 log(alpha_peak / alpha_threshold)`. The source-view trainer
model route still locks its learned q-UVT precision to isotropic unless that
model/init contract is changed.

Current-code precision use:

```text
CPU/reference cell renderer: consumes spatial_precision_uv
finite-exposure quadrature reference: consumes spatial_precision_uv
stale-support tail-alpha certificate: consumes spatial_precision_uv
production interval Metal forward/backward: consumes spatial_precision_uv
q-UVT compatibility lowering: isotropic by default; anisotropic opt-in tested
```

Precision-only metadata verification at that pass:

```text
143 passed in 16.99s
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_metal_precision_rerun/summary.md
```

The scalar rerun also includes a useful aggregation red-team: overlapping
omitted tails in the same missing tile sum to `0.01309515`, exceed the `0.001`
budget, and prevent reuse. A certificate must be per tile/sample aggregate,
not only per primitive.

The depth-side companion metadata now exists too. `ProjectiveTraceCellTraceAtlas`
has optional

```text
depth_affine_uv: Tensor[N,6] =
  [zu0, zu1, zu2, zv0, zv1, zv2]
```

which evaluates a tile-local screen-fiber depth plane:

```text
z(u,v,t) =
  z_c(t)
  + z_u(t) * (u - u_c(t))
  + z_v(t) * (v - v_c(t)).
```

`eval_projective_trace_cell_depth_at_uv_torch(...)` is the first certificate
helper for this model. Metadata validation rejects malformed tensors, and the
field is preserved through support rebinning, trainer refresh, quadrature
lowering, and CPU detach. The q-UVT producer now keeps this depth section when
explicitly opted in with `allow_depth_affine_uv=True`: nonzero
`depth_beta[:,0:2]` is rejected by default, but the opt-in path lowers
`beta_u,beta_v` into `depth_affine_uv` and adjusts the center-depth slope by
`beta_u velocity_u + beta_v velocity_v`.

Visibility certificates now evaluate the depth plane over tile pixel-corner
ranges. If an order flip can happen inside one tile even though center depths
look stable, the report marks that tile/sample ambiguous and the fallback
marker can flag it for live-depth reference sorting. The interval Metal hot
path now also consumes `depth_affine_uv` in its per-pixel dynamic selection
sort, with zero slopes supplied for legacy scalar-depth atlases. The current
contract treats depth-plane slopes as compiled metadata, not trainable VJP
parameters.

The depth plane now has an explicit UV visibility event report:

```text
projective_trace_cell_uv_visibility_event_report(...)
```

For each tile/sample/trace pair it computes the affine zero line of
`Delta z(u,v,t_k)` over the tile pixel-center rectangle. A crossing emits
`ProjectiveTraceCellUVVisibilityEvent` with the line coefficients and min/max
depth delta. This does not yet cut an oblique sub-tile cell; it gives the
compiler a named certificate for "spatial split or fallback is required here"
instead of only a generic ambiguous-depth bit.

The fallback marker now consumes that certificate. A cell whose UV order line
crosses its tile is marked with `visibility_uv_depth_line` in
`fallback_reasons`, alongside the generic range-overlap ambiguity when both
apply.

The first spatial split side also exists:

```text
split_projective_trace_cell_atlas_uv_visibility_events(...)
```

It retile-compiles the atlas onto a finer child tile grid and recomputes
per-child depth intervals/order. This is not an oblique halfspace-cell renderer
yet, but it is a real compiled split through the existing tile schema: the
simple crossing fixture turns one parent tile into two stable child cells,
left `(0,1)` and right `(1,0)`, with no fallback. If the line still crosses a
child tile, the event-driven fallback path remains explicit.

The first adaptive policy/report is now:

```text
adapt_projective_trace_cell_atlas_uv_visibility_events(...)
ProjectiveTraceCellUVVisibilitySpatialSplitReport
```

The packed atlas still has one global tile size, so the policy keeps the
parent atlas when no UV event exists, otherwise tries divisor child grids from
coarsest to finest. It accepts the first child grid whose residual UV
event-tile count is within budget; if the minimum child grid still crosses the
line, the report returns `accepted=False` and leaves
`visibility_uv_depth_line` fallback on unresolved child cells.

The report now carries parent/output fallback fractions. The high-motion row
now reads the checked-in smoke video and uses the strongest frame-difference
motion centroids as the UV roots of a diagnostic pairwise depth line. This is
video-derived motion-centroid trace geometry, not trained scene
reconstruction. On the current video, the selected adjacent pairs are
`7,8,9`, with root positions approximately `4.395, 4.055, 4.123` on the
8-pixel diagnostic tile. The parent grid has
`parent_uv_event_tile_samples=3` and `parent_fallback_fraction=1.0`; the
adaptive policy chooses child size `4` and reaches
`residual_uv_event_tile_samples=0` with `fallback_fraction=0.0`.

The same before/after report now has an orbit-parameterized fixture. It uses
`q = tan(theta/2)` plus the orbit depth polynomial from `_orbit_trace_coeffs`,
then moves the UV order root over `q in {-0.5,0,0.5}`. The 8-pixel parent tile
has `parent_uv_event_tile_samples=3` and `parent_fallback_fraction=1.0`; the
adaptive policy again chooses child size `2` and reaches
`fallback_fraction=0.0`.

The measurement is now available as a JSON report builder:

```text
research_experiments/star_uvt_feature_tubes/projective_uv_visibility_split_report.py
outputs/projective_uv_visibility_split_report.json
```

The saved report has schema `projective_uv_visibility_split_report_v1`, status
`ok`, `max_parent_fallback_fraction=1.0`, `max_output_fallback_fraction=0.0`,
and `max_cell_growth=4.0`. It includes
`high_motion_video_centroid_line_sweep`, sourced from
`data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4`,
with decoded-frame count, selected pair indices, motion scores, centroids,
UV root positions, and fitted depth coefficients recorded in the row.

The next extraction step is a separate trace-geometry artifact:

```text
research_experiments/star_uvt_feature_tubes/projective_high_motion_trace_geometry_report.py
outputs/projective_high_motion_trace_geometry_report.json
```

It compiles actual STAR UVT trainer-harness tensors from the checked-in
high-motion smoke clip into the projective cell atlas at smoke scale
(`64` tubes, `16` frames, `64px`). The config-faithful row keeps
`velocity_init=zero`; the motion row uses `block_match_gated` initialization;
the trained row runs three dense CPU optimizer steps before extraction. Current
results: fallback fraction stays `0.0`; the zero-velocity row has
interval/dense tile-pair ratio `0.063`; the block-match init row has `58/64`
nonzero velocities, max motion `5.657 px/frame`, and interval/dense tile-pair
ratio `0.293`; the 3-step trained row reduces loss `0.30096 -> 0.29547`,
moves parameters by L1 `67.95` across `center_uv`, `center_t`, `velocity_uv`,
`raw_precision`, `raw_opacity`, and `raw_color` while `depth0` stays fixed
under the detached-sort dense harness, keeps fallback `0.0`, and has
interval/dense tile-pair ratio `0.294`.

The paired visibility stress now checks the complementary invariant: support
debounce is not visibility debounce. A tolerated `0.05px` support overshoot
still refreshes when the stored depth intervals imply the wrong front-to-back
order. The refreshed atlas repairs support, sorts the cells to the live order,
and returns zero order mismatches. This keeps the tolerance scoped to tile
support only; depth/order certificates still gate chart reuse.

The combined orbit regression now checks both conditions in the same
projective/gauged fixture. Two yaw-window traces share the small revolving
camera support drift; with `support_stale_overshoot_epsilon=0.10`, a
support-only refresh does not rebin, but the visibility-aware refresh sees four
stale order samples and rebuilds the atlas to live order `(1, 0)` with zero
remaining order mismatches. This is the current crisp answer to "does rich
projective math replace fallback?": it carries the orbit support, while
visibility remains an explicit certificate.

A second orbit visibility regression checks the actual event-stratification
case. Two rational yaw-window traces have depths that cross at sensor time
`t=0`, between the sampled frames. The compiler reports one continuous
visibility root, refreshes without support staleness, and splits the interval
into `(0,2)` with order `(0,1)` and `(2,4)` with order `(1,0)`. No fallback is
marked and the interval-to-dense trace-sample ratio remains `0.5`. This is the
stronger answer to revolving-camera visibility: use the projective gauge for
smooth support, then cut the trace arrangement on depth-order roots.

The first event-distance headroom allocator is:

```text
projective_interval.support_guard_policy = "slack_budgeted"
```

It uses the same cap-safety structure as `trace_budgeted`, but when a guarded
tile overflows it ranks extra trace ids by distance from the trace's base
support footprint to that exact tile's support event. The nearest-to-crossing
traces get the available guard slots first; primitive id is only the
deterministic tie-break. This turns support-event slack from telemetry into a
guard-allocation rule, though it still needs the harder-scene debounce/error
gate before promotion.

The first combined orbit stress now ties these mechanisms together. One
projective/gauged refresh sees a tail-alpha-certified support drift
(`alpha_tail` between `1e-4` and `3e-4`), a depth-order root that needs event
stratification, and a crowded guard tile where only `12` of `24` extra traces
fit under `tile_capacity=32`. `trace_budgeted` spends those slots on lower-id
far traces; `slack_budgeted` spends them on the near-boundary traces. The
refreshed atlas has zero visibility mismatches, no fallback, max per-cell
active count equal to the cap, and interval/dense ratio about `0.41`.

Current q-UVT smoke:

```text
max_rgb_error = 5.96e-08, zero overflow, pair_ratio = 0.5
anisotropic/depth-affine opt-in q-UVT producer/backend/Metal parity:
149 passed in 19.91s after rebuilding the STAR UVT native extension
UV visibility event certificate + forced interval-backward ABI rebuild:
150 passed in 14.23s
UV event-driven fallback marking:
151 passed in 10.09s
UV finer-grid spatial split:
153 passed in 15.90s
UV adaptive split-vs-fallback policy:
156 passed in 16.69s
UV high-motion split-vs-fallback measurement:
157 passed in 23.47s
UV orbit-parameterized split-vs-fallback measurement:
159 passed in 24.31s
UV split report artifact:
163 passed in 25.46s
video-derived UV split report extraction:
targeted report/adaptive checks: 4 passed in 4.53s
broad focused STAR UVT projective suite + report test:
166 passed in 93.12s
high-motion STAR UVT trace-geometry extraction:
targeted report tests: 2 passed in 35.36s
broad focused STAR UVT projective suite + trace reports:
168 passed in 231.68s
spatial precision VJP:
direct backward matches Torch autograd, q-UVT opt-in route backprops to q_uu/q_vv
broad projective/interval suite: 152 passed in 16.41s
source-view trainer anisotropic precision opt-in:
locked/unlocked bridge tests pass, broad suite: 153 passed in 20.31s
trainable rotated UV footprint:
SPD-safe q_uv cross precision, locked by default, trainable under anisotropic opt-in
focused cross/locked/unlocked tests pass, broad suite: 154 passed in 23.49s
exposure/rolling quadrature verifier:
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.md
contract locks finite-exposure rendered-field integration, rolling unique-time row weights, mixed fallback patching before accumulation, recomputed interval/dense ratios, fallback sample subset consistency, and recomputed Metal summary max/count
focused verifier: 11 passed in 34.79s, saved artifact verified by CLI
exposure/rolling backward verifier:
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.md
contract locks final-image adjoint lowering to sample adjoints by quadrature weights or row_weights, positive sample image/adjoint support, nonzero coeff/opacity/color reference gradients, recomputed rolling reuse ratio, recomputed Metal compare aggregates, and recomputed summary fields
focused verifier: 11 passed in 25.19s, saved artifact verified by CLI
projective interval trainer frame scaling:
actual run_training route with synthetic loader, frames 4/8/16, steps 4
cadence rebuilds 2/2/2, measured rebuilds 1/1/1, max measured-vs-cadence loss delta 0.0, zero tile overflow
synthetic trainer frame-scaling verifier:
projective_interval_trainer_frame_scaling_benchmark.py now exports verify/assert helpers plus --verify-report
contract locks actual run_training pass/loss decrease, cadence/measured loss match, measured rebuild reduction, live staleness checks, zero overflow/fallback/visibility stratification, support-rebin/stale-refresh consistency, and no-first-step timing wins on the synthetic MPS smoke
focused verifier: 6 passed in 11.58s, saved synthetic artifact verified by CLI, strict synthetic+real-video trainer verifier suite: 26 passed in 10.59s
real-video projective interval trainer frame scaling:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling/summary.md
checked-in high-motion clip, frames 4/8/16, size 64, tube_count 128, steps 4
cadence rebuilds 2/2/2, measured rebuilds 1/1/1, max loss delta 0.0, zero tile overflow
measured/cadence no-first-step ratios 0.881/0.352/0.692; support rebins still 3/3/3
guarded real-video trainer support-churn rerun:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard05_tail001/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard10_tail001/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard20_tail001/summary.json
slack_budgeted + tail001 guards 0.25/0.5/1/2px: measured rebuilds 1/1/1, support rebins 0/0/0, max loss delta 0.0
guard0.25 is smallest certified no-churn guard; current rerun ratios stay below cadence for all guards, with max guarded no-first-step ratio 0.590
aggregate guarded support matrix:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json
matrix summary: 5 artifacts, 15 measured rows, default measured support rebins 9, guarded measured support rebins 0, guarded stale refreshes 0, max guarded rebuild ratio 0.5
guard025/guard05/guard1/guard2 are included in the saved real-video verifier matrix and in the aggregate guarded-support artifact; goal + guarded matrix tests pass 39 passed in 0.99s
real-video multiscene trainer matrix:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json
three source-distinct checked-in segments, 6 cadence/measured rows, exact cadence-loss agreement, max no-first ratio 0.550, rebuild ratio 0.5, support rebins/stale refreshes/fallback/overflow/visibility stratifications all 0
real-video multiscene extended functional matrix:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json
five source-distinct checked-in segments, 10 cadence/measured rows, max motion score 7.018424034118652, exact cadence-loss agreement, rebuild ratio 0.5, support rebins/stale refreshes/fallback/overflow/visibility stratifications all 0; max no-first ratio 1.50811535915855, so this is functional broadening evidence, not a timing-win row
real-video multiscene frame-scaling matrix:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json
three source-distinct checked-in segments over frames 4/8/16, 18 cadence/measured rows, frame-growth factor 4.0, exact cadence-loss agreement within 2.98e-8, max no-first ratio 0.690, max no-first/frame-growth ratio 0.438, rebuild ratio 0.5, measured rebuild growth 1.0, support rebins/stale refreshes/fallback/overflow/visibility stratifications all 0
real-video multiscene extended frame-scaling diagnostic:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json
five source-distinct checked-in segments over frames 4/8/16, 30 cadence/measured rows, frame-growth factor 4.0, strict source status failed only the two expected timing gates, exact cadence-loss agreement, rebuild ratio 0.5, measured rebuild growth 1.0, support rebins/stale refreshes/fallback/overflow/visibility stratifications all 0; max no-first ratio 1.188933546093892 and max no-first/frame-growth ratio 1.0009153415685994, so this is diagnostic evidence, not timing-win evidence
real-video multiscene extended timing breakdown:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown/summary.json
pair-level breakdown of the failed five-source frame-scaling source: 15 cadence/measured pairs, 3 no-first pairs over 1.0, 1 normalized frame-growth scene over 1.0 by only 0.0009153415685994037, all failing pairs cache/support clean, max rebuild ratio 0.5, max loss delta 0.0
real-video multiscene extended phase profile:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile/summary.json
saved per-step timing profile for the three no-first misses plus the two growth endpoints, source/case no-first delta 0.0, max step ratio 1.188933546093892, max render-forward ratio 1.3566329017525305 on Bq4rmeIvJbs_seg_000 4f, max backward ratio 1.0839184402497806 on Bq4rmeIvJbs_seg_000 16f, no-first dominant phases render_forward_ms:2 and colorize_loss_ms:1, all profiled pairs cache/support clean
real-video Bq4 traced spike rerun:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun/summary.json
reruns the saved Bq4 4f/16f spike steps with trace_global_steps and projective interval substep timing; all expected steps are traced, cache/support remains clean, traced_bq4_spike_reproduced=false, max measured/cadence no-first ratio is 0.5785517503959672, max measured/cadence projective interval total ratio is 1.2736600499593582, and max feature-state-update ratio is 1.250134158419622
real-video Bq4 16f trace repeat stability:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability/summary.json
repeats the Bq4 16f cadence/measured traced pair three times; all expected steps are traced, cache/support remains clean, paired_repeat_count=3, no_first_spike_reproduced_count=0, projective_total_bump_count=0, feature_state_update_bump_count=0, max no-first ratio 0.45165397508134686, max projective interval total ratio 0.9101288137358652, and max feature-state-update ratio 0.7882220153002857
real-video Bq4 trace sequence order:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order/summary.json
two repeats of mixed_4_to_16 and reverse_16_to_4, all expected steps traced, cache/support clean, paired_16f_ratio_count=4, all 16f no_first_bump_count=0 with max no-first ratio 0.45600195672964483; mixed_4_to_16 max 16f projective-total ratio 0.9606946419165872 and feature-state ratio 1.0006466493572015, while reverse_16_to_4 max 16f projective-total ratio 1.844612661591509 and feature-state ratio 1.73336471126077
real-video Bq4 warmed trace policy order:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order/summary.json
warms with traced 4f/16f cadence/measured cases, then runs two repeats each of cadence_then_measured and measured_then_cadence 16f target pairs; all expected steps traced, cache/support clean, paired_ratio_count=4, no_first_bump_count=1, projective_total_bump_count=3, feature_state_update_bump_count=3, max no-first ratio 1.7836530508238704, max projective-total ratio 1.7184222253396344, max feature-state-update ratio 1.9605903379413647; measured_then_cadence has measured first and is worse, so the bump is not just a second-slot effect
real-video Bq4 fresh-process trace isolation:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
three fresh-process repeats over both 16f policy orders with warmup_discard_repeats=1, all expected steps traced, projective interval substeps present, cache/support clean, paired_ratio_count=6, no_first_bump_count=0, projective_total_bump_count=1, feature_state_update_bump_count=2, max no-first ratio 0.7087283466117477, max projective-total ratio 2.2454207580524894, max feature-state-update ratio 1.2948922914387324; post-warmup median acceptance passes with status pass, post_warmup_pair_count=4, median no-first ratio 0.5645123618278631, median projective-total ratio 0.8356591487478802, and median feature-state-update ratio 0.846418513757801, so timing acceptance should use fresh-process medians/warmup-discard while keeping max outliers as a caveat
real-video multiscene quality tether:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json
saved case payloads from the source-distinct frame-scaling matrix, 9 cadence/measured pairs, max loss-curve delta 0.0, max end-PSNR delta 0.0, min measured PSNR gain 0.02227306365966797, all gradient-flow flags present
real-video multiscene extended quality tether:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_quality_tether/summary.json
saved case payloads from the five-source extended functional matrix, 5 cadence/measured pairs over 5 distinct YouTube sources, max loss-curve delta 0.0, max end-PSNR delta 0.0, min measured PSNR gain 0.04466235637664795, all gradient-flow flags present
real-video multiscene media tether:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json
actual contact-sheet media writer on 3 source-distinct checked-in clips, 3 cadence/measured pairs, max contact-sheet pixel delta 0, matching PNG hashes, valid two-row target/pred layout, max contact-sheet payload-loss delta 0.001525666420389149, min target/pred row stds 0.1444/0.07265, max final full-RGB loss delta 0.0, max loss-curve delta 0.0, min measured PSNR gain 0.0451, all gradient-flow flags present
real-video multiscene extended media tether:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether/summary.json
actual contact-sheet media writer on 5 source-distinct checked-in clips, 5 cadence/measured pairs, max contact-sheet pixel delta 0, matching PNG hashes, valid two-row target/pred layout, max contact-sheet payload-loss delta 0.001525666420389149, min target/pred row stds 0.1444/0.07178, max final full-RGB loss delta 0.0, max loss-curve delta 0.0, min measured PSNR gain 0.0447, all gradient-flow flags present, rebuild ratio 0.5; max no-first ratio 1.2066 so this is media/quality evidence, not timing-win evidence
real-video acceptance envelope:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
consolidates eleven underlying verifiers, including broad10 quality/media
tethering and the Bq4 fresh-process median gate; broad quality distinct source
count 10, broad media distinct source count 10, functional/media scene count 5,
max support rebins 0, max rebuild ratio 0.5,
expected five-source timing failures preserved, fresh-process post-warmup
medians no-first/projective-total/feature-state-update =
0.5645/0.8357/0.8464, no-first bump count 0, strict timing win false,
fresh-process median timing win true, does_not_prove_completion true
real-video timing-variance envelope:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json
consolidates the strict timing failures, phase-shape diagnostics, Bq4 traced reruns/repeats/order probes, and fresh-process acceptance; source_scene_count 5, strict_failure_count 2, all cache/support clean, workload_explains_render_forward_miss_count 0, drop_spike_render_forward_ratio 0.8418254365135661, fresh-process timing status pass with median no-first ratio 0.5645123618278631, strict_timing_win_claimed false, does_not_prove_completion true
trained high-motion trace scaling:
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.md
tiny saved trainer smoke checkpoint from high-motion video, frames 4/8/16
trained dense tile pairs 1542 -> 6016, interval entries 392 -> 573, fallback 0
per-frame baseline at 16f replays 3862 entries; shared interval uses 573
tiny MPS diagnostic at 16f: shared forward/backward 57.0/60.3ms vs per-frame 355.2/367.9ms
larger 64px/128t trained high-motion smoke:
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.md
trained dense tile pairs 3578 -> 14363, interval entries 956 -> 1371, fallback 0
per-frame baseline at 16f replays 9605 entries; shared interval uses 1371
tiny MPS diagnostic at 16f: shared forward/backward 469.7/303.3ms vs per-frame 802.0/1779.1ms
larger 96px/256t cap256 trained high-motion smoke:
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.md
trained dense tile pairs 7820 -> 31255, interval entries 2045 -> 2831, fallback 0
per-frame baseline at 16f replays 20547 entries; shared interval uses 2831
tiny MPS diagnostic at 16f: shared forward/backward 117.1/169.9ms vs per-frame 1282.6/1810.5ms
trained high-motion scaling verifier:
locks top-level config/frame fields, exact trained/per-frame frame coverage, row ratio consistency, fallback-free trained rows, nonzero learned velocity, opacity bounds, positive timing-gradient signals, recomputed summaries, structural interval/trace wins across scales, and final-scale timing wins
trained high-motion + shared-work audit suite: 33 passed in 6.20s
saved artifacts verified:
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
revolving camera variable-segment fiber metric:
16 frames -> 4 temporal charts per tube, SPD q_uv changes sign across orbit
orbit tests: 7 passed in 2.18s, broad suite: 158 passed in 14.19s
revolving camera chart-size image sweep:
8 frames, segment ratios 1/0.5/0.25/0.125, shared-route mean abs error < 0.009
orbit tests: 8 passed in 14.11s, broad suite: 160 passed in 26.11s
revolving camera interval atlas / Metal sweep:
trace counts 16/8/4/2, fallback 0, interval ratio 1.0 -> <0.35
focused interval tests: 2 passed in 10.42s, broad suite: 163 passed in 33.16s
revolving camera interval backward:
Metal autograd backprops into orbit chart centers, color, opacity, and all q_uvt entries
focused backward: 1 passed in 3.37s, broad suite: 164 passed in 28.66s
revolving camera frame-growth work units:
8/16/32 frames keep 8 orbit charts and 8 atlas traces while per-frame route grows 16/32/64 segments
dense samples 156 -> 820, interval entries 99 -> 156, interval ratio 0.635 -> 0.190, fallback 0
focused frame-growth: 1 passed in 12.13s, orbit file: 13 passed in 33.10s, broad suite: 165 passed in 48.65s
revolving camera backward frame densification:
4/8 frames keep 4 orbit charts/traces while interval Metal VJP reaches ma, opacity, color, q_uv, and temporal q_uvt terms
focused backward gates: 2 passed in 4.28s, orbit file: 14 passed in 30.73s, broad suite: 166 passed in 49.64s
measured revolving-camera fixed-chart scaling artifact:
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.md
default scale is now 8/16/32/64 frames to avoid the too-small MPS launch-noise regime
fixed charts keep traces 8/8/8/8 and payload 608 bytes while per-frame grows traces 16/32/64/128 and payload 1216 -> 9728
fixed CPU compile 16.49 -> 22.41ms versus per-frame 37.19 -> 246.95ms; 64f fixed/per-frame compile ratio 0.091
fixed 64f forward/backward is 0.117/0.158 of the per-frame route on the small prewarmed MPS diagnostic
revolving fixed-chart scaling verifier:
projective_orbit_fixed_chart_scaling_benchmark.py now exports verify/assert helpers plus --verify-report
contract locks constant fixed chart/trace/payload counts, row-level interval-ratio and CPU-compile consistency, zero fallback, slower interval growth than dense samples, fixed/per-frame CPU/GPU timing ratios, direct Metal gradients into coeff/opacity/color/spatial precision, and autograd gradients into ma/opacity/color/q_uv/q_uvt
focused verifier: 10 passed, saved artifact verified by CLI, trained+shared audit suite: 33 passed in 6.20s, shared aggregate verifies with --verify-current-inputs, goal-progress aggregate verifies with --verify-current-inputs
```

Next implementation gate:

```text
run the tail-alpha certificate through broader scenes and image-error checks,
then make support guards motion/update-aware from overshoot margins, scale the
trained high-motion checkpoint geometry gate beyond tiny smoke settings,
decide whether an oblique/fiber halfspace cell is worth adding,
decide whether depth-plane slopes need gradients, move the revolving-camera
interval/fallback benchmark from synthetic orbit to real high-motion views, and
broaden the same trace object to WorldFoam cells
```
