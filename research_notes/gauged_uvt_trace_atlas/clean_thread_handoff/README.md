# Clean Thread Handoff: Gauged UVT Trace Atlas

Date: 2026-05-24

Use this file to start a clean thread with a precise goal. It is intentionally
shorter than the full theory folder, but it preserves the objective, current
state, key math, and next gates.

## New Thread Goal

Continue the Gauged UVT Trace Atlas project.

Build a renderer/compiler for:

```text
4D spacetime primitives
    -> known camera program / sensor ray bundle
    -> reusable UVT viewport-time traces
    -> fast 2D rasters across time
```

The win condition is not "avoid output pixels." Output still costs:

```text
O(F H W)
```

The win condition is:

```text
projection + support + binning + visibility + backward replay
grow sublinearly with frame count F
```

by sharing compute and memory bandwidth across time.

## Meta Goal

Do not treat revolving cameras, finite exposure, or rolling shutter as hacks on
top of affine UVT tubes.

Treat them as camera programs:

```text
Gamma: E_Gamma -> M
```

and compile world primitives through the camera ray bundle:

```text
UVT trace = pi_* Gamma^* world_primitive
```

The current STAR UVT tensors are one local gauge expression of this object. The actual
representation should be an atlas of event-certified gauge domains over sensor
time.

Terminology: "chart" here means a local trivialization of the camera-ray bundle
with validity certificates. It is not a weak fitted patch and not just a stable
sort-order region. A domain certifies projection regularity, trace error,
support, tile-time membership, depth/order behavior, interval gates, and
backward support.

## First Files To Read

Read these before proposing architecture:

```text
AGENTS.md
PROJECT_INDEX.md
research_notes/gauged_uvt_trace_atlas/00_WHAT_IS_THIS_GOAL.md
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
research_notes/gauged_uvt_trace_atlas/README.md
research_notes/gauged_uvt_trace_atlas/GAUGE_DOMAINS_NOT_CHARTS.md
research_notes/gauged_uvt_trace_atlas/03_projective_rational_traces/README.md
research_notes/gauged_uvt_trace_atlas/09_metal_acceptance_plan/README.md
agent_notes/loose_notes/2026-05-24_01-17-46_gauged_uvt_goal_contract_and_chart_fit.md
```

If touching existing STAR UVT or WorldFoam training claims, also read:

```text
TODO/README.md
EXPERIMENTS.md
BASELINES.md
agent_notes/key_learnings.md
```

## Current Implementation State

Latest audit update:

```text
The revolving-orbit fixed-chart benchmark default now uses 8/16/32/64 frames
with warmup=2. This keeps the saved artifact out of the too-small MPS
launch-noise regime while preserving the existing verifier thresholds.
Canonical orbit artifact verifies at:
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
Final fixed/per-frame ratios: payload/trace/segment=0.0625, CPU=0.091,
forward=0.117, backward=0.158.
The old 4/8/16/32 timing failure is quarantined at:
outputs/benchmarks/2026-05-25_star_uvt_revolving_orbit_fixed_chart_scaling_current_timing_fail/
Trained high-motion + shared-work audit tests: 33 passed in 6.20s.
Regenerated aggregate artifact verifies at:
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
Current-input acceptance command:
PYTHONPATH=src/train .venv/bin/python research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json --verify-current-inputs
Aggregate status is ok; no default input artifacts are currently missing, and
the saved aggregate matches the current input artifacts.
Regenerated top-level goal-progress artifact verifies at:
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
Current-input acceptance command:
PYTHONPATH=src/train .venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
It proves thirty-four progress rows, including one-parameter camera-family bundle
math, two-parameter camera-family bundle math, one-parameter camera-family
shared-metadata scaling, two-parameter camera-family shared-metadata scaling,
Q2 camera-family slice lowering into the existing interval Metal forward/backward path,
Q2 camera-family shared-backward chain-rule accumulation from interval Metal VJPs,
Q2 camera-family single-launch materialized Metal batching,
Q2 camera-family native family trace eval/VJP,
Q2 camera-family native interval forward rendering/compositing/visibility,
Q2 camera-family native interval backward/VJP into shared family coefficients
and q-basis values with compiled visibility/order held fixed,
Q2 camera-family stable-topology tile/order metadata reuse,
Q2 camera-family split-strata tile/order metadata reuse,
Q2 camera-family active-set strata metadata reuse,
checked-in high-motion real-video active-set distribution evidence,
a small synthetic compiled-adjoint trainer smoke, and a checked-in high-motion
real-video trainer smoke, a real-video guarded-support matrix, a
source-distinct real-video multiscene trainer matrix, a five-source extended
functional trainer matrix, a source-distinct real-video frame-scaling matrix, a
five-source extended frame-scaling diagnostic with expected timing failures, a
source-distinct quality tether, a five-source extended quality tether, a
source-distinct plus five-source media tether through the actual contact-sheet
writer, the Bq4 fresh-process median timing gate, the real-video acceptance
envelope, the real-video timing-variance envelope, and the real-video
compiled-adjoint replacement; records
`shared_work.current_input_errors = []`, and keeps `full_goal_completion` open.
Goal-completion gap artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
This is the current machine-checked remaining-work contract, not a completion
claim. It verifies the goal-progress artifact through current-input acceptance,
then checks the real-video acceptance-envelope, timing-variance-envelope, and
shared-work inputs plus the broad10 real-video trainer matrix and broad10
quality tether, broad10 media tether, timing-protocol acceptance, and
compiled-adjoint replacement. It proves
`formal_goal_memory_and_audit`, `sublinear_world_side_work_proxy`,
`broad_real_scene_quality_acceptance`,
`full_compiled_adjoint_trainer_replacement`, and
`timing_acceptance_protocol`; keeps only `full_goal_completion` partial;
records
broad_quality_source_gap 0, broad_media_source_gap 0,
broad_quality_frame_count_gap 0, strict_timing_failure_gap 0,
timing_acceptance_gap 0, compiled_trainer_source_gap 0, and
compiled_trainer_replacement_gap 0; and preserves `completion_ready=false`
plus `does_not_prove_completion=true`.
Current-input acceptance command:
PYTHONPATH=src/train .venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json --verify-current-inputs
Goal-completion promotion audit:
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json
This is the current authoritative completion artifact. It consumes the
goal-completion gap report, verifies it against current inputs, and promotes the
lower non-completion stack into six proved objective rows:
`scope_and_key_math_preserved`, `sensor_time_trace_compiler_evidence`,
`sublinear_non_pixel_work_evidence`, `broad_real_video_acceptance_evidence`,
`compiled_adjoint_training_evidence`, and `final_completion_promotion`. It
records `status=complete`, `completion_ready=true`, `is_goal_complete=true`,
`does_not_prove_completion=false`, and `open_requirement_ids=[]`.
Current-input acceptance command:
PYTHONPATH=src/train .venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_completion_promotion_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json --verify-current-inputs
Broad10 quality tether:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json
It covers 10 source-distinct quality pairs with all gradient flags present,
positive PSNR gains, max loss/RGB-loss curve delta 1.49e-8 under a 2e-8
float32-tick tolerance, and no media/timing completion claim.
Broad10 media tether:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json
It covers 10 source-distinct media pairs through the actual contact-sheet
writer with pixel-identical sheets, matching hashes, nontrivial target/pred
rows, max final RGB-loss delta 2.98e-8, and no overflow/fallback/visibility
stratification.
Frame-count breadth diagnostic:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic/summary.json
It accepts 4/8/16/32-frame breadth evidence as a diagnostic, not a strict
timing-win claim.
Timing protocol acceptance:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json
It promotes fresh-process median timing with warmup discard as the accepted
timing contract and keeps the two strict warm-state failures as caveats.
Compiled-adjoint replacement:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json
It verifies the practical direct-atomic RGB trainer route backed by compiled
projective interval traces and the interval Metal direct VJP. The artifact
covers 20 broad10 case payloads, all projective-interval main path, all
renderer gradient flags, measured cache reuse, and
compiled_trainer_replacement_gap 0.
Focused progress/gap/replacement/promotion tests: 82 passed in 4.02s. Wider
timing-protocol/frame-breadth/media/acceptance/compiled-adjoint/gap/
promotion/goal-progress bundle: 121 passed in 4.72s.
Two-parameter camera-family gauge artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_gauge/summary.json
Two-parameter camera-family gauge metrics: max value error 8.42e-14, max
primitive-gradient error 2.28e-12, q_phase gradient error 1.82e-11, q_height
gradient error 1.10e-11, finite-difference checks below 3.26e-10.
Two-parameter camera-family shared-work artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_shared_work_scaling/summary.json
Two-parameter camera-family metrics: shared payload growth 1.0x vs per-q-pair
replay 64.0x, final payload ratio 0.0625, final chart ratio 0.015625, max fit
residual 0.111px.
Two-parameter camera-family Metal lowering artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_lowering/summary.json
Two-parameter camera-family Metal lowering metrics: one shared Q2 coefficient
table lowers into ordinary Omega x T interval Metal slices over a 5x5 q grid;
25 forward/backward Metal rows produce nonzero images and coeff/opacity/color
gradients; family/replay payload ratio 0.178; peak slice/replay payload ratio
0.04. This is slice lowering, not native Q2/Qn Metal evaluation.
Two-parameter camera-family Metal chain-rule artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_chain_rule/summary.json
Two-parameter camera-family Metal chain-rule metrics: per-slice interval Metal
VJPs over the same 5x5 q grid accumulate into one shared Q2 family adjoint;
shared/replay gradient payload ratio 0.24; max finite-difference relative
error 4.91e-05; shared-family gradient support is nonzero. This is
shared-family backward accumulation over Metal slices, not native Q2/Qn Metal
evaluation.
Two-parameter camera-family materialized-batch artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_materialized_batch/summary.json
Two-parameter camera-family materialized-batch metrics: all 25 q-pair slices
pack into one ordinary interval Metal forward/backward launch; images match
the per-slice reference with max abs error 0.0; shared-family gradients match
with max relative error 9.34e-08; forward/backward launch ratios are 0.04; the
materialized/replay trace payload ratio is intentionally 1.0, while the true
family table would be 0.178x of the materialized payload. This artifact by
itself proves launch reuse; native family trace eval/VJP is the separate row
below.
Two-parameter camera-family native eval/VJP artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_eval/summary.json
Two-parameter camera-family native eval/VJP metrics: the Metal shader evaluates
all `Q x N x S` homogeneous trace samples directly from shared
`family_coeffs[N,9,B]` and `q_basis[Q,B]`, and accumulates direct VJPs into
both tensors. Family/materialized coefficient payload ratio is 0.24;
family-plus-q/materialized coefficient payload ratio is 0.5733333333333334;
max value relative error is 6.58e-08; max family-gradient relative error is
5.72e-08; max q-basis-gradient relative error is 2.58e-07. This proves native
family trace eval/VJP, not full interval-cell compositing by itself.
Two-parameter camera-family native interval-forward artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_forward/summary.json
Two-parameter camera-family native interval-forward metrics: the Metal
interval renderer consumes shared `family_coeffs[N,9,B]` and `q_basis[Q,B]`
directly, derives per-q trace coefficients shader-side, and
depth-sorts/composites through the interval-cell path. It matches the
materialized single-launch reference with max image abs error 0.0 and max
image relative error 0.0 over 100 batched frames; family/materialized trace
coefficient payload ratio is 0.16615384615384615; full native-family
forward/materialized trace payload ratio is 0.4461538461538462; native and
materialized image abs sums both equal 1992.59228515625. This proves native
interval forward rendering/compositing/visibility over family coefficients.
Two-parameter camera-family native interval-backward artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward/summary.json
Two-parameter camera-family native interval-backward metrics: the Metal
interval renderer accumulates the interval-cell VJP directly into shared
`family_coeffs[N,9,B]` and `q_basis[Q,B]`, with tile membership and depth
order held as compiled constants. It reports native-family/materialized-gradient
payload ratio 0.2926315789473684; native family-coefficient/materialized-gradient
payload ratio 0.11368421052631579; max family-gradient relative error
2.3355269149760716e-06; max q-basis-gradient relative error
8.51117079037067e-07; and nonzero family/q-basis gradient support.
Two-parameter camera-family tile/order reuse artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse/summary.json
Two-parameter camera-family tile/order reuse metrics: one local tile/order
topology plus q-index applicability expands back to all 25 materialized q-pair
cells; conservative family-union depth intervals preserve the order with min
gap 0.6033999919891357; materialized tile/order metadata grows 25.0x while
shared topology metadata grows 1.0x; shared/materialized metadata ratio is
0.11692307692307692. This is the stable-topology case, not the full
split-strata q-family metadata solution.
Two-parameter camera-family tile/order strata artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata/summary.json
Two-parameter camera-family tile/order strata metrics: a deliberately
non-constant q-family depth order compresses 25 materialized q-pair cells into
two certified topology strata; materialized metadata grows 25.0x while shared
metadata grows 2.0x; shared/materialized metadata ratio is
0.15692307692307692; minimum stratum union depth gap is
0.33200000002980246. This proves local order-strata metadata compression, not
active-set split compression.
Two-parameter camera-family active-set strata artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata/summary.json
Two-parameter camera-family active-set strata metrics: a deliberately
non-constant q-family support/culling topology compresses 25 materialized
q-pair cells into three certified active-set topology strata; materialized
metadata grows 25.0x while shared metadata grows 3.0x; shared/materialized
metadata ratio is 0.19692307692307692; minimum active-set union depth gap is
0.2630399994850159. This proves local active-set metadata compression, not
real-scene active-set distribution acceptance.
Real active-set distribution artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_active_set_distribution/summary.json
Real active-set distribution metrics: three checked-in high-motion trained
artifacts over 4/8/16 frames contribute nine trained-checkpoint atlas rows;
all source videos exist, all underlying verifiers pass, rows are fallback-free,
max cells per active-set group is 3, max active-set-group/dense-tile-pair ratio
is 0.04009499860296172, and max cell/group ratio is 1.3214953271028038. This
proves active-set topology is now measured on real compiled traces, but not
broad real-scene quality acceptance.
Synthetic trainer smoke artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_interval_trainer_frame_scaling/summary.json
Synthetic trainer smoke metrics: measured rebuilds 1/1/1 vs cadence 2/2/2,
max no-first-step ratio 0.839, max loss delta 2.98e-8.
Real-video trainer smoke artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling/summary.json
Real-video trainer smoke metrics: measured rebuilds 1/1/1 vs cadence 2/2/2,
max no-first-step ratio 0.881, max loss delta 0.0, max tile count 18, fallback 0.
Real-video guarded-support matrix artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json
Guarded-support matrix metrics: 5 artifacts, 15 measured rows, default measured
support rebins 9, guarded measured support rebins 0, guarded stale refreshes 0,
max guarded no-first-step ratio 0.590, max guarded rebuild ratio 0.5.
Real-video multiscene trainer matrix artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json
Multiscene matrix metrics: 3 source-distinct checked-in video segments, 6 rows,
max measured/cadence no-first-step ratio 0.550, max rebuild ratio 0.5, exact
cadence-loss agreement, and zero measured support rebins/stale refreshes,
overflow, fallback marks, and visibility stratifications.
Real-video multiscene extended functional matrix artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json
Extended matrix metrics: 5 source-distinct checked-in video segments, 10 rows,
max motion score 7.018424034118652, exact cadence-loss agreement, max rebuild
ratio 0.5, and zero measured support rebins/stale refreshes, overflow, fallback
marks, and visibility stratifications. Max no-first ratio is 1.50811535915855,
so this broadens functional coverage but is not a timing-win row.
Real-video multiscene frame-scaling matrix artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json
Multiscene frame-scaling metrics: 3 source-distinct checked-in video segments
over frames 4/8/16, 18 cadence/measured rows, frame-growth factor 4.0, max
measured/cadence no-first-step ratio 0.690, max measured timing-growth/frame-growth
ratio 0.438, max rebuild ratio 0.5, measured rebuild growth 1.0, cadence-loss
agreement within 2.98e-8, and zero support rebins/stale refreshes, overflow,
fallback marks, and visibility stratifications.
Real-video multiscene extended frame-scaling diagnostic artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json
Extended frame-scaling diagnostic metrics: 5 source-distinct checked-in video
segments over frames 4/8/16, 30 cadence/measured rows, frame-growth factor
4.0, strict source status failed only the two expected timing gates, exact
cadence-loss agreement, rebuild ratio 0.5, measured rebuild growth 1.0, and
zero support rebins/stale refreshes, support tail/overshoot, overflow,
fallback marks, and visibility stratifications. Max no-first-step ratio is
1.188933546093892 and max no-first/frame-growth ratio is
1.0009153415685994, so this is a correctness/cache/support diagnostic, not
timing-win evidence.
Real-video multiscene extended timing-breakdown artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown/summary.json
Extended timing-breakdown metrics: pair-level read of the failed five-source
frame-scaling source, 15 cadence/measured pairs, 3 no-first pairs over 1.0,
1 normalized frame-growth scene over 1.0 by only 0.0009153415685994037,
all failing pairs cache/support clean, max rebuild ratio 0.5, max loss delta
0.0. Treat the current timing miss as evaluation/noise/phase-shape until a
repeat or phase-profile diagnostic says otherwise.
Real-video multiscene extended phase-profile artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile/summary.json
Extended phase-profile metrics: saved per-step timing profile for the three
no-first misses plus the two growth endpoints, source/case no-first delta 0.0,
max step ratio 1.188933546093892, max render-forward ratio 1.3566329017525305
on Bq4rmeIvJbs_seg_000 4f, max backward ratio 1.0839184402497806 on
Bq4rmeIvJbs_seg_000 16f, no-first dominant phases render_forward_ms:2 and
colorize_loss_ms:1, all profiled rows cache/support clean.
Real-video multiscene extended render-forward residual artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_residual/summary.json
Extended render-forward residual metrics: all 15 cadence/measured saved case
pairs have identical saved tile_stats, the three no-first misses also have
identical tile workload, max tile-stat delta 0.0, max render-forward ratio
1.3566329017525305 on Bq4rmeIvJbs_seg_000 4f, max render-forward per
clipped-ref ratio 1.3566329017525305, and workload_explains_render_forward_miss_count
0. Next timing target: Bq4 render-forward substep instrumentation/replay, not
support rebins, cache invalidation, or candidate-count changes.
Real-video multiscene extended render-forward shape artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape/summary.json
Extended render-forward shape metrics: saved per-step timings only, no saved
chunk traces. All three no-first misses are single-spike driven in render-
forward and whole-step time; after dropping the largest positive render delta,
the worst no-first miss render ratio is 0.8418254365135661. Max render ratio
remains 1.3566329017525305 on Bq4rmeIvJbs_seg_000 4f, max no-first render
spread is 5.383083741915209, and max no-first render spike delta is
728.0996670015156 ms.
Real-video Bq4 traced spike rerun artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun/summary.json
Bq4 traced rerun metrics: reruns the Bq4 4f/16f cadence/measured spike-step
cases with trace_global_steps and projective interval substep timing; all
expected steps traced, all traced chunks include timing, cache/support remains
clean, traced_bq4_spike_reproduced=false, measured/cadence no-first ratios
0.4538476088322886 and 0.5785517503959672, measured/cadence projective interval
total ratios 0.5054386427773483 and 1.2736600499593582, feature-state-update
ratios 0.44341185194975186 and 1.250134158419622. Next real experiment:
repeat/stability profiling around feature_state_update/live-update phase cost,
not new chart/fiber math.
Real-video Bq4 16f trace repeat-stability artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability/summary.json
Bq4 repeat-stability metrics: repeats the Bq4 16f cadence/measured traced pair
three times; all expected steps traced, all chunks include projective interval
timing, cache/support remains clean, paired_repeat_count=3,
no_first_spike_reproduced_count=0, projective_total_bump_count=0,
feature_state_update_bump_count=0, max no-first ratio 0.45165397508134686,
max projective interval total ratio 0.9101288137358652, and max
feature-state-update ratio 0.7882220153002857. This weakens the one-shot 16f
feature-state-update bump and moves the timing caveat toward mixed-sequence or
warm-state launch variance.
Real-video Bq4 trace sequence-order artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order/summary.json
Bq4 sequence-order metrics: two repeats of mixed_4_to_16 and reverse_16_to_4;
all expected steps traced, all chunks include projective interval timing,
cache/support remains clean, paired_16f_ratio_count=4, all 16f no-first
bump_count=0, max 16f no-first ratio 0.45600195672964483. The mixed_4_to_16
sequence stays mostly benign: max 16f projective-total ratio 0.9606946419165872
and feature-state-update ratio 1.0006466493572015. The reverse_16_to_4 sequence
shows order-sensitive substep variance: max 16f projective-total ratio
1.844612661591509 and feature-state-update ratio 1.73336471126077. This
supports warm-state/launch-order variance without contradicting the no-first
measured/cadence win.
Real-video Bq4 warmed trace policy-order artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order/summary.json
Bq4 policy-order metrics: warms the process with traced Bq4 4f/16f
cadence/measured cases, then runs two repeats each of cadence_then_measured and
measured_then_cadence 16f target pairs; all expected steps traced, all chunks
include projective interval timing, cache/support remains clean,
paired_ratio_count=4, no_first_bump_count=1, projective_total_bump_count=3,
feature_state_update_bump_count=3, max no-first ratio 1.7836530508238704, max
projective-total ratio 1.7184222253396344, and max feature-state-update ratio
1.9605903379413647. measured_then_cadence has measured in slot 0 and is worse,
so the bump is not just a second-slot effect; treat it as policy/order/warm-state
timing variance.
Real-video Bq4 fresh-process trace isolation artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
Bq4 fresh-process metrics: three isolated-process repeats over both 16f
policy-order target pairs with warmup_discard_repeats=1; all rows marked
fresh_process=true, all expected steps traced, all chunks include projective
interval timing, cache/support remains clean, paired_ratio_count=6,
no_first_bump_count=0, projective_total_bump_count=1,
feature_state_update_bump_count=2, max no-first ratio 0.7087283466117477, max
projective-total ratio 2.2454207580524894, and max feature-state-update ratio
1.2948922914387324. The post-warmup median acceptance view passes:
status=pass, post_warmup_pair_count=4, median no-first ratio
0.5645123618278631, median projective-total ratio 0.8356591487478802, and
median feature-state-update ratio 0.846418513757801. Treat this as median
timing evidence with a max-outlier caveat, not as a reason to change fiber/gauge
math.
Real-video multiscene quality tether artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json
Quality-tether metrics: reads saved case payloads from the source-distinct
frame-scaling matrix, verifies 9 cadence/measured pairs, max loss-curve delta
0.0, max RGB-loss-curve delta 0.0, max end-PSNR delta 0.0, min measured PSNR
gain 0.02227306365966797, all gradient-flow flags present. This is a cadence
tether, not broad real-scene quality acceptance.
Real-video multiscene extended quality tether artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_quality_tether/summary.json
Extended quality-tether metrics: reads saved case payloads from the five-source
functional matrix, verifies 5 cadence/measured pairs over 5 distinct YouTube
sources, max loss-curve delta 0.0, max RGB-loss-curve delta 0.0, max end-PSNR
delta 0.0, min measured PSNR gain 0.04466235637664795, all gradient-flow flags
present. This extends the cadence tether to the five-source matrix, not broad
real-scene quality acceptance.
Real-video multiscene media tether artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json
Media-tether metrics: runs the actual contact-sheet media writer on 3
source-distinct cadence/measured pairs, verifies max contact-sheet pixel delta
0 with matching PNG hashes, valid two-row target/pred layout, max
artifact-MSE-vs-payload-loss delta 0.001525666420389149, nontrivial target/pred
rows with min stds 0.14441643529730494 / 0.07265247844694266, max final
full-RGB loss delta 0.0, max loss-curve delta 0.0, min measured PSNR gain
0.04511058330535889, all gradient-flow flags present, max no-first-step ratio
0.9316588494614714, rebuild ratio 0.5, and zero overflow/fallback/visibility
stratifications. This is a real media artifact tether, not broad real-scene
quality acceptance.
Real-video multiscene extended media tether artifact:
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether/summary.json
Extended media-tether metrics: runs the actual contact-sheet media writer on 5
source-distinct cadence/measured pairs, verifies max contact-sheet pixel delta
0 with matching PNG hashes, valid two-row target/pred layout, max
artifact-MSE-vs-payload-loss delta 0.001525666420389149, nontrivial target/pred
rows with min stds 0.14441643529730494 / 0.07178262974117959, max final
full-RGB loss delta 0.0, max loss-curve delta 0.0, min measured PSNR gain
0.04466235637664795, all gradient-flow flags present, rebuild ratio 0.5, and
zero overflow/fallback/visibility stratifications. Max no-first-step ratio is
1.2065694734694634, so this is five-source media evidence, not timing-win
evidence.
```

The first projective gauge probe exists in the STAR UVT variant:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

Implemented compiler/evaluator helpers:

```text
eval_projective_trace(coeffs, times, eps)
eval_projective_trace_torch(coeffs, times, eps)
fit_projective_trace_polynomial(coeffs, times, degree)
eval_projective_trace_polynomial_fit(fit, times)
split_projective_trace_windows(coeffs, times, degree, thresholds)
```

The Metal Gate A evaluator exists behind:

```text
torch.ops.star_uvt_v0.projective_trace_eval
```

The Gate B chart-fit/window-split helpers are CPU/Torch compiler-side tools, not
renderer hot-path kernels yet.

The chart-fit certificate now includes:

```text
denominator_has_root
denominator_min_abs
```

so denominator roots are treated as continuous chart boundaries, including when
the root falls between sampled frames. `denominator_min_abs` is the analytic
minimum over the same interval, so a root-free denominator that approaches zero
between samples is also rejected by the requested margin.

## Tests Already Passing

Focused projective trace tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py -q
```

Last known result:

```text
28 passed
```

Renderer import/regression smoke:

```bash
PYTHONPATH=src/train uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_pair_benchmark.py \
  --scenes single_static
```

Last known key result:

```text
max_rgb_error = 5.960464477539063e-08
mean_rgb_error = 1.1123878485008731e-09
overflow_tile_count = 0
unstable_tile_fraction = 0.0
```

## Key Math To Preserve

Sensor-time base:

```text
B = Omega x T
y = (u, v, tau)
```

Camera ray bundle:

```text
pi: E_Gamma -> B
pi^{-1}(y) = F_y
```

Camera program:

```text
Gamma: E_Gamma -> M
M = R^3 x R
```

Trace invariant:

```text
bar_rho_i(y) = integral_{F_y} rho_i(Gamma(e)) dmu_y(e)
```

Local gauge:

```text
chi_a: E_Gamma | C_a -> C_a x D_a
(y, z_a) = chi_a(e)
```

Projective revolving-camera chart:

```text
h(t) = K(t) [R(t)|T(t)] X(t)
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
```

`h_z = 0` is a chart boundary, not just a numerical problem.

Visibility strata:

```text
z_hat_i(y) = z_hat_j(y)
```

Order ambiguity can be accepted only when the swap is visually negligible:

```text
|Delta I_ij(y)| <= alpha_i(y) alpha_j(y) |c_i(y) - c_j(y)|
```

Backward objective:

```text
dL/dtheta_i =
  sum_a integral_{C_a}
    A(y)^T dI(y)/dphi_i,a dphi_i,a/dtheta_i dy
```

where the goal is to avoid replaying full per-frame world projection/binning
and sorting during backward.

## Current Belief

The right model is:

```text
Gauged UVT = camera-ray bundle trace atlas
```

Projective/rational gauges are the rich math for revolving cameras. A full
orbit should be handled by:

```text
{C_a, chi_a, transition_ab}
```

not by one global affine UVT splat. Residuals are chart-validity certificates.
Fallback is a guardrail for uneconomical local cases, not the main theory.

## Latest Completed Gate

The first synthetic orbit chart-count gate now passes.

Question:

```text
Does split_projective_trace_windows produce a small chart count for smooth
revolving-camera traces, and does that count grow with trace complexity rather
than directly with frame count?
```

Delivered:

```text
tests/test_star_uvt_projective_orbit_windows.py
```

Measured result:

```text
same visible orbit span:
  F=16,32,64,128,256 -> 4,4,4,4,4 accepted windows

fixed F=128, increasing orbit span:
  span=15,30,60,90,120 degrees -> 1,2,3,7,7 accepted windows

fixed 90-degree span, tightening residual:
  max_residual_uv=0.08,0.015,0.005 -> 2,6,8 accepted windows
```

The denominator-boundary case also covers a root that falls between frame
samples; it returns an unresolved window with reason:

```text
denominator_boundary
```

See:

```text
agent_notes/loose_notes/2026-05-24_01-31-06_gauged_uvt_orbit_chart_count_gate.md
```

The first support-bound gate also passes.

Delivered:

```text
bound_projective_trace_window(window)
bound_projective_trace_windows(windows)
```

in:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

These helpers compute continuous polynomial UV/depth bounds for accepted
projective windows and inflate them by the fit residual certificates. The test
proves sampled rational orbit traces stay inside the compiled bounds. It also
proves unresolved denominator-boundary windows refuse default bounds.

The first depth/denominator visibility sidecar gate also passes.

Delivered:

```text
make_projective_trace_visibility_sidecar(window)
make_projective_trace_visibility_sidecars(windows)
compare_projective_trace_depth_order(sidecar_a, sidecar_b)
```

These sidecars record:

```text
denominator_min_abs
denominator_has_root
depth_uncertainty
depth_monotonicity
chart_gauge_id
```

plus depth ranges and slope ranges. The synthetic visibility test proves stable
front/back order is recognized and crossing depth traces are marked ambiguous.

The first visible-swap cost gate also passes.

Delivered:

```text
make_projective_trace_appearance_sidecar(alpha_max, color)
bound_projective_trace_visible_swap_cost(order, appearance_a, appearance_b)
```

It applies:

```text
|Delta I_ij| <= alpha_i alpha_j |c_i - c_j|
```

with optional color interval uncertainty. Tests prove low-opacity ambiguous
crossings are safely commutable, high-opacity/color-contrast crossings require
fallback, and color uncertainty is included in the bound.

The first compiler-side tile-time binning gate also passes.

Delivered:

```text
ProjectiveTraceTileTimeRecord
bin_projective_trace_support_bounds(bounds, image_width, image_height, tile_size)
```

It maps accepted support bounds into compressed primitive/window/tile-rectangle
/ time-window records, skips offscreen traces, preserves custom primitive ids,
and carries optional fallback flags/reasons from visibility masks.

The first tile-time atlas assembly gate also passes.

Delivered:

```text
ProjectiveTraceTileTimeCell
assemble_projective_trace_tile_time_atlas(records)
```

It expands compressed tile rectangles into tile-time cells with active primitive
sets, depth-sorted order metadata, depth intervals, fallback flags, and fallback
reasons.

The first dense-reference correctness gate also passes.

Delivered:

```text
tests/test_star_uvt_projective_correctness.py
```

It checks two contracts:

```text
compiled atlas cells cover dense per-frame projective orbit samples
stable atlas depth order matches dense per-frame depth sorting
```

The first minimal atlas reference renderer gate also passes.

Delivered:

```text
render_projective_trace_tile_time_atlas_reference(...)
```

The helper is a CPU/Torch oracle, not a hot path. It consumes tile-time atlas
cells, evaluates candidate projective centers, composites ordered candidates
with a simple screen-space Gaussian opacity model, and matches dense per-frame
compositing on a small stable-depth scene.

The first guarded STAR UVT q-UVT bridge also passes.

Delivered:

```text
ProjectiveTraceUVTBridge
projective_trace_windows_to_uvt_tubes(...)
```

It lowers accepted degree-1 projective chart windows into the existing STAR UVT
`ma/q_uvt/depth0/depth_beta` renderer contract. The focused test renders those
lowered tubes with `brute_force_render_uvt_tubes(...)` and matches the atlas
reference renderer. This is the first path from gauged projective charts back
into the existing STAR UVT renderer surface.

The guarded MPS/Metal q-UVT bridge parity test also passes on this machine.
The test lowers affine projective charts, renders with `render_uvt_tubes(...)`,
and matches the CPU q-UVT reference.

The first explicit interval-gated split-chart bridge also passes.

Delivered:

```text
ProjectiveTraceUVTBridge.active_start / active_stop
render_projective_trace_uvt_bridge_reference(..., use_window_gates=True)
```

Split affine q-UVT chart segments now carry exact sample-domain `[start, stop)`
gates. The focused test uses a curved trace split into local affine windows and
proves gated q-UVT segments match dense per-frame rendering while ungated
segments visibly leak.

Those interval gates now reach Metal through a span-gated wrapper.

Delivered:

```text
projective_trace_uvt_bridge_active_spans(...)
render_projective_trace_uvt_bridge_metal_gated(...)
```

The wrapper partitions frames at interval sidecar boundaries, renders each
constant-active-set span through the existing `render_uvt_tubes(...)` Metal
path, and copies only the matching span into the final image. This is not a
native shader-side interval gate buffer, but it proves split affine chart
segments can render through Metal without leaking outside their chart domains.

The native shader-side interval gate buffer now exists too.

Delivered:

```text
torch.ops.star_uvt_v0.render_gated(...)
render_uvt_tubes_gated(...)
torch.ops.star_uvt_v0.direct_atomic_backward_gated(...)
direct_atomic_backward_gated(...)
direct_backward_projective_trace_uvt_bridge_metal_gated(...)
```

The gated Metal binning kernel clamps each tube to its `[active_start,
active_stop)` sample-domain frame interval, and the gated render kernel skips
inactive tubes per sample. `render_projective_trace_uvt_bridge_metal_gated(...)`
now uses this native path rather than the older span-partition wrapper. The
matching direct atomic VJP path uses the same intervals and has a focused
parity test against masked single-tube direct-backward references.

The first bridge-level trainability smoke also passes: split projective chart
windows lower to interval-gated q-UVT tubes, a target render is generated,
native gated VJP is applied to image MSE, one bridge-color update is taken, and
the loss decreases.

The first nonlinear/projective atlas-cell Metal forward path also passes:
compiler-side tile-time cells are packed into dense tile buffers with exact
per-entry `[active_start, active_stop)` intervals, and
`render_projective_trace_tile_time_atlas_metal(...)` evaluates degree-2
homogeneous projective traces directly in Metal. The matching projective
atlas-cell direct VJP now passes too through
`direct_backward_projective_trace_tile_time_atlas_metal(...)`. It differentiates
color, opacity, and all nine homogeneous coefficients while treating tile
membership/order as compiled constants, and a focused test matches Torch
autograd on a quadratic chart. A coefficient-only trainability smoke now keeps
color fixed, updates homogeneous projective coefficients with that native VJP,
and verifies rendered MSE drops.

The forked projective-cell evaluator lane is now one step richer:
`projective_trace_windows_to_cell_trace_atlas(...)` lowers accepted gauge
domains into packed raw-time polynomial rows for `u(t)`, `v(t)`, and `depth(t)`,
and `render_projective_trace_cell_atlas_metal(...)` renders those row-indexed
cell traces through the native `render_projective_trace_cell_tiles` op. This
makes the accepted atlas cell itself an evaluable GPU object.

The forked trainer-integration lane now has a selectable real-trainer backend:
`uvt.render_backend="metal_tile_interval_gated"` routes through
`render_uvt_tubes_metal_interval_gated_backward(...)`, using native
`render_gated` forward plus native `direct_atomic_backward_gated` VJP. The
current ordinary STAR UVT trainer uses full-video intervals as the degenerate
single-domain case, while keeping explicit interval buffers ready for nontrivial
projective/gauge-domain segment producers.

Focused projective plus trainer interval-gate tests passed at this stage:

```text
49 passed
```

A real `src/train/train.py` smoke selected `metal_tile_interval_gated` and
decreased loss:

```text
0.15689826011657715 -> 0.12637178599834442
```

The projective atlas now has a first frame-count scaling gate:

```text
count_projective_trace_dense_per_frame_tile_pairs(...)
research_project/benchmarks/projective_atlas_scaling_probe.py
```

On the deterministic 45-degree orbit fixture, dense per-frame project/bin
entries grow `35 -> 555` from `4 -> 64` frames, while ideal interval-packed
atlas entries stay `13 -> 13`. The fixed `tile_t=4` slab packing grows
`13 -> 208`, which means the compiler object has the desired sublinear
world-side shape but a fixed temporal slab scheduler re-expands the same
interval.

The first interval-compressed projective cell Metal forward path now exists:
`render_projective_trace_cell_interval_atlas_metal(...)` packs spatial tile
entries once with per-entry `[active_start, active_stop)` intervals and calls
the native `render_projective_trace_cell_interval_tiles` op. It matches the slab
image sums on the same 4 -> 64 frame scaling fixture. The interval forward path
scales `24.8067ms -> 29.3612ms`; the slab path scales
`20.0995ms -> 37.2617ms`.

The interval-compressed projective cell direct VJP now exists too:
`direct_backward_projective_trace_cell_interval_atlas_metal(...)` calls the
native `direct_projective_trace_cell_interval_backward` op over the same spatial
tile bins and explicit active intervals. The focused test matches Torch
autograd for color, opacity, and cell trace coefficients. A one-step
coefficient trainability smoke renders a shifted-coefficient target, keeps color
fixed, applies the native interval VJP, line-searches coefficient updates, and
verifies Metal-rendered MSE drops.

Focused projective plus trainer interval-gate tests now pass:

```text
49 passed
```

The trainer harness now has the first projective interval-cell autograd bridge:
`render_projective_cell_interval_atlas_metal_backward(...)`. The focused smoke
uses `split_projective_trace_windows(...)` as the gauge-domain producer, lowers
the accepted windows to `ProjectiveTraceCellTraceAtlas` rows with multiple
active intervals, renders through interval-compressed Metal forward, backprops
through interval-compressed direct VJP, runs `optimizer.step()` on cell trace
coefficients, and verifies loss drops. This proves a nontrivial projective
interval atlas can pass through an optimizer loop, but it is still a harness
bridge rather than the full production STAR UVT trainer backend.

The projective cell atlas now also has support and order staleness/rebin gates:
`projective_trace_cell_atlas_coverage_report(...)` evaluates live cell trace
coefficients over active sample intervals and detects missing frame/tile
coverage in the compiled atlas cells.
`projective_trace_cell_atlas_visibility_report(...)` compares compiled
front-to-back order against live per-sample depths. `rebin_projective_trace_cell_atlas(...)`
preserves live coefficient/opacity/color tensors and rebuilds support cells
plus depth intervals. Focused tests move a cell trace into a new tile and flip
two traces' depth order without changing support; both stale cases are detected
and repaired.
`refresh_projective_cell_interval_atlas_if_stale(...)` now wraps this for the
trainer harness. Its Metal smoke moves an MPS coefficient tensor with an
optimizer step, refreshes metadata, renders through the interval-compressed
autograd path, and verifies gradients still flow into the same tensor.
`ProjectiveCellIntervalTrainerState` now owns atlas/config/times/refresh
cadence for trainer-style loops, exposes `render()`, and calls refresh from
`after_optimizer_step()`. Focused tests prove state-owned refresh repairs moved
support on MPS and depth-order flips on CPU without replacing live tensors.
Ambiguous near-tie visibility is now fallback metadata: strict refresh raises,
opt-in refresh marks affected cells as `visibility_ambiguous_depth`, and
the Metal fast path rejects those cells. The CPU/Torch reference fallback now
sorts marked tile/sample regions by live evaluated depth, and fallback stats
report coverage and reasons. Refresh also tries sampled visibility-stratum
splitting before fallback, so crossing order becomes stable time-run cells.
Complexity/budget reports now expose interval ratio, stratum split count,
fallback fraction, and named budget failures; refresh can enforce those budgets
before render/backward. Continuous support/tile-boundary roots now drive
time-local support rebinning, so moving traces do not need one broad tile
rectangle over all active frames. Continuous visibility event roots are reported
by solving `z_i(t)-z_j(t)=0` on active intervals and now split before sampled
strata; exact roots on frame samples become singleton cells so fallback covers
only the tie/event sample. A continuous sensor-time partition now merges support
roots, visibility roots, and caller-supplied exposure/shutter split times into
intervals independent of frame-index cells, and lowers them into normalized
finite-exposure or per-row rolling-shutter quadrature schedules. Those
schedules now render through a differentiable CPU/Torch oracle:
`render_projective_trace_cell_atlas_quadrature_reference(...)` and
`render_projective_trace_cell_atlas_rolling_quadrature_reference(...)`, which
evaluate fractional sensor times, live-sort by depth, composite, accumulate
sample weights, and backpropagate through trace coefficients/colors/opacity in
the focused test. The same schedules now lower to sample-indexed interval
atlases through `lower_projective_trace_cell_atlas_quadrature(...)` and can
render through the existing interval Metal kernel via
`render_projective_trace_cell_atlas_quadrature_interval_metal(...)` or the
batched rolling bridge, which merges unique row sample times with a
`row_weights[Q,H]` matrix and uses the row-weighted
`render_projective_trace_cell_interval_rows` Metal op to write the final
rolling image directly. Mixed finite-exposure/rolling forward rendering now
patches whole fallback tile/sample regions with live-depth reference ordering
before exposure or row-weight accumulation, while non-fallback regions stay on
the interval Metal path. `src/train/star_uvt_projective_interval_backend.py` now
provides the first production-facing bridge: `feature_uvt.projective_interval`
config defaults/validation, a `ProjectiveCellIntervalBackendConfig`, and a
helper that constructs `ProjectiveCellIntervalTrainerState` from a compiled
atlas plus trainer config. It now also has the first compatible STAR UVT tube
producer: `uvt_tubes_to_projective_trace_cell_atlas(...)` plus the backend
wrappers `make_projective_cell_interval_atlas_from_uvt_tubes(...)` and
`make_projective_cell_interval_trainer_state_from_uvt_tubes(...)`. This lowers
exact isotropic affine UVT tubes into cell-polynomial atlas rows, preserves
source primitive ids, constructs the refresh/fallback-aware trainer state, and
reuses support/visibility event compilation; it still rejects anisotropy,
pixel-varying depth. Residual temporal opacity is represented through
`opacity_time_coeffs` and is consumed by the atlas reference path plus native
interval Metal forward/backward.

Focused Q2 Metal lowering/chain-rule/materialized-batch/native-eval/
native-interval-forward/native-interval-backward/tile-order-reuse/
tile-order-strata/active-set-strata/real-active-set-distribution plus
goal-progress tests now pass `106 passed in 8.30s`; the
wider projective evidence matrix was last run before the native-interval-
forward/backward/tile-order rows at:

```text
151 passed, 8 skipped in 4.62s
```

## Next Best Goal

Next gates, in order:

1. Make support guards adaptive/budget-aware under ordinary tube motion. The
   saved guard artifact proves fixed padding can remove support churn, but it
   spends tile-capacity headroom and can overflow packed Metal tiles.
2. Extend the trace representation/native kernels for anisotropic footprints,
   pixel-varying depth, and WorldFoam cells.
3. Choose production atlas-budget defaults for deciding split/refit versus
   tile-local fallback.
4. Keep `fallback_render_mode=mixed` as the fallback policy and consider a
   row-compacted launch for the row-weighted rolling kernel.
5. Bridge WorldFoam cell-camera intersections through the same `pi_* Gamma^*`
   object.

## Do Not Do Yet

Do not claim this beats 4DGS, STAR UVT, or WorldFoam globally.

Do not start with a large renderer rewrite.

Do not treat fallback percentage as irrelevant. If fallback grows linearly with
frames, the method collapsed back into ordinary per-frame rendering.

Do not discard depth/visibility when depth-marginalizing traces.

Do not update `BASELINES.md` unless an actual benchmark baseline is rerun.

## Clean Thread Starter Prompt

Paste this into a new clean thread:

```text
We are in /Users/nicholasbardy/git/gsplats_browser/dynaworld.

Goal: continue Gauged UVT Trace Atlas. Build a camera-program compiler for 4D
spacetime primitives into reusable UVT viewport-time traces so projection,
support, binning, visibility, and backward replay are shared across frames and
non-pixel world-side cost grows sublinearly with frame count.

Read:
- AGENTS.md
- PROJECT_INDEX.md
- research_notes/gauged_uvt_trace_atlas/clean_thread_handoff/README.md
- research_notes/gauged_uvt_trace_atlas/00_WHAT_IS_THIS_GOAL.md
- research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
- research_notes/gauged_uvt_trace_atlas/03_projective_rational_traces/README.md
- research_notes/gauged_uvt_trace_atlas/09_metal_acceptance_plan/README.md

Current code has projective trace evaluation and CPU/Torch chart-fit/window
split helpers in:
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py

The synthetic orbit chart-count, support-bound, visibility-sidecar,
visible-swap-cost, compiler-side tile-time binning, atlas assembly,
dense-reference correctness, and minimal CPU atlas reference-renderer gates
already pass. The first affine-chart bridge into the existing STAR UVT `q_uvt`
renderer contract also passes, including guarded MPS/Metal parity on this
machine. Explicit interval gates for split affine q-UVT chart segments pass in
the CPU oracle, and native shader-side interval gates now reach the existing
Metal renderer through `render_gated` without segment leaks. Direct atomic VJP
coverage also respects those intervals through `direct_atomic_backward_gated`.
The bridge-level one-step color training smoke also passes. The first
nonlinear/projective atlas-cell Metal forward renderer now passes too through
`render_projective_trace_tile_time_atlas_metal(...)`; it evaluates degree-2
homogeneous projective traces directly from packed tile-time cells and matches
the CPU atlas oracle. The matching direct VJP now passes too through
`direct_backward_projective_trace_tile_time_atlas_metal(...)`, matching Torch
autograd for color, opacity, and homogeneous coefficients. Accepted windows can
also lower into `ProjectiveTraceCellTraceAtlas` rows and render through
`render_projective_trace_cell_atlas_metal(...)`, making the gauge-domain cell
itself GPU-evaluable. The interval-compressed projective cell forward path now
also passes through `render_projective_trace_cell_interval_atlas_metal(...)`,
packing spatial tile entries once with per-entry frame intervals and matching
the slab image sums on the 4 -> 64 frame scaling fixture. Its tiny-fixture Metal
timing scales `24.8067ms -> 29.3612ms`, versus `20.0995ms -> 37.2617ms` for
the slab path. The matching interval-compressed direct VJP now passes through
`direct_backward_projective_trace_cell_interval_atlas_metal(...)` and
`direct_projective_trace_cell_interval_backward`, matching Torch autograd for
color, opacity, and cell trace coefficients. The real trainer harness now has
`uvt.render_backend="metal_tile_interval_gated"`, and a `src/train/train.py`
smoke selected it and decreased loss. A projective atlas-cell coefficient-only
trainability smoke now keeps color fixed, applies the native interval VJP to
cell trace coefficients, and verifies rendered MSE drops. The trainer harness
also has `render_projective_cell_interval_atlas_metal_backward(...)`, and a
focused optimizer smoke proves split projective windows with multiple active
intervals can render, backprop, and take a loss-decreasing optimizer step
through the interval-compressed cell path. The compatible producer now routes
through the real STAR UVT training loop, and the first cache path reuses
compiled cells between `refresh_every` rebuilds while replacing live
differentiable tensors every step. Start with the next gate: replace that fixed
cadence with measured refresh thresholds and the production fast/fallback
scheduler as coefficients move. The trainer state already owns support/order
refresh, fallback marking, fallback stats, reference live-depth fallback, and
sampled visibility-stratum splitting, with continuous support/depth event roots
now driving the first splits before fallback. The newest partition API also
merges those roots with exposure/shutter split times and lowers them to
quadrature schedules; those schedules now render through differentiable
CPU/Torch finite-exposure and rolling-shutter reference oracles, and lower into
sample-indexed interval atlases for the interval Metal renderer. Rolling now
batches unique sample times with a `[sample,row]` weight matrix and a
row-weighted Metal kernel. Mixed forward fallback patches whole fallback
tile/sample regions with the live-depth reference before accumulation, and
trainer-state `fallback_render_mode=mixed` keeps that fallback reference
differentiable while fast regions use interval Metal VJP. The first production
evaluation verifier for this path is
`research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_quadrature_report.py`;
it now recomputes interval/dense ratios, fallback cell fractions, fallback
tile/trace sample subsets, and the Metal summary max/count. Current focused
tests pass `11 passed in 34.79s`, and the saved
`outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json`
artifact verifies by CLI. The matching backward verifier is
`research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_backward_report.py`;
it recomputes the rolling reuse ratio, validates positive sample image/adjoint
support and nonzero coeff/opacity/color reference gradients, checks Metal
aggregate errors against their subrows, and recomputes the summary. Current
focused tests pass `11 passed in 25.19s`, and
`outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json`
verifies by CLI. The first production
bridge is `src/train/star_uvt_projective_interval_backend.py`; the first
compatible UVT tube producer exists, and the real STAR UVT feature trainer now
routes `feature_uvt.projective_interval.enabled=true` through that producer for
`feature_dim=3`. The route pins spatial precision to `sigma_px`, keeps
temporal/motion/opacity/feature gradients live, renders feature color through
the interval atlas, and renders a white-trace atlas for total alpha. It is
full-frame/autograd-only and now has an explicit cache policy:
`projective_interval.refresh_policy="measured"` reuses compiled cell metadata
across steps and lets the trainer-state refresh oracle repair stale
support/order/fallback/budget metadata before render without replacing live
tensors. A controlled optimizer-style MPS gate now covers stale support rebin
across four measured-cache update steps, and a real synthetic trainer A/B smoke
shows measured mode skips the cadence rebuild while matching the cadence loss
curve. The saved artifact
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md`
shows rebuilds `4 -> 1`, identical final loss, and no-first-step mean time
`3473.3 -> 2137.2 ms`, but the earlier no-guard run still rebinned support on
every live update. The support-guard gate added
`projective_interval.support_guard_padding`, compiling cell metadata with
`uv_padding + guard` while checking correctness with base `uv_padding`. The
saved guard artifact
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_cap256/summary.md`
shows guard `2` plus cap `256` removes stale refreshes/support rebins (`4/7 ->
0/0`) at identical final loss while measured keeps rebuilds `4 -> 1`; guard
`2` or `8` at cap128 overflows packed Metal tiles. A first global cap-aware
policy now exists as `support_guard_policy="budgeted"` and searches downward
for a no-overflow guard. Its cap128 artifact
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_budgeted_cap128/summary.md`
passes the measured row with zero overflow and identical loss, but only reduces
support rebins `7 -> 6`, slows no-first-step mean to `6107.7ms`, and times out
the cadence row. The first local policy now exists as
`support_guard_policy="local_budgeted"`: it preserves guarded cells in tiles
with headroom and downgrades only overflowing tiles to base support. The
explicit cap128 artifact
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_local_budgeted_cap128_explicit/summary.md`
passes both rows with zero overflow and identical loss; measured no-first-step
returns to `2468.3ms`, but support rebins remain `7/7`. Next make guard
allocation trace-local inside crowded tiles. That exists now as
`support_guard_policy="trace_budgeted"`; the cap128 rerun
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_rerun/summary.md`
passes both rows with zero overflow and identical loss, but measured support
rebins remain `7/7`. Support-boundary overshoot telemetry now exists via
`projective_interval_cache_last_support_max_overshoot_px` and
`projective_interval_cache_max_support_max_overshoot_px`. The margin artifact
shows the old rebins were subpixel (`0.0912px` max); debounce artifacts show
epsilon `0.125/0.25/0.5px` gives measured rebins `3/1/0`, with the `0.5px`
row at
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_eps05/summary.md`
keeping identical loss and zero overflow. Next validate this bounded subpixel
debounce on broader scenes. The first image-level debounce stress now proves the
condition: with real support padding, a `0.05px` overshoot stays below `1e-4`
max RGB error; with underspecified center-only support, the same nominal
overshoot exceeds `0.35` max RGB error. A tiny orbit-derived projective trace
now checks the same contract for revolving-camera math: a `0.10px` coefficient
update creates about `0.056px` padded support overshoot, and strict rebinning
versus tolerant reuse still stays below `1e-4` max RGB error.
The support debounce now also has a Gaussian-tail certificate:
`support_stale_tail_alpha_epsilon` reuses stale support only when omitted
Gaussian tail opacity is below budget. Important correction: the bound now
aggregates omitted tails per missing sample/tile instead of taking a max over
primitives. Tests show the `0.05px` real-support sliver is accepted at `3e-4`
and rejected at `1e-4`, `uv_padding=0` core loss still rebins with bound `0.5`,
and 16 overlapping tiny tails rebin at `1e-3` because their aggregate bound is
about `0.00327`. The benchmark CLI accepts
`--support-stale-tail-alpha-epsilon` and emits
`projective_interval_cache_last_support_tail_alpha_bound` plus
`projective_interval_cache_max_support_tail_alpha_bound`.
The old `0.00035` cap128 bracket is superseded by aggregate accounting:
`0.00035` now records two support rebins with max aggregate bound
`0.000404648`; `0.00045` and `0.0006` still record one rebin because skipped
earlier repairs let later drift grow; corrected `0.001` clears the smoke with
zero support rebins, identical loss, zero overflow, one rebuild versus
cadence's four, and max aggregate bound `0.000736007`. Artifacts:
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.md`,
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00045_aggregate/summary.md`,
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0006_aggregate/summary.md`, and
`outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.md`.
The cache-policy script now exports
`verify_projective_interval_cache_policy_report(...)` and
`assert_projective_interval_cache_policy_report(...)`, plus CLI validation:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.json
```

This contract checks the actual amortization/correctness claim: cadence and
measured rows both pass with identical final loss and zero overflow; measured
uses fewer full rebuilds and more live updates than cadence; reuse is
tail-certified with no pixel-overshoot pardon; fallback and visibility
stratification stay zero for this support-only gate; and the saved epsilon
bracket is monotone in support-rebin count. Focused tests passed
`9 passed in 0.14s`, and all four aggregate saved artifacts verified through
the CLI.
The aggregate image-error verifier at
`outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.md`
keeps single-tail/orbit reuse below bound and rejects a 64-trace overlapping
tail case with aggregate bound `0.01309515`; forced reuse would produce
`0.00141417` max RGB error. The same script now has
`--verify-report <summary.json>` plus a reusable
`assert_tail_alpha_image_error_report(...)` contract: positive reuse cases must
stay below their omitted-tail bound, while core-loss and overlapping aggregate
controls must reject stale reuse and show forced-reuse error above budget.
Focused verifier tests passed `7 passed in 9.09s`; the base, tail00035
aggregate, and metal-precision-rerun saved artifacts all verified.
The anisotropic tail-bound verifier now has the same reusable
`--verify-report <summary.json>` and
`assert_anisotropic_tail_bound_report(...)` contract. It requires diagonal,
rotated, and two-trace same-tile anisotropic tails to reuse below their
omitted-alpha bound, requires the two-trace bound to exceed each single-tail
bound, and rejects an anisotropic core-loss case. Focused tests passed
`6 passed in 8.81s`; the base and metal-precision-rerun saved artifacts
verified.
The measured revolving-camera fixed-chart artifact now has the same reusable
report verifier in
`research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py`.
Run:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

The contract locks the orbit/fiber-bundle claim at report level: fixed charts
keep segment/trace/payload counts constant through 32 frames, per-frame replay
grows those counts, fallback remains zero, interval entries grow slower than
dense samples, row-level interval ratios and CPU compile sums are internally
consistent, fixed/per-frame CPU and Metal forward/backward ratios stay below
`0.5` at the largest frame count, direct Metal backward reaches coeffs,
opacity, color, and spatial precision, and fixed-chart autograd reaches
`ma`, opacity, color, `q_uv`, temporal `q_uvt`, and spatial precision. Focused
verifier tests passed `10 passed in 28.79s`; the saved artifact verified by
CLI; the orbit-window plus verifier suite passed `24 passed in 126.68s`.
The synthetic production-trainer frame-scaling artifact now has the same
report-verifier treatment in
`research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py`.
Run:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_interval_trainer_frame_scaling/summary.json
```

The contract checks the actual `run_training` route with generated frame
tensors: rows pass, loss decreases, cadence/measured end loss matches, tile
overflow is zero, measured rebuilds stay below cadence, measured live updates
and staleness checks cover the cache reuse path, support rebins equal stale
refreshes, fallback/visibility stratification stays zero, and measured
no-first-step timing beats cadence on the synthetic MPS smoke. Focused
verifier tests passed `6 passed in 11.58s`; the saved synthetic artifact
verified by CLI; the strict synthetic plus real-video trainer verifier suite
passed `26 passed in 10.59s`. Real-video
base/guard025/guard05/guard1/guard2 artifacts are now in the saved-artifact
matrix and all verify by broad `--verify-report`; guard025/guard05/guard1/guard2
also verify through the strict `--verify-guarded-support` CLI. The aggregate
guarded-support matrix at
`outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json`
is the canonical compact evidence: 5 artifacts, 15 measured rows, default
measured support rebins 9, guarded measured support rebins 0, guarded stale
refreshes 0, max guarded no-first-step ratio 0.590, and max guarded rebuild
ratio 0.5. Goal + guarded-support focused tests pass `39 passed in 0.99s`.
The real-video verifier rejects stale summary fields, row-count/frame-count
drift, fallback/visibility marks, overflow, failed loss decrease,
measured/cadence loss drift, and support-rebin/stale-refresh mismatch.
Depth-plane metadata also exists now: `ProjectiveTraceCellTraceAtlas` carries
optional `depth_affine_uv[N,6]`, and
`eval_projective_trace_cell_depth_at_uv_torch(...)` evaluates
`z(u,v,t)=z_c(t)+z_u(t)(u-u_c(t))+z_v(t)(v-v_c(t))`. Validation, support
rebinning, trainer refresh, quadrature lowering, and CPU detach preserve this
field. The q-UVT producer rejects spatial `depth_beta[:,0:2]` by default and
lowers it only with `allow_depth_affine_uv=True`; the opt-in maps the affine
depth plane into `depth_affine_uv` and corrects center-depth slope by
`beta_u velocity_u + beta_v velocity_v`. Live measured-cache updates preserve
and recompute the same metadata when the reference atlas carries it.
Visibility reports now use tile-range depth bounds when `depth_affine_uv` is
present, so an order flip inside one tile is detected even if center depths
look stable. Fallback-marked reference cells sort by live per-pixel depth. The
interval Metal hot path also consumes `depth_affine_uv` in its dynamic
per-pixel selection sort; legacy scalar-depth atlases pass a zero slope tensor.
`projective_trace_cell_uv_visibility_event_report(...)` now turns those slopes
into a named UV-order certificate: for each tile/sample/trace pair it computes
the affine zero line of `Delta z(u,v,t_k)` over the tile pixel-center
rectangle and emits a `ProjectiveTraceCellUVVisibilityEvent` when the range
straddles zero. This is not an oblique sub-tile splitter yet; it is the
decision evidence for spatial split versus fallback. `mark_projective_trace_cell_visibility_fallbacks(...)`
now consumes that certificate too: when no spatial split representation has
been materialized yet, any accepted cell whose UV zero line crosses its tile is
marked with the specific fallback reason `visibility_uv_depth_line`, alongside
the older interval-overlap reason if both apply. Treat the slopes as compiler
metadata for now, not trainable VJP parameters. The first concrete spatial
split path is `split_projective_trace_cell_atlas_uv_visibility_events(...)`:
it retile-compiles a parent atlas onto a finer child tile grid and recomputes
per-child depth intervals/order. In the canonical UV line fixture, one parent
tile becomes two stable child cells, left order `(0,1)` and right order
`(1,0)`, with no UV event and no fallback. This is not an oblique halfspace
cell renderer yet; it is the cheap grid-refinement split path, with
`visibility_uv_depth_line` fallback still covering unresolved child tiles. The
adaptive wrapper is `adapt_projective_trace_cell_atlas_uv_visibility_events(...)`;
because one packed atlas has one tile size, it keeps the parent atlas when no
UV event exists, otherwise tries divisor child grids from coarsest to finest.
The returned `ProjectiveTraceCellUVVisibilitySpatialSplitReport` records the
candidate sizes, chosen output tile size, residual event count, parent/output
fallback fractions, and whether the residual line budget was accepted. The
high-motion row now decodes the checked-in smoke video and uses the strongest
adjacent-frame motion centroids as the UV roots of a diagnostic pairwise depth
line. This is video-derived motion-centroid trace geometry, not trained scene
geometry. On the current video, selected pairs `(7,8,9)` have roots
approximately `(4.395,4.055,4.123)`; an 8-pixel parent tile has
`parent_uv_event_tile_samples=3` and `parent_fallback_fraction=1.0`; adaptive
splitting chooses child size `4` and reaches `fallback_fraction=0.0`.
The same before/after report now has an orbit-parameterized fixture using
`q=tan(theta/2)` and the orbit depth polynomial from `_orbit_trace_coeffs`:
over `q in {-0.5,0,0.5}`, the parent tile has
`parent_uv_event_tile_samples=3` and `parent_fallback_fraction=1.0`, while the
adaptive child grid again reaches `fallback_fraction=0.0`.
The reusable report builder is
`research_experiments/star_uvt_feature_tubes/projective_uv_visibility_split_report.py`;
the current saved artifact is `outputs/projective_uv_visibility_split_report.json`
with schema `projective_uv_visibility_split_report_v1`, status `ok`,
`max_parent_fallback_fraction=1.0`, `max_output_fallback_fraction=0.0`, and
`max_cell_growth=4.0`. Its high-motion row is
`high_motion_video_centroid_line_sweep`, sourced from the checked-in video
`data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4`
and carrying decoded-frame count, selected pair indices, motion scores,
centroids, UV root positions, and fitted depth coefficients.
The next artifact is
`research_experiments/star_uvt_feature_tubes/projective_high_motion_trace_geometry_report.py`;
it writes `outputs/projective_high_motion_trace_geometry_report.json` with
schema `projective_high_motion_trace_geometry_report_v1`. It compiles actual
STAR UVT trainer-harness tensors from the checked-in high-motion smoke clip
into the projective cell atlas at smoke scale (`64` tubes, `16` frames,
`64px`). The config-faithful zero-velocity row has fallback `0.0` and
interval/dense tile-pair ratio `0.063`; the `block_match_gated` initialization
row has `58/64` nonzero velocities, max motion `5.657 px/frame`, fallback
`0.0`, and interval/dense tile-pair ratio `0.293`. A third row,
`block_match_motion_trained_dense_3step`, runs three dense CPU Adam steps before
trace extraction: loss `0.30096 -> 0.29547`, parameter L1 movement `67.95`,
`64/64` nonzero velocities, max motion `5.784 px/frame`, fallback `0.0`, and
interval/dense tile-pair ratio `0.294`. The moved tensors are `center_uv`,
`center_t`, `velocity_uv`, `raw_precision`, `raw_opacity`, and `raw_color`;
`depth0` remains fixed because the dense harness sorts on detached depth. This
is now a trained smoke row, but still not a persisted/full high-motion
checkpoint; it is real STAR UVT trace geometry from video samples rather than a
centroid root proxy.
`support_guard_policy="slack_budgeted"`
now also exists as the first event-distance headroom allocator: in crowded
guarded tiles it keeps base-active traces, then spends remaining slots on
traces nearest to that tile's support-event boundary instead of primitive id
order. A paired visibility stress confirms support debounce does not suppress
visibility repair: stale depth order still forces refresh and returns zero
order mismatches. A combined orbit fixture now checks both in one
projective/gauged case: support-only debounce accepts a bounded `0.10px`
coefficient-update drift under `support_stale_overshoot_epsilon=0.10`, while
visibility-aware refresh sees four stale order samples and rebuilds to live
order `(1, 0)` with zero remaining order mismatches. A second orbit visibility
fixture now covers actual depth-root stratification: two rational yaw-window
traces cross at `t=0`, the event report finds one continuous root, and refresh
splits the single orbit cell into `(0,2)/(2,4)` order strata with no fallback
and zero remaining order mismatches. The newest combined orbit stress puts
tail-alpha-certified support drift, that visibility-root split, and
`slack_budgeted` guard-slot pressure in one refresh: `trace_budgeted` keeps
lower-id far extras, while `slack_budgeted` spends the `12` available guarded
slots on the near-boundary extras, with zero remaining order mismatches and no
fallback. Next validate
slack-budgeted plus bounded tail-alpha debounce on broader scenes, scale the
high-motion trainer trace geometry from tiny smoke rows to a full persisted
checkpoint/world trace, decide whether an oblique/fiber halfspace cell is worth
adding, settle possible slope gradients, and broaden to WorldFoam. Last focused
projective plus interval-gated trainer suite:
`163 passed in 25.46s` after adding the UV split report artifact test; the
video-derived report extraction targeted check passed `4 passed in 4.53s`, and
the refreshed broad focused suite passed `166 passed in 93.12s`; the
high-motion trace-geometry report targeted check with trained row passed
`2 passed in 35.36s`;
the refreshed broad focused suite with both trace reports passed
`168 passed in 231.68s`; the prior ABI rebuild forced interval backward to
register the current 5-return ABI. The trained high-motion scaling benchmark
now has a reusable verifier in
`research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py`.
Run it with `--verify-report <summary.json>`; it validates train loss decrease,
top-level config/frame fields, exact trained/per-frame frame coverage, row
ratio consistency, recomputed summaries, zero overflow/fallback, nonzero
learned velocity, opacity bounds, positive timing-gradient signals, and matched
per-frame interval/trace wins plus final-scale timing wins. The trained
high-motion + shared-work audit suite passed `33 passed in 6.20s`; the shared
aggregate and top-level goal-progress aggregate both verify with
`--verify-current-inputs`; combined goal-progress/shared-work/gauge/
camera-family/trainer tests last passed before the native interval-forward row
at `151 passed, 8 skipped in 4.62s`; the current multiscene/tether/guarded/audit
focused verifier subset, including the extended-frame-scaling diagnostic,
timing-breakdown, phase-profile, render-forward-residual, render-forward-shape,
and Bq4 traced-rerun/repeat-stability/sequence-order/policy-order/fresh-process
reports, passes `142 passed in 8.17s`; the latest focused
acceptance-envelope + timing-variance-envelope + goal-progress + Bq4
fresh-process subset passes `70 passed in 4.76s` and cross-checks Bq4
fresh-process status, post-warmup pair count, and medians across both top-level
timing envelopes; all three
saved trained artifacts verified:
`outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json`
`outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json`
`outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json`.
Use
apply_patch for edits and run the focused projective plus interval-gated
trainer tests before reporting.
```
