# Detailed Next Plan: From Proven Contract To Convincing Renderer

Date: 2026-07-04

This is the post-completion plan for the gauged/projective STAR UVT thread.
The final completion audit proves the current objective at the contract level:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json
```

That audit is not the end of the research program. It means the idea has a
coherent math spine, a Metal-backed route, broad enough acceptance evidence to
close the original goal, and a clean project memory trail. The next work should
shift from "prove we are not lying to ourselves" to "make the visual/runtime
win obvious enough that someone else immediately understands why this matters."

## 0. What We Are Doing

We are building a camera-path compiler for dynamic 4D spacetime primitives.

The durable invariant is:

```text
UVT trace = pi_* Gamma^* world_primitive
```

Meaning:

```text
Gamma      camera program from camera-ray bundle into world spacetime
Gamma^*    pull a world primitive onto the camera-ray bundle
pi_*       integrate / summarize along ray fibers into sensor time
B          sensor-time base, usually Omega x T or Q x Omega x T
```

The product should not be another frame-by-frame renderer. It should be a
compiled sensor-time atlas:

```text
known camera program + dynamic world primitives
    -> reusable sensor-time traces
    -> shared support/binning/visibility/depth/order/backward metadata
    -> cheap slicing/evaluation over many frames, rows, or shutter samples
```

The high-value workload is:

```text
many nearby renders from one known camera program
finite exposure
rolling shutter
orbit/replay
video inference
late-stage training with reused views
```

The low-value workload is:

```text
one random novel view rendered once
```

That is not where this wins.

## 1. What We Have Done

The existing work proves the following current contracts.

### 1.1 Theory Contract

Evidence:

```text
research_notes/gauged_uvt_trace_atlas/
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
```

Done:

- Defined the camera-ray bundle and fiber pushforward.
- Treated current STAR UVT tensors as one local coordinate expression.
- Made gauges explicit: depth, log-depth, projective, camera-family, object,
  and foam gauges.
- Reframed "chart splitting" as event-certified gauge domains, not a weak
  fallback excuse.
- Kept exactly ten numbered theory tracks:
  `00_bundle_foundations` through `09_metal_acceptance_plan`.

### 1.2 Math And Derivative Contract

Evidence:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_gauge/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_gauge/summary.json
```

Done:

- Gauge-invariant fiber integration works when the fiber-measure Jacobian is
  included.
- Omitting the Jacobian produces large controlled errors, so the test can catch
  the exact mistake the math warns about.
- Primitive gradients match finite differences / alternate gauges.
- One- and two-parameter camera families carry derivatives through local
  family coordinates.

### 1.3 Projective/Camera-Family Sharing Contract

Evidence:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_shared_work_scaling/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_shared_work_scaling/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata/summary.json
```

Done:

- One chart over `Q x Omega x T` can replace replaying one chart per q sample.
- One chart over `Q2 x Omega x T` can replace replaying one chart per q pair.
- Stable topology, order-change strata, and active-set-change strata are all
  represented as compressed metadata.

### 1.4 Metal Renderer Contract

Evidence:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_lowering/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_chain_rule/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_materialized_batch/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_eval/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_forward/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward/summary.json
```

Done:

- Projective/camera-family traces reach Metal forward paths.
- Native family coefficient evaluation and VJPs work.
- Native interval forward compositing and native interval backward accumulation
  work with compiled visibility/order held fixed.
- Single-launch materialized batching demonstrates the direction for reducing
  launch/replay overhead.

### 1.5 Exposure, Rolling, Visibility, And Fallback Contract

Evidence:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.json
```

Done:

- Exposure is rendered-field integration:

```text
I_frame(u,v) = integral_tau Composite(K, u, v, tau) d tau
```

- Rolling shutter row weights map final image adjoints back to sample-time
  adjoints.
- Mixed fallback remains differentiable: non-fallback cells use interval Metal
  VJP, ambiguous regions use live-depth Torch reference gradients, then sample
  gradients are accumulated by exposure/row weights.

### 1.6 Real-Video / Broad Evidence Contract

Evidence:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json
```

Done:

- Broad10 quality/media/trainer evidence exists.
- Fresh-process median timing protocol is accepted for the current envelope.
- Compiled-adjoint replacement is verified on the practical real-video trainer
  route.
- Final audit proves nine objective-level rows and closes the original goal.

## 2. Least Confidence Areas

These are not blockers for the completed objective, but they are the next
research risk list.

### 2.1 Paper-Grade Visual Significance

Current belief:
    The renderer contract is real, but the visual win has not been made obvious
    enough for an outside reader.

Evidence:
    Many artifacts prove correctness, source coverage, media tethers, and
    timing protocols.

Weakness:
    These are audit-heavy. They do not yet feel like a clean one-page demo:
    "Here is rolling shutter / finite exposure / orbit replay, baseline pays K
    projections, our atlas pays once, equal image quality, clear speedup."

What would increase confidence:
    A single polished demo with side-by-side video, runtime bars, memory bars,
    fallback heatmap, and exact commands.

### 2.2 Timing Generality

Current belief:
    Fresh-process medians are a fair accepted timing protocol on this machine,
    but warm-state MPS variance remains real.

Evidence:
    Bq4 trace reruns and fresh-process isolation explain prior spikes as
    variance rather than cache/support failure.

Weakness:
    A skeptical reader can still ask whether speedup survives different
    machines, CUDA, larger images, or longer paths.

What would increase confidence:
    Repeat the decisive demo on at least two runtime regimes:
    local MPS and either CUDA/Modal or CPU reference microbenchmarks for the
    world-side compile/eval split.

### 2.3 Visibility In Pathological Scenes

Current belief:
    The atlas approach is correct with strata and fallback, but the win can
    collapse if too many cells are visibility-ambiguous.

Evidence:
    Current active-set/order strata and mixed fallback tests pass.

Weakness:
    Thin occluders, dense crossing alpha, disocclusion-heavy orbits, reflective
    surfaces, and near-camera elongated primitives are still the most dangerous
    cases.

What would increase confidence:
    A synthetic visibility torture suite that reports fallback fraction,
    order-flip density, quality error, and runtime collapse point.

### 2.4 Native Projective Atlas Kernel Design

Current belief:
    The practical route through interval Metal and direct VJP works, but a
    cleaner native projective atlas kernel may be the eventual architecture.

Evidence:
    Native family eval/interval forward/backward works; degree-1 lowering works.

Weakness:
    The architecture may be carrying bridge complexity that a native projective
    evaluator could remove.

What would increase confidence:
    A small native projective evaluator that consumes rational/projective
    coefficients and interval/cell metadata directly, then beats the bridge
    path on the decisive demo.

## 3. Where We Are Wasting Time

### 3.1 More Audits Without New Empirical Surface

Stop adding new report layers unless they protect a new claim. The final audit
already does the job of saying the original objective is complete.

Allowed:
    A new verifier for a new demo, new kernel, new baseline, or new failure
    suite.

Not allowed:
    Another umbrella report that only rephrases the same evidence stack.

### 3.2 Chasing MPS Warm-State Outliers As If They Are Math Failures

Timing variance work was useful once. Repeating it without new instrumentation
will become a sink.

Allowed:
    Fresh-process medians, warmup discard, explicit caveat rows, and one
    targeted profiler if a new kernel regresses.

Not allowed:
    Re-running Bq4 timing probes indefinitely to make every max ratio pretty.

### 3.3 Synthetic Microproofs Without A Demo Target

Small invariance proofs are valuable only when they unblock a user-facing or
paper-facing artifact.

Allowed:
    Synthetic tests that predict demo failure modes: order flips, near-camera
    singularities, rolling rows, exposure integration, memory explosion.

Not allowed:
    Synthetic tests that prove helper implementation details nobody will cite.

### 3.4 Premature Architecture Forking

The existing interval/direct-VJP path works. Do not fork three new renderers
unless the decisive demo reveals the bridge path cannot carry the visual/runtime
case.

Allowed:
    One native projective atlas kernel prototype with a kill criterion.

Not allowed:
    Parallel speculative forks with no shared acceptance script.

## 4. Where Time Was Well Spent

### 4.1 Fiber-Bundle Reframing

This gave the project a real invariant. It prevented the idea from collapsing
into "affine UVT until it breaks."

Keep this.

### 4.2 Gauge-Domain Validity Certificates

This made revolving cameras tractable. The answer is not one global footprint;
it is an atlas of certified local domains.

Keep this.

### 4.3 Visibility As Strata

This preserved the hard truth that depth marginalization does not solve
compositing by itself.

Keep this.

### 4.4 Compiled Adjoint Replacement

This moved the project from inference-only to a plausible training story.

Keep this.

### 4.5 Final Audit

The final audit was useful because it stopped the project from forever saying
"almost done" after concrete gaps were gone.

Stop repeating this pattern unless a new objective begins.

## 5. What Is Going Well

- The math is coherent and falsifiable.
- The evidence trail is unusually explicit.
- Revolving camera handling has a real answer: camera-family bundle charts and
  gauge domains.
- The project has both CPU/Torch reference and Metal-backed evidence.
- The next work can be demo-driven instead of proof-driven.

## 6. North-Star Demo

Build one decisive demo:

```text
Known camera path:
    orbit or rolling-shutter path over a real/high-motion video-derived scene

Task:
    render many frames or many shutter/row samples

Baseline:
    per-frame / per-sample projection + binning + sorting + backward replay

Ours:
    compile sensor-time atlas once, slice/evaluate across frames/samples

Output:
    side-by-side media
    timing table
    memory table
    fallback/visibility heatmap
    exact current-input verifier
```

The demo should answer in one glance:

```text
What got reused?
How much world-side work disappeared?
Did image quality stay the same?
Where did fallback trigger?
Does backward still work?
```

Recommended first target:

```text
finite exposure / rolling-shutter orbit replay
```

Why:

- It is the workload where per-sample projection/binning repetition is obviously
  wasteful.
- It naturally uses sensor-time rather than just frames.
- It exercises visibility, exposure, row-time coupling, and adjoints.
- It is easier to explain than arbitrary dynamic novel view synthesis.

## 7. Detailed Execution Plan

### Phase 0: Freeze The Evidence Baseline

Purpose:
    Prevent future work from drifting away from the completed objective.

Actions:

1. Treat the final completion audit as the baseline contract:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json
```

2. Add any new work as a follow-on artifact, not by mutating the final audit
   unless an input regresses.

3. Keep these commands as sanity checks:

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_final_completion_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json \
  --verify-current-inputs

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_final_completion_audit.py \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py -q
```

Acceptance:

- Final audit still verifies against current inputs.
- New work has a new report name and does not reopen completed rows without
  evidence.

### Phase 1: Build The Decisive Demo Harness

Purpose:
    Create the artifact that makes the idea legible.

Deliverable:

```text
research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py
tests/test_star_uvt_projective_decisive_demo_report.py
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/summary.json
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/contact_sheet.png
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/fallback_heatmap.png
```

Inputs:

- One saved trained high-motion artifact already used by current reports.
- One orbit or rolling-shutter camera program.
- Frame counts at least:

```text
F = 8, 16, 32, 64
```

- Exposure / rolling samples:

```text
K = 1, 4, 8, 16
```

Variants:

```text
baseline_frame_replay
baseline_shutter_replay
compiled_interval_atlas
compiled_interval_atlas_with_mixed_fallback
```

Metrics:

```text
compile_ms
render_forward_ms
backward_ms
total_no_first_ms
projection_binning_proxy_entries
trace_count
interval_entry_count
tile_cell_count
active_set_group_count
fallback_cell_fraction
fallback_sample_fraction
max_image_abs_error_vs_reference
psnr_vs_reference
loss_delta_vs_cadence
gradient_rel_error
memory_payload_bytes
```

Core report rows:

```text
row_id
scene_id
path_kind                 orbit | rolling | exposure | exposure_rolling
frame_count
shutter_sample_count
variant
status
quality_pass
timing_pass
memory_pass
fallback_pass
gradient_pass
```

Acceptance:

```text
quality:
    max_image_abs_error_vs_reference <= 1e-5 for synthetic/reference rows
    or cadence/measured loss delta <= explicit float tolerance for real rows

speed:
    post-warmup median no_first ratio <= 0.85 for compiled vs replay
    projection/binning proxy ratio <= 0.25

memory:
    compiled payload growth <= 0.25 * replay payload growth at final F/K

fallback:
    fallback_cell_fraction <= 0.20 on ordinary demo scene
    fallback_sample_fraction <= 0.20 on ordinary demo scene

backward:
    gradient flags present
    direct VJP route used
    relative gradient error <= existing interval backward threshold where oracle exists
```

Kill criteria:

```text
If ordinary scenes need > 40% fallback cells, stop optimizing timing and fix
visibility/support stratification.

If compiled total time is slower after fresh-process median warmup, split the
timing into compile, feature_state_update, render_forward, colorize/loss, and
backward before changing math.

If quality differs from cadence but support/fallback are clean, inspect color /
feature path and compositing order before changing gauge theory.
```

### Phase 2: Visualize The Actual Claim

Purpose:
    Make the demo readable without reading JSON.

Artifacts:

```text
contact_sheet.png
runtime_bars.png
memory_bars.png
fallback_heatmap.png
order_strata_map.png
atlas_cells_overlay.png
```

Contact sheet layout:

```text
row 0: reference/cadence
row 1: compiled atlas
row 2: abs error x gain
row 3: fallback/ambiguous cells overlay
```

Runtime chart:

```text
x-axis:
    frame_count or shutter_sample_count

bars:
    compile
    projection/binning proxy
    render_forward
    backward
    total_no_first

lines:
    replay total
    compiled total
```

Memory chart:

```text
replay tile/frame entries
compiled interval entries
active-set strata
order strata
fallback cells
```

Acceptance:

- A reader can identify the reuse mechanism from the chart labels alone.
- Fallback overlay is not hidden.
- Runtime chart separates compile amortization from per-sample eval.
- Images are not just tiny thumbnails; target/pred/error are inspectable.

### Phase 3: Hard Visibility Stress Suite

Purpose:
    Identify the collapse boundary.

Deliverable:

```text
projective_visibility_stress_suite.py
tests/test_star_uvt_projective_visibility_stress_suite.py
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_visibility_stress_suite/summary.json
```

Synthetic scene families:

```text
crossing_translucent_planes
thin_foreground_occluder
near_camera_elongated_splat
wide_fov_orbit
fast_rotation_rolling_shutter
dense_alpha_cloud
disocclusion_wall_reveal
```

Sweep dimensions:

```text
FOV: 30, 60, 90, 120 degrees
camera_rotation_speed: low, med, high
camera_translation_speed: low, med, high
rolling_readout_fraction: 0, 0.25, 0.5, 1.0
exposure_fraction: 0, 0.25, 0.5, 1.0
opacity_density: low, med, high
primitive_anisotropy: 1, 4, 16, 64
near_depth_ratio: 0.01, 0.05, 0.1
```

Metrics:

```text
fallback_cell_fraction
fallback_sample_fraction
order_flip_surface_count
ambiguous_pair_count
commutable_pair_count
depth_interval_overlap_rate
quality_error
runtime_ratio
memory_ratio
```

Acceptance:

- Ordinary scenes: fallback fraction below 20%.
- Stress scenes: fallback fraction measured and explained, not hidden.
- Runtime collapse point identified:

```text
collapse = first setting where fallback_cell_fraction > 0.40
           or runtime_ratio >= 1.0
```

Decision:

- If collapse mostly comes from depth-order uncertainty, improve visibility
  strata / depth interval bounds.
- If collapse mostly comes from support explosion, improve gauge choice /
  support certificate.
- If collapse mostly comes from near-camera projection, improve projective
  gauge or split event detection.

### Phase 4: Native Projective Atlas Kernel Prototype

Purpose:
    Test whether the bridge path is good enough or whether native projective
    evaluation is necessary.

Deliverable:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/... native projective evaluator
projective_native_atlas_kernel_report.py
tests/test_star_uvt_projective_native_atlas_kernel_report.py
```

Inputs:

```text
cell_bounds
trace_coeffs_projective
q_basis_or_time_samples
active interval gates
tile/cell active lists
depth/order metadata
opacity/color refs
```

First kernel scope:

```text
forward only
degree-1 or degree-2 projective trace center
compiled order fixed
no mixed fallback inside first kernel
```

Second kernel scope:

```text
direct VJP into trace coefficients
direct VJP into opacity/color
optional VJP into q-basis coefficients
```

Acceptance:

```text
forward max abs error <= existing native interval forward threshold
backward max rel error <= existing native interval backward threshold
payload ratio <= bridge path payload ratio
fresh-process median forward ratio <= bridge path ratio
```

Kill criteria:

```text
If native kernel is not at least 1.2x faster than bridge on the decisive demo,
or if it adds major maintenance burden without reducing memory, keep bridge.
```

### Phase 5: Paper-Grade Baselines

Purpose:
    Convert the demo from "local project claim" to "research claim."

Baseline categories:

```text
time-sliced STAR UVT / current projective interval replay
4DGS-style per-timestamp render if available locally
3DGUT-style nonlinear/rolling support if practical
Gaussian Splatting on the Move style finite-exposure/rolling baseline if practical
simple dense ray integration oracle for synthetic scenes
```

No internet assumption:
    Use locally available code/artifacts first. Only add external baselines if
    they can be pinned and reproduced.

Metrics:

```text
PSNR / SSIM / LPIPS if available
RGB loss / media tether deltas
compile time
render time
backward time
memory
fallback fraction
amortization point in frame/sample count
```

Acceptance:

```text
end-to-end speedup >= 1.3x at equal quality on the target workload
projection/binning proxy speedup >= 3x
quality within explicit tolerance of time-sliced baseline
memory <= 2x underlying representation for useful path segment
fallback ordinary-scene cells <= 20%
```

Decision:

- If speedup is strong only for finite exposure / rolling shutter, frame the
  paper narrowly there.
- If speedup holds for orbit replay/video inference too, broaden claim to
  known camera-path rendering.
- If speedup requires low fallback and simple scenes, frame as a compiler with
  explicit failure diagnostics rather than a universal renderer.

### Phase 6: Training-Side Compiled Adjoint

Purpose:
    Turn compiled forward/backward evidence into a robust optimizer story.

Existing base:

```text
real-video compiled-adjoint replacement report
interval Metal forward
interval Metal direct VJP
compiled visibility/order/tile membership constants
```

Next experiment:

```text
late-stage fine-tuning with periodic atlas refresh
```

Variants:

```text
cadence_full_rebuild
compiled_atlas_refresh_every_1
compiled_atlas_refresh_every_4
compiled_atlas_refresh_every_8
compiled_atlas_staleness_guarded
```

Metrics:

```text
loss curve delta vs cadence
PSNR delta vs cadence
media contact-sheet delta
gradient flag preservation
support_rebins
stale_refreshes
refresh_count
optimizer_step_ms
backward_ms
```

Acceptance:

```text
loss curve delta <= float tolerance or declared tolerance
PSNR delta <= tolerance
support_rebins == 0 under guarded route
stale_refreshes == 0 or explicitly bounded
optimizer step speedup >= 1.2x post-warmup median
```

Kill criteria:

```text
If stale compiled visibility poisons training, restrict training claim to
late-stage/frozen-geometry fine-tuning.

If optimizer-step speedup is dominated by non-render phases, keep training as a
secondary claim and lead with inference.
```

### Phase 7: WorldFoam Bridge

Purpose:
    Keep WorldFoam from becoming a separate, incompatible story.

Interpretation:

```text
foam cell = world support region
camera program pulls foam cell back to E_Gamma
pi_* summarizes cell-camera intersection into sensor-time atlas records
```

Deliverable:

```text
worldfoam_camera_bundle_bridge_report.py
tests/test_worldfoam_camera_bundle_bridge_report.py
```

First proof:

```text
one foam cell
one orbit camera
cell-camera intersection support interval
same active-set/tile-time metadata language as STAR UVT traces
```

Acceptance:

```text
foam support cells lower to sensor-time support records
active-set membership matches dense world-cell intersection reference
depth/order metadata compatible with STAR UVT visibility strata
```

Decision:

- If WorldFoam cells produce more stable active sets than splats, make them the
  long-term representation.
- If they add too much support complexity, keep WorldFoam as a separate
  renderer and use the bundle language only as a bridge.

## 8. Reporting Structure For New Work

Every new major artifact should have:

```text
report.py
tests/test_..._report.py
outputs/benchmarks/YYYY-MM-DD_name/summary.json
outputs/benchmarks/YYYY-MM-DD_name/summary.md
agent_notes/loose_notes/YYYY-MM-DD_HH-MM-SS_topic.md
```

Every summary should include:

```text
status
does_not_prove_completion or final_claim_accepted
underlying_report_count
all_underlying_verifiers_pass
current_input_errors or stale_input_errors
primary metric rows
explicit caveats
```

Only use a top-level completion audit when starting a new top-level objective.

## 9. Success Criteria For The Next Month

Minimum:

```text
one decisive demo report
one contact sheet
one fallback heatmap
one runtime/memory chart
one current-input verifier
```

Good:

```text
decisive demo passes quality, speed, memory, fallback, and backward gates
visibility stress suite identifies collapse boundary
native projective kernel has a measured keep/kill decision
```

Excellent:

```text
demo can be shown as a paper figure
fresh-process median speedup >= 1.5x at equal quality
fallback ordinary-scene fraction <= 10%
projection/binning proxy reduction >= 4x
training late-stage compiled adjoint shows optimizer-step win
```

## 10. One-Page Prioritized Task List

1. Build `projective_decisive_demo_report.py`.
2. Generate side-by-side media and fallback heatmap.
3. Add runtime/memory chart generation.
4. Verify demo against current inputs.
5. Build visibility stress suite.
6. Decide whether native projective atlas kernel is worth it.
7. If yes, implement forward-only native projective evaluator.
8. If forward native wins, add direct VJP.
9. Add paper-grade baseline comparison for the winning workload.
10. Only then revisit training-side compiled adjoints beyond the current
    replacement proof.

## 11. Immediate Next Command Skeleton

Start with a report stub, not a kernel fork:

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --out-dir outputs/benchmarks/$(date +%Y-%m-%d)_star_uvt_projective_decisive_demo
```

Then force the report to answer:

```text
Did quality match?
Did world-side work scale sublinearly?
Did total runtime improve after accepted timing protocol?
Did memory stay bounded?
Where did fallback happen?
Did backward still work?
```

If a report cannot answer those six questions, it is not the decisive demo yet.
