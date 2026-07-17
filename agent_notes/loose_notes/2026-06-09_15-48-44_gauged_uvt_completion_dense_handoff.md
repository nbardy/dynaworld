# 2026-06-09 15:48:44 Gauged UVT Completion Dense Handoff

## Context

This note is a dense handoff after returning to the Gauged UVT / Sensor-Time
Trace Atlas thread. The relevant goal was:

```text
4D spacetime primitives compiled through a known camera program into reusable
sensor-time traces for fast rasterization across time, with clean derivatives,
maximal compute reuse, memory-bandwidth reuse, and backward-pass reuse across
frames so non-pixel costs grow sublinearly with frame count.
```

The current authoritative completion artifact is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json
```

The current promotion summary, inspected on 2026-06-09, records:

```text
completion_ready = true
is_goal_complete = true
does_not_prove_completion = false
proved_requirement_count = 6
open_requirement_ids = []
source_gap_open_gap_ids = ["full_goal_completion"]
```

The lower gap report remains intentionally non-completion-scoped:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json

completion_ready = false
does_not_prove_completion = true
proved_requirement_count = 5
partial_requirement_count = 1
open_gap_ids = ["full_goal_completion"]
```

This is not contradictory. The gap report is the final pre-completion evidence
ledger. The promotion audit is the artifact that consumes that ledger and
closes the top-level goal. Future agents should not "fix" the gap report to say
complete; its refusal to claim completion is part of the proof ladder.

## One-Sentence Model

Gauged UVT is a camera-path compiler: it pulls spacetime world primitives
through a known camera program, pushes/marginalizes along depth fibers, stores
piecewise sensor-time trace records plus visibility/order metadata, and
evaluates many frames or shutter samples by slicing a compiled atlas instead of
repeating world-side projection/support/binning/sorting/backward work per
frame.

## Core Mathematical Object

The stable invariant is:

```text
trace = pi_* Gamma^* world_primitive
```

where:

```text
Gamma : B x D -> M
B = Omega x T                 sensor-time base
D                             depth fiber
M = R^3 x R                   world spacetime
pi : B x D -> B               depth-fiber projection
```

For sensor coordinate:

```text
y = (u, v, tau) in B
s in D
x = Gamma(y, s)
```

The world primitive is first pulled back:

```text
rho_i^Gamma(y, s) = rho_i(Gamma(y, s))
```

Then compiled into a sensor-time trace by depth pushforward:

```text
alpha_i(y) ~= int_D rho_i^Gamma(y, s) ds
```

For Gaussian primitives and locally affine camera gauge, this yields the
Schur-complement trace form. If:

```text
rho_i(x) = a_i exp[-1/2 (x - m_i)^T Lambda_i (x - m_i)]
Gamma(y, s) ~= x0 + J [dy, ds]
H = J^T Lambda_i J
g = J^T Lambda_i (m_i - x0)
```

with block split by `y` and `s`:

```text
H = [[H_yy, H_ys],
     [H_sy, H_ss]]
g = [g_y, g_s]
```

then the depth-marginalized local sensor trace has precision:

```text
S = H_yy - H_ys H_ss^-1 H_sy
```

and the conditional depth model is:

```text
s_hat_i(y) = s0 + H_ss^-1 (g_s - H_sy (y - y0))
Var(s | y) = H_ss^-1
```

That conditional depth is the visibility hook. The trace is not merely a
screen-space splat; it is a section of a camera-ray bundle with a pushed-forward
measure and an order model.

## Why "Gauge Domains" Won Over "Just Charts"

The user correctly objected that "charts" sounded like a weak local patching
story. The final formulation treats charts as coordinate expressions of richer
gauge domains:

```text
gauge domain = local trivialization of the camera-ray/depth bundle
```

The invariant is not a Gaussian in UVT. The invariant is the pull-push measure:

```text
pi_* Gamma^* rho
```

A local UVT Gaussian, polynomial trace, interval atlas record, or q-UVT family
record is only a coordinate representation of that invariant. Gauge changes are
allowed only when they preserve the measure/Jacobian contract and carry the
correct derivative map.

This matters for revolving cameras. A revolving path often destroys any one
global affine screen approximation, but it does not destroy the bundle object.
The orbit is handled by stratifying or gauging along the camera program:

```text
Q x Omega x T          optional camera-family base
Omega x T              path-fixed base
local gauge domain     valid projection/support/order neighborhood
event strata           support/order lifecycle partitions
```

Throwing away "charts" is valid only if the replacement still supplies:

```text
1. a local coordinate/gauge for evaluation,
2. conservative support bounds,
3. visibility/order certificates,
4. derivative transport,
5. event/fallback semantics where lifecycle changes occur.
```

The final implementation keeps this disciplined: the code uses interval
records and metadata strata, but the docs now frame them as gauge-domain
certificates, not ad hoc visual patches.

## Compiler Target

The compiled object is not a video and not a universal scene representation.
It is tied to a known camera program:

```text
K_Gamma = Compile_epsilon(W, Gamma)
```

The intended contents are:

```text
sensor-time trace records
active-set metadata
support intervals
depth/order metadata
visibility fallback masks
adjoint/backward accumulation structures
source primitive refs
error/fallback certificates
```

Evaluation is a slice:

```text
I_k(u, v) = Eval(K_Gamma, u, v, tau_k)
```

Finite exposure is an integral over sensor time:

```text
I_k(u, v) = int w_k(tau) Eval(K_Gamma, u, v, tau) d tau
```

Rolling shutter is a row/time-coupled map:

```text
tau = tau0 + r(v) + shutter_sample
```

The acceptance artifacts treat finite exposure and rolling shutter as weighted
sample/integration contracts, not as unrelated postprocessing.

## Visibility Model

The hard problem was not depth integration. It was visibility and lifecycle
stability. The final stack treats visibility as:

```text
piecewise stable depth/order metadata
+ bounded commutation / event masks
+ fallback for ambiguous regions
```

Depth order is locally derived from:

```text
s_hat_i(y)
```

Order changes occur when:

```text
s_hat_i(y) = s_hat_j(y)
```

For affine depth models this is a plane in sensor-time; for richer models it is
a curved event surface. The compiler does not pretend these surfaces disappear.
It encodes stable topology regions and split strata.

Important proof rows included:

```text
tile_order_reuse
tile_order_strata
active_set_strata
real_active_set_distribution
uv_visibility_split_report
depth_affine_interval_metal_order
uv_event_driven_fallback
```

The key failure mode remains: if a future change silently lets support/order
change inside a "stable" interval without a certificate or fallback mask, it
invalidates the completion proof. This is the most dangerous regression class.

## Backward / Adjoint Model

The backward pass is treated as part of the compiled atlas, not an afterthought.
Inside a stable-order tile, front-to-back alpha compositing gives local
derivatives:

```text
dI/dc_i     = T_i alpha_i
dI/dalpha_i = T_i (c_i - I_behind_i)
```

For Gaussian-like trace opacity:

```text
alpha_i(y) = A_i exp[-1/2 (y - mu_i)^T S_i (y - mu_i)]
dalpha/dA  = alpha / A
dalpha/dmu = alpha S (y - mu)
dalpha/dS  = -1/2 alpha (y - mu)(y - mu)^T
```

So the backward workload is structured as:

```text
int_C adjoint(y) * transmittance(y) * trace(y) * polynomial(y) dy
```

The practical route that actually closed the goal is not the deterministic
compact static-STAR backward. It is the direct-atomic RGB trainer path backed
by compiled projective interval traces and interval Metal direct VJP:

```text
trainer selects _render_projective_interval_feature_tubes_autograd
harness uses _ProjectiveCellIntervalBackward
forward calls render_projective_trace_cell_interval_atlas_metal
backward calls direct_backward_projective_trace_cell_interval_atlas_metal
visibility/tile membership are compiled constants
```

The final compiled-adjoint replacement artifact checks 20 broad10 case payloads
and requires:

```text
all projective-interval main path
all RGB direct-loss autograd
all renderer gradient flags present
forward/backward timings present
measured cache reuse ok
zero fallback/support churn
10 broad trainer sources
10 broad quality/media sources
four frame-count points
shared-work ratios below threshold
compiled_trainer_replacement_gap = 0
```

## What Was Actually Proved

The final promotion audit proves six rows:

```text
scope_and_key_math_preserved
sensor_time_trace_compiler_evidence
sublinear_non_pixel_work_evidence
broad_real_video_acceptance_evidence
compiled_adjoint_training_evidence
final_completion_promotion
```

The row meanings are:

```text
scope/key math:
    goal, meta-goal, key math, and camera-ray bundle theory framing survived.

sensor-time trace compiler:
    projective/gauged Metal paths, camera-family gauges, interval support,
    visibility metadata, and derivatives are present in the evidence stack.

sublinear non-pixel work:
    shared-work proxies stay under thresholds:
        orbit_payload_growth_ratio = 0.125 <= 0.20
        trained_interval_growth_ratio = 0.14836872087001554 <= 0.25
        max_backward_ratio = 0.09386445865404805 <= 0.25

broad real-video acceptance:
    10 broad quality sources, 10 broad media sources, 4 frame-count points,
    accepted fresh-process timing protocol, zero support/stale issues.

compiled-adjoint training:
    practical trainer replacement uses compiled interval traces and direct VJP
    as main path, with broad10 case payload coverage.

final promotion:
    all concrete gaps are zero and the only lower open row was the need for the
    final promotion artifact itself.
```

The concrete gap counters at completion:

```text
broad_quality_source_gap = 0
broad_media_source_gap = 0
broad_quality_frame_count_gap = 0
strict_timing_failure_gap = 0
timing_acceptance_gap = 0
compiled_trainer_source_gap = 0
compiled_trainer_replacement_gap = 0
```

## Timing Interpretation

Strict warm-state timing did not become the winning final claim. The accepted
protocol is:

```text
fresh-process median with warmup discard
```

The reason is empirical:

```text
strict warm-state misses existed
cache/support invariants stayed clean
workload did not explain render-forward misses
Bq4 spike behavior was not stable under rerun/repeat/order probes
fresh-process medians passed
```

The accepted timing medians are:

```text
fresh_process_median_no_first_ratio = 0.5645123618278631
fresh_process_median_projective_total_ratio = 0.8356591487478802
fresh_process_median_feature_state_update_ratio = 0.846418513757801
```

The lesson is not "timing does not matter." The lesson is that the proof
separates:

```text
cache/support/math correctness
from
MPS warm-state process variance
```

Future timing claims should not re-promote strict warm-state ratios unless they
also control launch/process/warmup variance. Use the fresh-process protocol or
a stronger replacement.

## Revolving Camera Status

Revolving camera concern was handled through several layers:

```text
orbit fixed-chart scaling
camera-family Q and Q2 gauge reports
q-UVT bridge
tile/order strata
active-set strata
real active-set distribution
shared-work audit
```

The final stance is:

```text
Revolving paths are not solved by one global UVT Gaussian.
They are handled by camera-program gauges plus interval/topology certificates.
```

The key orbit/shared ratios in the final evidence:

```text
orbit_payload_growth_ratio = 0.125
trained_shared_to_replay_interval_growth_ratio = 0.14836872087001554
max_trained_final_backward_ms_ratio = 0.09386445865404805
```

This is the exact sense in which non-pixel costs are sublinear: not that output
pixels vanish, but that world-side payload, interval metadata, and backward
work do not scale linearly with the number of frames/samples under the tested
known-path workloads.

## What Not To Reopen Accidentally

Do not rewrite the theory back to "a 4D Gaussian becomes a 3D Gaussian in UVT"
as the primary claim. The defensible claim is:

```text
A world primitive induces a reusable sensor-time trace under a camera program.
Local Gaussian/polynomial/interval records are coordinate approximations of
pi_* Gamma^* rho, with support/order/fallback certificates.
```

Do not make charts sound like mere fitting windows. Call them gauge domains or
local trivializations when discussing the math. Keep "chart" for implementation
records only.

Do not erase the distinction between:

```text
gap report = non-completion evidence ledger
promotion audit = authoritative completion artifact
```

Do not claim strict warm-state timing win. The accepted timing protocol is
fresh-process median with warmup discard, with strict warm-state misses kept as
caveats.

Do not claim universal novel-view synthesis. The compiler is path-specific or
local-camera-family-specific:

```text
single known trajectory -> cheap B = Omega x T atlas
nearby camera family -> higher-dimensional Q x Omega x T atlas
arbitrary novel view -> recompile or use a broader world representation
```

Do not conflate practical direct-atomic RGB completion with deterministic
compact static-STAR promotion. The former closed this goal; the latter remains
a possible future optimization path, not a required completion row.

## Current Artifact Ladder

Top:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json
```

Consumes:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

Gap consumes:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
```

The report scripts:

```text
research_experiments/star_uvt_feature_tubes/projective_goal_completion_promotion_audit.py
research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py
research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
research_experiments/star_uvt_feature_tubes/projective_real_video_compiled_adjoint_replacement_report.py
```

The newest focused test:

```text
tests/test_star_uvt_projective_goal_completion_promotion_audit.py
```

## Verification Commands

Promotion current-input verifier:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_goal_completion_promotion_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json \
  --verify-current-inputs
```

Focused promotion/progress/gap/replacement gate:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_completion_promotion_audit.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py -q
```

Last known result:

```text
82 passed in 4.02s
```

Wider cross-report gate:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_timing_protocol_acceptance_report.py \
  tests/test_star_uvt_projective_real_video_frame_count_breadth_diagnostic_report.py \
  tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py \
  tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py \
  tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_goal_completion_promotion_audit.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Last known result:

```text
121 passed in 4.72s
```

## If Future Agents Continue This Work

The next work should not be "prove completion again." Completion has an
authoritative artifact. Future useful directions are:

```text
1. Turn the promotion audit into a paper/table-friendly result summary.
2. Broaden to non-RGB feature-space trainer replacement beyond direct-atomic RGB.
3. Replace fresh-process median timing with a stricter controlled-runtime timing
   protocol if reproducibility improves.
4. Explore deterministic compact static-STAR backward as an optimization, not
   as a missing completion requirement.
5. Extend Q-family native Metal evaluation beyond slice lowering where it buys
   meaningful memory or launch reduction.
6. Convert gauge-domain theory into a concise paper section:
       pullback -> fiber pushforward -> interval atlas -> visibility strata
       -> adjoint trace integrals.
7. Build a smaller canonical demo that can be run by outsiders without the
   whole artifact ladder.
```

## Red-Team Checks

Ways the completion claim could become invalid:

```text
artifact drift:
    saved promotion audit no longer matches current gap/progress artifacts.

source-contract drift:
    trainer no longer selects projective interval autograd or direct Metal VJP.

visibility drift:
    active-set/order lifecycle changes inside an interval without split/fallback.

timing-protocol drift:
    someone cites strict warm-state speedups instead of the accepted
    fresh-process median protocol.

scope drift:
    someone presents this as arbitrary novel-view synthesis rather than a known
    camera-program compiler.

math drift:
    docs regress from pi_* Gamma^* rho to "UVT Gaussian fitting" as the core
    invariant.
```

Cheap falsification tests:

```text
run promotion current-input verifier
run focused promotion/progress/gap/replacement pytest
search for old final_goal_completion artifact names
check compiled replacement source-contract verifier
check gap concrete counters stay zero
check source_gap_open_gap_ids == ["full_goal_completion"]
check promotion open_requirement_ids == []
```

## Bottom Line

The project did not become a magical render-video-without-pixels machine. It
became something narrower and real: a known-camera-program compiler that
amortizes world-side trace/support/visibility/backward work over sensor time,
has a principled bundle/gauge mathematical framing, has Metal forward/backward
paths, handles finite exposure / rolling / visibility fallback contracts, and
has a verified completion artifact tying broad real-video and trainer evidence
into one audit ladder.
