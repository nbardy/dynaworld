# Gauged UVT / World Tubes Code Implementation Plan

Date: 2026-07-06

## Short Answer

Yes, we know the next code to implement.

The next code should **not** be a new renderer from scratch and should **not**
begin with a broad native Metal rewrite. The first extension should be a
testable decisive-demo/report layer over the projective interval machinery that
already exists.

The implementation spine is:

```text
projective_decisive_demo_report.py
    -> uses existing projective interval atlas render/backward APIs
    -> emits JSON + contact sheet + fallback/visibility overlays
    -> proves or falsifies the camera-path compiler story on replayable cases

projective_visibility_stress_suite.py
    -> sweeps orbit / rolling / exposure / occlusion pathologies
    -> measures when gauge domains, visibility strata, and fallback collapse

projective_native_atlas_kernel_report.py
    -> only after the demo identifies bridge overhead as the bottleneck
    -> small native evaluator first, direct VJP second
```

The current planning gap is narrower than it felt in conversation:

```text
Known:
    report shape
    existing renderer/backward APIs
    existing verifier/test style
    first demo metrics
    first stress metrics
    first pass/fail thresholds

Not yet fixed:
    exact native Metal shader entry points
    public-dataset paper baselines
    whether WorldFoam should be fused into this code path or stay second-paper
```

That is the right boundary. We should earn the native kernel by measurement,
not by aesthetic desire.

## Current Code Surfaces To Reuse

The projective/gauged code already exposes most of the machinery needed for a
first implementation. Do not duplicate these unless a report proves they are
the bottleneck.

### Report And Verifier Pattern

Use the established report style:

```text
research_experiments/star_uvt_feature_tubes/projective_goal_final_completion_audit.py
research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py
research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_quadrature_report.py
research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_backward_report.py
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py
research_experiments/star_uvt_feature_tubes/report_artifacts.py
```

The idiom is:

```python
def summarize(report: dict[str, Any]) -> dict[str, Any]: ...
def run_report(...) -> dict[str, Any]: ...
def verify_...(report: dict[str, Any]) -> list[str]: ...
def assert_...(report: dict[str, Any]) -> None: ...
def write_report(report: dict[str, Any], out_dir: Path) -> None: ...
def main() -> None: ...
```

Tests should mirror the existing report tests:

```text
tests/test_star_uvt_projective_goal_final_completion_audit.py
```

Specifically:

- build a valid in-memory fixture
- mutate one field at a time to prove stale/bad reports are rejected
- optionally read saved artifacts if present
- skip optional saved-artifact tests when files are missing
- do not require long training runs inside unit tests

### Projective Atlas APIs

Use the existing bridge API surface from:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py
tests/test_star_uvt_projective_correctness.py
```

Relevant imports already used in tests:

```python
split_projective_trace_windows
bound_projective_trace_windows
bin_projective_trace_support_bounds
assemble_projective_trace_tile_time_atlas
projective_trace_windows_to_cell_trace_atlas
lower_projective_trace_cell_atlas_quadrature
lower_projective_trace_cell_atlas_rolling_quadrature
mark_projective_trace_cell_visibility_fallbacks
stratify_projective_trace_cell_atlas_visibility
split_projective_trace_cell_atlas_fallback_cells
projective_trace_cell_atlas_visibility_report
projective_trace_cell_atlas_fallback_stats
projective_trace_cell_atlas_complexity_stats
projective_trace_cell_atlas_coverage_report
projective_trace_cell_atlas_budget_report
render_projective_trace_cell_interval_atlas_metal
render_projective_trace_cell_atlas_reference
render_projective_trace_cell_atlas_quadrature_reference
render_projective_trace_cell_atlas_rolling_quadrature_reference
render_projective_trace_cell_atlas_rolling_quadrature_interval_metal
render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal
direct_backward_projective_trace_cell_interval_atlas_metal
```

These are enough for:

- orbit traces
- finite exposure
- rolling shutter
- mixed fallback
- interval Metal forward
- interval Metal direct VJP
- visibility/fallback stats
- complexity/budget reports

## Implementation Unit 1: Decisive Demo Report

### Purpose

Make the research claim inspectable:

```text
compile a known camera program into reusable sensor-time traces
then slice/evaluate those traces faster than replaying projection/binning
```

This report should be the first thing to implement.

### New Files

```text
research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py
tests/test_star_uvt_projective_decisive_demo_report.py
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/summary.json
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/summary.md
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/contact_sheet.png
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/fallback_heatmap.png
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/runtime_bars.png
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/memory_bars.png
```

Do not add helper modules until the report becomes hard to read. If it does,
extract only fixture/media helpers:

```text
research_experiments/star_uvt_feature_tubes/projective_demo_fixtures.py
research_experiments/star_uvt_feature_tubes/projective_demo_media.py
```

### CLI Contract

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --out-dir outputs/benchmarks/$(date +%Y-%m-%d)_star_uvt_projective_decisive_demo \
  --fixture-only
```

Later, when saved real-video inputs are wired:

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --out-dir outputs/benchmarks/$(date +%Y-%m-%d)_star_uvt_projective_decisive_demo \
  --include-saved-real-video
```

Verification:

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --verify-report outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/summary.json
```

Fresh-input verification, only after saved-input dependencies are explicit:

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --verify-report outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_decisive_demo/summary.json \
  --verify-current-inputs
```

### Main Data Model

Use plain dictionaries or small dataclasses. Keep the written JSON plain.

```python
@dataclass(frozen=True)
class DemoCase:
    case_id: str
    scene_id: str
    path_kind: str                  # orbit | exposure | rolling | exposure_rolling
    frame_count: int
    shutter_sample_count: int
    image_width: int
    image_height: int
    tile_size: int
    primitive_count: int
    source: str                     # synthetic_fixture | saved_real_video


@dataclass(frozen=True)
class DemoVariant:
    variant_id: str                 # replay | compiled | compiled_mixed_fallback
    uses_projective_interval: bool
    uses_metal_forward: bool
    uses_direct_vjp: bool
    uses_mixed_fallback: bool


@dataclass(frozen=True)
class DemoCaseResult:
    case: DemoCase
    variant: DemoVariant
    metrics: dict[str, Any]
    artifacts: dict[str, str]
```

The JSON should write rows, not nested opaque blobs:

```json
{
  "benchmark": "star_uvt_projective_decisive_demo",
  "status": "ok",
  "rows": [
    {
      "row_id": "...",
      "case_id": "...",
      "scene_id": "...",
      "path_kind": "orbit",
      "frame_count": 32,
      "shutter_sample_count": 1,
      "variant": "compiled_interval_atlas",
      "status": "ok",
      "quality_pass": true,
      "timing_pass": true,
      "memory_pass": true,
      "fallback_pass": true,
      "gradient_pass": true
    }
  ],
  "summary": {}
}
```

### Functions To Implement

```python
def build_orbit_fixture_case(...) -> DemoCaseResult:
    ...

def build_exposure_fixture_case(...) -> DemoCaseResult:
    ...

def build_rolling_fixture_case(...) -> DemoCaseResult:
    ...

def render_replay_reference(...) -> dict[str, Any]:
    ...

def render_compiled_interval(...) -> dict[str, Any]:
    ...

def render_compiled_mixed_fallback(...) -> dict[str, Any]:
    ...

def measure_variant(...) -> dict[str, Any]:
    ...

def make_contact_sheet(...) -> Path:
    ...

def make_fallback_heatmap(...) -> Path:
    ...

def make_runtime_bars(...) -> Path:
    ...

def make_memory_bars(...) -> Path:
    ...

def summarize(report: dict[str, Any]) -> dict[str, Any]:
    ...

def verify_projective_decisive_demo_report(report: dict[str, Any]) -> list[str]:
    ...

def assert_projective_decisive_demo_report(report: dict[str, Any]) -> None:
    ...

def verify_projective_decisive_demo_current_acceptance(
    report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> list[str]:
    ...

def write_report(report: dict[str, Any], out_dir: Path) -> None:
    ...

def main() -> None:
    ...
```

### Metrics

Each result row should include:

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
visibility_strata_count
fallback_cell_fraction
fallback_sample_fraction
max_image_abs_error_vs_reference
mean_image_abs_error_vs_reference
psnr_vs_reference
gradient_rel_error
memory_payload_bytes
```

Report summary should recompute:

```text
case_count
row_count
compiled_row_count
replay_row_count
all_quality_pass
all_timing_pass
all_memory_pass
all_fallback_pass
all_gradient_pass
max_fallback_cell_fraction
max_fallback_sample_fraction
max_image_abs_error_vs_reference
min_psnr_vs_reference
max_gradient_rel_error
best_compiled_to_replay_total_ratio
worst_compiled_to_replay_total_ratio
projection_binning_proxy_ratio_at_max_frames
memory_growth_ratio_at_max_frames
```

### First Fixture Cases

Keep fixtures small enough for unit tests and meaningful enough to exercise the
geometry.

#### Fixture A: Clean Orbit

```text
path_kind: orbit
frames: 8, 16, 32
image: 64x64 or 96x96
tubes: 2 to 8
camera parameter: tan(theta / 2)
expected: no fallback, stable order, exact or near-exact reference match
```

Use the orbit coefficient helpers already visible in
`tests/test_star_uvt_projective_correctness.py` as the starting pattern.

#### Fixture B: Exposure Orbit

```text
path_kind: exposure
frames: 8, 16
shutter samples: 4, 8
expected: compiled quadrature route matches replay quadrature
```

Use:

```python
lower_projective_trace_cell_atlas_quadrature
render_projective_trace_cell_atlas_quadrature_reference
render_projective_trace_cell_atlas_quadrature_interval_metal
```

#### Fixture C: Rolling Orbit

```text
path_kind: rolling
frames: 8, 16
row/time coupling: mild
expected: batched rolling reference matches interval/mixed route
```

Use:

```python
lower_projective_trace_cell_atlas_rolling_quadrature
render_projective_trace_cell_atlas_rolling_quadrature_reference
render_projective_trace_cell_atlas_rolling_quadrature_batched_reference
render_projective_trace_cell_atlas_rolling_quadrature_interval_metal
render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal
```

#### Fixture D: Intentional Visibility Ambiguity

```text
path_kind: orbit
frames: 16
tubes: crossing depth order
expected: fallback/strata nonzero, report shows it visibly
```

This fixture should not be treated as a failure unless the fallback fraction
exceeds the explicit stress threshold.

### Pass/Fail Thresholds

For fixture rows:

```text
quality_pass:
    max_image_abs_error_vs_reference <= 1e-5

gradient_pass:
    gradient_rel_error <= existing interval backward tolerance
    or route flags prove direct VJP for rows without an oracle

fallback_pass ordinary:
    fallback_cell_fraction <= 0.20
    fallback_sample_fraction <= 0.20

timing_pass:
    compiled total_no_first_ms / replay total_no_first_ms <= 0.85
    only for rows where warm timing was actually measured

memory_pass:
    compiled payload growth <= 0.25 * replay payload growth at max F/K
```

For first implementation, timing gates can be marked:

```text
status = "not_measured"
timing_pass = null
```

That is acceptable for the verifier only when:

```text
report["mode"] == "fixture_correctness"
```

The full decisive report must not allow timing to be absent.

### Unit Tests

Create:

```text
tests/test_star_uvt_projective_decisive_demo_report.py
```

Required tests:

```python
def test_decisive_demo_accepts_valid_fixture_report() -> None: ...
def test_decisive_demo_rejects_missing_key_math() -> None: ...
def test_decisive_demo_rejects_stale_summary() -> None: ...
def test_decisive_demo_rejects_bad_quality_gate() -> None: ...
def test_decisive_demo_rejects_hidden_fallback_failure() -> None: ...
def test_decisive_demo_rejects_missing_artifact_path_for_media_mode() -> None: ...
def test_decisive_demo_fixture_run_smoke() -> None: ...
def test_saved_decisive_demo_artifact_satisfies_contract() -> None: ...
```

The smoke can run `run_report(mode="fixture_correctness")` and avoid long
training or saved real-video dependencies.

## Implementation Unit 2: Visibility Stress Suite

### Purpose

Answer the hard question directly:

```text
where do gauge domains and visibility strata stop carrying a revolving or fast
camera path, and where does fallback dominate?
```

### New Files

```text
research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py
tests/test_star_uvt_projective_visibility_stress_suite.py
outputs/benchmarks/YYYY-MM-DD_star_uvt_projective_visibility_stress_suite/summary.json
```

### Scene Families

```text
clean_orbit_ordered
crossing_translucent_planes
thin_foreground_occluder
near_camera_elongated_splat
wide_fov_orbit
fast_rotation_rolling_shutter
dense_alpha_cloud
disocclusion_wall_reveal
```

### Sweep Axes

Start narrow; expand only after the report works.

```text
fov_degrees: 30, 60, 90, 120
orbit_degrees: 30, 90, 180, 360
rotation_speed: low, medium, high
rolling_readout_fraction: 0.0, 0.25, 0.5, 1.0
exposure_fraction: 0.0, 0.25, 0.5, 1.0
opacity_density: low, medium, high
primitive_anisotropy: 1, 4, 16, 64
near_depth_ratio: 0.01, 0.05, 0.1
```

### Metrics

```text
fallback_cell_fraction
fallback_sample_fraction
order_flip_surface_count
ambiguous_pair_count
commutable_pair_count
depth_interval_overlap_rate
visibility_strata_count
max_cells_per_trace
max_active_set_group_count
quality_error
runtime_ratio
memory_ratio
collapse_reason
```

### Collapse Definition

```text
collapse = fallback_cell_fraction > 0.40
           or fallback_sample_fraction > 0.40
           or runtime_ratio >= 1.0
           or quality_error > threshold while fallback is disabled
```

The suite must report collapse, not hide it.

### Unit Tests

```python
def test_visibility_stress_accepts_valid_fixture_report() -> None: ...
def test_visibility_stress_rejects_missing_collapse_boundary() -> None: ...
def test_visibility_stress_rejects_unexplained_high_fallback() -> None: ...
def test_visibility_stress_rejects_stale_summary() -> None: ...
def test_visibility_stress_fixture_smoke_has_clean_and_ambiguous_rows() -> None: ...
```

## Implementation Unit 3: Real-Video Decisive Demo Rows

### Purpose

Lift the demo from synthetic correctness to the actual research claim:

```text
same representation, same source videos, replay versus compiled path
```

### Existing Inputs To Consume

Start from saved artifacts already used by:

```text
projective_real_video_multiscene_trainer_matrix.py
projective_real_video_multiscene_quality_tether_report.py
projective_real_video_multiscene_media_tether_report.py
projective_real_video_timing_protocol_acceptance.py
projective_real_video_compiled_adjoint_replacement_report.py
```

The decisive demo should not rerun broad training by default. It should read
existing payloads first and only run fresh cases behind an explicit flag.

### Current-Input Acceptance

Implement:

```python
def verify_projective_decisive_demo_current_acceptance(...):
    ...
```

It should reject:

- missing saved payloads
- mismatched source segment ids
- changed frame counts
- changed renderer route flags
- changed cadence/measured loss deltas
- changed contact-sheet hashes when media mode is enabled
- stale summary fields

This follows the pattern in final-completion and shared-work reports.

## Implementation Unit 4: Native Projective Atlas Kernel Keep/Kill

### When To Start

Do not start this first.

Start native kernel work only if the decisive demo shows:

```text
compiled correctness passes
fallback fraction is acceptable
but total speed is bridge-overhead limited
```

or:

```text
payload/memory is dominated by bridge representation conversion
```

### New Files

```text
research_experiments/star_uvt_feature_tubes/projective_native_atlas_kernel_report.py
tests/test_star_uvt_projective_native_atlas_kernel_report.py
```

Likely implementation area, to inspect before editing:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/
```

Do not hard-code the exact shader filename in this plan; inspect the current
extension layout before editing. The codebase has had multiple variant layouts,
and guessing here is how stale plans break builds.

### First Kernel Scope

Forward only:

```text
input:
    image size
    sample times
    tile/cell active ranges
    projective trace coefficients
    opacity/color
    fixed compiled order

output:
    image samples
```

No mixed fallback inside the first native kernel. Patch fallback outside the
kernel using the existing mixed route until forward parity is proven.

### Second Kernel Scope

Direct VJP:

```text
grad wrt projective trace coefficients
grad wrt opacity
grad wrt color/features
optional grad wrt q-basis/family coefficients
```

Visibility order and tile membership remain compiled constants in the first
VJP. This matches the current compiled-adjoint replacement contract.

### Acceptance

```text
forward max abs error <= existing interval forward threshold
backward relative error <= existing interval backward threshold
fresh-process median forward ratio < bridge path
payload ratio < bridge path
```

Kill it if:

```text
native path is not faster after warmup
or native path duplicates too much report/trainer bridge logic
or visibility/fallback complexity dominates before bridge overhead matters
```

## Implementation Unit 5: WorldFoam Bridge, Kept Separate

### Purpose

WorldFoam should share the camera-bundle language but not silently change the
World Tubes baseline-compatible claim.

### Code Plan

Do not fold WorldFoam into `projective_decisive_demo_report.py`.

Instead create, after the World Tubes decisive demo:

```text
research_experiments/dynamic_foam/worldfoam_camera_bundle_stress_report.py
tests/test_worldfoam_camera_bundle_stress_report.py
```

Its role:

```text
sigma(u,v,t,z) over the ray fiber
prefix optical depth / transmittance
crossing slab and translucent cloud synthetic scenes
same-representation replay baseline
gradient correctness for prefix/suffix transmittance
```

The bridge between papers is conceptual and metric-level:

```text
World Tubes:
    baseline-compatible sorting/compositing with visibility gauge atlas

WorldFoam:
    lifted transmittance over ray fiber, sort-free by construction
```

Do not claim WorldFoam quality parity unless its own report proves it.

## Test Matrix

### Fast Unit Tests

Run after report implementation:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_decisive_demo_report.py \
  tests/test_star_uvt_projective_visibility_stress_suite.py -q
```

### Existing Projective Correctness Tests

Run after touching projective atlas APIs:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py -q
```

### Final Audit Regression

Run before claiming the new work preserves the completed state:

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_final_completion_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json \
  --verify-current-inputs
```

and:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_final_completion_audit.py \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py -q
```

### Metal Build/Import Smoke

If native extension code changes, first confirm the active Python extension is
built for Python 3.11:

```bash
ls third_party/fast-mac-gsplat/variants/star_uvt_v0/**/_C.cpython-311-darwin.so
```

Then run the smallest import/render smoke before broad tests.

## First PR-Sized Slice

The first implementation slice should be:

```text
1. Add projective_decisive_demo_report.py with:
       - report schema
       - summarize / verify / assert / write
       - fixture_correctness mode
       - one clean orbit fixture row

2. Add tests/test_star_uvt_projective_decisive_demo_report.py with:
       - valid fixture accepted
       - stale summary rejected
       - bad quality row rejected
       - fixture smoke

3. Add contact_sheet.png only after the JSON report is stable.

4. Add exposure/rolling rows.

5. Add saved-real-video rows behind --include-saved-real-video.
```

This is the shortest path from theory to a real code artifact.

## What We Should Not Do Yet

Do not:

- implement a new native Metal renderer before the decisive demo
- replace current projective interval APIs wholesale
- merge WorldFoam transmittance into World Tubes compatibility tests
- make timing claims from first-run kernel compile effects
- add tests that only assert helper internals
- run broad public/SOTA baselines before the same-representation replay demo
- mutate the final completion audit unless one of its input artifacts regresses

## Confidence Table

| Area | Confidence | Reason |
| --- | --- | --- |
| Report/verifier implementation | High | Existing reports already use the exact pattern. |
| Clean orbit fixture | High | Existing projective correctness tests already construct orbit traces. |
| Exposure/rolling fixture | Medium-high | Existing quadrature/rolling APIs and reports exist; still needs clean demo packaging. |
| Direct VJP fixture | Medium-high | Existing interval backward is tested; demo must choose small oracle cases. |
| Real-video saved rows | Medium | Artifacts exist, but current-input acceptance must be carefully wired. |
| Native projective kernel | Medium-low | We know the shape, not the exact shader edit path; measure first. |
| WorldFoam bridge | Medium | Conceptual split is clear; quality and parity evidence remain separate blockers. |

## Immediate Answer To The User's Question

We have enough to implement the next extension cleanly and testably.

The next code is:

```text
projective_decisive_demo_report.py
tests/test_star_uvt_projective_decisive_demo_report.py
```

The extension after that is:

```text
projective_visibility_stress_suite.py
tests/test_star_uvt_projective_visibility_stress_suite.py
```

The native Metal extension is planned at the level of inputs, outputs, and
acceptance gates, but not at the level of exact shader entry points. That is
intentional. The codebase should first prove whether the current bridge is
actually the bottleneck.
