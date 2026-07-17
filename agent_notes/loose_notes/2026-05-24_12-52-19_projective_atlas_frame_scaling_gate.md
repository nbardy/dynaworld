# Projective Atlas Frame-Scaling Gate

## Context

The active Gauged UVT Trace Atlas goal requires more than projective correctness:
it needs evidence that world-side work such as projection, support, binning, and
visibility can be shared across time. The previous state had projective atlas
forward rendering, native direct VJP, coefficient trainability, and interval
gates, but the frame-count scaling claim was still only a theory/next-gate item.

## Current Model

The compiled projective atlas has two different scaling shapes:

```text
ideal interval atlas:
    one accepted gauge-domain interval can index tile support once across many
    sensor-time samples

fixed slab path:
    the same interval is still replicated into fixed tile_t slabs because the
    hot UVT runtime currently allows tile_t in {1,2,4}

new interval Metal forward path:
    spatial tile entries are packed once, each entry carries an
    [active_start, active_stop) frame interval, and the shader checks that
    interval per output sample
```

So the right next evidence is not just "Metal renders projective traces." It is
to measure the dense per-frame project/bin denominator, the ideal
interval-packed atlas entries, and the current `tile_t=4` slab expansion on the
same orbit fixture.

## Changes

Added:

```text
count_projective_trace_dense_per_frame_tile_pairs(...)
```

to count the ordinary time-sliced project-and-bin support entries.

Added:

```text
render_projective_trace_cell_interval_atlas_metal(...)
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_tiles
```

to evaluate projective cell traces from interval-compressed spatial tile bins
without fixed `tile_t=4` slab replication in the forward pass.

Added:

```text
research_project/benchmarks/projective_atlas_scaling_probe.py
```

to report dense per-frame pairs, ideal interval atlas entries, current
Metal-compatible slab entries, interval Metal render timings, and optional Metal
image-sum parity.

Added the focused pytest:

```text
tests/test_star_uvt_projective_binning.py::test_projective_interval_packing_scales_sublinearly_over_frame_count
```

## Evidence

Focused scaling test:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_binning.py::test_projective_interval_packing_scales_sublinearly_over_frame_count -q
```

Result:

```text
1 passed in 4.33s
```

Full projective binning file:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_binning.py -q
```

Result:

```text
5 passed in 1.87s
```

Combined focused projective plus interval-gated trainer suite before the
interval-backward follow-up:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
46 passed in 11.61s
```

Interval-compressed projective cell Metal forward parity:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_renders_in_metal_if_available -q
```

Result:

```text
1 passed in 6.67s
```

CPU scaling probe:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/projective_atlas_scaling_probe.py \
  --out-json outputs/benchmarks/2026-05-24_projective_atlas_scaling_probe.json
```

Key result:

```text
frames:                         4 -> 64
dense_per_frame_tile_pairs:     35 -> 555
interval_packed_tile_entries:   13 -> 13
metal_slab_packed_tile_entries: 13 -> 208
dense_pair_growth:              15.857142857142858
interval_entry_growth:          1.0
metal_slab_entry_growth:        16.0
interval_pair_ratio:            0.37142857142857144 -> 0.023423423423423424
```

Optional current-Metal slab probe:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/projective_atlas_scaling_probe.py \
  --frame-counts 4,8,16 \
  --run-metal \
  --iterations 1 \
  --warmup-iterations 1 \
  --out-json outputs/benchmarks/2026-05-24_projective_atlas_scaling_probe_metal_4_8_16.json
```

Key result:

```text
frames:                         4 -> 16
dense_per_frame_tile_pairs:     35 -> 139
interval_packed_tile_entries:   13 -> 13
metal_slab_packed_tile_entries: 13 -> 52
Metal image sums:               386.0363, 772.0728, 1544.1455
Metal render ms:                50.6553, 15.1279, 18.2319
```

The Metal timings are not a clean promotion benchmark because first-use MPS
effects dominate this tiny fixture. The important measured fact is the packed
entry shape: current Metal-consumable slab bins grow linearly with the number of
4-frame slabs, while the ideal interval atlas entries stay flat.

Interval Metal forward probe after adding
`render_projective_trace_cell_interval_tiles`:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/projective_atlas_scaling_probe.py \
  --run-metal \
  --iterations 1 \
  --warmup-iterations 1 \
  --out-json outputs/benchmarks/2026-05-24_projective_atlas_scaling_probe_interval_metal.json
```

Key result:

```text
frames:                         4 -> 64
dense_pair_growth:              15.857142857142858
interval_entry_growth:          1.0
metal_slab_entry_growth:        16.0
interval_pair_ratio:            0.37142857142857144 -> 0.023423423423423424
slab_pair_ratio:                0.37142857142857144 -> 0.3747747747747748
interval Metal image sums:      match slab image sums for every row
interval Metal render ms:       24.8067089705728 -> 29.36116699129343
slab Metal render ms:           20.09954204550013 -> 37.26170799927786
```

This is not yet a robust performance claim; it is a gate that proves the
forward shader can consume interval-compressed atlas cells without re-expanding
the support bins per temporal slab.

## Decision Implications

- Supported: the projective atlas compiler object can represent shared
  world-side support/binning over time; at fixed orbit support it keeps entries
  constant while dense per-frame binning grows with frame count.
- Strengthened: the projective Metal forward path can now consume interval
  compression directly through `render_projective_trace_cell_interval_tiles`.
- Strengthened again: the projective Metal direct VJP now consumes the same
  interval-compressed cell scheduler through
  `direct_projective_trace_cell_interval_backward`.
- At this point in the session, still open: the trainer path did not yet
  produce nontrivial projective or gauge-domain intervals, so the optimizer loop
  could only exercise this cell object through focused tests and probes. The
  follow-up below adds the trainer-harness bridge.

## Follow-Up: Interval Backward Gate

Added:

```text
torch.ops.star_uvt_v0.direct_projective_trace_cell_interval_backward
direct_backward_projective_trace_cell_interval_atlas_metal(...)
```

The new kernel dispatches over output samples, indexes spatial tile bins once,
checks per-entry `[active_start, active_stop)` intervals, and accumulates direct
VJP gradients for cell-local polynomial `u(t)` and `v(t)` coefficients, opacity,
and color. Depth coefficients are still treated as visibility/order metadata;
there is no differentiable depth-order term in this gate.

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_backward_matches_torch_autograd_if_available \
  tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_one_step_coeff_training_smoke_if_available -q
```

Result:

```text
2 passed in 1.82s
```

Focused correctness file:

```text
19 passed in 7.07s
```

Combined focused projective plus interval-gated trainer suite before the
trainer-harness producer follow-up:

```text
48 passed in 5.65s
```

New scaling artifact:

```text
outputs/benchmarks/2026-05-24_projective_atlas_scaling_probe_interval_backward_metal.json
```

Key result:

```text
frames:                         4 -> 64
interval_packed_tile_entries:   13 -> 13
metal_slab_packed_tile_entries: 13 -> 208
interval Metal image sums:      match slab image sums for every row
interval Metal render ms:       2.4674999876879156 -> 3.0619159806519747
slab Metal render ms:           6.024917005561292 -> 14.67433397192508
interval Metal backward ms:     6.7827089806087315 -> 35.68224998889491
```

Interpretation: the derivative path now consumes the interval-compressed atlas
object, so the support/binning side is shared across time. Total backward time
still grows with the number of output pixels, which is allowed by the objective;
the remaining research question at this point was whether a trainer-style
producer could preserve this reuse when chart windows are nontrivial and
changing under optimization. The follow-up below adds the first harness proof.

## Follow-Up: Trainer-Harness Producer Gate

Added:

```text
render_projective_cell_interval_atlas_metal_backward(...)
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_wrapper_uses_split_windows_and_trains
```

The wrapper is a PyTorch autograd bridge around the interval-compressed
projective cell forward and direct VJP. The test uses
`split_projective_trace_windows(...)` as the gauge-domain producer, verifies it
creates multiple active intervals, lowers the windows to a
`ProjectiveTraceCellTraceAtlas`, renders a shifted-coefficient target, runs
`loss.backward()`, and takes optimizer steps on cell trace coefficients. Loss
drops, proving the interval cell object can pass through a trainer-style
optimizer loop with nontrivial chart domains.

Focused trainer interval file:

```text
3 passed in 5.99s
```

Combined focused projective plus interval-gated trainer suite:

```text
49 passed in 5.61s
```

Remaining gap: this is a harness bridge, not the full production STAR UVT
trainer backend. The next implementation question is chart lifecycle: when
coefficients move enough to invalidate support/visibility bounds, the trainer
must refresh or recompile the affected intervals.

## Follow-Up: Support Staleness And Rebin Gate

Added:

```text
projective_trace_cell_atlas_coverage_report(...)
rebin_projective_trace_cell_atlas(...)
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_coverage_report_detects_motion_and_rebin_repairs
```

The coverage report is a compiler/training-control check for the static atlas
metadata around live differentiable cell coefficients. It evaluates current
cell traces over their active sample intervals, expands each sample by
`uv_padding`, and reports missing frame/tile coverage relative to the compiled
tile-time cells. The rebin helper keeps `coeffs`, `opacity`, and `color` as the
same live tensors and rebuilds only support cells plus conservative depth
interval metadata.

Focused evidence:

```text
initial compiled atlas: not stale, 4 checked tile pairs
after shifting u coefficient by +8.5 px: stale, 4 missing tile pairs
after rebin: not stale, missing tile pairs = 0
```

Combined focused projective plus interval-gated trainer suite:

```text
50 passed in 6.21s
```

Interpretation: chart lifecycle now has a minimal executable invariant for
support coverage. This does not yet solve visibility/order staleness; the next
trainer integration should decide when to call the report, when to rebin, and
when movement is large enough to require full chart refit rather than support
rebin.

## Follow-Up: Trainer Harness Refresh Gate

Added:

```text
refresh_projective_cell_interval_atlas_if_stale(...)
ProjectiveCellIntervalAtlasRefresh
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_lifecycle_rebins_after_optimizer_motion
```

The helper wraps the support lifecycle check for trainer loops. It returns the
old coverage report, a maybe-rebinned atlas, the post-refresh report, and a
`rebinned` flag. Rebinning preserves the live `coeffs`, `opacity`, and `color`
tensors, so optimizer-owned parameters are not replaced when static support
metadata changes.

Focused evidence:

```text
optimizer step moves a live MPS coeff tensor from tile 0 into tile 1
refresh.before: stale with 4 missing tile pairs
refresh.after: not stale
render_projective_cell_interval_atlas_metal_backward(...) renders right-tile energy
right_energy.backward() produces nonzero gradients on the same coeff tensor
```

Focused trainer lifecycle test:

```text
1 passed in 7.62s
```

Combined focused projective plus interval-gated trainer suite:

```text
51 passed in 7.21s
```

Interpretation: support staleness is no longer only an offline oracle. The
trainer harness can now preserve the single-step static-atlas autograd contract
across optimizer motion by refreshing support metadata between steps. Still
open at this point: production trainer ownership, thresholds/cadence, full
chart refit criteria, and visibility/order staleness.

## Follow-Up: Visibility Order Staleness Gate

Added:

```text
projective_trace_cell_atlas_visibility_report(...)
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_visibility_report_detects_depth_order_flip_and_rebin_repairs
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_refresh_rebins_depth_order_without_replacing_tensor
```

The visibility report reconstructs the same per-sample/per-tile active list the
renderer consumes: compiled cells contribute ordered ids and stale depth
intervals, then the report compares that compiled front-to-back order against
live depths evaluated from current cell coefficients. This catches a depth-only
staleness mode where screen support is still valid but the visibility order has
flipped.

Focused evidence:

```text
initial order: (0, 1)
after moving trace 0 depth from 1.0 to 3.0: stale with 4 order mismatches
after rebin: order becomes (1, 0), no mismatches
refresh.before coverage: not stale
refresh.visibility_before: stale
refresh.visibility_after: not stale
refresh.atlas.coeffs is the optimizer-owned tensor
```

Targeted order-staleness tests:

```text
2 passed in 3.21s
```

Combined focused projective plus interval-gated trainer suite:

```text
53 passed in 4.69s
```

Interpretation: lifecycle now covers two critical static-atlas invalidations:
support leaves compiled tile-time cells, and live depth order diverges from
compiled order. Remaining visibility work is harder: ambiguous-depth fallback,
visibility-stratum splitting, full chart refit criteria, and production trainer
ownership of refresh cadence.

## Follow-Up: Trainer-Owned Lifecycle State

Added:

```text
ProjectiveCellIntervalTrainerState
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_state_owns_support_refresh_and_render
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_state_owns_depth_order_refresh
```

The state object owns the current atlas, `times`, render config, `sigma_px`,
support/order refresh parameters, refresh cadence, step index, and last refresh
report. It exposes:

```text
render()
refresh(force=False)
after_optimizer_step()
```

Focused evidence:

```text
support case:
    optimizer moves live MPS coeff into a new tile
    state.after_optimizer_step() rebins support/order metadata
    state.render() uses interval Metal autograd
    backward writes nonzero gradients to the same coeff tensor

order case:
    refresh_every=2 skips first call and refreshes on the second
    depth-order flip is detected by visibility_before
    rebinned atlas order becomes (1, 0)
    state.atlas.coeffs is the optimizer-owned tensor
```

Targeted trainer-state tests:

```text
2 passed in 5.56s
```

Combined focused projective plus interval-gated trainer suite:

```text
55 passed in 8.42s
```

Interpretation: lifecycle ownership has moved from "manual helper the caller
might remember to call" to a trainer-style object with explicit cadence. This
is still harness-level, but it is the right shape for production integration:
the training loop can hold one state, render through it, and call
`after_optimizer_step()` after parameter updates.

## Follow-Up: Ambiguous Visibility Fallback Metadata

Added:

```text
mark_projective_trace_cell_visibility_fallbacks(...)
ProjectiveCellIntervalAtlasRefresh.fallback_marked
ProjectiveCellIntervalTrainerState.allow_ambiguous_fallback
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_visibility_ambiguity_marks_fallback_cells
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_trainer_state_marks_ambiguous_visibility_fallback
```

The visibility report now records `ambiguous_examples`. If depths remain within
`depth_epsilon` after rebin, strict refresh still raises: the atlas cannot
pretend that one static order is certified. If the trainer state opts into
`allow_ambiguous_fallback=True`, refresh marks affected cells with
`fallback=True` and reason `visibility_ambiguous_depth`. Ordinary pack/render
paths still reject fallback cells unless `allow_fallback_cells=True`, so this is
metadata for an explicit fallback path, not a hidden quality downgrade.

Focused evidence:

```text
near-tie depths 1.0 and 1.0005 with depth_epsilon=1e-3
visibility report: stale only because ambiguous_depth_samples=4
fallback marker sets cell.fallback=True and fallback_reasons=("visibility_ambiguous_depth",)
pack_projective_trace_tile_time_bins(...) rejects by default
pack_projective_trace_tile_time_bins(..., allow_fallback_cells=True) succeeds
trainer state strict mode raises on unresolved ambiguity
trainer state opt-in mode returns fallback_marked=True
```

Targeted ambiguity tests:

```text
2 passed in 3.69s
```

Combined focused projective plus interval-gated trainer suite:

```text
57 passed in 4.60s
```

Interpretation: ambiguous visibility is no longer conflated with order staleness
that rebin can repair. It now becomes a visible fallback contract. Remaining
work is to implement the actual fallback evaluator, track fallback fraction,
and decide when to split/refit rather than fallback.

## Follow-Up: Live-Depth Reference Fallback And Metrics

Added:

```text
ProjectiveTraceCellAtlasFallbackStats
projective_trace_cell_atlas_fallback_stats(...)
render_projective_trace_cell_atlas_reference(..., allow_fallback_cells=True)
ProjectiveCellIntervalTrainerState.fallback_stats()
ProjectiveCellIntervalTrainerState.render_reference_with_fallback()
```

The fallback path is intentionally reference-first. The interval Metal renderer
still rejects fallback cells, because the current hot kernel treats visibility
order as a compiled constant. The CPU/Torch cell atlas oracle now gives
fallback cells their intended semantics: for marked tile/sample regions, it
re-sorts active traces by live evaluated depth before compositing. Non-fallback
regions still use compiled interval order.

Focused evidence:

```text
strict render rejects fallback cells
fallback stats reports fallback_cells, fallback_tile_samples,
    fallback_trace_samples, fallback_fraction, and fallback_reasons
manual stale-order fallback case:
    static compiled order differs from dense cell rendering by > 1e-2
    live-depth fallback render matches dense cell rendering at 1e-6
trainer state reports fallback fraction and refuses Metal render on fallback
trainer state CPU reference fallback renders a nonzero image
```

Targeted fallback tests:

```text
3 passed in 3.14s
```

Combined focused projective plus interval-gated trainer suite:

```text
58 passed in 8.01s
```

Interpretation: fallback is no longer just a tag. We now have a measurable
fallback fraction and an executable dense-correct reference behavior. This does
not yet make fallback fast; it gives the exact oracle needed before deciding
between more gauge splitting, tile-local live sort, k-buffer/depth-bin
fallback, or a mixed Metal fast/fallback scheduler.

## Follow-Up: Visibility-Stratum Splitting Before Fallback

Added:

```text
stratify_projective_trace_cell_atlas_visibility(...)
ProjectiveCellIntervalAtlasRefresh.visibility_stratified
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_visibility_stratifies_depth_crossing_without_fallback
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_refresh_stratifies_visibility_crossing_without_fallback
```

This is the richer visibility path the theory wanted. After rebinning, if live
depth order still mismatches the compiled order, refresh now tries to split
tile-time cells into consecutive sample runs whose live depth order is stable.
The tensor payload stays fixed; only cell metadata changes. A crossing that
previously needed one stale `[0,4)` order now compiles into:

```text
[0,2): (0, 1)
[2,4): (1, 0)
```

Near-tie ambiguity still falls through to explicit fallback metadata. So the
policy is now:

```text
support/order rebin -> visibility-stratum split -> ambiguity fallback
```

Focused evidence:

```text
manual crossing atlas:
    before: order_mismatch_samples = 2
    after stratify: two stable cells, no fallback cells, visibility not stale
refresh lifecycle:
    visibility_stratified = True
    fallback_marked = False
    live coeff tensor preserved
```

Targeted tests:

```text
2 passed in 3.05s
```

Combined focused projective plus interval-gated trainer suite:

```text
60 passed in 5.47s
```

Interpretation: important order changes no longer automatically imply fallback
or per-frame sorting. They become event cells in sensor-time, which is the
actual camera-path compiler story. Remaining fallback work is for true
ambiguity, invalid depth, or cases where stratum count explodes.

## Follow-Up: Stratum And Fallback Budget Diagnostics

Added:

```text
ProjectiveTraceCellAtlasComplexityStats
ProjectiveTraceCellAtlasBudgetReport
projective_trace_cell_atlas_complexity_stats(...)
projective_trace_cell_atlas_budget_report(...)
ProjectiveCellIntervalTrainerState.complexity_stats()
ProjectiveCellIntervalTrainerState.budget_report(...)
```

The atlas now reports the quantities that decide whether the compiler is still
doing useful amortization:

```text
interval_trace_entries
dense_trace_samples
interval_to_dense_trace_sample_ratio
tile_active_set_groups
visibility_stratum_split_cells
max_cells_per_active_set_group
fallback_fraction
```

Focused evidence on the crossing fixture:

```text
tile_active_set_groups = 1
visibility_stratum_split_cells = 1
max_cells_per_active_set_group = 2
interval_trace_entries = 4
dense_trace_samples = 8
interval_to_dense_trace_sample_ratio = 0.5
fallback_fraction = 0.0
```

The budget helper returns named failures. A tight budget on the same fixture
fails with:

```text
("interval_to_dense_trace_sample_ratio", "max_cells_per_active_set_group")
```

Combined focused projective plus interval-gated trainer suite after this
diagnostic gate:

```text
60 passed in 10.76s
```

Interpretation: the lifecycle can now distinguish "good split" from "exploding
visibility complexity." This is the control signal needed before production
trainer integration: split/refit while interval ratio, stratum count, and
fallback fraction are within budget; escalate to fallback or rebuild when they
are not.

## Follow-Up: Budget Enforcement In Refresh Lifecycle

Added:

```text
ProjectiveCellIntervalAtlasRefresh.budget_after
ProjectiveCellIntervalTrainerState.enforce_complexity_budget
ProjectiveCellIntervalTrainerState.max_interval_to_dense_trace_sample_ratio
ProjectiveCellIntervalTrainerState.max_fallback_fraction
ProjectiveCellIntervalTrainerState.max_cells_per_active_set_group
refresh_projective_cell_interval_atlas_if_stale(..., enforce_complexity_budget=True, ...)
```

Every refresh now returns the post-refresh atlas budget report. If
`enforce_complexity_budget=True`, refresh raises a `RuntimeError` with named
budget failures instead of silently accepting an atlas that has collapsed toward
per-frame work or high fallback coverage.

Focused evidence:

```text
crossing fixture after stratum split:
    budget_after.within_budget = True
    interval_to_dense_trace_sample_ratio = 0.5
    visibility_stratum_split_cells = 1

same fixture with max_interval_to_dense_trace_sample_ratio = 0.40:
    refresh raises "projective cell atlas exceeded complexity budget"
    failure includes "interval_to_dense_trace_sample_ratio"
```

Targeted policy test:

```text
1 passed in 1.60s
```

Combined focused projective plus interval-gated trainer suite:

```text
60 passed in 5.94s
```

Interpretation: atlas complexity is now a lifecycle policy, not only a report.
The production trainer can keep working with cached gauge domains only while
the compiled atlas remains within an explicit amortization budget.

## Follow-Up: Continuous Visibility Event Roots

Added:

```text
ProjectiveTraceCellVisibilityEvent
ProjectiveTraceCellVisibilityEventReport
projective_trace_cell_visibility_event_report(...)
```

The prior visibility-stratum split was sample-run based: it could turn a
detected crossing into stable time intervals over the available sample grid.
This pass adds the continuous event diagnostic for cell-local depth models.
For each active pair inside an atlas cell, the report solves the polynomial
root of

```text
z_i(t) - z_j(t) = 0
```

on the closed active time interval. Linear depth crossings produce the exact
event time, and quadratic cell-local depth models can expose two crossing
events. The new tests cover:

```text
linear crossing root: 5 / 3
quadratic crossing roots: -1, 1
stable depth pair: no events
```

Focused gate:

```text
tests/test_star_uvt_projective_visibility.py
9 passed in 2.23s
```

Combined focused projective plus interval-gated trainer suite:

```text
63 passed in 5.86s
```

Interpretation: visibility is now represented at two levels. Sampled strata
are the current repair mechanism used by refresh; continuous event roots are
the chart certificate needed to choose exact split/refit boundaries for finite
exposure, rolling shutter, and revolving-camera paths. This keeps the math in
the gauge/projection layer: fallback remains a guardrail for unresolved or
budget-breaking regions, not the primary way the orbit is represented.

## Follow-Up: Event-Root Stratifier Enters Refresh

Added:

```text
stratify_projective_trace_cell_atlas_visibility_events(...)
refresh_projective_cell_interval_atlas_if_stale(...) prefers event-root split before sampled split
tests/test_star_uvt_projective_visibility.py::test_projective_cell_visibility_event_stratifier_isolates_exact_root_sample
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_refresh_uses_event_roots_to_localize_fallback
```

The continuous root report now has an executable repair path. The new
stratifier cuts each cell at active-span boundaries and at pairwise continuous
roots of

```text
z_i(t) - z_j(t) = 0
```

For roots that fall exactly on a frame sample, it isolates that sample as a
singleton interval. This matters because fallback should cover only the actual
tie/event sample, not the entire neighboring sample run. In the focused
fixture, depths `z_0=t` and `z_1=1` over frames `0,1,2,3` now split into:

```text
[0,1): order (0,1), no fallback
[1,2): order (0,1), fallback only for the exact tie
[2,4): order (1,0), no fallback
```

Focused evidence:

```text
tests/test_star_uvt_projective_visibility.py tests/test_star_uvt_trainer_interval_gated.py
20 passed in 6.51s
```

Combined focused projective plus interval-gated trainer suite:

```text
65 passed in 8.98s
```

Interpretation: this is the first actual continuous-event-cell compiler step.
The lifecycle is now:

```text
support/order rebin -> continuous event-root split -> sampled stratum split -> ambiguity fallback
```

That is closer to the fiber-bundle/gauge story: the projection/depth geometry
chooses chart boundaries first, and fallback only handles unresolved or
budget-breaking local regions.

## Follow-Up: Support Tile-Boundary Roots Enter Refresh

Added:

```text
ProjectiveTraceCellSupportEvent
ProjectiveTraceCellSupportEventReport
projective_trace_cell_support_event_report(...)
rebin_projective_trace_cell_atlas_support_events(...)
refresh_projective_cell_interval_atlas_if_stale(...) uses support-event rebin
tests/test_star_uvt_projective_binning.py::test_projective_cell_support_event_report_finds_tile_boundary_times
tests/test_star_uvt_projective_binning.py::test_projective_cell_support_event_rebin_splits_tile_runs
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_cell_refresh_uses_support_events_for_tile_runs
```

Before this pass, support refresh rebuilt one broad tile rectangle over the
whole sampled active interval. For a trace moving across the screen, that can
cover tile/time pairs where the trace is not actually present. That is exactly
the kind of hidden projection/binning overcoverage the camera-path compiler is
supposed to remove.

The new support event report solves roots of

```text
u(t) +/- padding = tile_boundary
v(t) +/- padding = tile_boundary
```

over each trace active interval. The support-event rebin maps those roots to
sample interval boundaries and computes conservative continuous polynomial
ranges inside each resulting interval. In the focused fixture, `u(t)=8+8t`
over frames `0,1,2,3` crosses tile boundaries at `t=1` and `t=3`.

The old sampled support rebin produced broad tile cells:

```text
tile 0: [0,4)
tile 1: [0,4)
tile 2: [0,4)
```

The event-root support rebin produces time-local tile cells:

```text
tile 0: [0,1)
tile 1: [1,3)
tile 2: [3,4)
```

Focused evidence:

```text
tests/test_star_uvt_projective_binning.py tests/test_star_uvt_trainer_interval_gated.py
18 passed in 5.36s
```

Combined focused projective plus interval-gated trainer suite:

```text
68 passed in 4.31s
```

Interpretation: the lifecycle is now:

```text
support/order rebin with continuous support roots
    -> continuous visibility event-root split
    -> sampled stratum split
    -> ambiguity fallback
```

This is a real movement toward sublinear frame scaling: support/binning is now
split by camera-path event roots instead of being stretched across all frames
in an active interval.

## Follow-Up: Continuous Sensor-Time Event Partition

Added:

```text
ProjectiveTraceCellSensorTimeInterval
ProjectiveTraceCellSensorTimePartition
projective_trace_cell_sensor_time_event_partition(...)
tests/test_star_uvt_projective_binning.py::test_projective_cell_sensor_time_partition_merges_support_visibility_and_exposure_events
```

This pass introduces the first continuous partition object that is not just a
frame-index cell. The function merges:

```text
support roots:     u/v support crosses tile/image boundaries
visibility roots:  z_i(t) - z_j(t) = 0
extra splits:      exposure/shutter/quadrature boundaries supplied by caller
```

and returns sorted sensor-time intervals. In the focused fixture it merges
support roots at `t=1` and `t=3`, a visibility root at `t=1.6`, and exposure
split times at `t=0.5` and `t=2.5`, producing:

```text
[0.0, 0.5)
[0.5, 1.0)
[1.0, 1.6)
[1.6, 2.5)
[2.5, 3.0)
```

Focused evidence:

```text
tests/test_star_uvt_projective_binning.py
8 passed in 3.57s
```

Combined focused projective plus interval-gated trainer suite:

```text
69 passed in 8.53s
```

Interpretation: this is the bridge from sampled cell maintenance to finite
exposure and rolling shutter. We now have a continuous event partition over
sensor time that can be lowered into quadrature intervals, rolling-shutter row
chunks, or mixed fast/fallback Metal scheduling later.

## Follow-Up: Exposure And Rolling Quadrature Lowering

Added:

```text
ProjectiveTraceCellSensorTimeQuadratureSample
ProjectiveTraceCellSensorTimeQuadrature
projective_trace_cell_sensor_time_partition_quadrature(...)
projective_trace_cell_sensor_time_partition_rolling_quadrature(...)
tests/test_star_uvt_projective_binning.py::test_projective_cell_sensor_time_partition_quadrature_clips_exposure_to_event_cells
tests/test_star_uvt_projective_binning.py::test_projective_cell_sensor_time_rolling_quadrature_offsets_rows
```

The continuous partition now lowers into executable sample schedules. The
finite-exposure helper clips an exposure window to event intervals, emits
midpoint samples, and normalizes weights by exposure duration. In the focused
fixture, exposure `[0.25,2.75]` over event cells `[0,0.5), [0.5,1),
[1,2.5), [2.5,3)` produces weighted samples:

```text
[0.25,0.5]   midpoint 0.375
[0.5,1.0]    midpoint 0.75
[1.0,2.5]    midpoint 1.75
[2.5,2.75]   midpoint 2.625
total_weight = 1.0
```

The rolling-shutter helper applies the same lowering per row with
row-dependent readout offsets. In the three-row fixture with
`exposure_duration=1` and `readout_duration=1`, the row windows become:

```text
row 0: [0.0,1.0]
row 1: [0.5,1.5]
row 2: [1.0,2.0]
```

Focused evidence:

```text
tests/test_star_uvt_projective_binning.py
10 passed in 2.01s
```

Combined focused projective plus interval-gated trainer suite:

```text
71 passed in 8.57s
```

Interpretation: finite exposure and rolling shutter are now concrete schedules
over the same event cells as support and visibility. The next implementation
step is to feed these schedules into the reference renderer, then decide how a
mixed Metal scheduler handles fast cells versus fallback cells.

## Follow-Up: Quadrature Reference Rendering

Added:

```text
render_projective_trace_cell_atlas_quadrature_reference(...)
render_projective_trace_cell_atlas_rolling_quadrature_reference(...)
tests/test_star_uvt_projective_correctness.py::test_projective_cell_quadrature_reference_matches_explicit_weighted_samples
tests/test_star_uvt_projective_correctness.py::test_projective_cell_quadrature_reference_backprops_to_trace_params
tests/test_star_uvt_projective_correctness.py::test_projective_cell_rolling_quadrature_reference_uses_per_row_schedules
```

This pass keeps the continuous-time exposure path separate from the existing
integer-frame tile renderer. The new oracle treats a cell atlas as a direct
sensor-time trace table, evaluates live depth at each quadrature sample, sorts
all active traces by that live depth, composites a screen-space Gaussian
footprint, and accumulates sample weights. The rolling helper applies the same
logic row by row, so each sensor row can use its own readout-shifted
quadrature schedule.

Important limitation: this is a correctness oracle for continuous schedules,
not the final interval/tile Metal schedule. It deliberately ignores
frame-indexed tile cell ranges, because fractional exposure samples live in
raw sensor time. Production lowering still needs a mixed scheduler that maps
event intervals and fallback cells onto the Metal fast path.

Focused evidence:

```text
tests/test_star_uvt_projective_correctness.py
27 passed in 3.35s
```

Combined focused projective plus interval-gated trainer suite:

```text
74 passed in 9.89s
```

Interpretation: finite exposure and rolling shutter now have a differentiable
CPU/Torch rendering oracle. This closes the gap between "we can build
quadrature schedules" and "we can render those schedules," while preserving the
larger goal: compile support/order over sensor time, then evaluate slices or
integrals without redoing world-side projection per frame.

## Follow-Up: Quadrature To Interval-Metal Lowering

Added:

```text
ProjectiveTraceCellQuadratureLowering
lower_projective_trace_cell_atlas_quadrature(...)
render_projective_trace_cell_atlas_quadrature_interval_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_metal(...)
tests/test_star_uvt_projective_correctness.py::test_projective_cell_quadrature_lowering_builds_sample_indexed_interval_atlas
tests/test_star_uvt_projective_correctness.py::test_projective_cell_quadrature_lowering_respects_domain_time_activity
tests/test_star_uvt_projective_correctness.py::test_projective_cell_quadrature_interval_metal_matches_reference_if_available
tests/test_star_uvt_projective_correctness.py::test_projective_cell_rolling_quadrature_interval_metal_matches_reference_if_available
```

The new lowering turns a continuous quadrature schedule into the integer
sample-index contract consumed by the interval Metal kernel:

```text
quadrature samples -> sample_times + sample_weights + sampled cell atlas
```

The sampled atlas keeps the original direct trace coefficients as raw
sensor-time functions, but its `active_start/active_stop` values and tile cells
are over quadrature sample indices. It then uses support-event rebinning plus
visibility event/sample stratification to build interval cells for those
samples. When `domain_times` is provided, split chart rows respect their
original active sample-domain validity; when it is omitted, rows are treated as
valid over the quadrature schedule.

Finite exposure can now render by:

```text
lower schedule -> interval Metal render [Q,H,W,3] -> weighted sum over Q
```

Rolling shutter currently calls the finite-exposure interval-Metal bridge per
row and takes the corresponding row. That is reference/bridge semantics, not
the final performance shape; the future production scheduler should batch or
row-mask the work rather than rendering a full frame per row.

Focused evidence:

```text
tests/test_star_uvt_projective_correctness.py
31 passed in 4.84s
```

Combined focused projective plus interval-gated trainer suite:

```text
79 passed in 9.85s
```

Interpretation: the finite-exposure / rolling-shutter path now reaches the
existing interval Metal renderer without pretending fractional sensor times are
ordinary frame indices. The remaining open piece is a true mixed fast/fallback
scheduler: fallback cells still need CPU/local live sort or a dedicated Metal
fallback path, and rolling rows need a less wasteful batching strategy.

## Follow-Up: Batched Rolling Quadrature Scheduler

Added:

```text
ProjectiveTraceCellRollingQuadratureLowering
lower_projective_trace_cell_atlas_rolling_quadrature(...)
render_projective_trace_cell_atlas_rolling_quadrature_batched_reference(...)
tests/test_star_uvt_projective_correctness.py::test_projective_cell_rolling_quadrature_batched_lowering_reuses_sample_times
tests/test_star_uvt_projective_correctness.py::test_projective_cell_rolling_quadrature_batched_reference_matches_rowwise_reference
```

The rolling path no longer needs one interval-renderer call per row. The new
lowering merges every row's quadrature schedule into a single sorted unique
sample-time axis:

```text
row schedules -> unique sample_times + row_weights[Q,H] + sampled cell atlas
```

The interval renderer evaluates all unique sample times once, and the output
image is:

```text
I[v,u,c] = sum_q row_weights[q,v] * rendered[q,v,u,c]
```

This still renders full frames for every unique rolling sample time, so it is
not the final row-masked Metal kernel. But it removes repeated atlas lowering,
packing, and kernel dispatch per row, and it shares duplicate sample times
across rows. The focused fixture proves the unique sample count is smaller than
the total row-sample count and that batched reference rendering matches the
older row-wise CPU reference.

Focused evidence:

```text
tests/test_star_uvt_projective_correctness.py
33 passed in 4.65s
```

Combined focused projective plus interval-gated trainer suite:

```text
81 passed in 7.47s
```

Interpretation: rolling shutter now has a scheduler object that shares sample
times across rows and calls the interval Metal path once per rolling image.
The next performance gate is a row-masked/row-tiled Metal kernel or sparse row
gather that avoids evaluating all rows for every rolling sample time.

## Follow-Up: Row-Weighted Interval Metal Kernel

Added:

```text
torch.ops.star_uvt_v0.render_projective_trace_cell_interval_rows(...)
render_projective_trace_cell_interval_atlas_rows_metal(...)
has_projective_trace_cell_interval_rows_metal()
```

The new Metal kernel dispatches one thread per output pixel, loops over the
unique rolling sample-time axis, skips `row_weights[q,row] == 0`, composites
the active interval atlas only for nonzero sample/row pairs, and writes the
final rolling image directly:

```text
I[v,u,c] = sum_q row_weights[q,v] * R(q,v,u,c)
```

This keeps the useful batched schedule object from the previous pass, but no
longer materializes or weighted-sums a full `[Q,H,W,3]` image in Python for
the rolling Metal path. It still performs a simple per-pixel loop over all
unique sample times, so sparse row tiles or a row-compacted launch can improve
it later, but it is now the right shape for shared rolling-shutter rendering:
one atlas pack, one kernel dispatch, no per-row interval dispatch.

The extension was rebuilt:

```text
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Focused evidence after rebuild:

```text
tests/test_star_uvt_projective_correctness.py
33 passed in 8.84s
```

Combined focused projective plus interval-gated trainer suite:

```text
81 passed in 11.86s
```

Interpretation: rolling shutter now reaches a dedicated row-weighted Metal
kernel. The remaining performance/fidelity work is mixed fast/fallback
handling and possibly a row-compacted launch so the kernel does not scan every
unique sample time for every output row.

## Follow-Up: Mixed Fast/Fallback Scheduler

Added a correctness-preserving mixed path for finite exposure and rolling
shutter:

```text
split_projective_trace_cell_atlas_fallback_cells(...)
projective_trace_cell_atlas_fallback_tile_sample_mask(...)
render_projective_trace_cell_atlas_quadrature_interval_mixed_metal(...)
render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal(...)
```

The key correction is compositional: fallback cells are not rendered as an
extra additive contribution. A fallback flag marks a whole tile/sample region
where compiled order is unsafe. The mixed renderer therefore:

```text
1. lowers the quadrature/rolling schedule into sample-indexed interval cells;
2. marks near-tie visibility cells as fallback;
3. renders non-fallback cells through the interval Metal path;
4. renders the full active list for fallback tile/sample regions with the
   live-depth CPU/Torch reference;
5. patches those whole regions before exposure or row-weight accumulation.
```

For rolling shutter, the no-fallback case still uses the row-weighted Metal
kernel directly. When fallback exists, the first safe implementation renders
sample images, patches fallback regions, then applies the row weights. This is
not the final fallback performance shape, but it preserves visibility semantics
and keeps the fast row-weighted path intact for ordinary no-fallback regions.

Focused evidence:

```text
tests/test_star_uvt_projective_correctness.py
36 passed in 12.29s
```

Combined focused projective plus interval-gated trainer suite:

```text
84 passed in 10.49s
```

Interpretation: the atlas now has a concrete mixed forward policy. The open
work moved from "how should fallback compose?" to "make fallback differentiable
and cheaper," plus production trainer routing for the mixed renderer.

## Follow-Up: Trainer-State Mixed Fallback VJP

Added a production-facing fallback render policy:

```text
feature_uvt.projective_interval.fallback_render_mode = "error" | "mixed" | "reference"
ProjectiveCellIntervalTrainerState.fallback_render_mode
ProjectiveCellIntervalTrainerState.render_mixed_fallback()
```

The default remains strict `error`, preserving the old invariant that fallback
cells cannot silently go through the native fast path. The new `mixed` mode
splits the atlas into non-fallback and fallback cells, renders non-fallback
regions with `render_projective_cell_interval_atlas_metal_backward(...)`, renders
the fallback tile/sample regions with the live-depth Torch reference, and patches
whole tile/sample regions. Unlike the previous quadrature mixed-Metal helper,
the trainer-state fallback reference is not detached, so gradients flow through
fallback opacity/color/footprint parameters while fast regions keep the native
interval Metal VJP.

Focused evidence:

```text
tests/test_star_uvt_render_configs.py tests/test_star_uvt_trainer_interval_gated.py
18 passed in 20.32s
```

Combined focused projective plus interval-gated trainer suite:

```text
90 passed in 20.37s
```

Interpretation: fallback now has a usable clean-derivative story in the trainer
state. The remaining work is production trainer routing, cheaper/native fallback
VJP for performance, and row-compacted rolling launch optimization.

## Follow-Up: Production Route Tripwire

Added:

```text
require_projective_interval_atlas_producer(...)
```

and called it from the first-class STAR UVT feature overfit trainer. The feature
trainer currently emits affine UVT feature tubes, not a
`ProjectiveTraceCellTraceAtlas`, and the projective interval Metal cell renderer
is RGB/cell-atlas shaped. So `feature_uvt.projective_interval.enabled=true`
must not silently run the existing feature-tube renderer and pretend it is the
camera-bundle atlas. The trainer now fails loudly unless a caller marks a real
atlas producer as available. Disabled configs still normalize the projective
interval backend and write the policy metadata (`enabled`, `fallback_render_mode`,
`tile_size`) into result rows.

Focused evidence:

```text
tests/test_star_uvt_render_configs.py tests/test_star_uvt_config_keys.py tests/test_star_uvt_trainer_interval_gated.py
25 passed in 10.59s
```

Combined focused projective plus interval-gated trainer suite:

```text
97 passed in 24.50s
```

Interpretation: production routing is now honest. The remaining bridge is the
actual `ProjectiveTraceCellTraceAtlas` producer for real STAR/WorldFoam models,
not another config flag.

## Follow-Up: Compatible STAR UVT Atlas Producer

Added the first real producer:

```text
uvt_tubes_to_projective_trace_cell_atlas(...)
make_projective_cell_interval_atlas_from_uvt_tubes(...)
```

The lowering is mathematical, not a wrapper. It completes each STAR UVT
quadratic in the spatial variables:

```text
q(u,v,t) = (x + A^{-1}b dt)^T A (x + A^{-1}b dt)
         + (q_tt - b^T A^{-1}b) dt^2
```

and extracts the moving screen center:

```text
x_c(t) = ma_uv - A^{-1}b (t - ma_t)
```

For compatible tubes this becomes a direct cell-trace row:

```text
[u0,u1,0, v0,v1,0, z0,z1,0]
```

Then the existing event compilers rebuild support cells, split tile-boundary
events, split depth-order events, and optionally mark ambiguity fallback.

The exact contract is still intentionally narrow because the current interval
cell Metal renderer has only one global isotropic `sigma_px` and one
center-depth model. By default the producer rejects:

```text
anisotropic UV precision
pixel-varying depth_beta[:,0:2]
```

The next follow-up removed the temporal-opacity restriction from the atlas
reference path by adding:

```text
ProjectiveTraceCellTraceAtlas.opacity_time_coeffs
```

with alpha evaluated as:

```text
alpha_i(x,t) =
    opacity_i
    exp(-0.5 * (k0 + k1 t + k2 t^2))
    exp(-0.5 * ||x - x_c(t)||^2 / sigma_px^2)
```

For STAR UVT residual temporal precision:

```text
lambda_tau = q_tt - b^T A^{-1}b
k0 = lambda_tau ma_t^2
k1 = -2 lambda_tau ma_t
k2 = lambda_tau
```

The CPU/Torch atlas reference now matches dense UVT with nonzero temporal
precision. The current interval Metal renderer rejects nonzero
`opacity_time_coeffs` rather than silently dropping the envelope. The producer
test also backprops through the reference atlas image into `q_uvt[:,5]`, so the
temporal opacity path is not forward-only.

Focused evidence:

```text
tests/test_star_uvt_projective_uvt_producer.py
8 passed in 3.45s
```

Combined focused projective plus interval-gated trainer suite:

```text
105 passed in 16.68s
```

Interpretation: the atlas producer is no longer purely hypothetical for STAR
UVT. The remaining work is to route this compatible-tube producer through the
real training loop and then extend the trace/native representation for
anisotropic footprints, pixel-varying depth, and
WorldFoam cell-camera intersections.

## Follow-Up: Native Interval Metal Temporal Opacity

The interval Metal path now consumes the same temporal opacity envelope as the
reference atlas instead of rejecting it. The op schema carries
`opacity_time_coeffs[N,3]` into:

```text
render_projective_trace_cell_interval_tiles
render_projective_trace_cell_interval_rows
direct_projective_trace_cell_interval_backward
```

The kernel evaluates:

```text
alpha = opacity
      * exp(-0.5 * (k0 + k1 t + k2 t^2))
      * exp(-0.5 * ||pixel - center(t)||^2 / sigma_px^2)
```

and the direct VJP returns `grad_opacity_time_coeffs` with:

```text
d alpha / d k_j = -0.5 * alpha * [1, t, t^2]_j
```

The trainer-harness interval autograd bridge now threads this tensor as a real
differentiable input when present, and uses a zero `[N,3]` payload for older
atlases. Focused evidence after rebuilding `star_uvt_v0`:

```text
tests/test_star_uvt_projective_uvt_producer.py
8 passed in 1.95s

tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_renders_in_metal_if_available
tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_backward_matches_torch_autograd_if_available
2 passed in 5.22s

projective focused suite
105 passed in 12.66s
```

Backtrack: "native temporal opacity" is no longer an interval-Metal blocker for
the compatible STAR UVT producer. The remaining producer gaps are anisotropic
screen precision, pixel-varying depth, real trainer-loop ownership, and
WorldFoam/instance cell traces.
