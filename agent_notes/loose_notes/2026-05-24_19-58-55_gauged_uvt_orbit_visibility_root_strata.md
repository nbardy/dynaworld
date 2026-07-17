# Gauged UVT Orbit Visibility Root Strata

## Context

The prior orbit regression proved support debounce does not suppress stale
visibility repair. The next stronger question was whether a revolving-camera
trace with a true order change inside the orbit window can be compiled into
stable visibility strata instead of marking the entire window as fallback.

## Regression Added

Added:

```text
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_orbit_visibility_crossing_splits_into_stable_strata
```

The fixture lowers two small yaw-window projective traces through the rational
trace path. Their conditional depth polynomials are:

```text
z_0(t) = 2 + 4t
z_1(t) = 2 - 4t
```

with sample times from a stereographic orbit coordinate:

```text
t = tan(theta / 2), theta in {-3deg, -1deg, 1deg, 3deg}
```

So the depth-order root is exactly `t=0`, between samples 1 and 2.

## Evidence

Observed contract:

```text
event_report.split_times = (0.0,)
refresh.before.stale = False
refresh.visibility_before.order_mismatch_samples = 2
refresh.visibility_stratified = True
refresh.fallback_marked = False
refresh.visibility_after.order_mismatch_samples = 0
refresh.budget_after.stats.visibility_stratum_split_cells = 1
refresh.budget_after.stats.interval_to_dense_trace_sample_ratio = 0.5
refresh cells:
    (0, 2, order=(0, 1))
    (2, 4, order=(1, 0))
```

Focused projective plus interval-gated trainer suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q

123 passed in 15.35s
```

## Interpretation

This is the cleanest current answer to the "charts versus richer math" concern:
the gauge handles smooth revolving-camera projection, but visibility is the
arrangement of traces over sensor time. A depth-order root is not a failure of
the gauge; it is an event boundary in the sensor-time atlas.

The method should be described as:

```text
projective/gauged trace domains + explicit support/visibility event certificates
```

not as:

```text
one global chart with generic fallback
```

## Next Test

Combine three pressures in one broader orbit fixture:

1. bounded support drift,
2. visibility depth-root stratification,
3. crowded-tile support guard allocation via `slack_budgeted`.

That would move this from a clean two-trace certificate toward a realistic
orbit-stress compiler gate.
