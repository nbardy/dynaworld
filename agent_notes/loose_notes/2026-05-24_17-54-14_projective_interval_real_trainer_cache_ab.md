# Projective Interval Real Trainer Cache A/B

## Context

The measured projective interval cache had two prior gates:

- policy-level rebuild decision tests
- helper/optimizer-style MPS tests proving support rebin during measured reuse

The missing bridge was the real STAR UVT feature trainer route. We needed to
show that `refresh_policy="measured"` changes cache behavior in
`run_training(...)` while preserving training outputs versus the old cadence
policy.

## Change

Added a real-trainer A/B test:

```text
tests/test_star_uvt_projective_uvt_producer.py::
    test_feature_overfit_trainer_measured_policy_reuses_atlas_vs_cadence_if_available
```

The test runs the same synthetic sequence, seed, target, four steps, and
`refresh_every=2` twice:

```text
cadence   -> fixed rebuild cadence
measured  -> one build, then measured refresh checks
```

It asserts:

```text
cadence:  rebuilds=2, live_updates=2, staleness_checks=2
measured: rebuilds=1, live_updates=3, staleness_checks=3
```

and verifies measured `losses` and `end_loss` match cadence within `1e-5`.

## Evidence

Focused node:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py::test_feature_overfit_trainer_measured_policy_reuses_atlas_vs_cadence_if_available -q
  # 1 passed in 6.63s
```

Producer/config:

```text
tests/test_star_uvt_projective_uvt_producer.py -q
# 16 passed in 15.13s

tests/test_star_uvt_render_configs.py -q
# 6 passed in 2.71s
```

Focused projective plus interval-gated bundle:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_config_keys.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
  # 113 passed in 10.45s
```

## Interpretation

This moves measured-cache evidence from "helper loop" to the real trainer
surface. It still does not prove the end goal. The next update below adds the
saved timing/quality artifact; after that, the remaining cache problem is
reducing metadata churn and extending the trace payload.

## Update: Saved 8-Step Artifact

Added and ran:

```text
research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py
```

Artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md
```

Result on the compatible 8f/64px full-frame projective interval route:

```text
cadence:  rebuilds=4, live_updates=4, staleness_checks=4
measured: rebuilds=1, live_updates=7, staleness_checks=7
end_loss_delta_measured_minus_cadence = 0.0
no_first_step_ms_delta_measured_minus_cadence = -1336.0325
```

The win is real but diagnostic: measured avoids full compatible-atlas rebuilds
and keeps the same final loss, but both policies still report support rebin and
stale refresh on every live update. The next cache problem is reducing that
metadata churn with better support intervals, padding, event margins, or a
less-invalidating live atlas update strategy.
