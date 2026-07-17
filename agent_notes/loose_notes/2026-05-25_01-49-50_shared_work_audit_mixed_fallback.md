# Shared-Work Audit Now Requires Mixed Fallback Backward

## Context

The active objective is still:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The standalone mixed fallback backward report proved differentiable finite and
rolling fallback patches, but the aggregate shared-work audit did not yet
require that evidence. That left a gap in the top-level "clean derivatives"
proof.

## Change

Updated:

```text
research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py
tests/test_star_uvt_projective_shared_work_goal_audit.py
```

The aggregate audit now loads and verifies:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.json
```

It adds `exposure_mixed_fallback_backward` to the report schema and summary.
The verifier now requires:

```text
finite_has_mixed_backward == true
rolling_has_mixed_backward == true
mixed_backward_case_count >= 2
rolling_unique_to_row_sample_ratio < 1
0 < finite_fallback_fraction < 1
0 < rolling_fallback_fraction < 1
max_mixed_output_abs_error <= 3e-4
max_mixed_grad_abs_error <= 1e-3
max_mixed_grad_rel_error <= 5e-3
```

It also rejects stale mixed-fallback summaries and underlying verifier failures.

## Evidence

Commands:

```text
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py tests/test_star_uvt_projective_shared_work_goal_audit.py

PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_shared_work_goal_audit.py -q
# 18 passed, 2 skipped in 8.98s
```

The two skipped tests are expected in this checkout because the default orbit
and trained high-motion audit inputs are absent. The exposure/rolling forward,
ordinary backward, and mixed fallback backward artifacts are present, but the
aggregate artifact still cannot be regenerated until the missing orbit/trained
inputs are restored or rerun.

## Implication

The top-level shared-work audit now matches the theory better: the objective is
not just sublinear payload/binning and non-fallback VJP. It explicitly includes
the hard visibility case where fallback could otherwise detach gradients.
