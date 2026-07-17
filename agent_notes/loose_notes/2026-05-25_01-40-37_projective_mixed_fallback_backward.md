# Projective Mixed Fallback Backward

## Context

The active thread is the STAR UVT / gauged camera-ray bundle goal:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The immediate gap was visibility fallback. The finite-exposure and rolling
reports already proved forward quadrature/row-weight semantics and shared
sample-adjoint backward through the interval-cell Metal VJP, but mixed fallback
forward had a deliberate detached CPU oracle for fallback patches. That is okay
for forward evidence, but not for the clean-derivatives meta-goal.

## Current Model

Fallback is not a replacement for rich projection/gauge math. It is an
evaluator choice on a marked visibility stratum:

```text
same lowered sensor-time atlas
fast cells     -> interval Metal VJP
fallback cells -> live-depth Torch reference VJP
accumulation   -> exposure weights or rolling row_weights
```

The important invariant is that fallback keeps gradients attached to the same
trace tensors. It must not become a detached oracle patch.

## Implementation

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_mixed_fallback_backward_report.py
tests/test_star_uvt_projective_exposure_rolling_mixed_fallback_backward_report.py
```

The report constructs a tiny atlas with one visibility-ambiguous fallback cell
and one fast cell, lowers finite-exposure and rolling schedules, and compares:

```text
mixed = interval Metal VJP fast samples patched with live-depth reference fallback samples
reference = full live-depth Torch reference samples
```

The patch happens before scalar exposure-weight or rolling row-weight
accumulation. The rolling schedule uses `11` unique sample times for `12` row
samples in this focused scene, so row-time reuse is present.

I briefly added public `*_mixed_autograd` bridge helpers, then removed them.
Reason: they called the raw interval Metal forward op, which emits a PyTorch
warning because that op itself has no registered autograd kernel. The correct
differentiable route is the existing trainer-harness wrapper:

```text
render_projective_cell_interval_atlas_metal_backward(...)
```

That wrapper owns the custom autograd function and calls the native direct VJP.

## Evidence

Saved artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.md
```

MPS summary:

```text
finite_has_mixed_backward:  true
rolling_has_mixed_backward: true
finite_fallback_fraction:   0.5
rolling_fallback_fraction:  0.5
rolling_unique_to_row_sample_ratio: 0.9166666666666666
max_mixed_output_abs_error: 5.960464477539063e-08
max_mixed_grad_abs_error:   2.1457672119140625e-06
max_mixed_grad_rel_error:   7.40632344786718e-07
```

Focused gates run:

```text
.venv/bin/python -m py_compile ...
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_exposure_rolling_mixed_fallback_backward_report.py -q
# 7 passed in 15.38s

PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_mixed_fallback_backward_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.json
# verified

PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_trainer_interval_gated.py -k mixed_fallback -q
# 2 passed, 26 deselected in 33.43s
```

## Decision Implication

Do not expose an "autograd" helper that wraps a raw forward-only Metal op. If
the bridge is public and differentiable, it must call the trainer-harness VJP
wrapper or register its own autograd kernel. Detached CPU fallback is acceptable
only for explicitly forward-only helpers.

The memory contract remains intact: rich gauges handle revolving/rolling
camera complexity; fallback is a differentiable local evaluator swap for
visibility ambiguity, not a surrender of the bundle formulation.
