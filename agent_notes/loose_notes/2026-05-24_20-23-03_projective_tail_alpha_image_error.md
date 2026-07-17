# Projective Tail-Alpha Image Error Verifier

## Context

Goal memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The cache-policy artifact showed that `support_stale_tail_alpha_epsilon=0.001`
can remove measured support rebins on the compatible 8f/64px route while
keeping final loss equal to cadence. The next question was whether the
certificate is actually tied to image error.

## Change

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py
```

The verifier compares strict support rebinning, tail-alpha-certified stale
support reuse, and one bad pixel-only overshoot tolerance in a missing-core
negative case.

It writes:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.json
```

## Result

Run:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py \
  --tail-alpha-epsilon 0.001 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error
```

Positive reuse cases:

```text
axis_r4_sigma1_opacity05:    tail 0.0002046117, max error 0.0000221119
axis_r5_sigma125_opacity08:  tail 0.0003146833, max error 0.0000550129
axis_r6_sigma15_opacity09:   tail 0.0003447873, max error 0.0000822361
orbit_rational_tail_reuse:   tail 0.0002094069, max error 0.0000227757
```

Negative control:

```text
core_loss_rejected:
    tail bound = 0.5
    tail-certified refresh rebins
    pixel-only 0.10px overshoot pardon would reuse
    forced bad max RGB error = 0.3987594
```

## Verification

Compile:

```text
.venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py
```

Targeted projective interval tests:

```text
3 passed in 4.19s
```

## Interpretation

This is still a local isotropic/projective suite, not a universal scene
guarantee. It does, however, make the acceptance boundary much sharper:
certified missing tails are allowed and remain below their alpha bound, while
missing core support is rejected despite having the same tiny pixel overshoot.

That is the right direction for the bigger renderer: use the camera gauge and
trace math to certify reusable support, and reserve fallback/refinement for
actual core loss or visibility events.
