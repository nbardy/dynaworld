# Projective Tail-Alpha Cache-Policy Bracket

Status: superseded. A later same-session pass found the max-per-trace
certificate was too weak for overlapping omitted tails. See
`agent_notes/loose_notes/2026-05-24_20-38-20_projective_tail_alpha_aggregate_certificate.md`
for the corrected aggregate certificate and updated artifacts.

## Context

The Gauged UVT Trace Atlas lane is trying to compile 4D spacetime primitives
through a known camera program into reusable sensor-time traces. The current
practical cache question is whether measured atlas refresh can skip support
rebinning when optimizer motion only loses a tiny Gaussian tail at a tile
boundary, without hiding real support/core loss or visibility repair.

Earlier cap128 smokes showed `support_stale_overshoot_epsilon=0.5` removed
support churn, but that was still a pixel-distance rule. The stronger rule is
the tail-alpha certificate:

```text
tail_alpha <= opacity * exp(-0.5 * (max(uv_padding - overshoot, 0) / sigma_px)^2)
```

`support_stale_tail_alpha_epsilon` allows stale support reuse only when the
omitted tail bound is below budget.

## New Evidence

Two real cache-policy artifacts now bracket the compatible slack-budgeted cap128
threshold.

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0003/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035/summary.md
```

Result:

```text
epsilon = 0.0003:
    max_support_tail_alpha_bound = 0.00032070223950928124
    measured stale_refreshes = 1
    measured support_rebins = 1
    measured atlas rebuilds = 1 vs cadence 4
    final loss = 0.08477679640054703, equal to cadence
    tile overflow = 0

epsilon = 0.00035:
    max_support_tail_alpha_bound = 0.00032070223950928124
    measured stale_refreshes = 0
    measured support_rebins = 0
    measured atlas rebuilds = 1 vs cadence 4
    final loss = 0.08477679640054703, equal to cadence
    tile overflow = 0
```

So this smoke's support-reuse threshold is not vague:

```text
0.0003 < observed omitted-tail alpha 0.0003207022 < 0.00035
```

The cache-policy Markdown formatter was patched to preserve sub-`1e-3`
significant digits. This matters because `0.0003`, `0.00035`, and
`0.000320702` used to be visually easy to collapse in reports.

## Commands

The focused formatter regression passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py::test_projective_interval_cache_policy_report_preserves_tail_alpha_precision -q
```

The focused projective suite passed after the formatter/test additions:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
125 passed in 19.59s
```

## Current Model

The useful cache invariant is now:

```text
support debounce may reuse stale support only under an omitted-alpha bound;
visibility debounce remains separate and must still split/refresh/fallback on
stale depth order.
```

This is exactly the chart/fiber compromise the theory wants: the camera-gauged
trace can be reused across time when the pullback support change is low-measure
in rendered opacity, while the visibility strata still act as hard events when
the conditional-depth order changes.

## Decision Implications

Use `support_stale_tail_alpha_epsilon` rather than a raw pixel overshoot as the
next serious support-debounce knob. On this smoke, `3.5e-4` is the first
measured budget that removes the last rebin; `3e-4` is intentionally too tight
and proves the bound is active.

Do not promote the threshold globally yet. The next falsification step is
broader-scene image/error validation, especially anisotropic traces, larger
support padding ranges, WorldFoam/instance cells, and crossing-occluder
visibility churn.

## Open Questions

- Does the same `~3e-4` alpha budget survive higher-opacity or denser scenes?
- Does anisotropic precision make the isotropic tail proxy over- or
  under-conservative?
- Can the certificate be made per-channel/per-feature for feature targets, or
  should it stay opacity-only?
- How should fallback regions report the split between support-tail reuse and
  visibility/order repair in one metric table?
