# UV visibility split report artifact

## Context

The adaptive UV split policy had focused tests for a high-motion proxy line
sweep and an orbit-parameterized line sweep. The next step was to make the
measurement reusable outside individual tests and give future agents a stable
JSON schema for parent-vs-child fallback fractions.

## Change

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_uv_visibility_split_report.py
```

The script writes:

```text
outputs/projective_uv_visibility_split_report.json
```

with schema:

```text
projective_uv_visibility_split_report_v1
```

It currently contains two cases:

1. `high_motion_proxy_line_sweep`
   - source: `synthetic_proxy_from_high_motion_video`
   - reference video: `data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4`
   - parent fallback fraction: `1.0`
   - output fallback fraction: `0.0`

2. `orbit_parameterized_line_sweep`
   - source: `synthetic_orbit_q_tan_half_angle`
   - parent fallback fraction: `1.0`
   - output fallback fraction: `0.0`

The report summary records:

```text
max_parent_fallback_fraction = 1.0
max_output_fallback_fraction = 0.0
max_cell_growth = 4.0
any_needs_oblique_halfspace = false
```

## Caveat

The high-motion row references the checked-in high-motion video as provenance,
but it does not parse that video or extract geometry from it. It is still a
synthetic proxy row. The point of this step is the report schema and reusable
before/after measurement path.

## Verification

Targeted report/adaptive checks:

```text
4 passed in 5.39s
```

Focused STAR UVT projective plus report suite:

```text
163 passed in 25.46s
```

Manual artifact generation:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_uv_visibility_split_report.py \
  --output outputs/projective_uv_visibility_split_report.json
```

## Next

Replace the proxy high-motion row with extracted high-motion trace geometry.
The decision threshold remains the same: if residual fallback or cell growth
stays high on extracted traces, that is evidence for an oblique/fiber
halfspace cell representation.
