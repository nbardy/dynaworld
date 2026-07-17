# Shared-Work Bandwidth Proxy

## Context

The user asked to keep the active "goal / meta goals / key math / theory" in
memory. The immediate risk was that the aggregate shared-work report only
implied memory-bandwidth sharing through payload bytes and interval entries.
That was too easy for future agents to miss.

## Current Model

Goal:

```text
fast 2D rasters across time from 4D spacetime primitives
```

Meta-goal:

```text
share projection/support/binning/visibility/backward work over time
```

Key math:

```text
UVT trace = pi_* Gamma^* world_primitive
```

Theory:

```text
STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The practical bandwidth proxy is not pixel writes. Pixel writes are still
O(FHW). The proxy is compiled-world traffic: atlas payload bytes, trace counts,
segments, interval entries, and the forward/backward work they drive. If those
grow much more slowly than per-frame replay, the report supports the active
goal.

## Changes

`projective_shared_work_goal_audit.py` now promotes these fields into the
top-level summary:

- `orbit_payload_growth_ratio`
- `orbit_final_trace_ratio`
- `orbit_final_segment_ratio`
- `orbit_final_forward_ms_ratio`
- `orbit_final_cpu_compile_ms_ratio`
- `max_trained_final_trace_count_ratio`
- `max_trained_final_forward_ms_ratio`
- `trained_shared_to_replay_interval_growth_ratio`

The verifier now rejects regressions in those explicit ratios, and the tests
cover orbit payload-growth ratio and trained shared/replay interval-entry
growth ratio failures.

## Evidence

Regenerated artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
```

Current summary highlights:

- orbit fixed payload growth `1.0x` vs per-frame payload growth `8.0x`
- orbit payload-growth ratio `0.125`
- final orbit payload/trace/segment ratios `0.0625`
- final orbit CPU/forward/backward ratios `0.091 / 0.117 / 0.158`
- trained shared interval-entry growth `<=1.462x`
- trained per-frame replay interval-entry growth `>=9.852x`
- trained shared/replay interval-entry growth ratio `0.148`
- trained final trace-count ratio `0.1`
- trained final forward/backward ratios `<=0.266 / <=0.208`
- exposure/rolling forward and backward artifacts still verify
- differentiable mixed-fallback backward still verifies

Commands:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py \
  tests/test_star_uvt_projective_shared_work_goal_audit.py

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_shared_work_goal_audit.py -q
```

Results:

```text
verified outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
22 passed in 5.02s
30 passed in 5.31s  # shared-work audit plus trained high-motion scaling verifier
```

## Decision Implication

When discussing memory bandwidth for STAR UVT, point to the explicit aggregate
ratio fields rather than hand-waving from timings. The honest claim is:

```text
compiled world-side payload/trace/entry work grows sublinearly over the tested
camera paths; output pixels still scale with the number of requested samples.
```

This keeps the goal alive without overstating it.
