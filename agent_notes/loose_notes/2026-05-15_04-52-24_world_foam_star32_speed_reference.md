# World Foam / STAR 32px speed reference

## Context

The fused World Foam status summary said not to claim STAR-UVT competitiveness,
but it did not include a fresh same-resolution STAR timing reference. Existing
fixed-step artifacts were either larger cases or used the older
`world_foam_lane2_v0` path, not the current fused fork.

## Build fix

The first STAR-only fixed-step probe failed because the active Python was 3.11
but the STAR variant only had a Python 3.14 extension built:

```text
AttributeError("'_OpNamespace' 'star_uvt_v0' object has no attribute 'render'")
```

Rebuilt STAR for the active interpreter:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

## Probe

Ran STAR-only direct-atomic fixed-step timing at 32px, 2/4/8/16 frames:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --cases 32x2,32x4,32x8,32x16 \
  --steps 5 \
  --warmup-steps 1 \
  --skip-world-foam \
  --skip-dynamic \
  --input-dir research_experiments/world_foam_lane2/results/fixed_step_speed_compare_inputs_star32_2_4_8_16 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fixed_step_speed_compare_star_directatomic_32px_2_4_8_16.json
```

Result:

```text
status: ok
STAR 2f mean step: 17.49 ms
STAR 4f mean step: 28.34 ms
STAR 8f mean step: 23.46 ms
STAR 16f mean step: 21.49 ms
```

Updated the fused World Foam status summary to include this speed reference:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
```

It compares the current fused World Foam winner's 16-frame total step
(`9.32 ms`) against the STAR 32px 16-frame mean step (`21.49 ms`), giving a
tiny-case step ratio of `0.43x`.

## Takeaway

This is useful speed context, not a full competitiveness claim. The comparison
is not matched on model capacity or training contract: current fused World Foam
is fixed-geometry/site-RGBA with 12 sites, while STAR UVT uses its world-tube
model. Keep `star_uvt_competitive_claim=false` in the canonical summary until a
matched quality/capacity comparison exists.
