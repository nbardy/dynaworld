# Fixed-Step Speed Compare Harness

## What changed

Added `fixed_step_speed_compare.py` for fixed-step timing across:

- STAR-UVT
- free dynamic GSplats
- World Foam Lane 2 CSR

Default cases are `128x8,128x16,128x32,256x32`, with one shared measured step
count and warmup count for all rows. The harness writes per-case manifests and
configs so 32-frame rows really load 32 frames instead of silently clipping to
the existing 16-frame validation manifest.

## Timing contract

- `steps` is the measured optimizer-step count.
- `warmup_steps` runs before measured rows and is excluded from `mean_step_s`.
- STAR-UVT defaults to full-sequence loss for one train camera per step.
- dynamic GSplats defaults to full-sequence loss for one train camera per step.
- World Foam currently renders all train-camera rays per step and is still
  fixed geometry/site-RGBA only, so it is not a full-trainer parity claim.

## Smoke

Command:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --cases 32x2 \
  --steps 1 \
  --warmup-steps 0 \
  --out-json /tmp/world_foam_fixed_step_compare_smoke.json \
  --input-dir /tmp/world_foam_fixed_step_compare_inputs
```

Result: passed. The JSON summary reported `loaded_frames=2` for all three
renderers and `status=ok`.

Real-shape smoke:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --cases 128x8 \
  --steps 1 \
  --warmup-steps 0 \
  --out-json /tmp/world_foam_fixed_step_compare_128x8_smoke.json \
  --input-dir /tmp/world_foam_fixed_step_compare_128x8_inputs
```

Result: passed. The JSON summary reported `loaded_frames=8` for all three
renderers and `status=ok`.
