# Dynamic Splat Direct-Atomic Scaling Check

Date: 2026-05-15

Question:

- Does direct-atomic execution alone make dynamic splats scale well with frame count?
- Or is the flat-ish scaling coming from direct atomic plus the STAR/UVT tube representation?

Commands run from the dynaworld root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --device mps \
  --cases 128x2,128x4,128x8,128x16,128x32 \
  --steps 8 \
  --warmup-steps 2 \
  --skip-star \
  --skip-world-foam \
  --splat-renderer fast_mac \
  --splat-loss-scope view_sequence \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_dynamic_splats_fastmac_direct_atomic_frame_scaling_128px_2_4_8_16_32.json \
  --input-dir research_experiments/world_foam_lane2/results/2026-05-15_dynamic_splats_fastmac_direct_atomic_frame_scaling_inputs
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --device mps \
  --cases 128x2,128x4,128x8,128x16,128x32 \
  --steps 8 \
  --warmup-steps 2 \
  --skip-dynamic \
  --skip-world-foam \
  --uvt-render-backend metal_tile \
  --uvt-sample-emission-mode direct_atomic \
  --uvt-loss-scope view_sequence \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_star_uvt_direct_atomic_frame_scaling_128px_2_4_8_16_32.json \
  --input-dir research_experiments/world_foam_lane2/results/2026-05-15_star_uvt_direct_atomic_frame_scaling_inputs
```

Mean step/render seconds:

| frames | dynamic fast_mac step | dynamic fast_mac render | STAR direct_atomic step | STAR direct_atomic render |
|---:|---:|---:|---:|---:|
| 2 | 0.062157 | 0.025863 | 0.020994 | 0.001504 |
| 4 | 0.081947 | 0.034153 | 0.035967 | 0.003196 |
| 8 | 0.162723 | 0.072019 | 0.032473 | 0.002871 |
| 16 | 0.336528 | 0.156292 | 0.023577 | 0.002017 |
| 32 | 0.658497 | 0.309206 | 0.043482 | 0.003544 |

Read:

- Dynamic fast_mac already uses a direct per-Gaussian atomic-style backward, but the `view_sequence` harness renders one dynamic splat frame per video frame, so step and render time still grow roughly with frame count.
- STAR direct-atomic keeps the same tube bank and one UVT sequence render, so the small fixed-step probe stays nearly flat from 2 to 32 frames.
- This supports the current interpretation: direct atomic fixes the backward memory/workspace failure mode; the STAR/UVT representation supplies the frame-count amortization.
