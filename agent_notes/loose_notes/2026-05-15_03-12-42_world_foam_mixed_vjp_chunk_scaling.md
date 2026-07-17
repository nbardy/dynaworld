# World Foam mixed VJP chunk scaling

Follow-up to the `world_foam_lane2_fused_slab_v0` mixed `num32_den16` VJP/autograd work.

## What changed

- Fixed the Python wrapper for `fused_slab_affine_num32_den16_vjp_reduce` to expose and validate `reduce_chunk_size`.
- Rebuilt the extension after the earlier dynamic chunk-size Metal/C++ changes.
- Ran the moving-camera render32 chunk sweep for VJP reduce chunk sizes 4, 8, and 16:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
# imported smoke_fused_slab_affine_realray_mps.run_smoke
# frame_counts=(2,4,8,16), render_size=32, site_count=12,
# include_vjp=True, vjp_reduce_chunk_size in (4, 8, 16)
PY
```

Artifacts:

- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_vjp_chunk4_render32_pertrack_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_vjp_chunk8_render32_pertrack_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_vjp_chunk16_render32_pertrack_2_4_8_16.json`

## Result

All three chunks were correct:

- status `ok`
- mixed VJP max error `0.0003178119659423828`
- gradients finite for all frame counts

Timing summary, ms:

| reduce chunk | 2f VJP | 4f VJP | 8f VJP | 16f VJP |
| --- | ---: | ---: | ---: | ---: |
| 4 | 4.387 | 4.589 | 5.717 | 9.520 |
| 8 | 3.935 | 4.376 | 5.428 | 10.085 |
| 16 | 4.376 | 4.774 | 6.209 | 10.441 |

## Interpretation

The mixed forward path is genuinely sublinear in this smoke, but the VJP/training path is only partially sublinear. Increasing the reducer chunk helps a little at small frame counts but does not fix the 16-frame cost. The remaining slope is therefore probably not just final-reduce overhead; it is the per-frame replay/VJP work inside the candidate traversal.

Current practical state:

- forward-only fused mixed path: sublinear across 2/4/8/16 frames
- frozen-geometry site-RGBA training: improves over older World Foam timings, but still grows with frame count
- not yet STAR-UVT-flat in practice

The next math/kernel target is to make the VJP accumulate temporal basis terms in a STAR-like way instead of replaying the full per-frame contribution path for every frame.
