# Gate4 Shared-Load Timing Probe

We resumed after the reflection stop to chase the apparent absolute-time gap in
the Gate4 affine moving-camera WorldFoam train/eval harness.

Changes made:

- Restored the inactive alpha/depth auxiliary-loss path to an exact RGB-only
  graph. When aux weights are zero, the loop now uses `loss = rgb_loss` instead
  of adding a zero aux tensor.
- Added `wall_timing` fields per frame-count row so the artifact separates data
  load/slice, synthetic motion, tape build, device transfer, optimizer init,
  train loop, final eval, and row total wall time.
- Changed the scale sweep to load the max frame count once per process and
  slice smaller frame-count rows from that shared data. This keeps per-row scale
  timing from being polluted by repeated video/multicam loading.
- Added `test_train_eval_fused_slab_mixed_mps.py` to verify that shared-data
  slicing preserves view-major prefix frames for train and heldout rows.

What the new artifact says:

- Artifact: `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16.json`
- Verifier: `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16_verifier.json`
- Verifier status: `ok`, failures `[]`.
- Shared max-frame multicam load: `42.116 s`.
- Per-row load/slice after sharing: `<= 0.003 s`.
- Per-row total wall after shared load at 2/4/8/16 frames:
  `3.887 / 3.223 / 3.303 / 4.103 s`.
- Total median step time at 2/4/8/16 frames:
  `68.496 / 87.971 / 72.016 / 86.986 ms`.
- Backward median step time at 2/4/8/16 frames:
  `32.051 / 40.144 / 32.343 / 39.666 ms`.
- Total median scale 2f->16f: `1.270x`; backward median scale: `1.238x`.
- Mixed train tape storage remains flat: `1.121612 MB -> 1.112348 MB`;
  explicit train ray storage is still `0.098304 MB -> 0.786432 MB`.

Interpretation:

The latest "absolute speed" concern was partly a benchmark-harness problem:
process wall was dominated by video/multicam loading, not by raster/VJP. After
shared loading, the row wall and per-step timings are honest shader/tape-loop
numbers. Those are still tens of milliseconds, so WorldFoam is still not
STAR-UVT-clean, but the specific 40-50 second wall gap was not shader time.

Tests:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

Result: 29 tests OK.
