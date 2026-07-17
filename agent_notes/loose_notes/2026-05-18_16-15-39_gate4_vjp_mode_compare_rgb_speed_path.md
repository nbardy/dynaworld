# Gate4 VJP Mode Compare And RGB Speed Path

We continued from the shared-load timing probe to isolate the remaining
30-40 ms render/backward loop cost.

Changes made:

- Added `tools/compare_fused_slab_vjp_modes_mps.py` in the
  `world_foam_lane2_fused_slab_v0` fork. It loads the max-frame multicam bundle
  once, then runs multiple VJP modes against the same cached tensors in one
  process.
- Added `test_compare_fused_slab_vjp_modes_mps.py` for parser and summary
  coverage.
- Changed the default `train_eval_fused_slab_mixed_mps.py --vjp-mode` to
  `direct_atomic_rgb_only`, because the harness default loss is RGB MSE and the
  measured RGB-only VJP is faster than the full RGBA/depth grad-only VJP.
- Updated `research_experiments/world_foam_lane2/README.md` so the current
  repeat20 speed gate points to the RGB-only artifact, while the owner-update
  alpha/depth aux path remains the correctness gate for nonzero alpha/depth
  adjoints.

Mode comparison artifact:

`research_experiments/world_foam_lane2/results/2026-05-18_gate4_vjp_mode_compare_direct_render32_site12_2_16.json`

Same-process 2f/16f results:

- `direct_atomic_grad_only`: total median `67.478 -> 86.495 ms`, backward
  median `31.555 -> 40.613 ms`, total scale `1.282x`.
- `direct_atomic_rgb_only`: total median `63.997 -> 78.474 ms`, backward median
  `28.598 -> 33.170 ms`, total scale `1.226x`.
- `direct_atomic_track`: total median `78.992 -> 80.475 ms`, backward median
  `36.164 -> 40.451 ms`, total scale `1.019x`.

The RGB-only mode preserves displayed train/heldout PSNR relative to the other
direct modes on this probe, so it is the current practical RGB speed path.
`direct_atomic_track` is interesting because total scaling is almost flat, but
its low-frame baseline is slower and backward is not faster.

Full RGB-only repeat20 artifact:

`research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_rgbonly_repeat20_render32_site12_2_4_8_16.json`

Verifier:

`research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_rgbonly_repeat20_render32_site12_2_4_8_16_verifier.json`

Verifier status: `ok`, failures `[]`.

- total median 2/4/8/16f: `65.028 / 78.987 / 78.954 / 79.031 ms`
- backward median 2/4/8/16f: `31.097 / 32.186 / 33.283 / 36.573 ms`
- total median scale 2f->16f: `1.215x`
- backward median scale 2f->16f: `1.176x`
- train PSNR: `13.845 / 13.869 / 13.918 / 13.998`
- heldout PSNR: `14.288 / 14.504 / 14.536 / 14.592`
- train mixed tape scale 2f->16f: `0.992x`; explicit ray scale: `8.0x`

Tests:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps \
  research_experiments.world_foam_lane2.test_compare_fused_slab_vjp_modes_mps -v
```

Result: 31 tests OK.

Next useful shader question:

The remaining bottleneck is not alpha/depth gradient work for RGB loss anymore.
For RGB-only mode, render is still about `25-31 ms` and backward about
`31-37 ms`. The next fork should focus on reducing per-track replay work or
kernel/sync overhead inside the RGB-only VJP path, not on loader timing or
nonzero alpha/depth adjoints.
