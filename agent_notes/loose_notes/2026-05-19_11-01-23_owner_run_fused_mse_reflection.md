# Owner-Run Fused-MSE Reflection

## Context

We stopped the shader iteration after adding and testing two RGB-only fused MSE paths in
`world_foam_lane2_fused_slab_v0`:

- `endpoint-run-fused-mse`: fuses loss and VJP over endpoint-run tape.
- `owner-run-fused-mse`: fuses loss and VJP over the more compact segment/owner-run tape.

Both modes target fixed-geometry RGB-only site RGBA optimization with manual VJP. They are not
full geometry-gradient, full-depth, or full-trainer claims.

## Verification

Owner-run fused MSE passed:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_endpoint_run_tape -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --tape-mode owner-run-fused-mse --frame-counts 2 --render-size 16 --site-count 8 --optimizer-mode manual-vjp --steps 1 --warmup-steps 0 --out-json research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_smoke_2f_render16_site8.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --tape-mode owner-run-fused-mse --frame-counts 2,4,8,16 --render-size 16 --site-count 24 --optimizer-mode manual-vjp --steps 3 --warmup-steps 1 --out-json research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_scale_2_4_8_16_render16_site24_warm3.json
```

The unit suite reported 4 tests OK. The scale run status was OK, but the benchmark
environment was contended by unrelated CPU jobs, so timing should be treated as directional.

## Main Results

At 16px, 24 sites, 2/4/8/16 frames:

| mode | total ms | backward ms | selected segments | selected storage |
| --- | --- | --- | --- | --- |
| owner-run fused, 2f | 3.402 | 2.609 | 1,866 | 26,492 B |
| owner-run fused, 4f | 4.384 | 3.254 | 4,477 | 61,920 B |
| owner-run fused, 8f | 3.365 | 2.598 | 9,287 | 127,832 B |
| owner-run fused, 16f | 4.506 | 3.600 | 18,862 | 259,116 B |

Endpoint-run fused on the same ladder:

| mode | total ms | backward ms | selected segments | selected storage |
| --- | --- | --- | --- | --- |
| endpoint-run fused, 2f | 2.859 | 2.229 | 5,039 | 64,568 B |
| endpoint-run fused, 4f | 3.423 | 2.596 | 10,616 | 135,588 B |
| endpoint-run fused, 8f | 2.878 | 2.384 | 21,548 | 274,964 B |
| endpoint-run fused, 16f | 2.606 | 1.851 | 43,351 | 552,984 B |

Owner-run fused is much more compact: at 16f it uses 18,862 selected segments and
259,116 bytes versus endpoint-run fused's 43,351 segments and 552,984 bytes. That is
about 2.3x fewer selected segments and 2.1x less selected storage.

The bad news is that owner-run selected storage still scaled 9.78x and selected
segments 10.11x over an 8x frame increase. The timed hot loop only scaled 1.32x
total / 1.38x backward, but that is a small contended benchmark and not enough to
claim a durable STAR-like frame-scaling law.

## Interpretation

This was a good fork, but not the breakthrough fork.

What improved:

- The RGB loss and VJP are now fused into one Metal pass for endpoint-run and owner-run tapes.
- Endpoint-run fused was a clear hot-kernel improvement over earlier track-MSE fused work.
- Owner-run fused proves that the compact owner-run representation can drive a direct RGB-MSE VJP correctly.
- Owner-run storage is materially lower than endpoint-run storage.

What did not improve enough:

- The tape itself is still frame-expanded. Even the compact owner-run tape grows roughly linearly with frame count on this fixture.
- The owner-run kernel was not faster than endpoint-run fused in the small 16px timing ladder despite using less storage.
- This does not port the core STAR UVT advantage yet. STAR stays cleaner because the work is organized around fixed temporal tubes / compact bins and then amortized through the selected shader path, while this WorldFoam lane still constructs and replays frame-local ray/owner events.

## Current Call

Keep the fused kernels, but do not call WorldFoam competitive with STAR UVT yet.

The next useful direction is not another minor owner-run replay tweak. It is to make
WorldFoam stop materializing frame-expanded tape for the optimization hot path:

- represent motion support as persistent tube/bin intervals,
- compute per-frame coefficients or endpoints on demand inside the fused pass,
- only store sparse boundary/change records if a visibility/owner relation actually changes,
- benchmark prepare time plus resident storage, not just the fused VJP kernel.

Until that exists, WorldFoam has a good local Metal VJP path and a compact owner-run
representation, but not STAR-style sublinear temporal structure.
