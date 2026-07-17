# World Foam Metal Segment Tape Shader

Followed the Torch segment-tape math probe with an actual compact-tape Metal
shader in the `world_foam_lane2_fused_slab_v0` fork.

Added ops:

- `segment_tape_rgba_depth_replay`
- `segment_tape_vjp_direct_atomic_grad_only`

Files touched inside the fork:

- `csrc/metal/world_foam_lane2_shared_replay_tensor.metal`
- `csrc/metal/world_foam_lane2_metal.mm`
- `csrc/bindings.cpp`
- `torch_world_foam_lane2_fused_slab/ops.py`
- `torch_world_foam_lane2_fused_slab/__init__.py`

Build command:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

The build completed for Python 3.11 and refreshed:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/_C.cpython-311-darwin.so
```

Cheap 2f/render16 smoke:

- status: `ok`
- Metal forward max error vs current mixed path: `7.241964340209961e-5`
- Metal VJP rel error vs current winner grad-only path:
  `1.1090331956350592e-5`
- Metal VJP rel error vs Torch tape: `1.158027661539514e-6`

Full render32 2/4/8/16 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json
```

Result: status `ok`.

Key full-run numbers:

- max Metal forward error vs current mixed path: `1.6695261001586914e-4`
- max Metal VJP rel error vs current reduce path: `7.851763826635618e-6`
- max Metal VJP rel error vs current winner grad-only path:
  `6.933987285378648e-6`
- 16f Metal tape forward: `1.407472009304911 ms`
- 16f Metal tape grad-only VJP: `6.212194663627694 ms`
- 16f compact tape storage: `15396108` bytes
- segment scale 2f -> 16f: `8.055867973756872x`

Interpretation:

- The compact Metal tape shader removes per-step depth sorting and owner lookup
  for the fixed-geometry/site-RGBA path.
- It is an isolated shader probe, not full trainer integration.
- The naive per-sample segment tape is still not STAR-UVT-clean: segment count
  and storage scale about linearly with frame count. To get STAR-like structure,
  we need a shared/evented tape layout instead of materializing every per-frame
  segment list.
