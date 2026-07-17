# World Foam Segment Tape Probe

Added `research_experiments/world_foam_lane2/probe_fused_slab_segment_tape.py`
to test the next structural idea after the fused-slab mixed VJP gate: precompute
a fixed-geometry segment tape with per-sample owner, length, and mid-depth, then
replay RGBA/density through that tape.

The probe compares differentiable Torch tape replay against the current fused
MPS mixed path and the current `direct_atomic_grad_only` VJP winner. It uses the
same moving first-person synthetic ray motion as the fused slab smoke and writes:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_fused_slab_segment_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --timing-iters 3 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json
```

Result: status `ok`.

Key numbers:

- max forward error versus current mixed shader: `1.6707181930541992e-4`
- max VJP relative error versus current reduce path: `9.934884841865477e-6`
- max VJP relative error versus current winner grad-only path:
  `8.546149366821135e-6`
- segment scale 2f -> 16f: `8.055867973756872x`
- Torch tape forward scale 2f -> 16f: `2.7982335280967674x`
- 16f total segments: `1272086`
- 16f compact CSR tape storage: `15396108` bytes
- 16f compact CSR storage ratio versus current mixed CSR plus affine rays:
  `13.286526472760913x`

Interpretation:

- The segment-tape math is compatible with the current forward/VJP contract for
  fixed geometry and trainable site RGBA/density.
- The probe is not a completion claim: it is Torch replay, not a fused Metal
  compact-tape shader.
- The naive per-sample tape removes depth sort and owner lookup from the step
  path, but its segment count grows essentially with frame count. This is not
  the clean STAR-UVT-style structural sublinearity we want by itself.
- Next real shader step is a compact Metal segment-tape forward and grad-only
  VJP kernel, but the larger research step is to avoid materializing one full
  segment list per frame if the target is STAR-like scaling.
