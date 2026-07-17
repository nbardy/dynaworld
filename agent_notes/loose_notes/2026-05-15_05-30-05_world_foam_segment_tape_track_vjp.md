# World Foam Segment Tape Track VJP

We extended the `world_foam_lane2_fused_slab_v0` fork with a second compact
segment-tape VJP kernel, `segment_tape_vjp_direct_atomic_track`. Unlike the
sample-atomic tape VJP, it assigns one thread per track and accumulates site
RGBA gradients across frames locally before atomically adding the 12-site
gradient vector.

The full render32 2/4/8/16 probe is green:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_fused_slab_segment_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --timing-iters 3 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json
```

Key numbers:

- status: `ok`
- max Metal-tape forward error vs current mixed shader: `1.6695e-4`
- max sample-atomic Metal tape VJP relative error vs current winner:
  `8.5461e-6`
- max track-accumulating Metal tape VJP relative error vs current winner:
  `6.0410e-6`
- max track-vs-sample-atomic tape VJP relative error: `8.9735e-6`
- 16f isolated Metal tape timings: `1.5297 ms` forward, `8.4577 ms`
  sample-atomic grad-only VJP, `4.3539 ms` track grad-only VJP
- segment scale 2f->16f: `8.0559x` for an `8x` frame-count increase
- 16f compact CSR tape storage: `15396108` bytes, `13.2865x` current mixed
  CSR plus affine-ray storage

Interpretation: the track VJP is a real useful shader variant for the isolated
tape replay, but the tape representation remains per-sample/per-frame in
cardinality. This improves execution of the current compact replay contract;
it does not create the STAR-UVT-style structural sublinearity we wanted.

Reporting updates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

The verifier is green with `failures: []`. The status summary still keeps
`completion_claim=false` and `star_uvt_competitive_claim=false`.
