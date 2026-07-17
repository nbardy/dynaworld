# World Foam Shader-Fork Completion Audit

Objective under audit: fork the World Foam shaders, try fused variants, run
tests/PSNR/speed across frame counts, and iterate until the path is fixed.

Prompt-to-artifact checklist:

- Forked shader work exists: yes. Current fork is
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0`.
- Three or more fused VJP variants were tested: yes. The status summary covers
  `reduce`, `direct_atomic`, `direct_atomic_track`, `direct_atomic_rgb_only`,
  and `direct_atomic_grad_only`.
- Speed was measured across frame counts: yes. The canonical sweep covers
  2/4/8/16 frames and reports `direct_atomic_grad_only` as the best current
  path: `7.17 ms` at 2f and `9.32 ms` at 16f, a `1.30x` total-step scale.
- Train/eval quality was measured: yes. The same sweep records PSNR; 16f
  heldout PSNR for the winner is `13.273961608131371`, with matched-frame
  PSNR spread across modes around `2.06e-6`.
- VJP correctness is covered: yes for site RGBA. RGB and RGBA/depth seeds,
  reduce/direct-atomic comparisons, and autograd wrapper checks are recorded in
  the fused-slab status summary.
- Negative shortcuts were tested: yes. Owner-update and ordered-append variants
  are rejected by saved artifacts.
- Compact segment-tape shader exists: yes. `segment_tape_rgba_depth_replay`,
  `segment_tape_vjp_direct_atomic_grad_only`, and
  `segment_tape_vjp_direct_atomic_track` are wired in the Metal fork and Python
  wrapper.
- Segment-tape shader correctness is covered: yes. The render32 2/4/8/16 probe
  is green; track VJP rel error versus current winner is `6.04e-6`.
- STAR-like structural sublinearity is achieved: no. Segment count grows
  `8.06x` from 2f to 16f for an `8x` frame-count increase, and 16f compact CSR
  tape storage is `15.4 MB`, `13.3x` current mixed CSR plus affine-ray storage.
- Matched STAR-UVT competitiveness is proven: no. The STAR result in the
  summary is only a small 32px speed reference, not a matched quality/capacity
  comparison.
- Full training-path integration for compact segment tape exists: no. The tape
  kernels are isolated probes, not promoted into the main trainer path.

Audit result: not complete if "fixed" means STAR-UVT-style sublinear World
Foam. The current fused path is a verified shader gate and is practically fast
on the small fixed-geometry/site-RGBA workload, but the remaining work is to
find a representation whose tape/cardinality grows better than per-frame.

Evidence commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Verifier status remains `ok` with `failures: []`, but the summary intentionally
keeps `completion_claim=false` and `star_uvt_competitive_claim=false`.
