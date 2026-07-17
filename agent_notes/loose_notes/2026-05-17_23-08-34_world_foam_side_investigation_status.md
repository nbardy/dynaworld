# WorldFoam Side Investigation Status

Date: 2026-05-17

Scope: notes/static audit only. I did not edit shared STAR UVT, fast-mac
benchmark audit, trainer, or baseline files. I did not launch MPS/Modal/GPU
benchmarks.

## Current Read

The current WorldFoam fixed-geometry shader gate is still:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse
```

The saved status summary is honest about scope: `status` is
`ok_current_shader_gate_with_structural_gap`, while `completion_claim=false` and
`star_uvt_competitive_claim=false`. The accepted render32/site12 loss-reduced
artifact remains a clean narrow speed-scale artifact under the robust verifier:
total mean scale `1.464x`, backward mean scale `1.536x`, and storage scale
`1.026x` from 16 to 128 frames.

That is real evidence for the fixed-geometry RGB-only site-RGBA replay kernel.
It is not a full trainer, geometry-gradient path, or BASELINES-style model
claim.

## What Not To Retry First

Recent forks give useful negative boundaries:

- Packed framegroup records are a narrow 16/32 speed candidate and storage win,
  but the 64/128 interleaved guard rejected broad promotion.
- Packed predecode and sentinel fast paths were reverted; extra threadgroup
  state did not fix the 128f loss.
- Producer/sidecar owner-reduce is correct on parity but not practical in the
  current i16x3 replay loop; the warmed 32f row was catastrophically slow.
- Compact-state reverse-pass reloads, framegroup64, i16cols, i16x4, and
  binary-search change selection were runtime negatives.

So the next real work should not mutate the promoted framegroup16 op again
unless the new mode is separately named and gated. The pattern is clear:
micro-edits around record packing or owner-list scans mostly reshuffle the same
per-frame replay work.

## Likely Next Shader Fix

The real missing shader bridge is between the Gate 4 moving-ray slab compiler
and the real-ray MPS CSR compositor/VJP path:

- Gate 4 compiles affine moving camera rays into one candidate tape per
  `(view, pixel, time_slab)` and keeps boundary tests flat across frames, but it
  is CPU accounting only.
- Gate 2B/2C/2D already prove true real-ray shared forward, fixed-segment VJP,
  and reduced site RGBA/density gradients on MPS.
- Gate 2F/2G prove CSR candidate storage and small frame scaling.
- Gate 3 proves frozen-geometry real-target train/eval through the CSR reduced
  VJP path, but only with fixed geometry and site RGBA/density updates.

The next useful shader is therefore a Metal CSR compositor/VJP that consumes
the Gate 4 compiled affine slab tape directly, without re-expanding candidate
work into per-frame boundary tests. The first version should stay deliberately
narrow: forward parity, then reduced site RGBA/density VJP parity, then a tiny
frozen-geometry autograd smoke. Do not add geometry/topology gradients in the
same step.

## No-GPU Validation Plan

Before any paid or long GPU work:

1. CPU/static gate: run `gate4_moving_ray_slab_compiler.py` at tiny frame counts
   and assert zero missing sample events, flat compiled boundary tests, finite
   candidate rows, and unchanged direct-vs-compiled accounting.
2. Import/static gate: `py_compile` any new Gate 4 CSR shader wrapper or
   verifier scripts.
3. CPU parity oracle: add a tiny pure-Python/CPU reference that consumes the
   compiled Gate 4 CSR tape and matches the existing per-frame real-ray
   reference on RGB/alpha/depth.
4. Tiny local MPS only after the CPU oracle is green: one 16px or smaller
   forward parity smoke, no timing promotion.
5. Backward only after forward parity: reduced site RGBA/density VJP parity
   against the CPU oracle and the existing Gate 2D reduction semantics.

Acceptance should stay file-backed and scoped: a new JSON artifact plus a
verifier that refuses to call the result a full trainer, geometry-gradient path,
STAR-UVT parity result, or BASELINES row.

## Lightweight Checks Run

Generated temporary outputs only:

```text
/tmp/world_foam_status_summary_check.json
/tmp/world_foam_framegroup16_robust_check.json
```

Commands:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py --out-json /tmp/world_foam_status_summary_check.json
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json --out-json /tmp/world_foam_framegroup16_robust_check.json
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py
```

Results: status summary regenerated, robust verifier returned `status=ok` with
`clean_speedscale_artifact=true`, and `py_compile` passed.
