# World Foam Coefficient Replay Probe

Context: after the LR-owner and track-loop CSR probes, I tried a lower-constant
candidate replay path for moving first-person rays. The prior slab CSR path
stored boundary IDs and recomputed the boundary plane dot products for every
candidate/frame sample. This pass precomputes affine depth coefficients per
track/candidate so replay only evaluates a small linear depth expression.

Files touched:

- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_metal.mm`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py`

What changed:

- Added `fused_slab_affine_coeff_realray_rgba_depth_replay`, which consumes
  fp32 per-candidate depth coefficients instead of candidate boundary IDs.
- Added `fused_slab_affine_coeff16_realray_rgba_depth_replay`, which consumes
  fp16 coefficient storage as a diagnostic speed/storage probe.
- Kept the coefficient route restricted to `layout=per-track`. The coefficients
  depend on the actual ray track, so tiled row sharing cannot reuse a single
  coefficient row without changing the math.
- Updated the smoke harness to time ID-CSR, fp32 coefficient CSR, and fp16
  coefficient CSR in one run, with strict acceptance tied only to the exact
  ID/fp32 paths. The fp16 path is now reported under `coeff16_diagnostics`
  because it is approximate.

Verification:

- Build passed:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

- `git diff --check -- third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0` passed.
- `py_compile` passed for the slab wrapper/smoke and the CSR smoke/probe scripts.
- The final render32 rerun exited `status: ok` for strict ID/fp32 correctness,
  while preserving the fp16 quality failure as diagnostic data.

Result JSONs:

- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_coeff_smoke_2f_site12_pertrack.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_coeff_site12_pertrack_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_coeff_site12_render32_pertrack_2_4_8.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_coeff16_smoke_2f_site12_pertrack.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_coeff16_site12_pertrack_2_4_8_16_ok.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_coeff16_site12_render32_pertrack_2_4_8_ok.json`

Key timings:

- site12, render16, per-track, slabs1:
  - 2f: direct 1.812 ms, ID-CSR 2.742 ms, coeff32 2.460 ms, coeff16 1.948 ms
  - 4f: direct 1.665 ms, ID-CSR 3.560 ms, coeff32 2.024 ms, coeff16 2.250 ms
  - 8f: direct 1.661 ms, ID-CSR 3.104 ms, coeff32 3.163 ms, coeff16 2.179 ms
  - 16f: direct 2.257 ms, ID-CSR 4.656 ms, coeff32 2.614 ms, coeff16 3.575 ms
- site12, render32, per-track, slabs1:
  - 2f: direct 2.082 ms, ID-CSR 3.221 ms, coeff32 2.543 ms, coeff16 2.621 ms
  - 4f: direct 1.858 ms, ID-CSR 2.791 ms, coeff32 2.521 ms, coeff16 2.525 ms
  - 8f: direct 2.101 ms, ID-CSR 3.441 ms, coeff32 2.522 ms, coeff16 2.977 ms

Interpretation:

- The fp32 coefficient path is exact within the existing strict tolerance and
  attacks the right constant factor. It is generally faster than the old ID-CSR
  replay, by about 1.1x to 1.8x in these rows.
- It still does not beat the direct per-frame real-ray scan in wall time. That
  means the current World Foam fork remains schedule/storage-sublinear in
  selected cases but not practically STAR-competitive.
- The fp16 coefficient path is interesting for bandwidth and sometimes gets
  near direct speed, but it is not quality-preserving. The render32 8f row had
  `coeff16_diagnostics.max_error = 0.02310839295387268`, driven by RGB error.
- A separate quantization probe showed denominator coefficient fp16 storage can
  have large depth outliers, so the fp16 failure is a math/conditioning issue,
  not just a smoke tolerance choice.
- The useful next direction is not another literal frame-loop shader. Track-loop
  proved that moving frame iteration inside one thread gives up too much GPU
  parallelism. The better target is to reduce candidate tape size and replay
  bandwidth while preserving normal pixel/frame parallelism.

No PSNR/training claim here. These are forward-only MPS correctness/timing gates.
