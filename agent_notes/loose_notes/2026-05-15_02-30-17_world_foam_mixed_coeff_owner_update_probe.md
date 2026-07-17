# World Foam Mixed-Coefficient and Owner-Update Probe

Context: the fp32 coefficient replay path was exact and reduced the old ID-CSR
constant, but it still carried four fp32 coefficient values per candidate. The
all-fp16 coefficient path was often fast but failed quality at render32 because
small depth perturbations changed segment owners and RGB.

What changed:

- Added `fused_slab_affine_num32_den16_realray_rgba_depth_replay`, which stores
  numerator coefficients in fp32 and denominator coefficients in fp16.
- Kept this mixed path in the normal smoke acceptance gate:
  `mixed_matches_explicit_realray`.
- Added `fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay`
  as an opt-in diagnostic path behind `--include-ownerupdate`. It tries to
  update the current site owner by crossing boundary left/right pairs instead
  of running all-sites owner lookup for every segment.
- Tried a threadgroup site/RGBA cache inside the mixed kernel, then reverted it
  because it slowed the measured rows.

Files touched:

- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_metal.mm`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py`

Verification:

- Build passed:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

- `py_compile` passed for the slab wrapper/smoke and the CSR smoke/probe scripts.
- `git diff --check` passed for the slab/CSR variants and loose notes.
- Clean render32 mixed sweep:
  `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_slabmid_site12_render32_pertrack_2_4_8_16_clean.json`
- Render16 mixed sweep:
  `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_slabmid_site12_pertrack_2_4_8_16.json`
- Owner-update diagnostic smoke:
  `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_ownerupdate_smoke_2f_site12_pertrack.json`
- Site-cache rejection probe:
  `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_sitecache_site12_render32_pertrack_8.json`

Key render32 timings, clean mixed path, site12, per-track, slabs1,
`slab-mid-depth`, timing iters 3:

- 2f: direct 1.812 ms, ID-CSR 2.386 ms, coeff32 2.370 ms, mixed 2.247 ms
- 4f: direct 2.025 ms, ID-CSR 3.316 ms, coeff32 1.771 ms, mixed 1.792 ms
- 8f: direct 2.469 ms, ID-CSR 3.475 ms, coeff32 2.563 ms, mixed 2.268 ms
- 16f: direct 3.476 ms, ID-CSR 4.178 ms, coeff32 3.221 ms, mixed 2.864 ms

Interpretation:

- The mixed path is the best exact forward-only path so far. It keeps strict
  correctness with max mixed error `0.00016689300537109375` in the clean
  render32 sweep.
- It beats direct at 4/8/16 frames in that clean render32 artifact, but not at
  2 frames. This is the first practical speed win across multiple frame counts,
  but it is still not STAR-like flat scaling.
- Storage becomes favorable only as frame count grows. Mixed fused storage vs
  explicit rays went from 12.19x at 2f to 1.47x at 16f in the clean render32
  artifact.
- All-fp16 remains rejected for quality. In the clean render32 artifact,
  `coeff16_diagnostics.max_error = 0.02310839295387268`.
- Owner-update was rejected: the 2f smoke was already outside strict RGB
  tolerance (`0.0011005699634552002`), and the render32/8f site-cache probe
  showed a much larger owner-update RGB error (`0.4236476719379425`).
- Threadgroup site/RGBA caching was also rejected for now. It was exact for the
  mixed path, but the measured render32/8f row slowed rather than improved.

Next useful direction:

- Keep the mixed `num32_den16` path as the baseline fused replay lane.
- Do not spend more time on literal track-loop, all-fp16 coefficients,
  owner-update topology, or per-threadgroup site caching without a narrower
  hypothesis.
- The remaining gap is candidate/segment replay constant and backward/training
  integration. This is now a real forward speed win at larger frame counts, but
  not a full STAR UVT competitiveness result.
