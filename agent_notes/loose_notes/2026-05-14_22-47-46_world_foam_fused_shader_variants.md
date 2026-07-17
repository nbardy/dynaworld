# World Foam fused moving-ray shader variants

Prompt: copy the World Foam fork and try three fused shader variants with subagents.

Starting point:
- Base variant copied from `third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/`.
- Prior Gate 4 CPU compiler had already shown moving affine ray tracks can keep compiled boundary tests flat across frame count, but replay was still frame-scaled and CPU-side.

Forks created:
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0/`
  - Package `torch_world_foam_lane2_fused_direct`.
  - Torch namespace `world_foam_lane2_fused_direct_v0`.
  - New op `fused_moving_realray_rgba_depth_replay`.
  - Shader receives affine ray coefficients `[track,12]` and computes `origin(t)` / `direction(t)` inside Metal, but still scans every boundary per frame.
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/`
  - Package `torch_world_foam_lane2_fused_csr`.
  - Torch namespace `world_foam_lane2_fused_csr_v0`.
  - New op `shared_affine_realray_rgba_depth_replay`.
  - Shader receives affine ray coefficients plus a per-track candidate bitset; it computes moving rays in Metal and iterates only set candidate bits.
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/`
  - Package `torch_world_foam_lane2_fused_slab`.
  - Torch namespace `world_foam_lane2_fused_slab_v0`.
  - New op `fused_slab_affine_realray_rgba_depth_replay`.
  - Shader receives affine ray coefficients plus tiled/per-track slab CSR rows.

Local rebuild commands, all passed:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Primary local reruns:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0/tools/smoke_fused_moving_realray_direct_mps.py --frame-counts 2,4,8,16 --render-size 16 --site-count 12 --timing-iters 5 --out-json research_experiments/world_foam_lane2/results/2026-05-14_fused_direct_moving_realray_smoke_2_4_8_16.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/tools/smoke_shared_affine_realray_fused_csr_mps.py --frame-counts 2,4,8,16 --render-size 16 --site-count 12 --timing-iters 5 --out-json research_experiments/world_foam_lane2/results/2026-05-14_fused_csr_moving_ray_mps_smoke_2_4_8_16.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py --frame-counts 2,4,8,16 --render-size 16 --site-count 12 --timing-iters 5 --out-json research_experiments/world_foam_lane2/results/2026-05-14_fused_slab_affine_realray_mps_smoke_2_4_8_16.json
```

Result shape:
- All three passed correctness against their explicit/direct references with max absolute errors under `5e-7`.
- Direct fused is the cleanest shader path but not sublinear: it computes rays in Metal but still performs `O(frames * boundaries)` scans. Timings were around break-even versus explicit ray materialization on the tiny smoke.
- Bitset CSR proves the Gate 4 idea in Metal: compiled boundary tests stayed flat at `33792` from 2 to 16 frames while direct scans grew `67584 -> 540672`; candidate iteration ratio stayed around `0.62x` direct scans. It was still slower than direct MPS on this tiny 66-boundary smoke, likely due to bit iteration overhead and too little work to amortize.
- Tiled slab CSR proves the strongest storage curve: candidate storage stayed about `5.8 KB`, affine ray storage stayed `24 KB`, and fused storage versus explicit rays dropped `1.24x -> 0.155x` from 2 to 16 frames. It was not consistently faster; tiled row union over-includes candidates and replay iterations were about `1.6x` the true per-frame event count.

Current read:
- Carry forward the CSR/bitset path for the next speed attempt because it is the closest to STAR UVT's clean "compile once, replay many frames" structure while avoiding tiled over-inclusion.
- Keep direct fused as the correctness oracle and minimal integration path.
- Treat slab/tiled as a storage proof, not the speed winner yet, unless row granularity is tightened or candidate replay is made much cheaper.

No PSNR/training claim was made here. These are forward-only MPS replay gates against synthetic affine moving rays.
