# World Foam CSR owner-update iteration

Prompt context: continue iterating on the fused World Foam shaders after the first three forks showed correct sublinear candidate/storage structure but no reliable speed win.

Change made:
- Added an experimental LR-owner replay path to `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/`.
- New Python API: `shared_affine_realray_rgba_depth_replay_lr`.
- New Torch op: `world_foam_lane2_fused_csr_v0::shared_affine_realray_rgba_depth_replay_lr`.
- New Metal kernel: `wf2_shared_affine_realray_rgba_depth_replay_lr_tensor`.
- The kernel takes `boundary_lr_i32 [B,2]` and stores boundary IDs with sorted depth cuts. It computes the segment owner once at the first segment, then updates ownership when crossing a boundary involving the current owner. This removes the old per-segment all-sites owner search from the LR path.

Validation:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/torch_world_foam_lane2_fused_csr/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/tools/smoke_shared_affine_realray_fused_csr_mps.py
git diff --check -- third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0
```

All passed.

Smoke outputs:
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_lr_smoke_site12_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_lr_smoke_site16_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_lr_smoke_site12_render32_2_4_8.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_lr_smoke_site12_slabs4_2_4_8_16.json`
- Also tested slab per-track layout:
  `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_pertrack_affine_realray_mps_smoke_2_4_8_16.json`
- Removed the stale forward-path 128-boundary-ID cap and raised the stored cut buffer to 256 cuts:
  - `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_uncapped_site24_2f.json` failed before the cut-buffer increase, proving truncation at site24/276 boundaries.
  - `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_uncapped_site24_2f_cut256.json` passed after the cut-buffer increase.
  - `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_uncapped_site24_slabs4_2_4_8.json` passed for 2/4/8 frames at 276 boundaries.

Result:
- LR-owner replay is exactly correct against direct MPS/CPU on the tested sweeps; errors stayed in the same `~1e-6` range as the original CSR path.
- It is not a reliable speed fix. It helps some rows but loses others:
  - site12 slabs1: LR speedup vs direct was `1.09x, 1.22x, 0.93x, 0.83x` for 2/4/8/16 frames.
  - site16 slabs1: LR speedup vs direct was `0.92x, 0.79x, 0.83x, 0.64x`.
  - site12 slabs4: LR speedup vs direct was `0.91x, 0.65x, 1.13x, 0.83x`.
- More time slabs reduce candidate overrun: candidate iteration ratio moved from about `0.62x` direct scans to about `0.57x`, and candidate/event ratio moved near `1.04x`. Wall-clock still did not reliably improve.
- Per-track CSR reduced over-inclusion but had poor storage and indirect-access cost; it was slower than explicit on all tested rows.
- The 256-cut fix is a real correctness/scaling fix: site24/276-boundary forward replay now matches CPU/direct MPS. It is not a speed fix; at site24, CSR was still slower than direct all-boundary replay (`0.71x` speedup at 2f with slabs1; `0.38x, 0.37x, 0.71x` for 2/4/8 with slabs4).
- Added `tools/probe_affine_replay_stage_timing_mps.py` to time normal full replay versus a depth-collection-only mode (`transmittance_threshold=2.0`, which skips owner/compositing in the current Metal branch). Outputs:
  - `research_experiments/world_foam_lane2/results/2026-05-15_affine_replay_stage_probe_site12_slabs1_2_4_8_16.json`
  - `research_experiments/world_foam_lane2/results/2026-05-15_affine_replay_stage_probe_site24_slabs4_2_4.json`
- Added `--candidate-order slab-mid-depth` to the slab smoke for per-track CSR lists. It feeds candidate IDs sorted by approximate slab-midpoint depth to test whether insertion-sort shifts are the bottleneck. Output:
  - `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_pertrack_depthordered_affine_realray_mps_smoke_2_4_8_16.json`
  - This was not a win versus boundary-ID order; the depth-ordered per-track list was slower on the tested rows.

Current interpretation:
- The world-foam candidate math is genuinely sublinear/flat in the compile side, but forward render speed is dominated by shader constant factors, indirect candidate access, sorting/insertion, and compositing. Direct all-boundary replay is very hard to beat on these tiny Metal tests because it is a compact contiguous loop.
- The LR-owner path should remain an experiment, not the default winner.
- The stage probe makes the next target narrower: depth collection/insertion and candidate memory access dominate more than owner/compositing. A STAR-like compact replay layout still needs ordered, coalesced candidate access; just reordering per-track CSR IDs by slab-midpoint depth did not solve it.
