# World Foam CSR Track-Loop Probe

Context: after the fused direct/CSR/slab variants and LR-owner replay, I tested a more literal STAR-like frame-sharing shape for moving first-person rays: one Metal thread per pixel track, with the thread looping over all frame times and reusing the compiled candidate tape.

Files touched:

- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/csrc/metal/world_foam_lane2_metal.mm`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/torch_world_foam_lane2_fused_csr/ops.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/torch_world_foam_lane2_fused_csr/__init__.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/tools/smoke_shared_affine_realray_fused_csr_mps.py`

What changed:

- Added `wf2_shared_affine_trackloop_realray_rgba_depth_replay_tensor`, which dispatches one thread per track and loops over all frames internally.
- Wired it through the Metal host, Torch binding, Python wrapper, and smoke timing/reporting as `shared_affine_trackloop_realray_rgba_depth_replay`.
- Kept it as a separate path so direct, normal CSR, LR-owner CSR, and track-loop CSR remain comparable in the same smoke JSON.

Verification:

- Build passed:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

- `git diff --check -- third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0` passed.
- `py_compile` passed for the edited Python wrapper and smoke script.
- Correctness passed against direct MPS and CPU reference in all track-loop smokes.

Result JSONs:

- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_trackloop_smoke_2f_site12.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_trackloop_site12_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_trackloop_site12_render32_2_4_8.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_fused_csr_trackloop_site24_slabs4_2_4.json`

Key timings:

- site12, render16, slabs1:
  - 2f: direct 1.828 ms, CSR 2.286 ms, LR 1.734 ms, track-loop 2.705 ms
  - 4f: direct 1.807 ms, CSR 1.721 ms, LR 1.722 ms, track-loop 3.776 ms
  - 8f: direct 1.631 ms, CSR 1.958 ms, LR 1.949 ms, track-loop 6.132 ms
  - 16f: direct 1.974 ms, CSR 2.391 ms, LR 2.366 ms, track-loop 11.165 ms
- site12, render32, slabs1:
  - 2f: direct 1.971 ms, CSR 2.495 ms, LR 2.360 ms, track-loop 4.608 ms
  - 4f: direct 2.501 ms, CSR 2.807 ms, LR 2.550 ms, track-loop 4.696 ms
  - 8f: direct 2.869 ms, CSR 3.125 ms, LR 3.705 ms, track-loop 8.686 ms
- site24, render16, slabs4:
  - 2f: direct 6.875 ms, CSR 10.313 ms, LR 15.573 ms, track-loop 12.008 ms
  - 4f: direct 4.229 ms, CSR 10.575 ms, LR 13.434 ms, track-loop 12.830 ms

Interpretation:

- The theory is still sublinear at the schedule/storage level for the simple site12/slabs1 case: compiled boundary tests stay fixed while direct scans grow with frame count. Example: site12/render16 compiled tests stay at 33,792 while direct scans grow from 67,584 to 540,672 from 2f to 16f.
- The practical wall time is not sublinear yet. Normal CSR is around break-even only in the easiest rows, LR-owner replay is mixed, and the literal track-loop port is worse because it gives up frame-level GPU parallelism.
- The site24/slabs4 result is a useful warning: time slabs can duplicate candidate work. At 2 frames compiled tests were 565,248 versus direct scans 282,624, so the compiled schedule was not smaller there.
- STAR UVT's practical cleanliness is not just "loop frames inside one shader." It has a lower-constant per-track schedule and a more compact reuse target. World Foam currently has expensive candidate insertion/sorting/owner evaluation and slab duplication, so the same high-level idea does not automatically carry.
- No PSNR/training claim here. These are forward-only MPS correctness/timing gates.
