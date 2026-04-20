# Taichi Mac Fork Submodule

Published the local `third_party/taichi-splatting` checkout as a standalone
GitHub fork:

- Repository: `git@github.com:nbardy/taichi-gsplat-differentiable-render-mac.git`
- Web URL: `https://github.com/nbardy/taichi-gsplat-differentiable-render-mac`
- Fork commit pushed to `main`: `37a83dd Add experimental Metal rasterizer path`
- Local branch remains `dynaworld-metal-reference`, tracking `origin/main`.
- Original upstream remote is preserved as `upstream`:
  `https://github.com/uc-vision/taichi-splatting.git`.

Fork README update:

- Added a Mac / Metal fork status section.
- Documented `metal_reference`, `pixel_reference`, `sort_backend` modes, and
  the fact that native Taichi sort experiments are correct but slower than the
  PyTorch/MPS sort plus Taichi/Metal raster reference.
- Called out the real next speed target: fused tile-local sort and raster using
  Metal threadgroup memory.

Parent repo submodule registration:

- Registered `third_party/taichi-splatting` in `.gitmodules` with the new SSH
  URL.
- The parent gitlink points at submodule commit `37a83dd`.
- Left unrelated dirty parent worktree files unstaged.

Validation before/around publishing:

- `git -C third_party/taichi-splatting diff --check` passed after fixing one
  trailing-space hunk.
- `uv run python -m py_compile ...` passed for the patched Taichi files and
  benchmark/accuracy harness files.
- Current Taichi experiment config still runs, with `taichi_metal_bucket_sort`
  and `taichi_metal_global_sort` slower than `taichi_metal_reference`.

Important result:

- This submodule is a stable baseline and upstreamable fork point, not the final
  fast Mac renderer. The current Taichi-native sort paths do not beat the
  reference path; the next speed attempt should be a raw fused Metal kernel or a
  Taichi backend change that exposes the needed threadgroup-memory pattern.

Follow-up README polish:

- Updated the fork README opening to lead with `Taichi Mac Splatting` and
  `Fast splatting on the Mac integrated with Taichi.`
- Pushed fork commit `244b002 Clarify Mac fork README`.
- Parent submodule pointer should advance from `37a83dd` to `244b002`.
