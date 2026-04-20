# Mac Renderer Repo Crosslinks

Updated both public Mac renderer repos so users can choose the right path:

- `taichi-gsplat-differentiable-render-mac`
  - README now says this fork is for Taichi compatibility.
  - README links to `https://github.com/nbardy/fast-mac-gsplat` as the
    recommended high-performance pure Metal/Torch path.
  - README records native Taichi batch speedups versus a direct packed-2D Torch
    reference:
    - `B=4`, 64x64, 128 splats: `7.6x` forward, `16.9x` forward+backward
    - `B=4`, 128x128, 128 splats: `30.8x` forward, `36.2x` forward+backward
  - Pushed commit: `c73625b Document Mac renderer choice and Torch speedups`

- `fast-mac-gsplat`
  - README now says it is the recommended speed-first Mac path when Taichi is
    not needed.
  - README links back to the Taichi-compatible fork.
  - Pushed commit: `477cd33 Link Taichi-compatible Mac renderer`

The raw Torch timing probe attempted a larger `B=4`, 128x128, 512-splat
forward+backward case first, but direct Torch was too slow to be useful
interactively, so that benchmark process was stopped and replaced with the
smaller README-sized probe above.
