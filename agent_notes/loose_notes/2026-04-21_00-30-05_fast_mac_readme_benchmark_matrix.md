# Fast Mac README Benchmark Matrix

Updated the `fast-mac-gsplat` README so the pure-Metal repo shows the three
renderer families side by side:

- direct Torch baseline for small scenes
- Taichi Mac compatibility path
- pure Metal `fast-mac` v3 path

The key numbers documented:

- 128x128 / 512 splats / sigma 1-5 px:
  - Torch forward `163.244 ms`, forward+backward `828.056 ms`
  - Taichi forward `9.812 ms`, forward+backward `18.400 ms`
  - fast-mac v3 forward `7.443 ms`, forward+backward `8.940 ms`
- 1024x1024 / 65,536 splats / sigma 1-5 px:
  - Torch skipped because it implies about `6.9e10` pixel-splat evaluations
  - Taichi forward `27.630 ms`, forward+backward `284.805 ms`
  - fast-mac v3 forward `7.519 ms`, forward+backward `17.704 ms`
- 4096x4096 / 65,536 splats:
  - Torch skipped because it implies about `1.1e12` pixel-splat evaluations
  - Taichi marked unsupported for the current 16x16 tile-key layout because
    4096x4096 creates 65,536 tiles and the Taichi fork packs tile id into a
    16-bit key range
  - fast-mac v3 sigma 1-5 px forward `12.410 ms`, forward+backward `47.872 ms`
  - fast-mac v3 sigma 3-8 px forward `13.702 ms`, forward+backward `60.738 ms`

The important interpretation is that direct Torch remains useful for correctness
at tiny sizes, Taichi is the compatibility path, and pure Metal is the path to
recommend for high-resolution training-scale renders.
