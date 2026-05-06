# PowerFoam Projected Bounds Probe

Date: 2026-05-03 01:40:17

## Context

After adding the full height+SV primitive, the streaming Metal path was still
far too slow at 4K. Before wiring the unfinished tiled kernel family, I tried
the cheaper optimization already supported by the streaming ABI: per-cell
screen bounds. Until this pass, every wrapper used full-screen bounds, so every
4K pixel checked every cell.

## Change

Added `_projected_screen_bounds(...)` in
`third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py` and routed all
streaming wrappers through it.

The bounds are conservative for the current pinhole rays:

- project cell center by ray-slope coordinates
- use the near side of the sphere for projected radius
- add a 3-pixel pad
- fall back to full-screen bounds for near-plane-intersecting spheres
- use empty bounds for cells fully outside the screen

This remains a candidate-culling optimization only. It does not change the
primitive math and it does not make the streaming kernel tiled; each pixel still
iterates through every cell and pays the bounds branch.

## Validation

Passed:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py
PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py
PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/backward_check.py
```

The full `linear_texture_check.py` still passed, including the newer
height+SV modes:

- `oriented_height_sv_texel_surface` features max error:
  `8.530914783477783e-07`
- `quaternion_height_sv_texel_surface` features max error:
  `9.238719940185547e-07`

## 4K Benchmark

Command:

```bash
PYTHONPATH=src/train .venv/bin/python \
  third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py \
  --cells 1024,4096 \
  --resolutions 4096x4096 \
  --feature-dim 3 \
  --neighbors 32 \
  --warmup 1 \
  --iters 2 \
  --foam-backward \
  --foam-height-sv-texel-surface \
  --json
```

Saved JSON:

`outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_projected_bounds_4k_1024_4096_2026-05-03.json`

Results after projected bounds:

- 1024 cells, 4096x4096: forward `1869.905 ms`, backward `11263.741 ms`,
  total `13133.646 ms`.
- 4096 cells, 4096x4096: forward `4435.052 ms`, backward `20126.446 ms`,
  total `24561.499 ms`.

Comparison to full-screen bounds from the previous note:

- 1024 cells total improved from `15356.079 ms` to `13133.646 ms`.
- 4096 cells total improved from `31491.203 ms` to `24561.499 ms`.

Conclusion: projected bounds help, especially forward at 4096 cells, but they
do not solve the 4K requirement. The streaming kernel is still structurally
wrong for 4K because every pixel loops over every cell. The next performance
work has to be a real tile/candidate-list path that changes the loop shape.
