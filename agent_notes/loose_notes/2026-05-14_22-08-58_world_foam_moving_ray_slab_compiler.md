# World Foam Moving-Ray Slab Compiler

Implemented the first CPU Gate 4 port of the STAR-UVT world-tube idea back into
World Foam.

The old World Foam Gate 2 real-ray sharing path assumes each train view's ray
bundle is static across time. That proved time-slab sharing on real camera rays,
but it did not answer the moving first-person camera case. The new
`research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py`
fits each `(view, pixel)` ray track as affine in normalized time and compiles
4D power-boundary candidates from the rational depth interval
`s(t) = -(n . o(t) + nt * t + b) / (n . d(t))`.

Files added:

- `research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py`
- `research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py`
- `research_experiments/world_foam_lane2/results/gate4_moving_ray_slab_compiler_affine_motion_smoke.json`
- `research_experiments/world_foam_lane2/results/gate4_moving_ray_slab_compiler_affine_motion_timed_2_4_8_16.json`

Validation run:

```bash
python3 -m py_compile \
  research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py

python3 -m unittest discover \
  -s research_experiments/world_foam_lane2 \
  -p 'test_gate4_moving_ray_slab_compiler.py' \
  -v

PYTHONDONTWRITEBYTECODE=1 python3 research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 12 \
  --time-slabs 1 \
  --origin-velocity-x 0.08 \
  --origin-velocity-z 0.02 \
  --direction-velocity-x 0.02 \
  --out-json research_experiments/world_foam_lane2/results/gate4_moving_ray_slab_compiler_affine_motion_timed_2_4_8_16.json
```

Result:

- status `ok`
- moving ray tracks present
- zero missing sample events for 2, 4, 8, and 16 frames
- direct boundary tests grow `8.0x` from 2 to 16 frames
- compiled boundary tests grow `1.0x`
- compiled boundary-test ratio improves from `0.5` to `0.0625`
- CPU candidate-tape compile time stays roughly flat:
  `0.055s -> 0.054s -> 0.065s -> 0.060s`
- compiled candidate replay iterations still grow `7.767859610335774x`

Interpretation:

The STAR-style compile-first idea transfers cleanly to World Foam candidate
generation for affine moving ray tracks. It collapses boundary-test growth, but
it does not yet collapse replay/compositing. The next gate should move this
compiled tape into Metal/CSR replay and measure whether the compositor can avoid
re-expanding work per frame.
