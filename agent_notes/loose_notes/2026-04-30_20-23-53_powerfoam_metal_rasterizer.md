# PowerFoam Metal Rasterizer Session

## What changed

- Downloaded the PowerFoam PDF to `research_notes/foam_papers/pdfs/powerfoam.pdf`.
- Extracted paper text to `research_notes/foam_papers/text/powerfoam.txt`.
- Added paper-to-implementation notes at
  `research_notes/foam_papers/powerfoam_rasterizer_notes.md`.
- Added an isolated forward-only Metal package under `third_party/powerfoam-metal/`.

## Paper reading summary

The useful rasterizable core is the bounded power-cell interval clip. Each
cell starts as a sphere interval along a ray, then clips that interval against
radical planes from neighboring overlapping cells. Painter order is by power
distance from the camera origin, `||c - p_i||^2 - r_i^2`. The paper uses a
Cech-style neighbor graph or conservative superset; false-positive edges should
not change the answer, only the amount of clipping work.

The full PowerFoam system has more than this first rasterizer: oriented dipole
surfaces, detail-site displacement/radiance texture, densification, and a
training recipe. This session implemented only the core bounded-cell renderer.

## Implementation

The package exposes:

```python
from torch_powerfoam_metal import FoamRasterConfig, rasterize_power_foam
```

Inputs are positions, radii, densities, arbitrary `F`-channel features,
CSR-style adjacency, and `[B,H,W,6]` rays. Outputs are `[B,H,W,F]` features and
`[B,H,W]` alpha, so this already supports a feature-foam raster surface for the
Dynaworld feature-to-color path.

The implementation is deliberately isolated from the current Gaussian renderer
variants. It does not edit `src/train/renderers/fast_mac.py` or the
`third_party/fast-mac-gsplat` variants.

## Validation

Built with:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/powerfoam-metal && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Reference checked with:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/reference_check.py
```

Observed:

```text
features max error: 5.8710575103759766e-06
alpha max error: 7.867813110351562e-06
powerfoam Metal reference check passed
```

## Remaining work

- Add backward or decide on a finite-difference/autograd host fallback only for
  debugging.
- Add a foam-token model head and conversion from decoded tokens to
  `(points, radii, densities, features, adjacency, offsets)`.
- Build a real neighbor graph path from overlapping spheres.
- Add a simple trainer smoke once tokens exist.
- Add tile/bin acceleration if dense `pixels * cells * neighbors` is too slow.

## Follow-up: random PNG and benchmarks

Added:

- `third_party/powerfoam-metal/torch_powerfoam_metal/random_scene.py`
- `third_party/powerfoam-metal/tests/render_random_png.py`
- `third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py`

The PNG smoke writes both RGB and alpha outputs. The first visual check used:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/render_random_png.py --cells 384 --height 192 --width 192 --seed 3 --neighbors 48 --adjacency overlap --out third_party/powerfoam-metal/outputs/random_foam_384_192.png
```

It wrote:

```text
third_party/powerfoam-metal/outputs/random_foam_384_192.png
third_party/powerfoam-metal/outputs/random_foam_384_192_alpha.png
cells=384 resolution=192x192 adjacency=overlap avg_degree=7.45
```

The first forward comparison against `v5_features` GS:

```text
powerfoam_metal    128x128 N=256  fwd_med=4.378ms
gsplat_v5_features 128x128 N=256  fwd_med=4.077ms
powerfoam_metal    128x128 N=1024 fwd_med=3.846ms
gsplat_v5_features 128x128 N=1024 fwd_med=3.138ms
powerfoam_metal    256x256 N=256  fwd_med=4.100ms
gsplat_v5_features 256x256 N=256  fwd_med=2.699ms
powerfoam_metal    256x256 N=1024 fwd_med=11.404ms
gsplat_v5_features 256x256 N=1024 fwd_med=3.506ms
```

A larger single case showed the expected dense-renderer scaling problem:

```text
powerfoam_metal    512x512 N=1024 fwd_med=18.341ms
gsplat_v5_features 512x512 N=1024 fwd_med=3.321ms
```

Backward status: the local Metal op has no registered autograd kernel. I added a
guard in `rasterize_power_foam` so differentiable inputs now raise instead of
returning silent `None` gradients. The official CUDA/Warp PowerFoam repo does
have a custom backward in `powerfoam/rasterize.py`: it saves forward tile lists
and log transmittance, traverses primitives in reverse, and accumulates sphere,
density/normal, and texture gradients.
