# STAR UVT Multi-Center K8 Radius/Opacity Sweep

## Question

The first multi-center gate showed that `farthest_xy` with `K=8`, `32` fixed
births, `r64`, and cap `128` moved dense alpha `>0.1` to `0.4309`, but the
target-background oracle fell to `23.965`. This sweep asks whether radius and
born opacity can recover oracle without giving back the new multi-center
coverage gain.

## Command

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  --target-sources uncovered_brightness \
  --reallocate-tubes 32 \
  --support-radii 48,56,64,72 \
  --center-strategies farthest_xy \
  --center-counts 8 \
  --opacities 0.4,0.6,0.8 \
  --tile-capacities 128 \
  --out-base outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128
```

Artifacts:

- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128.md`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128.json`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128_dense_support.md`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128_dense_support.json`

## Results

All 12 rows passed, all with zero overflow. Tile max/p95/cap stayed
`100-101/71/128`.

Best coverage row:

- `uncovered_brightness_n32_r72_farthest_xy_c8_o0p8_cap128`
- alpha `>0.1`: `0.431797`
- alpha `>0.5`: `0.163643`
- normal PSNR: `5.871`
- forced-alpha PSNR: `14.605`
- oracle PSNR: `23.670`
- step/backward: `183.8ms` / `70.1ms`

Best balance row:

- `uncovered_brightness_n32_r64_farthest_xy_c8_o0p4_cap128`
- alpha `>0.1`: `0.429806`
- alpha `>0.5`: `0.138504`
- normal PSNR: `5.789`
- forced-alpha PSNR: `14.620`
- oracle PSNR: `24.805`
- step/backward: `167.9ms` / `58.1ms`

Highest-oracle multi-center row:

- `uncovered_brightness_n32_r48_farthest_xy_c8_o0p4_cap128`
- alpha `>0.1`: `0.417831`
- normal PSNR: `5.731`
- oracle PSNR: `25.214`

## Read

Multi-center radius/opacity behaves as a clean frontier:

- lowering opacity recovers oracle at almost every radius
- raising radius/opacity improves normal PSNR and high-alpha coverage
- cap128 remains safe for this K8/32-birth setting

The important new row is `r64/o0.4`: it keeps almost all of the K8 coverage
gain (`0.4298` versus `0.4309` for the original K8 r64/default row), recovers
oracle by about `+0.84` PSNR (`23.965 -> 24.805`), and has the fastest measured
mean step among the balanced rows (`167.9ms`).

This still does not solve visual quality; forced-alpha remains around
`14.6` and dense RGB is below RGB STAR. But the support primitive is no longer
stuck at the `0.411-0.420` alpha band. Multi-center K8 with lower born opacity
is the current best fixed-budget support bridge.

## Next

Run a short media gate from the selected balanced row:

- `uncovered_brightness`, `K=8`, `32` births, `r64`, opacity `0.4`, cap `128`
- 20 steps first; promote only if feature/probe losses stay monotonic and media
  improves visibly
- if it holds, run a 50-step continuation or compare `r72/o0.4`

Do not return to single-center radius, scalar opacity, or one-line ellipse
sweeps unless the multi-center path regresses under longer training.
