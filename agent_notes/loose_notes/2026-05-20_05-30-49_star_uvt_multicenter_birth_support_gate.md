# STAR UVT Multi-Center Birth Support Gate

## Question

The single-line anisotropic support gate failed: all rows passed, but alpha
coverage fell to `0.408-0.409`, below the isotropic `0.411`. The likely problem
was not ellipse math; it was fitting one global trajectory through a broad
target field. This gate tests the next primitive: split target points into
multiple spatial centers, then birth the same fixed tube budget around those
centers.

## Implementation

`support_birth_split` now supports:

- `center_strategy`: `global_line` or `farthest_xy`
- `center_count`: number of requested spatial centers

The default is `global_line` with `center_count=1`, preserving the old path.
`farthest_xy` deterministically selects farthest-point centers in target
screen-space, assigns target points to the nearest selected center, fits a
line per group, and allocates the fixed `reallocate_tubes` budget
proportionally across groups with at least one tube per group.

The sweep harness now accepts:

```bash
--center-strategies
--center-counts
```

Focused validation:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py \
  research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_visibility_support_bridge.py -q
```

Result: `37 passed`.

## Sweep

Command:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  --target-sources uncovered_brightness,low_alpha \
  --reallocate-tubes 32 \
  --support-radii 64 \
  --center-strategies farthest_xy \
  --center-counts 4,8 \
  --tile-capacities 128 \
  --out-base outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128
```

Artifacts:

- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128.md`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128.json`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128_dense_support.md`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128_dense_support.json`

## Results

Best row:

- `uncovered_brightness_n32_r64_farthest_xy_c8_cap128`
- pass: true
- mean step/backward: `181.1ms` / `63.5ms`
- tile max/p95/cap: `101/71/128`
- tile overflow: `0`
- alpha `>0.1`: `0.4309`
- alpha `>0.5`: `0.1550`
- normal PSNR: `5.843`
- forced-alpha PSNR: `14.608`
- target-background oracle: `23.965`

Other passing rows:

- `low_alpha_n32_r64_farthest_xy_c4_cap128`: alpha `>0.1` `0.4246`, oracle `23.670`
- `low_alpha_n32_r64_farthest_xy_c8_cap128`: alpha `>0.1` `0.4261`, oracle `23.725`

The `uncovered_brightness`/`K=4` row failed the trainer loss gate despite zero
overflow: weighted loss worsened `0.937047 -> 0.946383`, feature loss worsened
`0.641269 -> 0.642324`, and probe loss worsened `0.007394 -> 0.007601`.

## Read

This is real progress. Multi-center birth/split moves dense alpha coverage more
than the previous radius, opacity, or single-ellipse gates at the same cap:

- isotropic uncovered birth/split baseline: alpha `>0.1` `0.411`
- best intermediate single-center radius at cap128: `0.417` at `r88`
- best safe cap128 radius row: `0.420` at `low_alpha_n32_r96_cap128`
- single-line anisotropic ellipse: `0.408-0.409`
- multi-center `K=8`: `0.431`

The cost is lower target-background oracle (`23.965` versus `25.319` for
uncovered single-center `r64`), so it is not a visual-quality solution yet.
But it changes the right variable: spatial coverage under the same fixed tube
budget and cap128.

Next experiment should sweep multi-center support, not return to single-center
shape:

- `K=8` with radii `48/56/64/72` to find the oracle/coverage frontier
- `K=8` with opacity `0.4/0.6/0.8` to see if oracle can recover without losing
  all of the coverage gain
- maybe `K=16` only after checking tile count and loss stability

Current state: keep the active goal open. The next promising primitive is
multi-center fixed-budget birth/split.
