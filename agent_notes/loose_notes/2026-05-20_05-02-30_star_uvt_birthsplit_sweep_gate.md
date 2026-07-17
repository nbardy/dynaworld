# STAR UVT Birth/Split Sweep Gate

Date: 2026-05-20 05:02 +07

## Why

After the uncovered-brightness target sampler passed mechanically but left alpha `>0.1` at `0.411`, the next gate was to sweep target source, reallocated tube count, support radius, and tile capacity.

## Harness

Added:

`research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py`

The script generates per-row configs under `outputs/benchmarks/*_work/configs/`, runs the STAR UVT feature trainer, runs the dense alpha/support diagnostic on passing rows, and writes JSON/Markdown summaries. It also sets `STAR_UVT_TILE_CAPACITY` to match each row's config; without that, cap-256 rows fail runtime validation before training.

## Results

Cap-128 64/128-tube sweep:

- Report: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row.md`
- Dense diagnostic: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row_dense_support.md`
- Result: all 64/128 reallocation rows fail the no-overflow gate at cap 128.
- `64` births: max/p95/cap `132/103/128`, overflow tiles `12`.
- `128` births: max/p95/cap `196/167/128`, overflow tiles `16384`.

Cap-256 64/128-tube sweep:

- Report: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row_cap256.md`
- Dense diagnostic: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row_cap256_dense_support.md`
- Result: all rows pass with zero overflow.
- Best alpha `>0.1`: `low_alpha_n128_r96_cap256` at `0.422`.
- That row: normal PSNR `5.878`, forced-alpha PSNR `14.603`, target-background oracle `23.623`, alpha `>0.5` `0.157`, max tile `196/256`, mean step `212.8ms`.

Safe cap-128 32-tube radius/source sweep:

- Report: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_n32_radius_cap128.md`
- Dense diagnostic: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_n32_radius_cap128_dense_support.md`
- Best alpha `>0.1`: `low_alpha_n32_r96_cap128` at `0.420`.
- That row: normal PSNR `5.825`, forced-alpha PSNR `14.591`, target-background oracle `24.226`, alpha `>0.5` `0.144`, max tile `100/128`, mean step `210.2ms`.
- Radius `32` rows are negative for coverage (`0.406-0.407` alpha `>0.1`) even though their oracle is higher (`25.704-25.716`).

## Read

This is real progress because it isolates the current birth/split lever:

- target source alone does not fix coverage
- wider support radius does increase dense alpha support
- cap 128 can keep `32` births but cannot keep `64+`
- cap 256 makes `64/128` births valid but does not give a qualitatively better coverage jump than `32/r96/cap128`
- the best coverage rows trade away target-background oracle, so this is still not a quality promotion

Next gate should not scale to 50 steps yet. The next useful short run is to test whether an intermediate radius (`72` or `80`) keeps most of the alpha coverage bump while recovering oracle/content, using `low_alpha`, `32` births, cap `128` first.
