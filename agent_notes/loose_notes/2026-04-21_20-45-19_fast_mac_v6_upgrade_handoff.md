# fast-mac v6 upgrade handoff

## Context

The chief scientist sent two renderer bundles through the upload folder:

```text
torch_metal_gsplat_v6_upgrade.tar.gz
torch_metal_gsplat_v7_hardware__1_.tar.gz
```

The v7 hardware upload had the same SHA-256 as the source artifact already
preserved in `third_party/fast-mac-gsplat/source_artifacts`, so the locally
fixed `variants/v7` was not overwritten.

The v6 upgrade archive was new and was added to fast-mac-gsplat as:

```text
source_artifacts/torch_metal_gsplat_v6_upgrade.tar.gz
variants/v6_upgrade/
```

## Integration Choice

`variants/v6_upgrade` was kept alongside `variants/v6` instead of replacing it.
Reason: existing `variants/v6` already had local engineering fixes that were not
all present in the handoff, including tighter active-policy behavior and
pair-budget chunking. Replacing current v6 with the raw upgrade would have lost
those decisions.

The full benchmark harness was updated to include:

```text
v6_upgrade_direct
v6_upgrade_auto
```

The harness now lazily imports each renderer inside the per-cell subprocess.
That matters because `variants/v6` and `variants/v6_upgrade` both expose the
same Python package name and Torch op namespace:

```text
torch_gsplat_bridge_v6
gsplat_metal_v6
```

Subprocess isolation makes side-by-side benchmark rows possible without trying
to load both op libraries into one Python process.

## Bug Found During Audit

The same saturated-backward Metal bug found in v5 was also present in current
v6 and the new v6 upgrade handoff.

Pattern:

```text
for chunk_end in end_i:
    threadgroup_barrier(...)
```

`end_i` is per pixel because forward compositing stops when transmittance
saturates. In a crowded tile, different pixels can stop at different Gaussian
indices. A threadgroup barrier inside a per-pixel loop bound is invalid control
flow and can silently corrupt gradients.

Patched both `variants/v6` and `variants/v6_upgrade`:

```text
tile_fast_backward_saved:   loop over uniform stop_count
tile_active_backward_saved: loop over uniform stop_count
tile_overflow_backward:     loop over uniform count
```

The per-pixel condition remains as `global_i < end_i` inside the loop.

## Verification

Both variants built:

```bash
cd third_party/fast-mac-gsplat/variants/v6
python setup.py build_ext --inplace

cd ../v6_upgrade
python setup.py build_ext --inplace
```

`variants/v6_upgrade` produced two existing unused-variable warnings in
`gsplat_metal.mm`, but no build failure.

Both variants passed reference checks after adding saturated 64-splat cases:

```bash
python tests/reference_check.py
```

Representative saturated errors:

```text
image:          2.086162567138672e-07
means grad:     2.3283064365386963e-10
conics grad:    1.1920928955078125e-07
colors grad:    3.725290298461914e-09
opacities grad: 9.313225746154785e-10
```

Full-matrix smoke:

```bash
python benchmarks/benchmark_full_matrix.py \
  --resolutions 128x128 \
  --splats 64 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random,clustered_hot_tiles \
  --renderers v6_direct,v6_upgrade_direct,v6_upgrade_auto,v7_hardware \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 90 \
  --output-md benchmarks/full_rasterizer_benchmark_v6_upgrade_smoke.md
```

All 16 cells returned `status=ok`.

## Smoke Results

| Distribution | Mode | Winner | Mean ms |
|---|---|---|---:|
| uniform random | forward | v6_direct | 4.296 |
| uniform random | forward+backward | v6_direct | 6.133 |
| clustered hot tiles | forward | v6_direct | 4.231 |
| clustered hot tiles | forward+backward | v7_hardware | 9.323 |

Do not overread those numbers. It was a small integration smoke, not the full
marketing benchmark.

## Next Decision

Keep `variants/v6` as the current v6 baseline. Treat `variants/v6_upgrade` as a
preserved handoff and benchmark target. Only replace or merge the two if a larger
quiet-machine benchmark shows the upgrade handoff beats the locally evolved v6.
