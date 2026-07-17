# STAR UVT Kernel Refresh And CLI Completion

Date: 2026-07-18 00:02:37 +09

## Goal

Finish the pending trainer CLI entrypoint cleanup, refresh practical STAR UVT
backward timing evidence, and make the renderer-scaling report usable after
historical benchmark-artifact retention changed.

## Completed Code Work

- `src/train/train_cli.py` now has `run_config_main(...)`, which accepts a
  config dict/path for programmatic callers or parses the CLI path when called
  with `None`.
- The thin Token-GS, multicam, PowerFoam, Dynamic Gauge, and STAR UVT wrapper
  entrypoints now use that one route. Their runner and command usage strings
  are unchanged.
- `tests/test_train_cli.py` covers dict, path, and CLI-argument behavior.
- `research_experiments/renderer_scaling_report.py` now tolerates absent
  historical CSV/JSONL sources and ingests current STAR kernel matrix CSVs.
  Missing sources are printed as `missing; omitted`, rather than failing the
  whole report.

## Validation

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_cli.py tests/test_trainer_registry.py -q
25 passed in 3.44s
```

`py_compile` for all changed wrappers/report code and `git diff --check` also
passed.

## Fresh GPU Evidence

All screen-space rows use Apple M4/MPS, 64 frames, 8192 RGB tubes, three timed
repeats after one warmup:

| resolution | fastest measured practical backward | direct atomic backward | artifact |
| --- | ---: | ---: | --- |
| 128 | direct fixedpoint, 38.2 ms | 51.0 ms | `outputs/benchmarks/2026-07-17_star_uvt_backward_matrix_128px_64f_8192t/summary.md` |
| 256 | direct atomic, 116.6 ms | 116.6 ms | `outputs/benchmarks/2026-07-17_star_uvt_backward_practical_256px_64f_8192t/summary.md` |
| 512 | direct serial, 79.6 ms | 263.5 ms | `outputs/benchmarks/2026-07-17_star_uvt_backward_practical_512px_64f_8192t/summary.md` |

The 128px all-mode matrix timed 17/19 modes. `atomic_append` and `with_keys`
both OOMed trying to allocate another 3 GiB after MPS had allocated about
26 GiB; retain those as explicit negative rows in the source matrix.

Projective STAR UVT is not valid at the same high-tube setting: train-step tile
configuration is verified only through 512 tubes and its fused-MSE policy only
through 1024. At the supported 256px/64f/512t envelope, tile-pixel atomic
backward is 377.8 ms and fused MSE is 362.9 ms; direct serial and tile-pair
exceeded the 45-second timeout. This is a bounded research/correctness route,
not the high-tube fast trainer.

## Current State And Next Decision

`direct_atomic + index_add` remains the robust RGB STAR trainer default. The
512px direct-serial result is interesting but needs a trainer-level parity and
repeat gate before changing the default, because this is an isolated kernel
probe and the ranking reverses across resolutions.

The existing feature-tube performance evidence remains current code context:
the best practical feature training surface is batched target/probe VJP, but
single-video feature media quality is still the blocker. Do not spend the next
GPU block repeating the background ablation: its 100-step/128px/256px records
already show renderer- and resolution-specific choices. The next experiment
should be a short trainer-level `direct_serial` versus `direct_atomic` 512px
parity/timing gate, followed by the selected visual-quality bridge only if it
does not regress the old objective.

The matching `64f/8192` dynamic RGB and F32 feature raster rows were then
refreshed at 128/256/512px and are stored at
`outputs/benchmarks/2026-07-18_fastmac_rgb_f32_64f_8192t_128_256_512.jsonl`.
The consolidated report now reads them. Historical 32768-tube inputs remain
absent, so the new all-renderer table is an 8192-primitive benchmark rather
than an invented reconstruction of the old 32768 table.
