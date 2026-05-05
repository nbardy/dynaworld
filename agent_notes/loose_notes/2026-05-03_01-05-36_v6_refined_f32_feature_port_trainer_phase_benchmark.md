# v6_refined F32 feature port and trainer phase benchmark

## What changed

This session focused on the narrow restart goal: make `v6_refined_features`
exercise the real v6 active-tile / adaptive-stop surface for arbitrary feature
channels, then add a trainer-side phase table so raster speed can be interpreted
against the rest of the step.

The fast-mac submodule changes are confined to
`third_party/fast-mac-gsplat/variants/v6_refined_features/`:

- `torch_gsplat_bridge_v6_refined_features/rasterize.py`
  - added active-tile config knobs for feature splatting
  - added adaptive stop-count metadata plumbing
  - chooses direct vs active paths for train and eval
  - exposes stop/active telemetry in `profile_projected_gaussians`
- `csrc/shared/common.h`, `csrc/bindings.cpp`, `csrc/metal/gsplat_metal.mm`
  - added active eval/state/backward operator declarations, dispatch, and Metal
    wrappers
- `csrc/metal/gsplat_v6_refined_features_kernels.metal`
  - added arbitrary-F active eval/state/backward kernels with alpha output and
    `grad_alpha`
- `tests/feature_contract_check.py`, `tests/alpha_output_check.py`
  - added active F32 feature and alpha/gradient checks
- `benchmarks/benchmark_mps.py`
  - added CLI switches for active policy and stop-count mode
- `README.md`, `ENGINEERING_NOTES.md`
  - updated the status from "not yet ported" to "ported but active mode is not
    globally faster"

The dynaworld root changes for this task are:

- `src/train/renderers/fast_mac.py`
  - forwards the v6 active/stop knobs only when
    `feature_variant == "v6_refined_features"` so the existing `v5_features`
    config surface stays compatible
- `src/benchmarks/trainer_phase_benchmark.py`
  - adds a one-step trainer phase benchmark for sample, encode, project,
    raster forward, loss, backward, and optimizer

## Validation

Build:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
```

Result: success, rebuilt `_C.cpython-311-darwin.so` for the active Python.

Python compile:

```bash
rtk /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python -m py_compile ...
```

Result: success for modified Python files and the new benchmark.

Feature contract:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/feature_contract_check.py )'
```

Result: success. The added active F32 gradient check reported max absolute
error `6.9849193e-10`; the F32 no-NaN smoke passed for both direct and active
policies.

Alpha output:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/alpha_output_check.py )'
```

Result: success. Tests A-F passed, including active-vs-direct F32 feature,
alpha, and gradient parity.

Reference check:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/reference_check.py )'
```

Result: success. Image, gradient, presorted, and overflow checks passed.

Trainer smokes:

```bash
rtk env PYTHONPATH=src/train WANDB_MODE=offline WANDB_SILENT=true /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python src/train/train.py /tmp/dynaworld_smoke_f3_v6_refined.jsonc
rtk env PYTHONPATH=src/train WANDB_MODE=offline WANDB_SILENT=true /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python src/train/train.py /tmp/dynaworld_smoke_f32_v6_refined_features.jsonc
rtk env PYTHONPATH=src/train WANDB_MODE=offline WANDB_SILENT=true /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python src/train/train.py /tmp/dynaworld_smoke_multicam_rgb_pyramid.jsonc
```

Result: all passed after patching the copied `/tmp` configs to
`"steps": 1`.

- F=3 fast-mac v6_refined: completed 1 step on
  `test_data/test_video_small_128_4fps.mp4`; final logged loss `0.3845`.
- F32 `v6_refined_features`: completed 1 step on
  `test_data/test_video_small_64_4fps.mp4`; final logged loss `0.2428`.
- Multicam rgb-pyramid smoke: cache hit, completed 1 step, and logged both
  train-view and heldout-camera eval metrics.

Whitespace:

```bash
rtk git diff --check -- src/train/renderers/fast_mac.py src/benchmarks/trainer_phase_benchmark.py
rtk git diff --check -- variants/v6_refined_features
```

Result: success.

## Timing notes

Small F32 dense-ish case:

```bash
benchmark_mps.py --height 512 --width 512 --gaussians 8192 --batch-size 2 --feature-dim 32 --case sparse_sigma_1_5 --backward --warmup 1 --iters 3 --profile --active-policy off --json
```

Result: mean `69.446 ms`, forward `18.100 ms`, backward `51.346 ms`.

The same case with `--active-policy on` was slower: mean `81.616 ms`, forward
`32.006 ms`, backward `49.610 ms`.

Clustered / overflow case:

```bash
benchmark_mps.py --height 512 --width 512 --gaussians 8192 --batch-size 2 --feature-dim 32 --case quadrant_cluster --backward --warmup 1 --iters 3 --profile --active-policy auto --json
```

Result: mean `191.715 ms`, forward `118.616 ms`, backward `73.099 ms`,
selected active path, active fraction `0.12598`, overflow tile count `64`,
mean stop ratio `0.06199`.

The same clustered case with `--active-policy off` was slightly faster overall:
mean `184.951 ms`, forward `99.300 ms`, backward `85.651 ms`.

Interpretation: the port is real and validated, but active mode is not a global
speed win on these small F32 MPS probes. Keep the knobs config-driven and select
from measured cases rather than promoting active globally.

## Trainer phase table

Command:

```bash
rtk env WANDB_MODE=disabled WANDB_SILENT=true PYTHONPATH=src/train /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python src/benchmarks/trainer_phase_benchmark.py src/train_configs/local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4_v6_refined_features.jsonc --warmup 1 --iters 2 --json-output benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_smoke.json
```

Result:

```text
| phase | mean_ms | median_ms | pct_total |
|---|---:|---:|---:|
| sample | 1.549 | 1.549 | 0.8% |
| encode | 69.707 | 69.707 | 34.1% |
| project | 7.013 | 7.013 | 3.4% |
| raster_forward | 10.540 | 10.540 | 5.2% |
| loss | 6.109 | 6.109 | 3.0% |
| backward | 106.511 | 106.511 | 52.1% |
| optimizer | 2.820 | 2.820 | 1.4% |
| total | 204.250 | 204.250 | 100.0% |
```

The smoke JSON was written to
`benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_smoke.json`.

The important readout is that on this tiny F32 smoke, `raster_forward` is only
about `5.2%` of the measured step while model backward and encode dominate.
Faster raster still matters for larger/high-overlap cases, but trainer-level
decisions should use the per-phase table instead of assuming raster is the
whole bottleneck.
