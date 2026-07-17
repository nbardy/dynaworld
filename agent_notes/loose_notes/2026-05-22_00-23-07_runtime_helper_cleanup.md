# Runtime Helper Cleanup

## Context

After the artifact-writer cleanup, the live train/probe scan still showed
small runtime helper drift:

- `run_alpha_background_ablation.py` manually imported W&B only to finish the
  trainer run.
- PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic Gauge
  Foam each kept a local one-line `resolve_device(...)` wrapper around
  `train_devices.resolve_torch_device(...)`.
- `run_dust3r_video.py` had its own MPS/CUDA/CPU auto-device branch.

## Change

- `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`
  now calls `train_logging.finish_wandb_run(...)`.
- `src/train/train_powerfoam_direct.py`,
  `src/train/train_dynamic_powerfoam_metal.py`,
  `src/train/train_powerfoam_metal.py`, and
  `src/train/train_dynamic_gauge_foam.py` now call
  `resolve_torch_device(...)` directly at their run entrypoints. The
  PowerFoam-family trainers keep `auto_cuda=False`; Dynamic Gauge keeps
  `auto_cuda=True`.
- `src/train/run_dust3r_video.py` now delegates auto-device selection to
  `resolve_torch_device(..., auto_cuda=True)` and returns the string expected by
  DUSt3R.

## Validation

Validation passed:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/run_dust3r_video.py \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  src/train/train_devices.py \
  src/train/train_logging.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_devices.py tests/test_train_logging.py -q

PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py tests/test_multicam_video_data.py -q

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py --help
```

The focused tests passed with 16 train-device/logging tests and 25 PowerFoam
tests passing with 30 expected skips. Targeted scans found no remaining local
`resolve_device(...)` wrappers in the four PowerFoam-family trainer files and no
remaining `wandb.finish()` call in the alpha-background ablation.

## STAR Kernel Benchmark Follow-up

The STAR UVT feature-kernel benchmark cluster now uses shared helpers too:

- `direct_feature_kernel_benchmark.py`
- `feature_autograd_overfit_benchmark.py`
- `sparse_hidden_sigmoid_mse_kernel_benchmark.py`
- `sparse_hidden_target_area_kernel_benchmark.py`

Each still owns its kernel-specific parity and timing code, but local
`torch.mps.synchronize()` calls now route through `sync_torch_device(...)`, and
`--out-json` writes now route through `write_report_json(...)`.

Validation passed:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py \
  research_experiments/star_uvt_feature_tubes/sparse_hidden_sigmoid_mse_kernel_benchmark.py \
  research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  src/train/train_devices.py

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4 --skip-timing --out-json /tmp/direct_feature_kernel_benchmark_shared_helpers.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/sparse_hidden_sigmoid_mse_kernel_benchmark.py \
  --feature-dims 4 --hidden-dim 4 --skip-timing --out-json /tmp/sparse_hidden_sigmoid_mse_shared_helpers.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 4 --hidden-dim 4 --skip-timing --out-json /tmp/sparse_hidden_target_area_shared_helpers.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py \
  --frames 2 --size 8 --tubes 3 --feature-dim 4 --steps 2 --chunk-size 1 \
  --tile-capacity 128 --out-json /tmp/feature_autograd_overfit_shared_helpers_2step.json

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_devices.py tests/test_star_uvt_report_artifacts.py -q
```

The three parity-only kernel checks reported `pass=true`, the tiny feature
autograd overfit reported `pass=true`, all four `/tmp` JSON artifacts were
written, and the focused pytest slice passed with 10 tests.
