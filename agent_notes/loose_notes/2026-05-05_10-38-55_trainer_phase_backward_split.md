# Trainer phase benchmark backward split

## Why this changed

The original `trainer_phase_benchmark.py` row named `backward` was easy to
misread as raster backward. It was actually the whole `loss.backward()` call for
the training graph: loss/colorizer, raster, projection, model, regularizers, and
autograd/MPS scheduling.

The row is now named `autograd_backward_total` in the main training-step table.
That table still measures a real optimizer step.

## Split strategy

Added `--backward-breakdown` for separate diagnostic VJP probes:

- `loss_colorize_backward_probe`
- `raster_backward_probe`
- `project_backward_probe`
- `model_backward_probe`
- `regularizer_backward_probe`

These probes are deliberately separate from the optimizer-step timing. They use
autograd gradients at detach-style boundaries so the rows do not pretend to be
one physical backward pass. The JSON payload includes the same warning:
breakdown rows are not expected to sum exactly to `autograd_backward_total`.

The project/model split includes camera scalar and transform inputs when those
tensors require grad, so implicit-camera configs do not silently drop camera-head
projection gradients from the model probe.

## Validation

Compile:

```bash
rtk /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python -m py_compile src/benchmarks/trainer_phase_benchmark.py
```

F32 feature-splat smoke:

```bash
rtk env WANDB_MODE=disabled WANDB_SILENT=true PYTHONPATH=src/train /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python src/benchmarks/trainer_phase_benchmark.py src/train_configs/local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4_v6_refined_features.jsonc --warmup 1 --iters 2 --backward-breakdown --json-output benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_split_warm.json
```

Warm result:

```text
Training step phases
| sample | 0.750 ms | 0.8% |
| encode | 31.510 ms | 32.1% |
| project | 4.930 ms | 5.0% |
| raster_forward | 8.275 ms | 8.4% |
| loss | 3.529 ms | 3.6% |
| autograd_backward_total | 47.867 ms | 48.7% |
| optimizer | 1.407 ms | 1.4% |
| total | 98.269 ms | 100.0% |

Backward breakdown probes
| loss_colorize_backward_probe | 8.222 ms | 16.9% |
| raster_backward_probe | 8.263 ms | 17.0% |
| project_backward_probe | 6.737 ms | 13.8% |
| model_backward_probe | 18.754 ms | 38.5% |
| regularizer_backward_probe | 6.743 ms | 13.8% |
| total | 48.719 ms | 100.0% |
```

RGB fast-mac smoke:

```bash
rtk env WANDB_MODE=disabled WANDB_SILENT=true PYTHONPATH=src/train /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python src/benchmarks/trainer_phase_benchmark.py src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_v6_refined_8192splats.jsonc --warmup 1 --iters 1 --backward-breakdown --json-output benchmark_outputs/trainer_phase/unconditioned_rgb_v6_refined_split_smoke.json
```

Result: passed. This checked the F=3 path where there is no alpha output and no
feature colorizer.
