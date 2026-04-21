# Debug Metric Sets

Refactored inline 128px NaN diagnostics out of `dynamicTokenGS.py` into
`src/train/debug_metrics.py`.

The new module provides:

- `MetricConfig` and `metric_config_from_logging(...)`
- `with_metrics(config, renderer=..., optimizer=..., every=..., ...)` for
  in-memory trainer config overrides
- `dense_render_diagnostics(...)` for renderer/projection metrics
- `optimizer_diagnostics(...)` for grad/parameter summaries
- summary formatting and print helpers

The trainer now reads `logging.with_metrics`:

```jsonc
"with_metrics": {
  "renderer": true,
  "optimizer": false,
  "every": 25,
  "print_summary": true,
  "wandb": true,
  "fail_fast": true
}
```

`local_mac_overfit_prebaked_camera_128_4fps.jsonc` now enables renderer metrics
with summaries and fail-fast enabled. A disabled-W&B smoke with
`with_metrics(..., renderer=true, optimizer=true, every=1, wandb=false)` caught
the expected 128px failure on frames `[39, 40, 41, 42]` and printed:

- `RenderDiag/CameraZMin=-1.052`
- `RenderDiag/FrontGaussiansMin=20`
- `RenderDiag/NearOrBehindGaussiansMean=457.8`
- `RenderDiag/PowerMax=1.521e31`
- `RenderDiag/AlphaPreNonfiniteCount=3.797e6`

No stabilization behavior was changed. The new path only makes metric collection
and fail-fast reporting reusable.

Then ran the real 128px training command:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh src/train_configs/local_mac_overfit_prebaked_camera_128_4fps.jsonc
```

W&B run: `https://wandb.ai/nbardy/dynaworld/runs/jn8q1e6s`

It failed at step 1, before optimizer update, on frames `[35, 36, 37, 38]`:

- `RenderDiag/RenderNonfiniteCount=9.83e4`
- `RenderDiag/CameraZMin=-0.7978`
- `RenderDiag/CameraZMax=1.008`
- `RenderDiag/FrontGaussiansMin=106`
- `RenderDiag/NearOrBehindGaussiansMean=195.8`
- `RenderDiag/PowerMax=1.014e31`
- `RenderDiag/PowerGt80Count=7.035e5`
- `RenderDiag/AlphaPreNonfiniteCount=7.035e5`
- `RenderDiag/RawDetMin=-2.535e30`
- `RenderDiag/RawDetNegativeCount=26`

This confirms the actual training run hits the same renderer/projection failure
mode found by the diagnostic probes.
