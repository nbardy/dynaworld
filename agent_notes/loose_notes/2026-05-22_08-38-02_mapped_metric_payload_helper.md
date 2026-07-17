# Mapped Metric Payload Helper

## Context

After the RGB+alpha media cleanup, the adjacent W&B scalar payloads still had
large repeated blocks of:

```python
payload["Some/WandbKey"] = metrics["some_metric_key"]
```

Some fields were required and should still fail loudly if missing; other fields
were branch-optional and should be skipped when absent. The active repeats were
in Direct PowerFoam, shared PowerFoam eval artifacts, Dynamic PowerFoam Metal,
and Dynamic Gauge Foam.

## Changes

- `src/train/train_logging.py`
  - Added `mapped_metric_payload(metrics, key_map, require=True)`.
  - Required maps preserve the old `KeyError` behavior for missing required
    metrics.
  - Optional maps use `require=False` and skip absent metrics.
- `src/train/train_powerfoam_direct.py`
  - Uses the helper for eval and heldout eval scalar keys.
- `src/train/powerfoam_eval_artifacts.py`
  - Uses the helper for required eval/state keys and optional heldout/state
    extension keys.
- `src/train/train_dynamic_powerfoam_metal.py`
  - Uses the helper for required eval/state keys and optional temporal/camera
    extension keys.
- `src/train/train_dynamic_gauge_foam.py`
  - Uses the helper for required eval/state keys.
- `tests/test_train_logging.py`
  - Covers required-missing and optional-skip behavior.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/train_logging.py tests/test_train_logging.py src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_train_logging.py tests/test_wandb_media.py tests/test_video_io.py tests/test_powerfoam_direct.py tests/test_dynamic_gauge_foam.py tests/test_dynamic_powerfoam_metal.py -q'`
  - `100 passed, 1 skipped`

## Notes

This keeps the metric schemas local to each trainer family. The shared behavior
is only the copy loop and the required-versus-optional missing-key policy.
