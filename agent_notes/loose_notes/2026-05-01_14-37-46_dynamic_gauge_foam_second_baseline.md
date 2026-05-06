# Dynamic Gauge Foam Second Baseline

## Context

The user provided `/Users/nicholasbardy/Downloads/dynamic_gauge_foam_scaffold.zip`
from another engineer and asked to run it alongside our PowerFoam work as a
second baseline.

The external scaffold is not PowerFoam. It is a dynamic disk-chart model:

- canonical centers
- time-varying SE(3) gauge frames
- per-primitive feature atlases
- RGB MLP conditioned on feature, view direction, normal, and time
- a Swift/Metal forward renderer sketch, but no Metal backward or Torch custom op

## Implementation

Ported the scaffold into repo-native training code:

- `src/train/dynamic_gauge_foam.py`
- `src/train/train_dynamic_gauge_foam.py`
- `src/train_configs/local_mac_dynamic_gauge_foam_video_1024_smoke.jsonc`

Added trainer dispatch:

- `arch: dynamic_gauge_foam` routes to `train_dynamic_gauge_foam`

Added smoke coverage:

- `tests/test_dynamic_gauge_foam.py`
- dispatch coverage in `tests/test_powerfoam_direct.py`

The local adaptation keeps the core SE(3) disk-chart/atlas/MLP idea but adds
our small-video initialization, W&B media logging, config schema, and output
artifact layout.

## Validation

`pytest` is not installed in the active uv/.venv environment, so the focused
tests were run manually:

```bash
PYTHONPATH=src/train rtk uv run python - <<'PY'
...
PY
```

Result:

```text
manual dynamic gauge tests passed
```

Syntax check passed:

```bash
rtk uv run python -m py_compile \
  src/train/dynamic_gauge_foam.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train.py \
  tests/test_dynamic_gauge_foam.py \
  tests/test_powerfoam_direct.py
```

Smoke after radius-initialization fix:

```text
64 prims, 32px, 2 frames, 1 step:
step 0 eval_l1 = 0.25302
step 1 eval_l1 = 0.22039
```

## W&B Runs

First online run before radius-init cleanup:

```text
run:  t3er0wwt
name: dynamic-gauge-foam-video-1024-120step
final Eval/L1 = 0.04516
final Eval/MSE = 0.00606
```

Superseding checked-code run:

```text
run:  b5h34jnk
name: dynamic-gauge-foam-video-1024-120step-radiusfix
url:  https://wandb.ai/nbardy/dynaworld/runs/b5h34jnk
final Eval/L1 = 0.04515
final Eval/MSE = 0.00603
train-loop wall = 12.91 s
```

For comparison, the frame-aware PowerFoam Metal run was:

```text
run:  ecysgsk8
name: powerfoam-metal-material-frame-1024-120step
final Eval/L1 = 0.01984
final Eval/MSE = 0.00143
train-loop wall = 3.08 s
```

## Takeaway

The dynamic gauge scaffold is valuable as a representation baseline, but the
current direct-fit result is worse than our PowerFoam Metal path on the tiny
same-source video at the same 1024 primitive / 120 step / 64px budget.

It starts from a poor blurry init (`Eval/L1 ~= 0.279`) and learns quickly to
`0.045`, but does not approach the bounded-cell PowerFoam run (`0.01984`).
The current bottleneck is not Swift/Metal availability; it is that the trainable
path is still a Torch reference disk-chart renderer, not a memory-safe Metal
backward path, and the disk-chart primitive is less geometrically constrained
than bounded power cells.

## Next

- Inspect W&B videos for whether failure is blur, wrong motion, or opacity wash.
- Try a longer run or stronger RGB/atlas initialization only if the videos look
  promising.
- If we merge concepts, use their SE(3) temporal controls on top of our bounded
  power-cell rasterizer rather than replacing the rasterizer with disk charts.
