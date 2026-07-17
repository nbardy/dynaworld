# Fused Adam Optimizer Helper

## Goal

Continue trainer modularization by extracting only the optimizer behavior that
was truly repeated across active trainers.

## Audit Result

- Base Token-GS and relative-pose-only training both constructed
  `torch.optim.Adam(..., fused=self.device.type in {"cuda", "mps"})`.
- PowerFoam, Dynamic PowerFoam, Dynamic Gauge Foam, STAR UVT, and gauge-field
  research scripts have different optimizer contracts: param-group schedules,
  AdamW, camera/colorizer LR multipliers, probe-specific trainable subsets, or
  argparse-first research semantics.
- Therefore the safe helper is only the Token-GS fused-Adam policy, not a broad
  universal optimizer factory.

## Changed

- Added `src/train/train_optim.py`.
- Added `adam_with_device_fused(params, *, lr, device)`.
- Updated `Trainer.__init__` in
  `src/train/train_video_token_implicit_dynamic.py`.
- Updated relative-pose-only scope in
  `src/train/train_multicam_relative_pose_implicit_dynamic.py`.
- Added `tests/test_train_optim.py`.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_optim.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  tests/test_train_optim.py
```

Passed.

First pytest attempt used a stale relative-pose test node and failed with
`not found`; no code failure occurred.

Corrected focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_optim.py \
  tests/test_config_factory_helpers.py::test_train_router_accepts_star_uvt_video_overfit_config \
  -q
```

Passed: `3 passed in 4.60s`.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_multicam_relative_pose_trainer.py -q
```

Passed: `13 passed in 3.13s`.

## Remaining

Do not broaden this into a universal optimizer factory unless a future trainer
really shares the same param-group and LR-multiplier semantics. The current
useful boundary is the small device-fused Adam policy for Token-GS-style
optimizers.
