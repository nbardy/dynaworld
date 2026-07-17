# W&B Media Boundary Split

## Goal

Continue trainer modularization by separating scalar/logging policy from
low-level W&B media constructors. After the previous scalar-payload move,
`train_logging.py` still owned `make_wandb_video(...)`,
`make_preview_image(...)`, and `build_validation_video_payload(...)`, which made
the logging module a mixed policy/media helper.

## Changed

- Added `src/train/wandb_media.py`.
- Moved these helpers from `train_logging.py`:
  - `make_wandb_video(...)`
  - `make_preview_image(...)`
  - `build_validation_video_payload(...)`
- Updated PowerFoam-family trainers and `pipeline.validation_media` to import
  media helpers from `wandb_media`.
- Left `train_logging.py` responsible for W&B run init, log cadence, scalar
  payloads, row-output flattening, and existing-file W&B media attachment.
- Added `tests/test_wandb_media.py` to protect the `Render_Video` and
  `Render_GT_Video` side-by-side payload contract.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/wandb_media.py \
  src/train/train_logging.py \
  src/train/pipeline/validation_media.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_wandb_media.py \
  tests/test_train_logging.py \
  tests/test_pipeline_helpers.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_wandb_media.py tests/test_train_logging.py tests/test_pipeline_helpers.py -q
```

Passed: `16 passed in 2.56s`.

## State After This Slice

- `train_logging.py` is now narrower: W&B run init, cadence, scalar payloads,
  row metric flattening, and existing artifact attachment.
- `wandb_media.py` is the low-level W&B tensor-to-media boundary.
- `pipeline.validation_media` owns higher-level trainer validation payload
  assembly.
- PowerFoam-family trainers still choose their metric/media keys locally, but
  they no longer import image/video constructors from the scalar logging module.

## Remaining

- The next cleanup should probably target render-dispatch convergence or the
  `pipeline.render.prepare_clip` compatibility re-export, after checking current
  imports.
- Do not delete legacy trainers until config usage is audited.
