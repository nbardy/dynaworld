# PowerFoam Direct Loss Objective Boundary

## Context

The trainer cleanup pass is trying to keep train files as orchestration while
moving repeated or pure math into owning helper modules. Direct PowerFoam had
already moved defaults to `powerfoam_direct_config.py`, shared schedules to
`powerfoam_objectives.py`, shared training data to `powerfoam_training_data.py`,
and shared eval rendering/artifacts to helper modules, but its train loop still
kept the full direct loss formula inline as `compute_powerfoam_loss(...)`.

## Change

- Moved the Direct PowerFoam loss assembly into
  `powerfoam_objectives.direct_powerfoam_loss(...)`.
- Updated `train_powerfoam_direct.py` to call that objective helper from the
  train loop.
- Removed the trainer-local `torch.nn.functional` and `ssim_per_image` imports
  that only supported the inline loss formula.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md` so future cleanup treats Direct
  render orchestration as trainer-local but loss math as objective-module owned.

## Why

This keeps another behavior-bearing formula out of the train-loop file without
changing the schedule, metrics keys, or artifact paths. It is a small boundary
cleanup, not a convergence or quality claim.

## Validation

- `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py tests/test_train_artifacts.py -q`
  passed: `51 passed, 1 skipped`.
- `py_compile` passed for `powerfoam_objectives.py`,
  `train_powerfoam_direct.py`, and `tests/test_powerfoam_direct.py`.
- `git diff --check` passed for the tracked touched files.
- Trailing-whitespace scan passed for the touched tracked files plus this loose
  note and `CODE_ORGANIZATION.md`.
