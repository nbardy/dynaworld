# PowerFoam Direct test helper import cleanup

## Context

After Direct PowerFoam started using the shared PowerFoam loss schedule,
`tests/test_powerfoam_direct.py` still imported `LOSS_DEFAULTS` and
`scheduled_loss_weights` through `train_powerfoam_direct.py`.

That kept the trainer file acting as a generic helper namespace even though the
owning modules now exist:

- Direct defaults: `powerfoam_direct_config.py`
- Shared schedule: `powerfoam_objectives.py`

## Change

- `tests/test_powerfoam_direct.py` imports `DIRECT_LOSS_DEFAULTS` from
  `powerfoam_direct_config.py`.
- The test uses `powerfoam_objectives.scheduled_loss_weights(...)` for both
  Direct and Metal schedule checks.
- `train_powerfoam_direct.py` now imports only `resolve_config(...)` from
  `powerfoam_direct_config.py`; it no longer imports all default dictionaries
  solely as compatibility exports.

## Validation

- `rg` confirmed no remaining `from train_powerfoam_direct import ...` helper
  imports.
- Focused PowerFoam-family pytest gate passed after the cleanup.
