# Token-GS Low-Use Config Alias Cleanup

## What changed

- `src/train/train_video_token_implicit_dynamic.py` no longer stores thin
  `Trainer.__init__` copies for:
  - `feature_dim`
  - `recon_backward_strategy`
  - `temporal_microbatch_size`
  - `profile_timing_sync`
  - `profile_timing_log_every`
  - `profile_backward_split`
- The trainer now reads those normalized values from `self.model_cfg` or
  `self.train_cfg` at the use sites.
- Heavily used section aliases such as `self.model_cfg`, `self.train_cfg`,
  `self.logging_cfg`, and `self.loss_cfg` remain. The cleanup target was
  one-off or two-off value aliases, not useful section handles.

## Why

This follows the current trainer unification rule: do not create a new
intermediate name solely to thread a config value across a function boundary or
store it once in `__init__`. Keeping the value at its normalized source makes
it harder for defaults, validators, and status prints to drift.

## Validation

```bash
rtk .venv/bin/python -m py_compile src/train/train_video_token_implicit_dynamic.py
rtk rg -n "self\\.(recon_backward_strategy|temporal_microbatch_size|profile_timing_sync|profile_timing_log_every|profile_backward_split|feature_dim)" src/train/train_video_token_implicit_dynamic.py
```

The compile passed. The alias scan returned no remaining Token-GS references
for those fields.
