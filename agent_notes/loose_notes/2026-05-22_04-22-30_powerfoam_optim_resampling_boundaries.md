# PowerFoam Optim And Resampling Boundaries

## Context

Continuation of the trainer modularization goal. The PowerFoam Metal trainer
had already shed config, raster-config, geometry, adjacency, point-cloud,
objective, eval-color, and training-primitive helpers into smaller modules.
The next live-file scan showed four remaining pure contracts embedded in
`src/train/train_powerfoam_metal.py`:

- optimizer LR schedule helpers
- official-style resample cadence / target-cell schedule helpers
- checkpoint/best-metric artifact helpers
- no-grad eval sample rendering helper

The first two were trainer-independent and already had focused tests in
`tests/test_powerfoam_direct.py`. The checkpoint helpers were validated with a
runtime smoke instead of adding a permanent unit test.

## Changes

- Added `src/train/powerfoam_optim.py`.
  - Owns `cosine_scheduled_lr(...)`.
  - Owns `powerfoam_group_initial_lr(...)`,
    `powerfoam_group_final_lr(...)`,
    `powerfoam_group_warmup_steps(...)`, and
    `powerfoam_group_lr_metadata(...)`.
  - Owns `update_powerfoam_learning_rates(...)`.
- Added `src/train/powerfoam_resampling.py`.
  - Owns `scheduled_resample_target_cells(...)`.
  - Owns `should_resample_powerfoam_step(...)`.
- Added `src/train/powerfoam_checkpoints.py`.
  - Owns `select_best_metric(...)`.
  - Owns `save_powerfoam_checkpoint(...)`.
  - Owns `maybe_save_best_powerfoam_checkpoint(...)`.
- Added `src/train/powerfoam_eval_render.py`.
  - Owns `render_powerfoam_samples(...)`.
- Updated `src/train/train_powerfoam_metal.py` to import and re-export those
  names for compatibility.
- Updated the color-affine and heldout-error diagnostics to import
  `render_powerfoam_samples(...)` from the light helper instead of reaching
  through `train_powerfoam_metal.py` for the no-grad renderer.
- Updated the two focused schedule tests to import the pure helpers directly,
  with config defaults from `powerfoam_metal_config.py`. These tests no longer
  need to import the full Metal trainer just to prove LR and resample schedule
  behavior.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  with the new helper boundaries.

## Validation

Commands run from repo root:

```bash
rtk .venv/bin/python -m py_compile src/train/powerfoam_optim.py src/train/powerfoam_resampling.py src/train/train_powerfoam_metal.py tests/test_powerfoam_direct.py
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q -k "lr_schedule or resample_schedule"
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -c "import train_powerfoam_metal as m; print('powerfoam_metal_import_ok', m.cosine_scheduled_lr(1.0, 0.0, 1, 2), m.scheduled_resample_target_cells({'resample_target_cells': 3, 'resample_final_cells': None, 'resample_from_step': 0, 'resample_until_step': None}, initial_cells=1, current_cells=1, step=0, total_steps=1))"
rtk .venv/bin/python -m py_compile src/train/powerfoam_checkpoints.py src/train/train_powerfoam_metal.py
rtk env PYTHONPATH=src/train .venv/bin/python -c "from pathlib import Path; import tempfile, json, torch; from torch import nn; from powerfoam_checkpoints import maybe_save_best_powerfoam_checkpoint, select_best_metric; print('select', select_best_metric({'eval_psnr': 1.0, 'heldout_eval_psnr': 2.0})); d=Path(tempfile.mkdtemp()); m=nn.Linear(1,1); v=maybe_save_best_powerfoam_checkpoint(m, {'logging': {'output_dir': str(d)}}, d, step=3, metrics={'eval_psnr': 1.25}, best_metric_value=None); print('checkpoint_smoke', v, (d/'checkpoint_best.pt').exists(), json.loads((d/'best_metrics.json').read_text())['best_metric_name'])"
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -c "import train_powerfoam_metal as m; print('powerfoam_checkpoint_aliases_ok', m.select_best_metric({'eval_psnr': 4.0})[0])"
rtk .venv/bin/python -m py_compile src/train/powerfoam_eval_render.py src/train/train_powerfoam_metal.py research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -c "import train_powerfoam_metal as m; print('render_alias_ok', callable(m.render_samples))"
rtk env PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal .venv/bin/python -c "import diagnose_powerfoam_color_affine as c; import diagnose_powerfoam_heldout_error as h; print('powerfoam_diagnostics_import_ok', callable(c.render_powerfoam_samples), callable(h.render_powerfoam_samples))"
```

Results:

- `py_compile` passed.
- Focused pytest: `2 passed, 43 deselected in 2.95s`.
- Compatibility import smoke printed `powerfoam_metal_import_ok 0.5 3`.
- Checkpoint runtime smoke printed `checkpoint_smoke 1.25 True eval_psnr`.
- Checkpoint compatibility import smoke printed
  `powerfoam_checkpoint_aliases_ok eval_psnr`.
- Eval-render compatibility smoke printed `render_alias_ok True`.
- Diagnostic import smoke printed `powerfoam_diagnostics_import_ok True True`.

The `uv run` command emitted the known parent `pyproject.toml` warning about
the missing `[project]` table in `/Users/nicholasbardy/git/gsplats_browser`;
the command still exited successfully.

## Next

Continue live-file-driven cleanup. Good next candidates are not another base
trainer class; they are remaining pure helper surfaces such as checkpoint/best
metric helpers, render/eval artifact helpers, or data-loading boundaries if
they can be separated without hiding same-view versus heldout semantics.
