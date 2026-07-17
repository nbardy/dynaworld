# PowerFoam Training Data Boundary

## Context

Continuation of the trainer modularization goal. After the optimizer,
resampling, checkpoint, and eval-render helpers were split out, the remaining
large pure helper in `train_powerfoam_metal.py` was
`load_powerfoam_training_data(...)`.

That function was a useful boundary to extract because it does not depend on
the Metal model or rasterizer extension. It only normalizes PowerFoam training
input data into the existing dict contract:

- `targets`
- `sample_frame_indices`
- optional `sample_rays`
- optional heldout targets/frame indices/rays
- init frames, frame count, source label, FPS
- train/heldout view names and pose source
- multicam world-to-model and point-cloud visibility metadata

The important constraint was not to hide same-view versus heldout semantics.
The extracted helper keeps the existing explicit keys rather than introducing
a vague loader abstraction.

## Changes

- Added `src/train/powerfoam_training_data.py`.
  - Owns `load_powerfoam_training_data(...)`.
  - Handles both `frame_source == "multicam_val"` and explicit-video inputs.
  - Keeps the old dict shape unchanged.
- Updated `src/train/train_powerfoam_metal.py` to import and re-export
  `load_powerfoam_training_data(...)` for compatibility.
- Rerouted direct helper imports in:
  - `research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py`
  - `research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py`
  - `research_experiments/world_foam_lane2/gate1_realdata_feeder_smoke.py`
  - `research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py`
- `gate4_moving_ray_slab_compiler.py` and downstream World Foam Lane 2 probes
  continue to receive the same callable through the Gate 1 reference module,
  but the source is now the light training-data helper rather than the full
  PowerFoam Metal trainer.

## Validation

Commands run from repo root:

```bash
rtk .venv/bin/python -m py_compile src/train/powerfoam_training_data.py src/train/train_powerfoam_metal.py research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py research_experiments/world_foam_lane2/gate1_realdata_feeder_smoke.py research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -c "from powerfoam_training_data import load_powerfoam_training_data; import train_powerfoam_metal as m; print('powerfoam_training_data_import_ok', load_powerfoam_training_data is m.load_powerfoam_training_data)"
rtk env PYTHONPATH=src/train:research_experiments/world_foam_lane2 .venv/bin/python -c "import gate1_realray_per_sample_reference as g1; import gate4_moving_ray_slab_compiler as g4; print('worldfoam_loader_imports_ok', g1.load_powerfoam_training_data is g4.load_powerfoam_training_data)"
rtk env PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal .venv/bin/python -c "import diagnose_powerfoam_color_affine as c; import diagnose_powerfoam_heldout_error as h; print('dynamic_foam_loader_imports_ok', callable(c.load_powerfoam_training_data), callable(h.load_powerfoam_training_data))"
rtk env PYTHONPATH=src/train:research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/gate1_realdata_feeder_smoke.py --max-frames 1 --render-size 8
```

Results:

- `py_compile` passed.
- Trainer compatibility smoke printed `powerfoam_training_data_import_ok True`.
- World Foam import smoke printed `worldfoam_loader_imports_ok True`.
- Dynamic Foam import smoke printed `dynamic_foam_loader_imports_ok True True`.
- Gate 1 real-data feeder smoke exited `0` with `status: ok`, one frame,
  render size 8, train views `camera_0001`/`camera_0015`, heldout view
  `camera_0040`, finite train/heldout targets/rays, and distinct train-view
  rays.

## Next

The remaining PowerFoam Metal trainer code is now more model/training-loop
specific. Continue with live-file checks before extracting anything else.
Good candidates to inspect are eval artifact payload construction and any
remaining diagnostics that import the full trainer for non-model helpers.
