# PowerFoam Direct Loader And Render Reuse

## Context

Continuation of the trainer modularization goal. After the Metal trainer data
and eval-artifact boundaries were split out, `train_powerfoam_direct.py` still
had two duplicated helper bodies:

- a local single-video / multicam training-data loader
- a no-grad batch render loop that differed from the Metal eval renderer only
  because `DirectPowerFoamVideo.forward(...)` returns extra tensors after
  `(rendered, alpha)`

The goal was to reuse the shared helper modules without changing Direct's
public imports or the old training-data key set.

## Changes

- Updated `powerfoam_eval_render.render_powerfoam_samples(...)` to accept
  PowerFoam-style tuple/list outputs with extra values and use the first two
  tensors as `(rendered, alpha)`.
- Updated `train_powerfoam_direct.render_all(...)` to call
  `render_powerfoam_samples(...)`.
- Updated `train_powerfoam_direct.load_direct_powerfoam_training_data(...)` to
  call `powerfoam_training_data.load_powerfoam_training_data(...)` and then
  prune to the historical Direct key set.
- Restored Direct compatibility re-exports for
  `flatten_multiview_powerfoam_samples` and
  `powerfoam_rays_from_camera_grid` after the full direct pytest collection
  caught those public imports.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

Commands run from repo root:

```bash
rtk .venv/bin/python -m py_compile src/train/powerfoam_eval_render.py src/train/train_powerfoam_direct.py
rtk env PYTHONPATH=src/train .venv/bin/python - <<'PY'
import torch
from torch import nn
from powerfoam_eval_render import render_powerfoam_samples
class FourTupleFoam(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
    def forward(self, frame_indices, rays=None):
        batch = int(frame_indices.numel())
        return torch.ones(batch, 3, 2, 2), torch.full((batch, 2, 2), 0.5), torch.zeros(batch), torch.zeros(batch)
rgb, alpha = render_powerfoam_samples(FourTupleFoam(), torch.arange(3), batch_size=2)
print('render_powerfoam_samples_four_tuple_ok', list(rgb.shape), list(alpha.shape), float(alpha.mean()))
PY
rtk env PYTHONPATH=src/train .venv/bin/python - <<'PY'
import torch
from train_powerfoam_direct import load_direct_powerfoam_training_data, resolve_config
cfg = resolve_config({
    'data': {'video_path': 'test_data/test_video_small_128_4fps.mp4', 'max_frames': 1},
    'model': {},
    'render': {'render_size': 8},
    'train': {},
    'losses': {},
    'logging': {},
})
data = load_direct_powerfoam_training_data(cfg, torch.device('cpu'))
print('direct_training_data_keys_ok', sorted(data), list(data['targets'].shape), data['sample_rays'] is None)
PY
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
```

Results:

- `py_compile` passed.
- Four-tuple renderer smoke printed
  `render_powerfoam_samples_four_tuple_ok [3, 3, 2, 2] [3, 2, 2] 0.5`.
- Direct data wrapper smoke printed the historical Direct key set, target shape
  `[1, 3, 8, 8]`, and `sample_rays is None`.
- Full `tests/test_powerfoam_direct.py`: `43 passed, 2 skipped in 9.06s`.
  The run emitted the known parent `pyproject.toml` warning but exited `0`.

## Next

The remaining Direct logger still has local artifact payload logic. Do not
force it into the Metal artifact helper unless the next live-file check shows a
small generic media helper that improves multiple trainers without hiding their
different metrics.
