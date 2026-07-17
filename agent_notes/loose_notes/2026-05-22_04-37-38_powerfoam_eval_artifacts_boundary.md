# PowerFoam Eval Artifacts Boundary

## Context

Continuation of the trainer modularization goal. After moving the data loader,
the largest remaining top-level helper in `train_powerfoam_metal.py` was
`log_artifacts(...)`. It was not core optimizer or model code; it assembled the
eval-time render, metrics, preview/video artifacts, optional eval color
calibration, aux/drift metrics, and W&B payload.

This is a useful helper boundary because the contract is "PowerFoam-like eval
module" rather than "Metal trainer internals": the model only needs
`parameters()`, callable render, `aux_metrics(...)`, and
`parameter_drift_metrics()`.

## Changes

- Added `src/train/powerfoam_eval_artifacts.py`.
  - Owns `log_powerfoam_artifacts(...)`.
  - Uses `powerfoam_eval_render.render_powerfoam_samples(...)`.
  - Uses shared background compositing, eval color calibration,
    reconstruction metrics, artifact writing, cadence, and W&B media helpers.
- Updated `src/train/train_powerfoam_metal.py` to import
  `log_powerfoam_artifacts as log_artifacts` for compatibility.
- Removed eval-artifact-only imports from the Metal trainer body.

## Validation

Commands run from repo root:

```bash
rtk .venv/bin/python -m py_compile src/train/powerfoam_eval_artifacts.py src/train/train_powerfoam_metal.py
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -c "import train_powerfoam_metal as m; from powerfoam_eval_artifacts import log_powerfoam_artifacts; print('powerfoam_eval_artifact_alias_ok', m.log_artifacts is log_powerfoam_artifacts, callable(m.render_samples))"
rtk env PYTHONPATH=src/train .venv/bin/python - <<'PY'
from pathlib import Path
import tempfile
import torch
from torch import nn
from powerfoam_eval_artifacts import log_powerfoam_artifacts

class DummyPowerFoam(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
    def forward(self, frame_indices, rays=None):
        batch = int(frame_indices.numel())
        rendered = torch.full((batch, 3, 5, 5), 0.25) + self.weight * 0.0
        alpha = torch.full((batch, 5, 5), 0.75)
        return rendered, alpha
    def aux_metrics(self, frame_indices, targets, rays=None):
        return {"aux_visible_fraction": 1.0}
    def parameter_drift_metrics(self):
        return {
            "state_mean_center_delta": 0.0,
            "state_p95_center_delta": 0.0,
            "state_max_center_delta": 0.0,
            "state_mean_xy_delta": 0.0,
            "state_mean_z_delta": 0.0,
            "state_mean_radius_delta": 0.0,
            "state_mean_density_delta": 0.0,
            "state_mean_feature_delta": 0.0,
            "state_cell_count": 1.0,
        }

cfg = {
    "train": {"frames_per_step": 2, "steps": 10},
    "logging": {"video_log_every": 999, "always_log_last_step": False},
    "render": {"background": [0.0, 0.0, 0.0], "background_mode": "fixed", "eval_color_calibration": "none"},
    "losses": {"ssim_window_size": 3, "ssim_c1": 0.01 ** 2, "ssim_c2": 0.03 ** 2},
}
out = Path(tempfile.mkdtemp())
targets = torch.full((2, 3, 5, 5), 0.3)
metrics = log_powerfoam_artifacts(DummyPowerFoam(), targets, cfg, 1, out, None)
print("eval_artifact_smoke", (out / "preview_step_0001.png").exists(), round(metrics["eval_l1"], 6), metrics["aux_visible_fraction"])
PY
```

Results:

- `py_compile` passed.
- Compatibility smoke printed `powerfoam_eval_artifact_alias_ok True True`.
- Runtime artifact smoke printed `eval_artifact_smoke True 0.05 1.0`.

## Next

The remaining top-level Metal trainer helpers are mostly model class,
`rays_for_sample_batch(...)`, and the train loop. Further extraction should be
done only after checking whether a helper is reused across PowerFoam-family
trainers or diagnostics; avoid turning the model class into scattered pieces.
