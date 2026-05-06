# PowerFoam grid artifact follow-up

Date: 2026-05-01 11:45

## Trigger

The online run `gtyrpqv0` showed a very visible grid in the render and alpha
videos. The run was the slow Torch `powerfoam_direct` path, not the Metal path.

## Diagnosis

The artifact is real. The current image-seeded reference init places `64` cells
on an `8x8` image-plane lattice at one depth, samples colors at those lattice
points, and trains only 20 CPU steps. Since the alpha is close to saturated, the
output exposes the power-cell partition almost directly.

This is a capacity/init/trainer limitation of the Torch reference path, not a
W&B logging artifact and not evidence that the forward-only Metal prototype was
used.

## Patch

Added `model.image_init_jitter` to break exact image-grid centers while keeping
deterministic reproducibility. The smoke configs now set it to `0.2`. A focused
test verifies deterministic jitter and that it moves points off the exact
lattice.

## Probes

Commands:

```bash
rtk env PYTHONPATH=src/train WANDB_MODE=offline uv run python src/train/train.py src/train_configs/local_mac_powerfoam_direct_video_full_tiny_smoke.jsonc
rtk env PYTHONPATH=src/train WANDB_MODE=offline uv run python -c 'from config_utils import load_config_file; from train_powerfoam_direct import run_training; cfg=load_config_file("src/train_configs/local_mac_powerfoam_direct_video_full_tiny_smoke.jsonc"); cfg["model"]["image_init_jitter"]=0.2; cfg["logging"]["output_dir"]="outputs/powerfoam_direct/local_mac_powerfoam_direct_video_jitter02_probe"; cfg["logging"]["wandb_enabled"]=False; run_training(cfg)'
rtk env PYTHONPATH=src/train WANDB_MODE=offline uv run python -c 'from config_utils import load_config_file; from train_powerfoam_direct import run_training; cfg=load_config_file("src/train_configs/local_mac_powerfoam_direct_video_full_tiny_smoke.jsonc"); cfg["model"]["cells"]=256; cfg["model"]["neighbor_count"]=32; cfg["model"]["radius_scale"]=0.52; cfg["model"]["density_init"]=12.0; cfg["train"]["steps"]=10; cfg["logging"]["output_dir"]="outputs/powerfoam_direct/local_mac_powerfoam_direct_video_256cell_probe"; cfg["logging"]["wandb_enabled"]=False; run_training(cfg)'
```

Results:

- `64` cells with jitter `0.2`: step-20 eval `L1 = 0.04715`, still visibly
  cell-boundary dominated.
- `256` cells with low radius/density probe: step-10 eval `L1 = 0.29323`,
  under-covered and much worse.

## Current conclusion

Jitter is worth keeping as an option, but it does not solve the artifact. The
actual next step is not another tiny CPU run. We need either:

- a tuned higher-capacity direct reference run, likely too slow on CPU, or
- the real Metal training path so we can use enough cells, run enough steps, and
  add the missing official densification/resampling behavior.

