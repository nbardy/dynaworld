# Multicam Train2 / Heldout1 Split

This folder tracks the first five-sample calibrated multicam split where each
sample trains on two cameras and evaluates one novel camera.

The split lives in:

```text
src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f.jsonc
src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f_manifest.jsonl
```

Samples:

| dataset | scene | train cameras | heldout | train sep | heldout-to-train |
|---|---|---|---|---:|---:|
| AIST | `gBR_sBM_d04_mBR0_ch01` | `c01`, `c03` | `c02` | 90.25 | 45.77 / 44.51 |
| Neural3D | `coffee_martini` | `cam04`, `cam09` | `cam06` | 41.30 | 21.05 / 20.32 |
| Neural3D | `coffee_martini` | `cam13`, `cam20` | `cam16` | 41.45 | 20.26 / 21.36 |
| DeepView | `03_Dog` | `camera_0001`, `camera_0015` | `camera_0013` | 30.87 | 32.85 / 38.02 |
| DeepView | `03_Dog` | `camera_0003`, `camera_0021` | `camera_0010` | 41.77 | 22.27 / 22.30 |

Smoke the loader:

```bash
uv run python -m research_experiments.multicam_train2_holdout1.smoke_load_split
```

Notes:

- This is a companion matrix, not a replacement for `multicam_val_v1`.
- The current multicam trainer still runs one selected sample at a time.
- Training configs should point `data.multicam_manifest` at the manifest above,
  set `data.multicam_sample_id`, and use the matching `train_cameras`,
  `heldout_camera`, `anchor_camera`, `condition_camera`, and `camera.rig_init`.
