# ViVo Scaffold And Disk Budget

Added a separate ViVo external-dataset lane.

Source facts checked from `https://vivo-bvicr.github.io/`:

- Project page says download requires MS Form submission and checking email for the link.
- Temporary Google Drive folder is linked at `https://drive.google.com/drive/folders/1uG4JB2GDWrIRMqmbI6NCP2kA0jUDAbvp?usp=sharing`.
- Code repository is `https://github.com/azzarelli/ViVo-DataProcessing`.
- Dataset contains `14x` RGB/depth video-camera pairs.
- Raw data includes RGB/depth videos/images plus per-frame intrinsics/extrinsics metadata.
- Docs describe a `10x` train camera and `4x` test camera layout.
- Catalogue explicitly says the entire dataset is `> 1 TB zipped`.

Repo changes:

- Added `src/dataset_configs/vivo_seed.jsonc`.
- Added `src/dataset_pipeline/vivo.py` with `info` and `inspect` stages.
- Added `src/dataset_scripts/inspect_vivo_seed.sh`.
- Documented ViVo in `data/external/README.md`.
- Ignored `data/external/vivo/raw/`, `extracted/`, and `logs/`.

Local validation:

- `.venv/bin/python src/dataset_pipeline/vivo.py info --config src/dataset_configs/vivo_seed.jsonc` prints source/access metadata.
- `.venv/bin/python src/dataset_pipeline/vivo.py inspect --config src/dataset_configs/vivo_seed.jsonc` correctly reports no local scenes yet and writes an empty inventory.
- `.venv/bin/python -m py_compile src/dataset_pipeline/vivo.py` passes.

Disk budget as of this note:

- Main filesystem: `460Gi` total, `358Gi` used, `65Gi` available, `85%` full.
- Repo total: `3.9G`.
- `data/`: `2.2G`.
- `data/external/neural_3d_video`: `2.2G` from one downloaded/extracted `coffee_martini` scene.

Conclusion:

- We will run out of local disk quickly if we bulk-download multi-view datasets.
- Neural 3D Video full release is roughly `10GB` compressed before extraction, likely around `20GB+` with raw+extracted copies if both are kept.
- ViVo full dataset is not viable locally (`>1TB zipped`).
- Future workflow should be one-scene-at-a-time, delete archives after extraction when safe, and create tiny derived training subsets under `clip_sets/` rather than keeping every raw source locally.
