# Ex4DGS Pretrained Assets

User asked to download three Ex4DGS release assets for validation setup:

- `coffee_martini.zip`
- `Birthday.zip`
- `Fabien.zip` (asset spelling is `Fabien`, not `Fabian`)

Added a checked-in config/script/pipeline:

- `src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc`
- `src/dataset_scripts/ex4dgs_pretrained_val_seed.sh`
- `src/dataset_pipeline/ex4dgs_pretrained.py`

Local generated files are ignored under `data/external/ex4dgs_pretrained/`.
The selected assets downloaded and extracted successfully:

- raw zips: `data/external/ex4dgs_pretrained/raw/` (~322M)
- extracted bundles: `data/external/ex4dgs_pretrained/extracted/` (~379M)
- inventory: `data/external/ex4dgs_pretrained/metadata/inventory.json`

Important distinction: these release assets are pretrained checkpoint/eval
bundles, not raw multi-camera videos or image sequences. Each bundle contains
`cameras.json`, `mean_metrics.json`, `all_metrics.json`, `input.ply`, and final
iteration PLY point clouds. They should not be folded into the raw
`multicam_val_v1` manifest until we add a renderer/evaluator path that consumes
Ex4DGS checkpoint outputs directly, or until we fetch the corresponding raw
Technicolor/Neural 3D Video data.

Inventory after extraction:

- `coffee_martini`: 5400 camera records, final iteration 40000
- `Birthday`: 800 camera records, final iteration 40000
- `Fabien`: 800 camera records, final iteration 30000

Validation run:

```bash
./src/dataset_scripts/ex4dgs_pretrained_val_seed.sh inspect
bash -n src/dataset_scripts/ex4dgs_pretrained_val_seed.sh
uv run python -m py_compile src/dataset_pipeline/ex4dgs_pretrained.py
jq -c . data/external/ex4dgs_pretrained/metadata/inventory.json >/dev/null
git diff --check -- .gitignore data/README.md src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc src/dataset_scripts/ex4dgs_pretrained_val_seed.sh src/dataset_pipeline/ex4dgs_pretrained.py
```
