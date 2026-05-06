# PowerFoam W&B media harness

## What changed

- The direct PowerFoam trainer now follows the existing trainer split:
  - trainer-local `render_all(...)` calls the model forward across all frames
  - shared `train_logging.py` helpers build W&B video/image payloads
- `src/train/train_powerfoam_direct.py` logs:
  - `Train/Loss`, `Train/L1`, `Train/MSE`, `Timing/ElapsedSeconds`
  - `Eval/L1`, `Eval/MSE`
  - `Preview`
  - `Render_Video`, `Render_GT_Video`, `GT_Video`, `Alpha_Video` on the configured video cadence
- `src/train_configs/local_mac_powerfoam_direct_128_smoke.jsonc` now enables W&B for the PowerFoam direct smoke config.

## Verification

Commands run from the dynaworld root:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_powerfoam_direct.py src/train/powerfoam_direct.py src/train/train.py
uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
WANDB_MODE=offline PYTHONPATH=src/train uv run python src/train/train.py /tmp/powerfoam_direct_wandb_smoke.json
```

Results:

- `tests/test_powerfoam_direct.py`: 2 passed.
- Offline W&B one-step run completed and wrote media under `wandb/offline-run-20260430_213037-ji816yek/files/media/`:
  - `Preview_1_2eef4b184d540083d5b6.png`
  - `Render_Video_1_ef82de7b960b0ae535fb.mp4`
  - `Render_GT_Video_1_815707a3649691a7c1c1.mp4`
  - `GT_Video_1_71d795ef8000abf21cd7.mp4`
  - `Alpha_Video_1_1e9fde4786982e9cd730.mp4`
- The same one-step smoke wrote local artifacts under `/tmp/powerfoam_direct_wandb_smoke/`.

## Remaining gap

This logs through W&B, but the renderer is still the temporary Torch direct renderer. It is useful for trainer/harness validation; it is not the final Metal PowerFoam rasterizer path.
