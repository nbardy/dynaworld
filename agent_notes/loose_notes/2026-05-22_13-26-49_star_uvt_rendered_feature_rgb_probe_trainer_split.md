# STAR UVT Rendered-Feature RGB Probe Trainer Split

## Context

The trainer unification pass is moving orchestration into owner modules while
leaving historical `train_*.py` entrypoints as thin CLI/backcompat wrappers.
The target-grid RGB probe already followed that pattern; the rendered-feature
RGB probe still needed the same owner boundary.

## Changes

- `src/train/star_uvt_rendered_feature_rgb_probe_trainer.py` owns
  `run_probe(...)`: config resolution, STAR checkpoint restore, sparse rendered
  feature sampling, colorizer training, optional checkpoint/media writes, W&B
  row logging, and output JSON.
- `src/train/train_star_uvt_rendered_feature_rgb_probe.py` is now a thin CLI
  wrapper and re-exports `run_probe` for backcompat imports.
- `src/train/trainer_registry.py` routes
  `star_uvt_rendered_feature_rgb_probe` to
  `star_uvt_rendered_feature_rgb_probe_trainer.run_probe`.
- `tests/test_trainer_registry.py` covers the rendered-feature probe registry
  route.
- `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` now record
  the owner/wrapper split and smoke evidence.

## Validation

Commands run from the dynaworld root:

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/star_uvt_rendered_feature_rgb_probe_trainer.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  src/train/trainer_registry.py \
  tests/test_trainer_registry.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_rgb_probe.py \
  tests/test_star_uvt_checkpoints.py \
  tests/test_trainer_registry.py -q
```

Result: `23 passed`.

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from star_uvt_rendered_feature_rgb_probe_trainer import run_probe as owner_run
from train_star_uvt_rendered_feature_rgb_probe import run_probe as wrapper_run
from trainer_registry import trainer_entry_for_arch
assert owner_run is wrapper_run
entry = trainer_entry_for_arch("star_uvt_rendered_feature_rgb_probe")
assert entry.module == "star_uvt_rendered_feature_rgb_probe_trainer"
assert entry.runner == "run_probe"
print("star uvt rendered feature rgb probe wrapper reexports owner run_probe and registry routes to owner module")
PY
```

Runtime smoke through `src/train/train.py` also passed with:

- config: `/tmp/dynaworld_star_uvt_rendered_feature_rgb_probe_owner_split_smoke.jsonc`
- output row: `/tmp/dynaworld_star_uvt_rendered_feature_rgb_probe_owner_split_smoke_result.json`
- checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`
- settings: 4 frames, 64px, one step, `sample_grid_shape=[4,16,16]`,
  8192 tubes retained for checkpoint compatibility, W&B disabled, no media or
  checkpoint writes.

The smoke exercised registry dispatch, MPS sparse rendered-feature forward,
checkpoint load, colorizer backward, optimizer step, full render evaluation,
and output row serialization. The row had `colorizer_grad_seen=true` and
`model_grad_seen=false`, as expected for the default frozen STAR model probe.
The row's `pass=false` is expected for a one-step smoke because loss decrease
cannot be established from a single recorded point with
`require_loss_decrease=false`.

## Follow-Up

No new `key_learnings.md` entry was added; this was a structural split and
validation of the existing owner/wrapper pattern, not a surprising shader or
training lesson.
