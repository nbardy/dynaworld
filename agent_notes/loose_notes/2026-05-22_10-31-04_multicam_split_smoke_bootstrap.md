# Multicam Split Smoke Bootstrap

## Context

Continuation of the trainer/code-organization cleanup pass. The broad goal is
still not to rewrite training, but to remove small repeated boundaries where
scripts own repo-root lookup, `src/train` import setup, or artifact I/O locally.

## Change

- Added `chdir_root()` to `research_experiments/report_artifacts.py`.
- Routed `research_experiments/multicam_train2_holdout1/smoke_load_split.py`
  through `research_experiments.report_artifacts` for repo-root `chdir` and
  config path resolution.
- Removed the smoke script's local `Path(__file__)`, `os.chdir(ROOT)`, and
  `sys.path.insert(0, ROOT / "src/train")` preamble.
- Updated the split README smoke command to module execution:

```bash
uv run python -m research_experiments.multicam_train2_holdout1.smoke_load_split
```

The direct `uv run python research_experiments/.../smoke_load_split.py` form
fails before helper import because Python starts import resolution from the
script directory. The module form keeps the repo root importable without
reintroducing another local bootstrap block.

## Validation

- `rtk .venv/bin/python -m py_compile research_experiments/report_artifacts.py research_experiments/multicam_train2_holdout1/smoke_load_split.py`
- `rtk uv run python -m research_experiments.multicam_train2_holdout1.smoke_load_split`

The smoke loaded all five train2/holdout1 samples at `(2, 16, 3, 128, 128)` for
train views and `(1, 16, 3, 128, 128)` for heldout views, with AIST,
Neural3D, and DeepView pose sources intact.

## Handoff

This is another small shared-boundary slice, not a full trainer rewrite.
Remaining cleanup should keep favoring narrow contracts:

- retire or archive stale one-off benchmark/probe scripts only after their
  replacement artifact path is documented;
- continue moving repo/bootstrap and artifact writes into helpers;
- avoid adding tests unless they protect a behavior users care about; use
  runtime smokes for path/setup and trainer-call-graph changes.
