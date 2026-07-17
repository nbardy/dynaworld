# PowerFoam Direct Render Options Boundary

## Goal

Continue the trainer modularization pass by removing one more pure helper from
`train_powerfoam_direct.py` without changing the Direct PowerFoam train loop,
loss math, artifact schema, or W&B payloads.

## Change

- Moved Direct PowerFoam config-to-render-options construction into
  `src/train/powerfoam_direct.py` as
  `direct_powerfoam_render_options(render_cfg)`.
- `direct_powerfoam_render_options(...)` is colocated with the
  `PowerFoamRenderOptions` dataclass it builds.
- Updated `src/train/train_powerfoam_direct.py` to import and call the helper
  instead of keeping `make_render_options(...)` locally.
- Removed the Direct trainer's local `render_all(...)` pass-through wrapper;
  eval and heldout artifact rendering now call
  `powerfoam_eval_render.render_powerfoam_samples(...)` directly.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  so future cleanup starts from the live ownership boundary.

## Why This Boundary

This is a small behavior-preserving cleanup. The Direct trainer still owns
sampling, optimizer stepping, scalar payloads, and Direct artifact policy. The
render-options dataclass and its normalized construction now live together in
the Direct PowerFoam model/render module, and eval batch rendering routes
directly through the shared PowerFoam eval renderer. This reduces
trainer-as-helper drift without introducing a base trainer or broad framework.

## Validation Plan

- Compile the touched PowerFoam modules.
- Run the focused Direct PowerFoam pytest gate.
- Check there are no remaining `make_render_options(...)` call sites.
- Check there are no Direct-local `render_all(...)` call sites.
- Run whitespace/diff checks on touched files.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/powerfoam_direct.py src/train/train_powerfoam_direct.py tests/test_powerfoam_direct.py` passed.
- `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q` passed: `44 passed, 1 skipped`.
- Direct helper smoke import/call passed for `direct_powerfoam_render_options(...)`.
- `rtk rg -n "def render_all\\(|render_all\\(|make_render_options|direct_powerfoam_render_options|render_powerfoam_samples" ...` shows no remaining Direct-local `render_all(...)` or `make_render_options(...)` symbols, with the Direct trainer calling `direct_powerfoam_render_options(...)` and `render_powerfoam_samples(...)` directly.
- Touched-file trailing-whitespace scan passed.
- `rtk git diff --check -- src/train/powerfoam_direct.py src/train/train_powerfoam_direct.py CODE_ORGANIZATION.md TODO/trainer_landscape_unification.md` passed for tracked touched files.
