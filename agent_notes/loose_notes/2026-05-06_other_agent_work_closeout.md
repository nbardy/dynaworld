# Other-Agent Work Closeout - 2026-05-06

## Scope

This note records the commit closeout for the dirty `dynaworld` submodule
state. The work was already present in the workspace; I audited the scope,
verified focused gates, left ignored/generated payloads out, and committed the
source, configs, notes, tests, and small fixtures.

## What Was Packaged

### PowerFoam and dynamic gauge lane

- PowerFoam Metal/CUDA runbooks, TODOs, baselines, acceptance checks, and
  completion-audit notes.
- Dynamic foam experiment scripts for Modal CUDA smoke, Metal/CUDA comparison,
  clean-init coverage, external blocker checks, motion-vs-repaint comparison,
  4K benchmark/trainability verification, topology/support diagnostics, and
  artifact-pixel validation.
- Dynamic gauge/PowerFoam training entrypoints and local train configs.
- Small PowerFoam parity fixtures and the `powerfoam_sfm_tiny_ascii.ply` test
  fixture.
- Tests for PowerFoam direct training, CUDA smoke plan, paper acceptance,
  eval-color calibration, dynamic gauge foam, relative pose, source-relative
  cameras, sequence/data IO, multicam video data, pipeline helpers, and video
  feature cache.

Ignored generated data remained untracked through `.gitignore`, including
`research_experiments/dynamic_foam/artifacts/`, PowerFoam build/outputs,
downloaded foam paper PDFs/text, pyc files, and larger local data caches.

### Multicam and relative-camera training

- Multicam loader and validation-media changes.
- Same-time/multicam precomputed-feature trainer updates for camera-swap modes,
  learned residual relative pose, heldout best-metric tracking, and validation
  rendering.
- Config-factory support and tests for feature-token stride, bf16 projected
  feature output, and explicit bf16 AMP dtype.
- Additional notes around multicam validation, PowerFoam, fluid/wave hypotheses,
  and remaining work.

## Verification

Syntax:

```bash
git ls-files -m -o --exclude-standard -z '*.py' | xargs -0 .venv/bin/python -m py_compile
git diff --check
```

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_config_factory_helpers.py tests/test_multicam_video_data.py tests/test_pipeline_helpers.py tests/test_sequence_data_single_frame.py tests/test_video_feature_cache.py tests/test_relative_pose.py tests/test_source_relative_cameras.py tests/test_config_and_dataset_io.py -q
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_direct.py tests/test_dynamic_gauge_foam.py tests/test_powerfoam_cuda_smoke.py tests/test_powerfoam_eval_color_calibration.py tests/test_powerfoam_paper_acceptance.py -q
```

Results:

- Python compile check passed.
- `git diff --check` passed.
- Multicam/config/relative-camera/data tests: 36 passed.
- PowerFoam/dynamic-gauge tests: 63 passed, 1 skipped.

## Caveats

- This commit records tooling, configs, notes, and focused gates. It does not
  claim full PowerFoam paper-quality reproduction.
- CUDA/Modal claims still need their saved summary JSON and verifier commands
  before being cited as fresh evidence.
- The multicam learned-residual relative-camera path has unit coverage and
  saved notes, but heldout-quality ranking still depends on the referenced run
  artifacts and W&B/summary evidence.
