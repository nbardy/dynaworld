# CUDA Smoke Report Artifact Boundary

## Context

Continuation of the trainer-landscape cleanup goal. The useful rule remains:
share repeated primitives when the contract is actually common, and leave
experiment or upstream-runner internals local.

I inspected `powerfoam_cuda_smoke_runner.py` after a raw sync scan showed one
remaining `torch.cuda.synchronize()` call. That call lives inside the embedded
`SMOKE_ENTRY` script that is written into a cloned upstream PowerFoam checkout.
Importing Dynaworld's `train_devices` helper there would make the upstream smoke
depend on this repo's source layout, so I left that embedded timing helper
self-contained.

## Change

- Routed top-level PowerFoam CUDA smoke `summary.json` writes through
  `research_experiments.dynamic_foam.report_artifacts.write_report_json(...)`.
- Routed Modal wrapper `modal_return.json` writes through the same helper.
- Kept lane `settings.json`, copied remote JSON files, embedded entrypoint
  generation, and upstream metrics writes local because they are execution
  inputs/outputs inside the cloned upstream repo, not Dynaworld report
  artifacts.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  to record the boundary and the reason for not moving the embedded CUDA sync.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  research_experiments/dynamic_foam/report_artifacts.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train:research_experiments/dynamic_foam \
  .venv/bin/python research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py --help
```

Passed.

```bash
rtk uv run --with pytest python -m pytest \
  tests/test_powerfoam_cuda_smoke.py \
  tests/test_dynamic_foam_report_artifacts.py -q
```

Passed: 13 tests.

## Handoff

This did not change CUDA execution behavior. It only removes one more local
parent-directory-plus-sorted-JSON report write from a reusable Dynamic Foam
runner.
