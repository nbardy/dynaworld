# Dynamic Foam external-blocker config writer cleanup

## Context

The Dynamic Foam helper layer already owns sorted parent-safe JSON object
writes through `research_experiments/dynamic_foam/report_artifacts.py`.
`run_powerfoam_external_blockers.py` was still hand-writing the generated
PowerFoam train config with `mkdir + write_text(json.dumps(...))`.

## Change

- `run_powerfoam_external_blockers.py` now imports `write_report_json`.
- The generated training config uses
  `write_report_json(config_output, cfg, sort_keys=False)`, preserving the
  previous unsorted config order while sharing parent creation and newline
  behavior.

## Left local on purpose

- Dry-run stdout stays as an inline `json.dumps(...)` print.
- Modal/remote manifests, settings, copied artifacts, and upstream runner
  internals stay local in their respective scripts because they are execution
  inputs or byte-preserving transfers, not generic Dynaworld report artifacts.

## Validation

- `py_compile` covered the updated script.
- Direct CLI help was checked with `--help`.
