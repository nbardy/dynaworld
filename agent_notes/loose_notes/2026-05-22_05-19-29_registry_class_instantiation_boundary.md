# Registry Class Instantiation Boundary

## Context

After the PowerFoam helper split, the remaining trainer-as-helper scan still
found benchmark/probe code importing concrete precomputed and multicam trainers
only to instantiate a trainer from a config:

- `src/benchmarks/trainer_phase_benchmark.py`
- `src/benchmarks/camera_swap_variant_parity.py`
- `research_experiments/vjepa_performance/benchmark_multicam_vjepa.py`

That bypassed the registry boundary even though those scripts are config-driven.

## Change

Extended `trainer_registry.TrainerEntry` with an optional `trainer_class` field
and added `instantiate_trainer_for_config(...)`.

The registry now resolves class-based trainer configs for:

- `precomputed_feature_implicit_camera`
- `ltx_feature_implicit_camera`
- `wan_vace_feature_implicit_camera`
- `multicam_precomputed_feature_implicit_camera`
- `mixed_same_heldout_precomputed_feature_implicit_camera`
- `multicam_relative_pose_implicit_camera`

Then rerouted the three benchmark/probe scripts above through registry
instantiation. `trainer_phase_benchmark.py` keeps a local
`trainer_uses_multicam_phase(...)` capability check instead of importing the
multicam class just for `isinstance(...)`.

## Validation

```bash
PYTHONPATH=src/train:src/benchmarks .venv/bin/python -m py_compile \
  src/train/trainer_registry.py \
  tests/test_trainer_registry.py \
  src/benchmarks/trainer_phase_benchmark.py \
  src/benchmarks/camera_swap_variant_parity.py \
  research_experiments/vjepa_performance/benchmark_multicam_vjepa.py

PYTHONPATH=src/train:src/benchmarks uv run --with pytest python -m pytest \
  tests/test_trainer_registry.py tests/test_temporal_sampling.py -q

PYTHONPATH=src/train:src/benchmarks uv run --with pytest python -m pytest \
  tests/test_trainer_registry.py -q

PYTHONPATH=src/train:src/benchmarks .venv/bin/python \
  src/benchmarks/trainer_phase_benchmark.py --help
PYTHONPATH=src/train:src/benchmarks .venv/bin/python \
  src/benchmarks/camera_swap_variant_parity.py --help
PYTHONPATH=src/train:src/benchmarks .venv/bin/python \
  research_experiments/vjepa_performance/benchmark_multicam_vjepa.py --help
```

Results:

- Registry plus temporal sampling: `19 passed`.
- Registry focused rerun: `9 passed`.
- All three CLI help/import smokes passed after adding the missing `Path`
  import to `trainer_phase_benchmark.py`.

## Handoff

The trainer-as-helper import scan for Token-GS/precomputed/multicam now leaves
only `tests/test_temporal_sampling.py`, which intentionally imports concrete
classes to call class methods on object-shell instances. The next meaningful
cleanup is trainer-loop duplication, not more registry routing.
