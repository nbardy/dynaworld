# WorldFoam optical-transfer fixture runner

## Context

The active goal is to build full runner coverage for the World Tubes and
WorldFoam papers. After landing the World Tubes decisive-demo and visibility
stress fixtures, this chunk implemented the first WorldFoam paper-math gate
from `research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`.

## Implemented

Added:

- `research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py`
- `research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py`

The fixture implements:

- `TransferElement(beta, m)`
- `compose(front, back)`
- `scan(elements)`
- `decode(element, background)`
- `constant_run_element(sigma, length, color)`
- `render_word(...)`
- `render_word_from_elements(...)`
- `make_two_run_fixture()`
- `make_three_run_fixture()`
- `same_representation_replay_fixture()`
- `analytic_prefix_suffix_vjp(...)`
- `finite_difference_vjp(...)`
- `commutator_swap_probe()`
- `run_all_checks()`
- `write_summary_json(...)`

The saved artifact is:

- `outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json`

## Verification

Passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py -q
```

Result:

```text
8 passed in 1.21s
```

Passed:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py
```

Passed:

```bash
PYTHONPATH=src/train uv run python research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py \
  --out outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json

PYTHONPATH=src/train uv run python research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py \
  --verify-report outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json
```

Saved summary:

- status: `ok`
- dtype: `float64`
- checks: all `ok`
- replay render error: `0.0`
- replay element error: `0.0`
- max VJP finite-difference error: `2.4557592070983958e-11`
- commutator error: `5.551115123125783e-17`

## Important boundary

This is a pure CPU/Torch math fixture. It proves the optical-transfer monoid,
constant-run alpha equivalence, same-representation replay, analytic VJP, and
commutator probe. It does not prove WorldFoam Metal trainability, real-video
quality, or parity against World Tubes.

## Next work

1. Extend `projective_decisive_demo_report.py` with real-video/media rows.
2. Add WorldFoam owner-run/Metal comparison rows that consume or mirror this
   optical-transfer contract.
3. Add a shared paper table/chart generator consuming saved World Tubes,
   WorldFoam, and dynamic 3DGS reports.
