# Gate4 Vector-Gather CSR Negative

## Context

After the fast-skip vector replay cleanup, I tried to remove another Python loop
from `_iter_single_slab_sorted_depth_id_chunks(...)`: the per-row padded CSR
materialization loop that copies each row's candidate coefficients and ids into
a fixed `[chunk, max_candidates]` block before depth evaluation and sorting.

The hypothesis was that a NumPy gather grid would be faster than looping through
up to 128 rows per chunk.

## Attempt

Temporary patch:

- computed `row_begins`
- built a `[row, slot]` gather index from `row_begins + arange(max_count)`
- gathered `coeffs[gather_index]` and `candidate_ids[gather_index]`
- used the existing validity mask to ignore padded slots

The parity path still passed:

- `py_compile` passed
- focused high-cap Gate4 test passed
- full Gate4 compiler unit passed `8/8`
- framegroup16 promotion wrapper unit passed `46/46`
- native sorted/emitted smoke produced
  `research_experiments/world_foam_lane2/results/2026-05-19_vectorgather_tensornative_sorted_emitted_smoke_2f_render16_site24.json`
  with `status=ok`

The smoke timing is not meaningful because the benchmark environment was
contended by unrelated `ai_trader` training jobs plus `MTLCompilerService`.

## Negative Result

An isolated CPU micro-benchmark matching the current CSR scale
(`1024` rows, roughly `110-225` candidates per row, `16` frames, chunks of
`128`) showed the gather variant was slower:

```text
old row-fill loop: median 0.907 ms
new gather grid:   median 2.075 ms
```

The gather version allocates and touches a full index grid, so it loses to the
simple row-copy loop for this shape.

## Resolution

I reverted the gather patch and kept the previous row-fill loop. Syntax after
the revert passed:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
```

Future setup/compiler work should skip this gather-grid approach. The next
viable step is a real native direct-from-CSR delta builder that avoids
materializing sorted depth/id tensors in Python/NumPy, not a larger NumPy gather
inside the existing helper.
