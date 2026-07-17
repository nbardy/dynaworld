# Gate4 Vectorized Affine Ray Fit

## Context

The benchmark preflight is still contaminated by unrelated `ai_trader` CPU
jobs, so I avoided another speed-promotion claim. I kept working on the setup
path that feeds the fast endpoint-record shader.

The broad `build_endpoint_record_sequences_s` timer hid two costs:

- building the Gate4 affine candidate tape
- converting that tape into endpoint delta records

The affine builder also fit one linear ray track at a time in Python, using a
per-track torch call for every view/y/x ray. That is avoidable host work.

## Change

- Added `_fit_all_linear_ray_tracks(...)` in
  `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`.
- `build_gate4_affine_slab_tape(...)` now fits all track rays in one vectorized
  tensor pass and materializes `ray_coeff` / `explicit_rays` from that layout.
- Preserved track ordering: view, y, x, frame.
- Added split prepare-timing fields in
  `research_experiments/world_foam_lane2/train_eval_owner_run_tape.py`:
  - `build_gate4_affine_endpoint_tape_s`
  - `build_gate4_endpoint_delta_replace_tape_s`
  - `build_gate4_endpoint_run_sequences_s` for non-delta endpoint modes
- Kept the old aggregate `build_endpoint_record_sequences_s` field intact for
  existing verifiers and result readers.

## Validation

Passed:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
```

Passed:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `8/8`.

Passed:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Result: `46/46`.

Integrated smoke:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 24 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_vectorfit_tensornative_sorted_emitted_smoke_2f_render16_site24.json
```

Result: `status=ok`, gradients nonzero, parameters updated, outputs finite.
The smoke artifact includes the new split timings. On the train row:

- `build_endpoint_record_sequences_s`: `0.1757s`
- `build_gate4_affine_endpoint_tape_s`: `0.1560s`
- `build_gate4_endpoint_delta_replace_tape_s`: `0.0197s`

## Interpretation

The split confirms that, at this small smoke size, affine tape construction is
the dominant setup cost after the tensor-native delta accumulation fork. The
vectorized fit removes one obvious Python loop from that path, but clean
promotion timing is still blocked by external contention. Next useful compiler
work should target the remaining affine tape row/candidate materialization
loop.
