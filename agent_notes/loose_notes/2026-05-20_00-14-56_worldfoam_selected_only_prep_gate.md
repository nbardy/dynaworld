# WorldFoam selected-only owner-run delta prep

Context: after framebitmask correctness reached render64/site24/16f, the next
blocker was prep wall time. The 16f artifact showed the shader was no longer
the only issue: baseline accounting spent minutes building the full segment
tape and compacting baseline tapes before the selected owner-run delta shader
could run.

What changed:

- Added `--experimental-selected-only-owner-run-delta-prep` to
  `train_eval_owner_run_tape.py`.
- The flag is deliberately narrow: it is valid only for `slow-owner-run`
  owner-run delta packed modes.
- It skips `build_segment_tape` and `compact_baseline_tapes` and builds only the
  selected owner-run delta tape needed by the fused-MSE shader.
- The artifact is marked with:
  - `experimental_selected_only_owner_run_delta_prep=true`
  - `train_baseline_segment_metrics_built=false`
  - `heldout_baseline_segment_metrics_built=false`
- To avoid lying about semantic row count, selected-only prep derives expanded
  per-frame row lengths from the delta table via
  `build_delta_replace_frame_row_descriptors`. This keeps
  `selected_segments` aligned with the full owner-run metric without rebuilding
  the full segment tape.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_selected_only_framebitmask_prep_skips_baseline_segment_tape \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays -v
```

Result after the follow-up metadata-skip patch: `3` tests passed in `74.745s`.

Path smoke:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 4 \
  --render-size 64 \
  --site-count 24 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode manual-vjp \
  --tape-mode owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid \
  --endpoint-record-source slow-owner-run \
  --experimental-selected-only-owner-run-delta-prep \
  --out-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_prep_4f_smoke.json
```

Result: `status=ok`, but end snapshot was contended by unrelated `ai_trader` /
Metal work, so it is path/correctness evidence only. The important prep fields
are:

- train prep includes `build_endpoint_record_sequences_s=54.092s`,
  `pack_endpoint_record_delta_replace_s=0.070s`, and
  `move_endpoint_record_delta_replace_to_mps_s=0.874s`
- train prep does not include `build_segment_tape_s`
- train prep does not include `compact_baseline_tapes_s`
- selected expanded segments still match the previous full-metric 4f row:
  `81,570`

Interpretation:

- This removes the separate baseline segment-tape accounting cost from
  selected framebitmask timing runs.
- It does not solve the larger remaining bottleneck: `slow-owner-run`
  endpoint-record sequence construction still took `54.1s` at 4f/render64 in a
  contended rerun and
  was `221s` at 16f in the previous artifact.
- The next real prep fork should move/cache/native-build owner-run endpoint
  sequence construction itself. Another shader replay micro-fork is less
  aligned until that is fixed.
