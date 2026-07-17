# 2026-05-15 repeat-loaded 32f WorldFoam harness

The paired WorldFoam train/eval comparison could not run `32f` directly from
the current multicam fixture because the fixture is only 16 real frames. The
old failure was late and indirect:

```text
ValueError: sample count must be view_count * frame_count
```

The failure came from asking the segment-tape builder to interpret a 16-frame
heldout sample stream as 32 frames.

I added an explicit `--repeat-loaded-frames` opt-in to
`research_experiments/world_foam_lane2/train_eval_owner_run_tape.py` and
threaded it through
`research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py`.
Strict mode remains the default. With the flag enabled, the harness repeats the
loaded view-major frames and assigns sequential frame ids. This is only a
synthetic speed-scaling smoke; it is not a real longer-video quality claim.

I also fixed a comparison-script bug exposed by adding 32f: `summary_16f` used
to take the last row, so a `2,4,8,16,32` run silently summarized 32f as 16f.
It now selects the actual 16f row and writes `summary_by_frame` for all
requested frame counts.

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 \
  -p 'test_compare_endpoint_run_record_edit_train_eval.py'

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 \
  -p 'test_probe_endpoint_record_edit_replay.py'
```

All three passed.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_repeat_loaded_scaling_smoke_render16_2_4_8_16_32.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_blockcoeff_repeat_loaded_warm1_render16_16_32.json
```

The first artifact is a broad execution smoke. It proves endpoint-run,
endpoint-record-edit, block4, and f32 block-coeff all execute through 32f under
the repeated-fixture mode, but its one-step no-warmup timings are too cold and
noisy for speed claims.

The second artifact is a narrower warmed 16/32 synthetic smoke. It intentionally
exited nonzero because the acceptance booleans reject the edit path as
non-sublinear/slower in that run. Key 16f/32f total times:

- endpoint-run: `8.33 ms -> 16.52 ms` (`1.98x`, roughly linear for a 2x frame
  increase)
- endpoint-record-edit: `34.43 ms -> 99.83 ms` (`2.90x`, not sublinear)
- f32 block-coeff: `27.78 ms -> 34.60 ms` (`1.25x`, sublinear across 16->32
  but still slower than endpoint-run at both frame counts in this warmed tiny
  setup)

Current interpretation: the 32f harness blocker is fixed for synthetic
speed-shape testing, but this does not rescue the WorldFoam competitive claim.
STAR UVT remains cleaner because it has a real frame-scaling path; WorldFoam
now has partial 32f execution evidence and a negative warmed timing read.
