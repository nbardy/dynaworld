# WorldFoam real-frame gate contract

## Context

After the render96/site48 i32-base-offset gate, the remaining evidence gap was
not another local shader micro-variant. The useful next proof is a real
longer-than-16f fixture or a quality-linked gate. The repeated 32f smoke already
records `loaded_frame_count=16`, so it must remain synthetic speed-scaling
evidence only.

## Fixture inventory

Checked manifests currently show no real multicam validation fixture beyond
16f:

- `data/multicam_val/clip_sets/multicam_val_v1_chunked_128_4fps_16f/manifest.jsonl`
  has 14 records at 16f.
- `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl`
  has 8 records at 16f.
- The available 64f manifests are single-video pretraining manifests, not the
  heldout-multicam train/eval fixture WorldFoam uses for this gate.

That means a true 32f/64f WorldFoam promotion is data-blocked today unless we
build or register a longer heldout-multicam fixture.

## Harness changes

Updated
`research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py`:

- `--worldfoam-config` forwards a custom config into
  `train_eval_owner_run_tape.py`.
- `--star-video-path` forwards a custom source video into
  `compare_star_uvt_worldfoam_scale.py`.
- `--require-real-loaded-frames` records a strict promotion requirement and is
  mutually exclusive with `--repeat-loaded-frames`.

Updated
`research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py`:

- When `require_real_loaded_frames=true`, reject a summary that also requested
  repeated loaded frames.
- Reject WorldFoam artifacts whose rows use repeated frames, lack
  `loaded_frame_count`, or have `loaded_frame_count < frame_count`.
- Reject STAR artifacts whose rows use repeated frames, lack loaded-frame
  metadata, or have `loaded_frame_count < requested_frames`.

## Verification

Focused tests pass:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate
PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
.venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py
```

The wrapper suite now has 17 tests and the verifier suite has 6 tests.

## Next

Do not claim real long-frame WorldFoam scaling from `--repeat-loaded-frames`.
For the next promotion attempt, first create or locate a heldout-multicam
fixture with real 32f/64f records, then run the native-cutwalk wrapper with
`--worldfoam-config`, a matched `--star-video-path`, and
`--require-real-loaded-frames --verify-promotion`.
