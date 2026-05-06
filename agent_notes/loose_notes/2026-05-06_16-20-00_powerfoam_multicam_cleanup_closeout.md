# PowerFoam / Multicam Cleanup Closeout

This is a cleanup and handoff note, not a claim that the whole checkout is ready
for a single commit. The worktree is intentionally broad and dirty across
PowerFoam, dynamic PowerFoam, multicam configs, tests, TODO docs, baselines, and
experiment artifacts. Do not bundle these lanes into one cleanup commit.

## Repo State

- `dynaworld` is on `main` at `bf5ac18`, tracking `origin/main` with no local
  ahead/behind divergence at the start of this cleanup pass.
- Existing nested repos `third_party/fast-mac-gsplat`,
  `third_party/taichi-splatting`, `third_party/dust3r`, and
  `third_party/dust3r/croco` were clean when inspected.
- `third_party/powerfoam-metal/` and
  `third_party/dynamic-powerfoam-metal/` are plain untracked source trees, not
  registered submodules.
- Generated local artifacts should stay out of commits by default:
  `outputs/`, `wandb/`, `data/youtube_curated_spans/high_motion_smokes/`,
  `research_experiments/dynamic_foam/artifacts/`, PowerFoam build outputs, and
  extracted paper PDFs/text.

## PowerFoam State

- Current short phrasing: audit green, raw gate red.
- The default completion audit is green under calibrated eval semantics, but the
  strict raw-quality audit remains red.
- The nearest0040 calibrated row passes PSNR/SSIM only after heldout-blind
  train-fit RGB matrix calibration. Raw uncalibrated metrics remain below
  `13.0 / 0.15`, and no post-initial raw row passes.
- P0.1 raw/calibrated verifier split, P0.2 same-split splat comparator, P0.3
  Metal dynamic-geometry explicit-video proof, and P0.4 CUDA dynamic-geometry
  micro smoke are done as gates/smokes, not solved quality.
- Dynamic-video result: geometry-only PowerFoam beats fixed-geometry repaint on
  high-motion YouTube 128px and 512px probes. Full-clip all-enabled F32 feature
  foam trains and writes valid H.264 MP4s, but remains visibly cellular/coarse.
  Treat support, initialization, and hierarchical representation as the next
  lever, not another appearance switch.
- MP4 artifact caveat is resolved for current checked paths: the green-video
  symptom was local playback/backend related. Decoded pixel gates pass, and
  `src/train/video_io.py` is the shared H.264 artifact helper.

## Multicam State

- Loader split-overlap guards exist: train and heldout camera names are rejected
  if they overlap or duplicate.
- DeepView `03_Dog` better-overlap trio is `camera_0005`, `camera_0006`,
  `camera_0014`.
- Preferred interpolation split:
  - train: `camera_0006`, `camera_0014`
  - heldout novel camera: `camera_0005`
  - anchor/condition: `camera_0006`
- The good-set configs exist:
  - `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_learned_residual_relpose_small_bf16_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc`
  - `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats_oracle_relative_camera_goodset_train0006_0014_holdout0005.jsonc`
- No completed good-set training metric, W&B/offline run, output summary, or
  emitted good-set media was found. Do not update `BASELINES.md` for the
  good-set until a real run finishes.
- Existing code should emit per-view `TrainView*`, `Heldout0_camera_0005`, and
  `Multicam_GT_Splat_Alpha_Feature_Grid_Video` diagnostics once the good-set
  run is launched.

## Verification During Cleanup

- `git diff --check`: pass.
- Changed Python compile gate: pass.
- Focused local pytest run from this pass:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_multicam_video_data.py tests/test_pipeline_helpers.py tests/test_source_relative_cameras.py \
  tests/test_config_factory_helpers.py tests/test_relative_pose.py tests/test_sequence_data_single_frame.py \
  tests/test_powerfoam_cuda_smoke.py tests/test_powerfoam_direct.py -q
```

Result: `80 passed, 1 skipped in 31.46s`.

Subagent verification also reported:

- config/multicam/media pytest batch: `31 passed in 7.50s`
- focused PowerFoam pytest batch: `70 passed, 1 skipped in 29.55s`
- changed JSON/JSONC parse check: `136 JSON/JSONC files`
- changed Python compile check: `91 Python files`

No long training was launched in this cleanup pass.

## Next Actions

- PowerFoam: strict raw SSIM diagnostic, raw-quality improvement, feature-foam
  contract/doc, better 2048+ support/coverage init or multires foam, and later
  multicam dynamic-geometry loader/camera support.
- Multicam: run the good-set learned-residual and oracle-relative configs to
  completion, log best-heldout metrics, then update `BASELINES.md` with overlap
  classification and final-vs-best heldout metrics.
- Commit hygiene: split future commits by lane. In particular, stage
  `BASELINES.md`, `AGENTS.md`, `agent_notes/key_learnings.md`, and
  `research_experiments/dynamic_foam/` carefully because they are mixed across
  implementation, benchmark evidence, and future-work notes.
