# PowerFoam OPENCV_FISHEYE Capacity And Feature Controls

Date: 2026-05-05

Scope: continue the PowerFoam Metal paper-reproduction lane after the
distortion-consistent OPENCV_FISHEYE clean init became the best clean DeepView
row.

## What changed

- Added a 2714-cell OPENCV_FISHEYE config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_2714cells_40step_lowgeom_noaux.jsonc`.
- Ran it with W&B disabled. It consumed all `2714` train-visible filtered
  points from
  `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.ply`.
- Result: selected/final step 40, heldout PSNR `8.5095`, L1 `0.314076`, SSIM
  `0.0260`, source PSNR `7.7709`. This is far below the 1024-cell
  OPENCV_FISHEYE row (`10.5931`), so keeping every filtered point is not the
  missing quality lever.
- Made `verify_powerfoam_clean_init_coverage.py` lens-aware by using
  `project_points_camera(...)` with DeepView `opencv_fisheye` metadata rather
  than pinhole-only `K/z` projection.
- Added direct PowerFoam W&B heldout metric logging for multicam direct runs.
- Added DeepView bundle-level lens metadata test coverage and included
  `tests/test_multicam_video_data.py` in the AGENTS PowerFoam gate.

## Stronger Feature Attempts

- ALIKED smoke command with `--allow-onnx-models` failed locally:
  `ALIKED feature extraction requires ONNX support`. This needs an ONNX-enabled
  pycolmap host.
- Covariant SIFT smoke worked, so a full 8-camera x 4-frame OPENCV_FISHEYE
  affine/domain-size SIFT artifact was built:
  `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_affine_minucam2.ply`.
- Geometry was worse before training: `2087` filtered points vs `2821`, reproj
  median/p90 `2.84/5.21px` vs `2.72/5.19px`, unique-camera p90 still `2`.
  It was not trained.

## Plane-Sweep Geometry Control

- Upgraded `research_experiments/dynamic_foam/build_multiview_plane_sweep_point_cloud.py`
  from the older pairwise prototype into a lens-aware all-train-camera builder.
  It uses `project_points_camera(...)`, DeepView `opencv_fisheye` metadata,
  configurable `--target-size`, `--source-views`, `--min-support`, and
  `--max-error`, then sorts points by mean train-view color consistency.
- Full artifact:
  `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_multiview_plane_sweep_frame0_128px_opencv_fisheye_stride4_support4_8192pts.ply`.
  It used all 8 train cameras, frame 0, 128px, depth range `0.25..8.0`,
  48 depths, stride 4, and `min_support=4`. It wrote `7830` points with
  support mean/median/p90 `5.78/6/7` and median/p90 color error
  `0.120/0.241`.
- Matched PowerFoam config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_multiview_plane_sweep_frame0_128px_opencv_fisheye_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
  The trainer saw `7818` train-visible points and kept the first/top 1024 by
  consistency error.
- Result: selected/final step 40, heldout PSNR `8.2487`, L1 `0.327659`,
  SSIM `0.0160`, source PSNR `8.8386`; the train loop reported `13.67s`.
  This is far below the selected OPENCV_FISHEYE pycolmap row (`10.5931`), so
  naive local plane sweep is not the missing clean-geometry lever.
- Added experimental score controls to the same builder:
  `--score-mode center_l1|mean_l1|patch_l1|zncc`, `--patch-radius`,
  `--min-patch-std`, and `--support-error`. Tiny 32px real-data smokes passed
  for `patch_l1`, `mean_l1`, `zncc`, and `center_l1 --support-error 0.2`.
  Larger 96px patch/ZNCC smokes were stopped after roughly 2 minutes without
  producing artifacts, so the patch-sampling implementation is not viable for
  full local sweeps until it is vectorized.
- Built one cheap full 8-camera center-L1 photometric-inlier artifact:
  `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_multiview_plane_sweep_frame0_128px_opencv_fisheye_stride4_support4_inlier02_8192pts.ply`.
  It uses the same 128px/frame0/48-depth/stride4 setup as the negative
  plane-sweep row, but counts target support only when per-view color error is
  `<=0.2`. It wrote `6450` points, median/p90 error `0.0680/0.1151`, and
  support mean/median/p90 `4.63/4/6`. This is cleaner than the raw-support
  artifact (`7830` points, median/p90 `0.120/0.241`) but is not trained yet.
- Added the matching ready-to-run config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_multiview_plane_sweep_frame0_128px_opencv_fisheye_inlier02_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
  After clearing generated Python caches, the 40-step run completed with W&B
  disabled. It selected step 0 at heldout PSNR `9.0679`, L1 `0.291473`, SSIM
  `0.0458`; final step 40 was heldout PSNR `8.8609`, L1 `0.297917`, SSIM
  `0.0390`. This is an improvement over raw plane sweep (`8.2487`) but still
  below the OPENCV_FISHEYE pycolmap row (`10.5931`) and still selects the
  initial checkpoint.

## Verification

- `py_compile`: passed for touched Python files.
- `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_multicam_video_data.py tests/test_powerfoam_direct.py -q`
  passed: `43 passed, 3 skipped`.
- `verify_powerfoam_paper_acceptance.py --allow-incomplete` now selects the
  W&B-offline appearance-only OPENCV_FISHEYE clean row and fails only the
  expected official fixture and clean-quality gates: missing official CUDA/Warp
  fixture, PSNR `<13`, and SSIM `<0.15`.
- Lens-aware `verify_powerfoam_clean_init_coverage.py --allow-incomplete` still
  fails the filtered-point retention diagnostic because the selected 1024-cell
  row samples only `1024/2714` filtered points.
- The plane-sweep builder was exercised by a 2-source smoke and by the full
  8-source artifact build above; the matched PowerFoam training run completed
  and wrote `best_metrics.json`.
- `verify_powerfoam_paper_acceptance.py --allow-incomplete` and
  `verify_powerfoam_clean_init_coverage.py --allow-incomplete` now include the
  plane-sweep and appearance-only candidates in their clean-candidate inventory.
  The selected clean candidate is the OPENCV_FISHEYE pycolmap artifact plus the
  W&B-offline appearance-only Metal run at heldout PSNR `10.8536`; the
  plane-sweep rows remain negative clean branches at heldout PSNR `8.2487` and
  `9.0679`.
- Latest scoped gate after the inlier-support training and verifier updates:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest -p no:cacheprovider tests/test_multicam_video_data.py tests/test_powerfoam_direct.py -q`
  passed with `43 passed, 3 skipped`. Paper acceptance still reports `ok=false`
  with failed gates: official CUDA/Warp fixture missing, clean PSNR `<13`, and
  clean SSIM `<0.15`. Clean-init coverage still reports `ok=false` because the
  selected row samples only `1024/2714` filtered points.
- Added `research_experiments/dynamic_foam/run_powerfoam_external_blockers.py`
  so the remaining nonlocal blockers are executable on a capable host. It now
  checks for `uv`, defaults the clean-geometry branch to ALIKED/LightGlue, and
  gives matcher-specific output filenames so brute-force and LightGlue artifacts
  cannot overwrite each other. It also has explicit `write-train-config` and
  `train-aliked` tasks for the matched W&B-backed Metal training run. Those
  tasks template the selected OPENCV_FISHEYE config, require the artifact PLY
  and JSON summary before writing/running, require `point_count > 0`, require
  the summary matcher to match the CLI matcher, and write matcher-specific
  output dirs/run names. Dry-run examples passed locally for official fixture
  generation, official parity tests, the OPENCV_FISHEYE ALIKED/LightGlue
  pycolmap artifact command, and the generated training config/command. Local
  `check` correctly reports `torch.cuda.is_available=False`, missing `warp`,
  and missing `pycolmap` in the venv. If the official fixture is generated on a
  Linux/CUDA host, copy it back to this Mac before running the Metal-vs-official
  fixture test, because that test still requires MPS to exercise the Metal path.
- Updated the paper-acceptance and clean-init coverage verifiers so future
  ALIKED/LightGlue and ALIKED/bruteforce clean candidates are picked up once
  their artifact JSON/PLY and run `best_metrics.json`/`resolved_config.json`
  exist. Missing optional ALIKED candidates are reported but do not break the
  current verifier state.
- Added `research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py`
  as the prompt-level gate for the active objective. It calls the saved 4K
  verifier and paper-acceptance verifier, records the focused local Metal test
  command (or runs it with `--run-local-tests`), checks official fixture
  presence, checks W&B backing, and repeats the heldout PSNR/SSIM thresholds.
  The audit is intentionally stricter than any single verifier so local Metal
  parity or saved 4K performance cannot be mistaken for full/proper PowerFoam.
  A current run with `--run-local-tests --allow-incomplete` reports `ok=false`:
  local Metal pytest passed (`43 passed, 3 skipped`), the targeted local Metal
  fixture-backward node passed, the low-level raytrace parity script passed,
  saved 4K passed, and the blocker list is now exactly official fixture
  missing, targeted official Direct/Metal parity nodes not run, paper acceptance
  still failing, and heldout PSNR/SSIM below threshold. The selected paper row
  now passes the W&B-backed and post-initial checkpoint gates.

## 4K Optimizer-Step Trainability

- Added/saved the focused verifier evidence from
  `research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py`.
  Default artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_trainability_1024cells_2026-05-05.json`.
- The verifier generated and verified a real `3840x2160`, `1024`-cell,
  `cech_aabb`, `oriented_height_sv_texel_surface` MPS optimizer step. This
  closes the gap between saved backward timing and one-step trainability
  evidence on the synthetic 4K lane.
- Successful metrics: `ok=true`, `loss_before=0.07926274836063385`,
  `loss_after=0.0789613351225853`, `loss_ratio=0.9961972900980273`,
  `grad_abs_max=0.010738098062574863`,
  `density_update_abs_max=0.0026845335960388184`,
  `forward_ms=1195.255124999676`,
  `backward_ms=2181.0507500013046`,
  `after_forward_ms=1217.3038750006526`.
- This still does not complete full/paper PowerFoam acceptance. The official
  CUDA/Warp fixture and official parity nodes remain missing/skipped; the
  paper-scale selected row is now W&B-offline-backed and selected at step `40`,
  but heldout quality remains below threshold at PSNR `10.853645324707031`,
  SSIM `0.07661956548690796`.

## Appearance-Only Clean Row Follow-Up

- Added eval/train JSONL history instrumentation to
  `src/train/train_powerfoam_metal.py`. New runs write
  `eval_metrics_history.jsonl` and `train_metrics_history.jsonl`, and
  `checkpoint_final.pt` now stores final metrics when the final step is
  evaluated.
- Ran the selected OPENCV_FISHEYE clean artifact with geometry frozen and only
  SV RGB trainable:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc`.
  It used W&B offline run `wandb/offline-run-20260505_223541-j0u3b4up` and
  selected step `40` with heldout PSNR `10.853645324707031`, L1
  `0.23019294440746307`, SSIM `0.07661956548690796`. This is now the selected
  clean DeepView candidate, so the W&B and post-initial gates pass.
- Ran the `first1024` version:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_first1024_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc`.
  It was worse: best heldout PSNR `10.053520202636719`, L1
  `0.24847407639026642`, SSIM `0.060867756605148315` at step `20`. The simple
  lowest-reprojection-point subset is not the missing lever.
- ALIKED/LightGlue remains external-host work. A local 128px/two-camera smoke
  with `--feature-type aliked_n16rot --matcher-type aliked_lightglue
  --allow-onnx-models` aborted with `ALIKED feature extraction requires ONNX
  support`, confirming the Mac pycolmap wheel cannot generate that candidate.

## Material-Only Clean Row Follow-Up

- Ran a narrow follow-up to test whether the frozen RGB-only row was leaving
  easy heldout quality on the table by freezing too much of the surface
  material frame:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_materialonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc`.
- This config freezes centers, radii, density, and quaternions, but trains
  texel sites, texel heights, SV axes, and SV RGB. W&B offline run:
  `wandb/offline-run-20260505_230827-c4reurai`.
- Result: negative for the paper gate. `best_metrics.json` selected step `0`
  by heldout PSNR: PSNR `10.851705551147461`, L1 `0.23040729761123657`, SSIM
  `0.07515893876552582`. Final step `40` had heldout PSNR
  `10.838074684143066`, L1 `0.23040194809436798`, SSIM
  `0.07918539643287659`. The material frame moved, but PSNR fell and SSIM
  remains far below `0.15`.
- Interpretation: allowing texel sites/heights/SV axes to move is not the
  missing local lever for this clean artifact. The selected paper row remains
  the RGB-only appearance run; the remaining quality gap still points to
  stronger geometry/track support or a genuinely new representation idea.

## Heldout Error / View Coverage Diagnostic

- Added and ran
  `research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py` on
  the selected RGB-only appearance row. It renders train and heldout splits
  from `checkpoint_best.pt`, writes per-split alpha/error bins, per-frame rows,
  per-view rows, and worst-sample labels to
  `outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/heldout_error_diagnostics.json`.
- The heldout view is not blank: `camera_0040` has heldout PSNR
  `10.853645324707031`, L1 `0.23019294440746307`, alpha mean
  `0.632867693901062`, alpha `<0.05` fraction `0.33056640625`, and alpha
  `>0.5` fraction `0.65423583984375`.
- The train split exposes a stronger support failure than the heldout scalar
  alone showed. Per-view train rows sorted by L1:
  `camera_0013` has alpha mean `0.0`, alpha `<0.05` fraction `1.0`, L1
  `0.38534069061279297`; `camera_0021` also has alpha mean `0.0`, alpha
  `<0.05` fraction `1.0`, L1 `0.38149338960647583`. `camera_0015` is nearly
  blank too with alpha mean `0.06388688087463379` and alpha `<0.05` fraction
  `0.9326171875`.
- Worst train samples are zero-alpha rows from named cameras, not anonymous
  sample-index noise: the top two are `camera_0021` frames `0` and `1`; the
  next several are `camera_0013` frames `4`-`8`.
- Interpretation update: this is not just a heldout-only generalization miss
  and not a material-frame LR problem. The selected clean artifact is
  over-covering the heldout view relative to some train cameras
  (`heldout/train alpha_mean ratio 4.59x`) while entirely missing other train
  views. The next local experiment should diagnose the camera pose/support
  distribution and cell initialization visibility by view before another broad
  schedule sweep.

## Raytrace Support-Gap Diagnostic

- Added and ran
  `research_experiments/dynamic_foam/diagnose_powerfoam_raytrace_support_gap.py`
  on the same selected RGB-only appearance row. It does not rerender; it
  decodes checkpoint centers/radii from `checkpoint_best.pt`, projects those
  centers through the configured OPENCV_FISHEYE train/heldout cameras, and
  joins the projection counts with `heldout_error_diagnostics.json`.
- Output:
  `outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/raytrace_support_gap_diagnostics.json`.
- Result: the only strict support-gap views are `camera_0021` and
  `camera_0013`. They have many projected checkpoint centers but saved
  raytrace alpha `0.0`:
  `camera_0021` has visible-center mean `873.0`, center-pixel mean `762.0`,
  center-pixel coverage `0.0465087890625`, raytrace alpha mean `0.0`;
  `camera_0013` has visible-center mean `890.0`, center-pixel mean `810.0`,
  center-pixel coverage `0.0494384765625`, raytrace alpha mean `0.0`.
- Heldout `camera_0040` has visible-center mean `803.0`, center-pixel mean
  `700.0`, center-pixel coverage `0.042724609375`, and raytrace alpha mean
  `0.632867693901062`, so center projection itself is not enough to explain
  alpha.
- Two independent read-only probes also ruled out the easy plumbing failures:
  the train/heldout camera order, view-major flattening, OPENCV_FISHEYE ray
  construction, and model-frame point-cloud handling look internally
  consistent; forcing the same checkpoint through the non-raytrace streaming
  renderer gives nonzero alpha for `camera_0021` and `camera_0013`.
- Interpretation update: the selected paper row is now blocked by raytrace
  traversal/start-cell/connectivity on oblique train views, not by missing
  decoded centers in those cameras. The suspicious surface is the raytrace
  wrapper's single origin-seeded `start_id` per image batch. Do not treat the
  saved 4K synthetic raytrace gate as proof that this real-scene traversal case
  is accurate.

## Raytrace Default-Start Patch

Context:

- The Metal ABI and kernels still take one `start_id` per image batch, not one
  per pixel. This is visible in `rasterize.py`, the C++ shape check, and the
  Metal kernels that read `start_ids[batch]`.
- Hypothesis: the zero-alpha train views can occur when the camera-origin
  nearest-power cell is unsupported by the rendered ray bundle or disconnected
  under the selected adjacency. With only one start cell, a bad origin seed can
  make the ray walk terminate before reaching visible cells even when many
  centers project into the image.
- Backtrack: this does not prove per-ray start ids are unnecessary. It is a
  Python-wrapper robustness patch that preserves the `[B]` ABI for the fast
  current kernels. True per-ray/tiled starts still require Python, C++, and
  Metal ABI changes.

Change:

- Updated `third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py` so
  `_default_start_ids` keeps the origin-nearest cell when sampled rays actually
  support it, but switches to the sampled visible cell with larger ray-support
  count when the origin choice has weaker/no support.
- The support count samples up to a `9x9` grid of rays per batch image and
  counts nearest cells whose closest approach to a sampled ray intersects the
  radius. This avoids a 4K-sized per-pixel prepass while making the default
  camera-ray-aware.

Falsification test:

- Added an unsupported-origin fixture to
  `third_party/powerfoam-metal/tests/raytrace_check.py`: two cells, one nearer
  to the camera origin but outside the oblique ray support, one visible in the
  image, and empty adjacency so the old origin start cannot walk to the visible
  cell.
- Ran:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/powerfoam-metal:src/train .venv/bin/python third_party/powerfoam-metal/tests/raytrace_check.py
```

- Result: pass. The new fixture printed old forced-start alpha max `0.0`, new
  default alpha max `0.580209493637085`, forward feature/alpha/normal-distance
  max error `0.0`, and unsupported-origin grad max errors within
  `1.341104507446289e-07`. Existing all-pairs height+SV forward/backward still
  passed; regular-triangulation still skipped locally because SciPy is absent.

Real-checkpoint refresh attempt:

- Tried to rerun `diagnose_powerfoam_heldout_error.py` on the selected
  OPENCV_FISHEYE appearance-only checkpoint after the patch. It exceeded two
  minutes without output and was killed. Treat the synthetic low-level verifier
  as the current regression evidence, not as proof that `camera_0021` and
  `camera_0013` are fixed in the saved 128px checkpoint.
- Added a non-rendering real-scene start verifier:
  `research_experiments/dynamic_foam/verify_powerfoam_raytrace_start_support.py`.
  It reconstructs train rays from the multicam camera matrices, decodes
  checkpoint centers/radii, and records origin start support vs patched default
  start support without invoking the Metal raytrace kernel.
- Ran:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_raytrace_start_support.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc
```

- Output:
  `outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/raytrace_start_support_diagnostics.json`.
- Result: for the two known zero-alpha train views, every sampled frame now
  switches away from the origin-nearest cell. `camera_0013`: switch fraction
  `1.0`, origin support mean `0.0`, default-start support mean `11.0`.
  `camera_0021`: switch fraction `1.0`, origin support mean `0.0`,
  default-start support mean `8.0`. This is real-view support evidence for the
  patch, still not a rendered-alpha proof.

## Real-View Rendered-Alpha Diagnostic

Context:

- The non-rendering support verifier was not enough: it proved the origin start
  had no sampled support, but not that the ray-walk output matched the streaming
  renderer.
- Added `research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py`.
  It decodes `checkpoint_best.pt` directly, reconstructs the OPENCV_FISHEYE
  rays from `multicam_matrices(cfg)`, samples a small pixel grid from the
  configured 128px camera, and calls the same height+SV Metal primitive with:
  patched default starts, forced old origin starts, and the non-raytrace
  streaming renderer.

Commands:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc --sample-size 9 --frames 0 4 --adjacency-mode cech_aabb
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc --sample-size 9 --frames 0 4 --adjacency-mode all_pairs --output outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/raytrace_real_view_alpha_allpairs_diagnostics.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy python research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc --sample-size 9 --frames 0 4 --adjacency-mode regular_triangulation --output outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/raytrace_real_view_alpha_regular_diagnostics.json
```

Result:

- Cech/AABB with per-ray default start fixes the strict zero-alpha symptom but
  not parity. For `camera_0021` / `camera_0013` frames `0` and `4`, old origin
  alpha mean is `0.0`; patched alpha mean rises to `0.617` / `0.694`, but
  streaming alpha mean is `0.893` / `0.899`. Mean alpha error remains
  `0.300` / `0.323`, with max alpha error `1.0`.
- All-pairs and regular triangulation show that the old origin start is often
  closer to streaming than the visible per-ray start when connectivity is rich
  enough. All-pairs old-origin means are `0.892` / `0.979`; regular old-origin
  means are `0.772` / `0.843`; the visible per-ray default stays around
  `0.647` / `0.695`.
- Interpretation update: the initial zero-alpha bug was real, but the deeper
  correctness blocker is now ray-walk start semantics plus graph connectivity.
  A visible-cell per-ray start is a rescue heuristic, not the official power
  diagram start. For real parity, the raytrace path needs a robust way to start
  in the power cell at the ray near point and traverse a graph that contains the
  required power-face neighbors. Cech/AABB is too sparse/incorrect for these
  real oblique views; all-pairs is accurate-ish but not a fast training graph.

## Next

Do not spend the next turn on more cell-count, LR, or plane-sweep scoring
tweaks unless there is a specific hypothesis beyond raw support and
photometric-inlier support. The immediate local follow-up is to redesign the
raytrace start/connectivity contract, not to keep sweeping material LRs. Viable
branches: near-plane power-cell starts plus regular-triangulation traversal,
faster regular graph construction/cache, a low-res streaming fallback that is
clearly labeled as non-raytrace, or a tile/per-ray start kernel that preserves
official ordering. After that, use an ONNX-enabled ALIKED/LightGlue host, a CUDA dense COLMAP
path, or a better multi-view track builder that improves distinct camera
support beyond p90 `2`. Use the external blocker runner first on that host so
fixture parity and stronger clean geometry are captured with comparable
commands.

## Near-Plane Start And Regular-Graph Diagnostic

Follow-up time: 2026-05-06 03:05 +07.

The previous rendered-alpha diagnostic was too coarse: it compared only forced
origin start and the then-current default per-ray closest-line start. I
extended
`research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py`
so each sampled real-view row can compare start modes in a loop:

- `origin`: one batch start from power distance at the camera origin.
- `default_per_ray`: the renderer's implicit default.
- `near_plane`: per-ray power distance at `origin + near_plane * dir`.

It can now also evaluate multiple adjacency modes in one run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal uv run --with scipy python \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc \
  --sample-size 9 \
  --frames 0 4 \
  --adjacency-modes cech_aabb regular_triangulation all_pairs \
  --output /tmp/raytrace_start_modes_20260506.json
```

Before changing the renderer default, the best sampled start was
`near_plane`: mean alpha error `0.01797` and feature max error `0.2354` on the
best row, versus the default closest-line start staying around alpha means
`0.647` / `0.695` for the regular/all-pairs graph. This falsifies the
closest-line start heuristic for this real scene: it can jump too far down the
ray and skip front power cells that the streaming renderer integrates.

I then patched
`third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py` so
`_default_per_ray_start_ids` uses the near-plane power cell when
`config.near_plane > 0`, while preserving the old closest-line fallback for
zero-near-plane synthetic fixtures. Re-running the same diagnostic wrote:

```text
/tmp/raytrace_start_modes_after_nearplane_default_20260506.json
```

Key sampled rows after the patch:

```text
camera_0021 regular_triangulation stream_alpha_mean=0.8927 default_alpha_mean=0.9107 alpha_mean_error=0.01797 feature_max_error=0.2354
camera_0013 regular_triangulation stream_alpha_mean=0.8992 default_alpha_mean=0.9788 alpha_mean_error=0.07962 feature_max_error=0.5359
camera_0021 cech_aabb              stream_alpha_mean=0.8927 default_alpha_mean=0.5223 alpha_mean_error=0.37442 feature_max_error=0.9068
camera_0013 cech_aabb              stream_alpha_mean=0.8992 default_alpha_mean=0.4897 alpha_mean_error=0.40956 feature_max_error=0.5665
```

Interpretation: near-plane starts fix the default-start semantics for positive
near planes, but Cech/AABB still undercovers the bad oblique views. The
promising local path is near-plane per-ray start plus regular-triangulation
connectivity. All-pairs does not improve over regular on the sampled rows, so
regular has the needed graph edges here without the dense edge count.

Verification after the renderer patch:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/powerfoam-metal .venv/bin/python \
  third_party/powerfoam-metal/tests/raytrace_check.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/powerfoam-metal uv run --with scipy python \
  third_party/powerfoam-metal/tests/raytrace_check.py
```

Both raytrace checks passed. The SciPy-enabled run included regular
triangulation and matched dense/raster forward and backward on the tiny
constant and height+SV fixtures.

I tried a 1-step trainer smoke by generating
`/tmp/powerfoam_regtri_nearstart_1step_20260506.jsonc` from the selected
DeepView config with `model.adjacency_mode="regular_triangulation"`,
`train.steps=1`, and W&B disabled, then launching:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal WANDB_MODE=offline uv run --with scipy python \
  src/train/train_powerfoam_metal.py \
  /tmp/powerfoam_regtri_nearstart_1step_20260506.jsonc
```

I interrupted it after about `2:21` with no trainer output. The traceback
showed it was still in multicam OpenCV frame loading, not Qhull or the
regular-triangulation raytrace path:

```text
load_multicam_video_bundle -> load_camera_video -> _sample_video_frames -> capture.set(...)
KeyboardInterrupt
```

So we have a fast rendered-alpha and low-level parity gate, but not yet a
completed regular-triangulation trainer smoke. The next compute-efficient
follow-up should either cache/use the already decoded tiny sampled rays for a
renderer-level loss smoke, or run the full trainer with enough wall time and
explicit startup logging so data-loading silence does not look like a graph
hang.

## Regular-Triangulation Trainer Probe

Follow-up time: 2026-05-06 02:51 +07.

First, I timed the suspected loader hang directly. Loading one 16-frame
DeepView camera through the current OpenCV path at 128px takes about
`23-25 s`:

```text
camera_0001 (16, 3, 128, 128) 24.80 s
camera_0021 (16, 3, 128, 128) 25.10 s
camera_0013 (16, 3, 128, 128) 23.46 s
camera_0040 (16, 3, 128, 128) 24.44 s
```

So the earlier `2:21` interrupted trainer smoke was still plausibly in normal
data loading: 8 train videos plus 1 heldout video can be minutes of silent
startup.

I then ran a one-frame `regular_triangulation` trainer probe from a temp config
with `data.max_frames=1`, `data.frame_indices=[0]`, `train.steps=1`, and
`train.frames_per_step=1`. It passed the real trainer path with
`regular_triangulation`, no Qhull failure, and no replay-cap failure. Step-0
heldout was already PSNR `12.3569`, SSIM `0.1133`; step 1 was effectively
unchanged at PSNR `12.3579`, SSIM `0.1135`.

Then I ran the matched full 16-frame selected-row probe:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=offline \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy python \
  src/train/train_powerfoam_metal.py \
  /tmp/powerfoam_regular_nearplane_16f_40step_20260506.jsonc
```

The durable checked-in config is:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc
```

Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux
wandb/offline-run-20260506_024505-qsfn7j80
```

Result:

- Step 0 selected by heldout PSNR: `heldout_eval_psnr=12.509903907775879`,
  `heldout_eval_l1=0.17935504019260406`,
  `heldout_eval_ssim=0.11691904067993164`.
- Step 40 final slightly overfit source: train PSNR rose to `12.9898`, but
  heldout fell to PSNR `12.4527`, L1 `0.180542`, SSIM `0.1160`.
- Regular graph stats at init: avg degree `14.076`, max degree `54`,
  required-overlap edges `8468`, missing-overlap edges `1690`.
- Train loop elapsed at step 40: `247.11 s`; progress-bar wall including eval
  and media was about `5:05`, plus the silent full-res video decode startup.

I added this run to `BASELINES.md` and to
`CLEAN_DEEPVIEW_CANDIDATES` in
`research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py`.
The paper verifier now selects the regular-triangulation run as the best clean
DeepView candidate. It is a major improvement over the prior selected Cech/AABB
appearance-only row (`10.8536` / `0.0766`), but it still fails paper acceptance:

```text
clean_heldout_psnr_threshold: actual 12.5099, required 13.0
clean_heldout_ssim_threshold: actual 0.1169, required 0.15
```

The completion audit now has a more precise remaining blocker: paper-scale
heldout quality is close but still not over threshold, and the selected best
checkpoint is step 0 rather than a post-initial improvement. This confirms the
raytrace/connectivity diagnosis while showing that source-view appearance
training alone does not close the last paper-quality gap.

## Background And All-Filtered-Point Checks

Follow-up time: 2026-05-06 03:15 +07.

I checked whether the remaining regular-row gap was just black-background
compositing through residual low-alpha pixels. A heldout-only eval loaded only
`camera_0040`, rendered the regular step-0 checkpoint once, then recomposited
with several fixed backgrounds and a least-squares per-channel constant
background. Output:

```text
/tmp/powerfoam_regular_background_sweep_20260506.json
```

Results:

```text
black background:        PSNR 12.5099, SSIM 0.1169, L1 0.1794
gray 0.25 background:   PSNR 12.7425, SSIM 0.1244, L1 0.1741
gray 0.50 background:   PSNR 12.7849, SSIM 0.1213, L1 0.1728
best L2 RGB background: PSNR 12.7958, SSIM 0.1228, L1 0.1720
```

So background explains only about `+0.29 dB` and does not get SSIM close to
`0.15`. The alpha mean on the raytrace heldout render is already `0.978`, so
the remaining gap is mostly foreground geometry/color, not empty background.

I also tested whether retaining all `2714` train-visible filtered
OPENCV_FISHEYE points with regular connectivity would close the gap. Temp
config:

```text
/tmp/powerfoam_regular_2714_1step_zero_lr_20260506.jsonc
```

This uses `model.cells=2714`, `regular_triangulation`, `train.steps=1`,
`frames_per_step=1`, and all LRs zero, so it is effectively a step-0/forward
probe plus one no-op backward smoke. Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_128_16f_2714cells_step0_noaux
```

It passed the trainer path and one backward step without replay-cap failure.
Graph stats: avg degree `14.30`, max degree `66`, overlap edges `32634`,
missing overlap edges `13178`. Metrics:

```text
train PSNR 13.1448, SSIM 0.1376
heldout PSNR 12.6368, SSIM 0.0942, L1 0.1814
```

This improves heldout PSNR by only `+0.127 dB` over the 1024-cell regular row
and hurts SSIM badly (`0.1169 -> 0.0942`). Do not promote this all-filtered
regular point-count probe into `CLEAN_DEEPVIEW_CANDIDATES`: it is not
W&B-backed, does not improve the paper gate, and would distract the selector
because it has slightly higher PSNR but worse SSIM and no training evidence.
