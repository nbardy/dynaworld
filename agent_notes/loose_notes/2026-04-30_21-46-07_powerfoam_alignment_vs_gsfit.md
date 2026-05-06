# PowerFoam alignment vs GS fit

## User observation

The first online PowerFoam video looked like large random colors and did not line up well with the target clip.

## What was verified

- The direct `FreeDynamic3DGS` fit baseline initializes from first-frame pixels:
  - `research_experiments/gauge_fields/data.py::initialize_material_points_from_first_frame`
  - `research_experiments/gauge_fields/train_splat_baseline.py` passes `init_rgb` into `FreeDynamic3DGS`, which repeats it across frames.
- The token-GS configs are different. For example `local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc` uses `rgb_init: "uniform"`, so a step-1 token-GS video can be random. Do not cite token-GS step-1 visuals as evidence that direct GS fit starts from image colors.
- Official PowerFoam is not a Mac-runnable pure Torch baseline:
  - README requires Linux, CUDA 12.x, and a CUDA GPU.
  - `powerfoam/rasterize.py` uses Warp CUDA kernels inside a custom `torch.autograd.Function`.
  - the forward/backward paths call `torch.cuda.current_stream()`.

## Local comparison

PowerFoam direct before image init:

- W&B run: `bbdtbwnu`
- 32 cells, 8 neighbors, random positions/colors
- step 100 eval: `eval_l1=0.1303576082`, `eval_mse=0.0320682749`

PowerFoam direct after image-seeded init:

- W&B run: `wv3tvwq9`
- 32 cells, 8 neighbors, image-grid positions/colors
- step 100 eval: `eval_l1=0.1035678983`, `eval_mse=0.0213639084`
- local preview: `/tmp/powerfoam_direct_image_init_100/preview_step_0100.png`

PowerFoam direct after adding step-0 logging:

- W&B run: `q3v0ybut`
- 32 cells, 8 neighbors, image-grid positions/colors
- step 0 eval: `eval_l1=0.2057821155`, `eval_mse=0.0586132519`
- step 100 eval: `eval_l1=0.1035678983`, `eval_mse=0.0213639084`
- Interpretation: the W&B init view is now explicitly logged, and it confirms the initialization itself is weak rather than just missing from the old videos.

Direct GS fit baseline:

- Command used a temp config `/tmp/gsfit_same_source_100.json`.
- 2048 per-frame splats, fast-mac, 100 steps, same 16-frame 128px clip.
- Final metrics: `eval_l1=0.0751253963`, `eval_mse=0.0143629592`, `eval_psnr=18.4275608063`.
- local preview: `/tmp/gsfit_same_source_100/preview.png`

Single-image PowerFoam baseline:

- Config: `src/train_configs/local_mac_powerfoam_direct_single_image_128_smoke.jsonc`
- W&B run: `wysnb33d`
- 1 frame, 64 cells, 8 neighbors, image-seeded init, 100 steps.
- Final metrics: `eval_l1=0.0619022548`, `eval_mse=0.0092814444`.
- Local preview: `outputs/powerfoam_direct/local_mac_powerfoam_direct_single_image_128_smoke/preview_step_0100.png`

Single-image PowerFoam after adding step-0 logging:

- W&B run: `9r354vsw`
- 1 frame, 64 cells, 8 neighbors, image-seeded init, 100 steps.
- step 0 eval: `eval_l1=0.2558635175`, `eval_mse=0.0818815604`
- step 100 eval: `eval_l1=0.0619022548`, `eval_mse=0.0092814444`

Single-image direct GS fit baseline:

- Command used temp config `/tmp/gsfit_single_image_100.json`.
- 1 frame, 2048 splats, fast-mac, 100 steps.
- Final metrics: `eval_l1=0.0381829441`, `eval_mse=0.0037491247`, `eval_psnr=24.2607002258`.
- Local preview: `/tmp/gsfit_single_image_100/preview.png`

## Interpretation

The random-looking PowerFoam web video was partly a bad initialization artifact. The image-seeded init improved the final score, but the new step-0 logs show it is still a poor visual/metric initialization.

It still does not match GS fit because the current PowerFoam trainer is a minimal Torch prototype, not the official PowerFoam model:

- it uses only 32 cells, versus GS fit's 2048 splats and official PowerFoam's tens of thousands to millions of points;
- it uses fixed KNN adjacency, while official PowerFoam rebuilds a Cech/power adjacency from an AABB tree;
- it uses one constant RGB per cell, while official PowerFoam uses normals/tangents plus texel sites and learned view-dependent color;
- it omits the official surface-plane clipping/color query inside each power cell;
- exact all-neighbor clipping (`neighbor_count=31`) was started but stopped after 31/100 steps because it was around 4-5x slower in the Torch prototype.

The next correctness step should be a small isolated reference test that compares our Torch segment clipping against the official `rendering_math.py` formulas, then a real Metal/Warp-style renderer with official attributes. Increasing cell count in the Torch prototype is not the right long-term path.

The single-image run is useful as a fast sanity gate: if a future foam renderer cannot beat `~0.062` L1 on this one frame at 64 cells, its geometry/color/gradient path is probably wrong. The GS fit one-frame number remains much lower at `~0.038` L1.
