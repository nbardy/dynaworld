# Scale Training Notes Monitor

Date: 2026-05-15

Role:

- Notes/Monitor agent for the scale-training setup pass.
- Write scope was limited to `agent_notes/loose_notes/*` and `agent_notes/key_learnings.md`.
- I did not edit trainer configs, launchers, dataset scripts, shader forks, benchmark scripts, or result artifacts.

## Context Checked

Recent notes and standings inspected:

- `agent_notes/loose_notes/2026-05-15_16-02-15_dynamic_splat_direct_atomic_scaling.md`
- `agent_notes/loose_notes/2026-05-15_06-07-49_world_foam_owner_run_rgb_train_eval.md`
- `agent_notes/loose_notes/2026-05-15_05-32-37_world_foam_completion_audit.md`
- `BASELINES.md`
- `agent_notes/key_learnings.md` before edit: 199 lines, already at the line-limit edge.

Current code/config surface inspected:

- `src/train/train_multicam_relative_pose_implicit_dynamic.py`
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
- `src/train/multicam_video_data.py`
- `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_tokenbudget_world4_fast_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc`
- `src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f.jsonc`
- `src/dataset_configs/youtube_scene_distinct_30_256_4fps_16f.jsonc`
- `src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc`
- `src/train_scripts/build_100_clip_dataset.sh`
- `src/train_scripts/build_local_mac_30_clip_dataset.sh`
- `src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh`
- `src/dataset_scripts/youtube_scene_distinct_30_256_seed.sh`
- `src/dataset_scripts/youtube_scene_distinct_30_seed.sh`

## STAR / direct_atomic: What Worked

The cleanest scaling evidence is the matched 128px frame-count probe:

| frames | dynamic fast_mac step | dynamic fast_mac render | STAR direct_atomic step | STAR direct_atomic render |
|---:|---:|---:|---:|---:|
| 2 | 0.062157 | 0.025863 | 0.020994 | 0.001504 |
| 4 | 0.081947 | 0.034153 | 0.035967 | 0.003196 |
| 8 | 0.162723 | 0.072019 | 0.032473 | 0.002871 |
| 16 | 0.336528 | 0.156292 | 0.023577 | 0.002017 |
| 32 | 0.658497 | 0.309206 | 0.043482 | 0.003544 |

Artifacts:

- `research_experiments/world_foam_lane2/results/2026-05-15_dynamic_splats_fastmac_direct_atomic_frame_scaling_128px_2_4_8_16_32.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_star_uvt_direct_atomic_frame_scaling_128px_2_4_8_16_32.json`

Current model:

- `direct_atomic` is the execution/memory result: it avoids large emitted sample-row workspaces and makes backward feasible where older sample-emission paths blow up.
- STAR/UVT is the frame-amortization result: the representation carries a tube through time instead of rendering an independent dynamic splat set per frame.
- Therefore, direct-atomic execution can be ported to older dynamic-splat formulations for scratch and backward-memory relief, but it will not by itself remove per-frame work. The old formulation still pays roughly with frame count.

Important boundary:

- The STAR evidence above is for the current affine `star_uvt_v0` direct-atomic path, not a complete moving-camera PRT claim.
- The world-foam owner-run lane is promising but still separate: the saved notes show compact owner-run RGB train/eval can be fast, but its record count still scales with frame count and depth replay semantics remain a representational choice, not a solved STAR-like sublinearity proof.

## Latest V-JEPA / Static-Dynamic Multicam Trainer

The script that matches the user's description is:

```bash
PYTHONPATH=src/train uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_tokenbudget_world4_fast_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc
```

What this lane actually does today:

- `arch = "multicam_relative_pose_implicit_camera"`.
- Inherits the multicam precomputed-feature trainer.
- Requires `train.camera_swap_mode = "learned_residual"`.
- Uses precomputed V-JEPA features, static/dynamic tokens, F32 feature splatting, colorization, camera-swap reconstruction, and heldout validation media/metrics.
- Current inspected config uses DeepView `03_Dog`, train cameras `camera_0006` and `camera_0014`, heldout `camera_0005`, condition/anchor `camera_0006`.
- Current inspected config uses `96` static tokens, `32` dynamic tokens, plus token-layout world/register/detail tokens, `64` gaussians per decoded token, V-JEPA 2.1 ViT-B/384 precomputed features, and multires render sampling at `64/128/256/512` with weights `0.25/0.45/0.25/0.05`.
- It writes a checkpoint under `outputs/multicam_relative_pose/.../checkpoint_final.pt`.

Semantic caveat:

- This is not a pure "source image only -> novel view" eval. The relpose lane can use heldout/query camera V-JEPA features during predicted-relpose evaluation. That is a valid query-conditioned novel-view task, but it must be labeled that way if the parent agent scales it.
- If the goal is source-only 3D export plus camera-only novel-view render, the parent agent should include a calibrated/no-query heldout eval or an explicit ablation where heldout RGB features are unavailable to the relpose head.

## Prepared Data Inventory

Current local manifests counted:

| artifact | records |
|---|---:|
| `data/clip_sets/local_mac_30_64_4fps_16f/manifest.jsonl` | 30 |
| `data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_256_4fps_16f/manifest.jsonl` | 30 |
| `data/youtube_curated_spans/clip_sets/youtube_curated_spans_64_4fps_16f/manifest.jsonl` | 19 |
| `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl` | 8 |
| `data/multicam_val/clip_sets/multicam_val_v1_chunked_128_4fps_16f/manifest.jsonl` | 14 |
| `src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f_manifest.jsonl` | 5 |

Other local data:

- `data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_256_4fps_16f` has `30` clip dirs and `480` PNG frames.
- `data/blender_synthetic/sintel/renders` exists and includes the `02_a_full` render with many frames plus `cameras.json`, but I did not find a first-class train manifest that plugs it into the current V-JEPA/static-dynamic trainer.
- `src/train_scripts/build_100_clip_dataset.sh` can build a `local_100_128_4fps_46f` clip set from a source video or directory, but that is not the same as a 1k mixed YouTube/synthetic/multicam scale contract.

Data conclusion:

- The repo has enough prepared data for a 30-ish same-source video scaling smoke and small multicam heldout probes.
- The repo does not yet have an inspected first-class 1k-item train manifest that mixes all prepared YouTube clips, curated spans, synthetic renders, and multicam samples with source/camera-disjoint train/test rules.

## Monitor Risks

Dirty tree:

- `third_party/fast-mac-gsplat` is modified.
- `research_experiments/world_foam_lane2/`, `research_experiments/star_uvt_notes.md`, `src/benchmarks/world_foam_gate0_paired_benchmark.py`, and many STAR/world-foam loose notes are untracked.
- I did not alter any of those.

Concurrent work:

- At inspection time, two Dynaworld `research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py` processes were still running.
- That is outside this notes lane, but it means parent/worker agents should avoid treating world-foam results as settled until their JSON outputs are complete and verified.

Long-training boundary:

- Do not launch the 1k-item training pass until workers first land:
  - a concrete manifest and config pair,
  - a dataset-count/audit command,
  - one offline 1-step runtime smoke for the exact script/config path,
  - and an explicit W&B-on training config for the real run.
- Per project rules, W&B should stay enabled for benchmark/training runs. Disable or offline mode is acceptable for mechanical smokes only.

Leakage and eval semantics:

- Multi-view camera splits must enforce no overlap between train and heldout camera names. `multicam_video_data.py` has a validation check for overlap, but new manifest builders should also audit it before training.
- YouTube source-distinct train/test is a same-source-video generalization benchmark, not a novel-view benchmark. Do not mix those metrics with multicam heldout PSNR in `BASELINES.md`.
- Query-conditioned heldout relpose eval should not be described as no-target-image novel-view synthesis.

Validation gates to ask from workers:

1. Manifest audit: total records, train/test counts, source IDs, camera IDs, duplicates, train/heldout overlaps, target size, fps, frame count.
2. Config resolve smoke: load the final JSONC through the actual trainer class.
3. Runtime smoke: 1-step offline `train_multicam_relative_pose_implicit_dynamic.py` for multicam, plus a tiny dataset smoke for the same-source YouTube path if it is included.
4. Baseline bookkeeping: if a meaningful run completes, append a dated row to `BASELINES.md`; do not overwrite prior rows.
5. Media check: validation grids should show train and heldout/query views separately so source-only, query-conditioned, and calibrated-heldout claims cannot blur.

## Key Learnings Update

I recompressed the final F32 optimization bullets in `agent_notes/key_learnings.md` and added only two new high-signal bullets:

- Direct atomic is the backward memory valve, while STAR/UVT is the frame-count amortizer.
- The current V-JEPA/static-dynamic multicam lane is real but small and query-conditioned; scaling to 1k needs a first-class manifest/eval contract before quality claims.
