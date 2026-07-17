# Gate 3 CSR 256px/16f Quality Status - 2026-05-12

## Goal

Produce the first same-split World Foam train/eval heldout metric artifact for
the STAR-UVT and dynamic-splat comparator split, while keeping the scope honest:
fixed geometry, site RGBA/density only, tiled CSR candidate storage, no
geometry/topology gradients, and no full trainer claim.

## Commands

Build the local extension:

```bash
cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
```

Run the tiny comparator-split smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/train_eval_shared_realray_csr_mps.py \
  --max-frames 2 \
  --render-size 16 \
  --steps 1 \
  --site-count 12 \
  --time-slabs 1 \
  --tile-h 8 \
  --tile-w 8 \
  --train-ppm-out /tmp/world_foam_csr_train_smoke.ppm \
  --heldout-ppm-out /tmp/world_foam_csr_heldout_smoke.ppm \
  --out-json /tmp/world_foam_csr_train_eval_smoke.json
```

Run the 256px/16f artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/train_eval_shared_realray_csr_mps.py --steps 5
```

Regenerate the paired report:

```bash
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/src/benchmarks/world_foam_gate0_paired_benchmark.py \
  --star-comparison-json dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle/comparison_report.json \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_9_paired_with_star_dynamic_heldout_pilot.json
```

Run the target-resolution 8-to-16-frame CSR scaling check:

```bash
cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
PYTHONDONTWRITEBYTECODE=1 python3 tools/smoke_shared_realray_csr_scaling_mps.py \
  --config /Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc \
  --frame-counts 8,16 \
  --render-size 256 \
  --site-count 12 \
  --time-slabs 1 \
  --tile-h 8 \
  --tile-w 8 \
  --timing-iters 1 \
  --out-json /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/world_foam_lane2/results/gate2g_mps_shared_realray_csr_scaling_256px_8_16f.json
```

## Result

Saved artifacts:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_csr_quality_256px_16f.json
dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_csr_quality_256px_16f_train.ppm
dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_csr_quality_256px_16f_heldout.ppm
dynaworld/research_experiments/world_foam_lane2/results/gate2g_mps_shared_realray_csr_scaling_256px_8_16f.json
dynaworld/research_experiments/world_foam_lane2/results/gate0_9_paired_with_star_dynamic_heldout_pilot.json
```

Key numbers:

- split: train `camera_0006`, `camera_0014`; heldout `camera_0005`;
- target shape: 256px, 16 frames;
- fixed representation: 12 sites, 66 boundaries, one time slab;
- status: `ok`;
- train PSNR: `10.504453575214932` -> `12.552081185301754`;
- heldout PSNR: `12.703601126620978`;
- train shared scan ratio: `0.0625`;
- heldout shared scan ratio: `0.0625`;
- train tiled CSR storage/bitset ratio: `0.5828577677408854`;
- heldout tiled CSR storage/bitset ratio: `0.5909423828125`;
- missing sample events: zero for train and heldout;
- elapsed: `134.28264766700158` seconds total, `4.296580165995692`
  seconds inside the train loop.
- target-resolution scaling check status: `ok`;
- 256px `8 -> 16` direct scan growth: `2.0x`;
- 256px `8 -> 16` shared scan growth: `1.0x`;
- 256px `8 -> 16` tiled candidate-iteration growth: train
  `1.9840766633277915x`, heldout `1.982501348355007x`;
- 256px scaling max tiled CSR versus bitset MPS output/gradient error: `0.0`.

The regenerated paired report now includes row
`world_foam_mps_shared_realray_csr_quality_256px_16f` with status `ok`.
Comparator numbers in the same report:

- STAR-UVT selected heldout PSNR: `13.888997077941895`;
- STAR-UVT final heldout PSNR: `13.835267066955566`;
- dynamic splat heldout PSNR: `11.190529823303223`.

## Interpretation

This removes the previous blocker that World Foam had no same-split 256px/16f
heldout metric row. The result is a concrete limited comparison: this fixed
World Foam variant beats the older dynamic-splat heldout PSNR in this report
but remains below STAR-UVT selected/final heldout PSNR.

The claim is still narrow. The current artifact optimizes site RGBA/density
through tiled CSR frozen-geometry autograd only. It does not implement
geometry/topology gradients, train site positions/weights, produce a production
trainer, or prove a quality win over STAR-UVT.
