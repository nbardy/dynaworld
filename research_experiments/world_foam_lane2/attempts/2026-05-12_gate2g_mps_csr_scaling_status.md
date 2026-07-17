# Gate 2G MPS CSR Scaling Status - 2026-05-12

## Scope

Gate 2G extends Gate 2F from a 16px/2-frame CSR parity smoke to a small
32px frame-count sweep on the shared real-ray reduced-VJP path.

It is still fixed-geometry evidence:

- fixed 4D sites and weights;
- fixed 4D power boundaries;
- fixed segment ownership and sorting;
- gradients only for site RGBA/density;
- no production trainer;
- no heldout-quality claim.

## Commands

From the repo root:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_shared_realray_csr_candidate_storage_mps.py \
  --max-frames 2 --render-size 16 --time-slabs 1 --tile-h 8 --tile-w 8 --timing-iters 5 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2f_mps_shared_realray_csr_candidate_storage_smoke.json
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_shared_realray_csr_scaling_mps.py \
  --frame-counts 2,4,8 --render-size 32 --time-slabs 1 --tile-h 8 --tile-w 8 --timing-iters 3 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2g_mps_shared_realray_csr_scaling_smoke.json
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/src/benchmarks/world_foam_gate0_paired_benchmark.py \
  --star-comparison-json dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle/comparison_report.json \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_9_paired_with_star_dynamic_heldout_pilot.json
```

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  dynaworld/src/benchmarks/world_foam_gate0_paired_benchmark.py \
  dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/torch_world_foam_lane2/ops.py \
  dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_shared_realray_csr_scaling_mps.py \
  dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_shared_realray_csr_candidate_storage_mps.py
python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/static_validate.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s dynaworld/research_experiments/world_foam_lane2
```

`static_validate.py` passed host checks and skipped offline Metal compilation
because `xcrun` could not find the Metal compiler. Runtime MPS dispatch did run.

## Result

Saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate2g_mps_shared_realray_csr_scaling_smoke.json
```

Status: `ok`.

Acceptance:

- `all_csr_rows_valid=true`;
- `all_rows_zero_missing=true`;
- `tiled_csr_matches_bitset_mps=true`;
- `tiled_csr_storage_below_bitset=true`;
- `shared_scan_growth_sublinear=true`;
- `tiled_candidate_iterations_sublinear=true`;
- `outputs_are_finite=true`.

Key numbers:

- frame counts: `2,4,8`;
- render size: `32`;
- time slabs: `1`;
- sites/boundaries: `12 / 66`;
- max tiled CSR vs bitset MPS output/gradient error: `0.0`;
- train direct scan growth from 2 to 8 frames: `4.0x`;
- train shared candidate-build scan growth: `1.0x`;
- train tiled CSR candidate-iteration growth: `3.9218009478672986x`;
- heldout direct scan growth from 2 to 8 frames: `4.0x`;
- heldout shared candidate-build scan growth: `1.0x`;
- heldout tiled CSR candidate-iteration growth: `3.915492957746479x`;
- 8-frame train tiled CSR storage ratio vs bitset: `0.6080729166666666x`;
- 8-frame heldout tiled CSR storage ratio vs bitset: `0.5651041666666666x`.

The paired report now includes
`world_foam_mps_shared_realray_csr_scaling_status=ok` in:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_9_paired_with_star_dynamic_heldout_pilot.json
```

## Sidecar Audits

Five read-only sidecars were used:

- Gate 2G design audit: confirmed Gate 2F was the right base and emphasized
  separating shared traversal accounting from per-frame replay/reduction work.
- Dynamic baseline audit: identified the paired 256px/16f STAR/dynamic report
  as the current same-video heldout baseline source, but confirmed World Foam
  has no matched quality row yet.
- STAR-UVT audit: identified the train-gain selected 256px/16f report as the
  fairer STAR comparator; the heldout-selected report is a ceiling, not a fair
  selector.
- CSR ABI audit: found that public CSR wrapper validation was shape-only. This
  was fixed in `torch_world_foam_lane2/ops.py` by validating row-index bounds,
  row-offset monotonicity, final offset, and candidate boundary id bounds before
  dispatch.
- Completion audit: confirmed the lane is still incomplete for the original
  objective because the World Foam side lacks a matched 256px/16f heldout
  train/eval artifact and full geometry/topology gradients.

## Remaining Blockers

- No same-split 256px/16f World Foam heldout training/eval row yet.
- No geometry, topology, site-position, site-weight, ray, or camera gradients.
- Tiled CSR reduces storage and candidate-build work, but compositor/reduction
  replay still has per-frame work; Gate 2G reports this explicitly.
- Dynamic splat baseline is available through STAR-UVT paired reports. The
  follow-up Gate 3 CSR 256px/16f train/eval artifact now provides a directly
  comparable fixed-geometry World Foam quality row, but not a full
  geometry/topology-gradient trainer.
