# Gate4 coeff16 sample-parallel promotion

## What changed

The earlier coeff16 reflection was based on the track-level VJP mode
`gate4-affine-candidate-coeff16-trackmse-fused-mse`. That kernel launches one
thread per track and serializes all frames inside the thread, so it preserved
the flat tape/storage result but still looked compute-bound across frame count.

The already-added sample-parallel coeff16 kernel,
`gate4-affine-candidate-coeff16-fused-mse`, is the better execution path. It
launches per rendered sample and exposes frame parallelism to MPS. Running the
same 2/4/8/16, render16, site24, 3-step/1-warmup ladder produced flat practical
timing while keeping the same storage and PSNR behavior.

## WorldFoam result

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3.json
```

| frame | total ms mean | backward ms mean | storage bytes | segments | train PSNR | heldout PSNR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.218 | 4.705 | 708604 | 84930 | 14.204 | 15.126 |
| 4 | 4.071 | 3.560 | 706044 | 84609 | 14.267 | 15.138 |
| 8 | 4.166 | 3.596 | 702756 | 84196 | 14.414 | 15.200 |
| 16 | 4.134 | 3.683 | 703020 | 84225 | 14.540 | 15.324 |

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3.json \
  --allow-contended
```

Verifier result:

- status `ok`
- benchmark environment `contended`
- total mean scale `0.792`
- backward mean scale `0.783`
- total median scale `0.870`
- backward median scale `0.861`
- resident storage scale `0.992`
- candidate count scale `0.992`

The verifier now defaults to this sample-parallel coeff16 artifact and requires
the coeff16/sample-mode flags by default. The older track mode remains
available, but the focused test now rejects it when the sample-parallel gate is
required.

## STAR comparison

Matched small-MPS comparison artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_star_uvt_vs_worldfoam_gate4_coeff16_sample_scale_mps_steps3_warm1.json
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py \
  --worldfoam-artifact research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3.json \
  --frame-counts 2,4,8,16 \
  --steps 3 \
  --warmup-steps 1 \
  --star-target-size 16 \
  --star-tube-count 224 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_star_uvt_vs_worldfoam_gate4_coeff16_sample_scale_mps_steps3_warm1.json
```

The comparison status is `ok`.

- WorldFoam total median scale: `0.870`; STAR total median scale: `0.749`.
- WorldFoam backward median scale: `0.861`; STAR backward median scale: `0.852`.
- STAR/WF total median ratio by frame: `1.52`, `1.46`, `1.36`, `1.31`. On
  this tiny speed gate, WorldFoam is faster on total step at every frame.
- STAR/WF backward median ratio by frame: `0.80`, `0.86`, `0.82`, `0.79`.
  STAR still has faster backward at every frame.

## Interpretation

This corrects the previous read. Gate4 coeff16 is not just a storage keeper: the
sample-parallel VJP makes it a real small-MPS speed keeper for this frozen-site
RGB-MSE gate. The important distinction is execution geometry, not math format:
track-level VJP serializes frames and hides the intended frame parallelism,
while sample-level VJP lets MPS schedule frame samples independently.

This still is not a full STAR replacement. STAR remains cleaner in backward
time and architecture because tubes are the primary primitive; WorldFoam still
replays candidate depths and scans owners per sample. The next shader fork
should target backward only: reduce candidate replay or owner scans while
keeping the sample-parallel launch shape.
