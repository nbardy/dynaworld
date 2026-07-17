# STAR UVT vs WorldFoam Gate4 Matched Scale Gate

## Scope

After fixing the WorldFoam Gate4 fused RGB-MSE affine-clear bug, I added and ran
a small matched scale gate:

```text
research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py
```

This is a small-MPS speed comparison only. It compares:

- STAR UVT direct-atomic/index-add source-video train-step timing at 32px,
  224 tubes, 2/4/8/16 frames.
- WorldFoam Gate4 fused-MSE moving-camera frozen-geometry train/eval artifact at
  32px, 12 sites, 2/4/8/16 frames.

It is not a quality/capacity parity proof and it is not a full WorldFoam trainer
claim.

## Commands

Unit test for the comparison summarizer:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale -v
```

Result: `Ran 2 tests ... OK`.

Matched gate:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py \
  --frame-counts 2,4,8,16 \
  --steps 20 \
  --warmup-steps 5 \
  --star-target-size 32 \
  --star-tube-count 224 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_scale_32px_224t_vs_12site_2_4_8_16.json
```

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_scale_32px_224t_vs_12site_2_4_8_16.json
```

Result: `status ok`, `failures []`.

## Result

Warm-step medians:

```text
frame  STAR total  WF total  STAR/WF total  STAR backward  WF backward  STAR/WF backward
2      5.719 ms    1.522 ms  3.758x         2.937 ms       1.212 ms     2.424x
4      6.464 ms    1.705 ms  3.792x         3.609 ms       1.387 ms     2.602x
8      4.834 ms    1.893 ms  2.554x         2.476 ms       1.559 ms     1.589x
16     7.604 ms    2.224 ms  3.420x         4.101 ms       1.877 ms     2.185x
```

First-to-last scale:

```text
STAR total median:       1.329x
STAR backward median:    1.396x
WorldFoam total median:  1.461x
WorldFoam backward med:  1.549x
WorldFoam mixed storage: 0.992x
WorldFoam explicit rays: 8.000x
```

## Interpretation

At this tiny matched 32px gate, the fixed WorldFoam fused-MSE warm step is
faster than STAR UVT direct-atomic. This is useful: the fused shader is not just
correct, it is competitive on the local warm-step timing surface.

The broader thesis is still not closed. WorldFoam still has explicit-ray
storage growing exactly with frame count and train/heldout tape-build wall time
growing from roughly `0.33/0.15s` at 2f to `1.03/0.46s` at 16f. STAR's
representation is still cleaner as a time-tube model; this gate only says the
current WorldFoam fused warm-step kernel is no longer obviously behind.

Next evidence needed before promoting any "competitive with STAR UVT" claim:

- repeat at a less tiny resolution/tube-cell budget
- include WorldFoam tape-build amortization policy explicitly
- use quality/capacity matched runs, not only warm-step timing
- keep STAR direct-atomic and WorldFoam fused-MSE both on their best stable
  practical path
